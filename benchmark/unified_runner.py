from __future__ import annotations

import argparse
import gc
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import torch

from benchmark.runner import BenchmarkRunner


Row = Dict[str, Any]
RunOnceReturn = Union[Row, List[Row], None]


def _to_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)


class UnifiedRunner:
    def __init__(
        self,
        *,
        base: BenchmarkRunner,
        spec_name: str,
        out_dir: str,
        args: argparse.Namespace,
    ):
        self.base = base
        self.spec_name = str(spec_name)
        self.out_dir = str(out_dir)
        self.args = args
        self.rows: List[Row] = []

        os.makedirs(self.out_dir, exist_ok=True)

        args_dict = {k: _to_jsonable(v) for k, v in vars(args).items() if str(k) != "_spec_run"}
        self.base.write_json(
            os.path.join(self.out_dir, "meta.json"),
            {
                "spec": self.spec_name,
                "args": args_dict,
                "env": self.base.env_info(),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    @staticmethod
    def resolve_media_path(manifest_path: str, path: Optional[str]) -> str:
        if path is None:
            return ""
        p = str(path).strip()
        if not p:
            return ""
        if os.path.isabs(p):
            return p
        base = os.path.dirname(os.path.abspath(str(manifest_path)))
        return os.path.normpath(os.path.join(base, p))

    def run(
        self,
        *,
        cases: List[Row],
        repeats: int,
        warmup: int,
        run_once: Callable[[Row], RunOnceReturn],
        clear_cache: bool = True,
    ) -> pd.DataFrame:
        rows: List[Row] = []

        t0 = time.perf_counter()
        total_cases = int(len(cases))

        for case_i, case in enumerate(cases):
            if case_i % 10 == 0:
                elapsed_s = float(time.perf_counter() - t0)
                print(f"[{self.spec_name}] progress: case {case_i}/{total_cases} elapsed={elapsed_s:.1f}s", flush=True)
            for r in range(int(max(1, repeats))):
                do_record = int(r) >= int(max(0, warmup))

                if clear_cache:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                try:
                    out = run_once(case)
                except Exception as e:
                    out = {"error": f"run_once_failed: {type(e).__name__}: {e}"}

                if out is None:
                    continue

                out_list = out if isinstance(out, list) else [out]
                if not do_record:
                    continue

                for o in out_list:
                    if o is None:
                        continue
                    row = {
                        "spec": self.spec_name,
                        "case": int(case_i),
                        "repeat": int(r),
                        **case,
                        **o,
                    }
                    rows.append(row)

        self.rows = rows
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.out_dir, "results.csv"), index=False)
        return df
