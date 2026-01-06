#!/usr/bin/env python3

import argparse
import csv
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}


def _ffprobe(cmd: Sequence[str]) -> str:
    return subprocess.check_output(list(cmd), stderr=subprocess.STDOUT, text=True)


def probe_has_audio(media_path: str) -> Optional[bool]:
    """Returns True/False if ffprobe is available, else None."""
    try:
        out = _ffprobe(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(media_path),
            ]
        )
    except Exception:
        return None

    types = {ln.strip() for ln in out.splitlines() if ln.strip()}
    return "audio" in types


def probe_duration_s(media_path: str) -> Optional[float]:
    try:
        out = _ffprobe(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ]
        ).strip()
        if not out:
            return None
        return float(out)
    except Exception:
        return None


@dataclass
class Item:
    sample_id: str
    video_path: str
    question: str
    has_audio: Optional[bool]
    duration_s: Optional[float]


def iter_videos(video_dir: Path) -> Iterable[Path]:
    for p in sorted(video_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def write_manifest_csv(items: List[Item], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "video_path",
                "question",
                "has_audio",
                "duration_s",
            ],
        )
        w.writeheader()
        for it in items:
            w.writerow(
                {
                    "sample_id": it.sample_id,
                    "video_path": it.video_path,
                    "question": it.question,
                    "has_audio": ("" if it.has_audio is None else str(bool(it.has_audio)).lower()),
                    "duration_s": ("" if it.duration_s is None else f"{float(it.duration_s):.6f}"),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build manifest.csv from a local video directory.")
    ap.add_argument("--video-dir", required=True, help="Directory containing videos")
    ap.add_argument("--out", required=True, help="Output manifest.csv path")
    ap.add_argument(
        "--question",
        default="Describe what you see and hear in this video.",
        help="Default question for all samples",
    )
    ap.add_argument("--limit", type=int, default=0, help="If >0, randomly sample N videos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-audio", action="store_true", help="Skip videos without audio")
    ap.add_argument("--absolute-path", action="store_true", help="Write absolute video paths")

    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    out_path = Path(args.out)

    if not video_dir.exists():
        raise SystemExit(f"video-dir not found: {video_dir}")

    vids = list(iter_videos(video_dir))
    if not vids:
        raise SystemExit(f"No videos found under: {video_dir}")

    if int(args.limit) > 0 and int(args.limit) < len(vids):
        random.seed(int(args.seed))
        vids = random.sample(vids, int(args.limit))

    items: List[Item] = []
    skipped_no_audio = 0
    for p in vids:
        has_audio = probe_has_audio(str(p))
        if args.require_audio and has_audio is False:
            skipped_no_audio += 1
            continue
        duration_s = probe_duration_s(str(p))
        sample_id = p.stem
        video_path = str(p.resolve()) if args.absolute_path else str(p)
        items.append(
            Item(
                sample_id=str(sample_id),
                video_path=video_path,
                question=str(args.question),
                has_audio=has_audio,
                duration_s=duration_s,
            )
        )

    if not items:
        raise SystemExit("No valid samples after filtering.")

    write_manifest_csv(items, out_path)
    print(f"Wrote {len(items)} samples -> {out_path}")
    if args.require_audio:
        print(f"Skipped no-audio videos: {skipped_no_audio}")


if __name__ == "__main__":
    main()
