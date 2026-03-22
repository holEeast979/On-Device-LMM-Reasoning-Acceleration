#!/usr/bin/env python3
"""
Prepare Video-MME dataset for benchmark experiments.

Video-MME HF split provides metadata (e.g. url/videoID/question/options), not embedded video files.
This script downloads videos from YouTube via yt-dlp, validates integrity, and generates manifest.csv.

Usage:
    python tools/prepare_video_mme.py --out-root /root/autodl-tmp/data --max-samples 100
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class VideoMMESample:
    sample_id: str
    video_id: str
    question_id: str
    domain: str
    sub_category: str
    task_type: str
    question: str
    options: List[str]
    answer: str
    video_path: str


def _ensure_dir(p: str) -> str:
    """Create directory if it doesn't exist."""
    os.makedirs(p, exist_ok=True)
    return p


def _safe_str(x: Any) -> str:
    """Convert to string safely."""
    if x is None:
        return ""
    return str(x)


def _run_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run command and return exit code, stdout, stderr."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(p.returncode), str(p.stdout), str(p.stderr)


def _check_dependencies() -> Dict[str, bool]:
    """Check if required tools are available."""
    deps = {}
    
    # Check ffprobe for validation
    code, _, _ = _run_command(["which", "ffprobe"])
    deps["ffprobe"] = code == 0

    # Check yt-dlp for downloading
    code, _, _ = _run_command(["which", "yt-dlp"])
    deps["yt_dlp"] = code == 0

    # Check ffmpeg for merging/remux
    code, _, _ = _run_command(["which", "ffmpeg"])
    deps["ffmpeg"] = code == 0
    
    return deps


def download_youtube_video(*, url: str, out_video_path: str, timeout_sec: int = 600) -> Tuple[bool, str]:
    """Download a YouTube video to mp4 using yt-dlp (high quality for 90GB RAM)."""
    if os.path.exists(out_video_path):
        return True, "already_exists"
    
    print(f"  ⬇️  Downloading: {url}")
    out_tpl = os.path.splitext(out_video_path)[0] + ".%(ext)s"
    
    # High-quality settings for 90GB RAM environment
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--socket-timeout", "60",
        "--retries", "3",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--concurrent-fragments", "4",  # Higher concurrency for faster downloads
        "-o", out_tpl,
        url,
    ]
    
    # Use timeout to prevent hanging
    import signal
    def timeout_handler(signum, frame):
        raise TimeoutError("Download timeout")
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)
        
        print(f"  ⏳ Running: {' '.join(cmd[:8])}...")
        code, out, err = _run_command(cmd)
        signal.alarm(0)  # Cancel timeout
        
        if code != 0:
            return False, (err or out)[:200]
        if not os.path.exists(out_video_path):
            return False, "output_not_found"
        
        print(f"  ✅ Downloaded: {os.path.basename(out_video_path)}")
        return True, "success"
        
    except TimeoutError:
        return False, f"timeout_after_{timeout_sec}s"
    except Exception as e:
        return False, f"exception: {str(e)[:100]}"
    finally:
        signal.alarm(0)  # Ensure timeout is cancelled


def validate_video_file(video_path: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    
    code, out, err = _run_command(cmd)
    if code != 0:
        return False, {"reason": "ffprobe_failed", "error": err[:100]}
    
    try:
        info = json.loads(out)
    except Exception as e:
        return False, {"reason": "json_parse_failed", "error": str(e)[:100]}
    
    streams = info.get("streams", [])
    format_info = info.get("format", {})
    
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    
    try:
        duration = float(format_info.get("duration", 0))
    except (ValueError, TypeError):
        duration = 0
    
    if not has_video:
        return False, {"reason": "no_video_stream"}
    
    if duration <= 0:
        return False, {"reason": "zero_duration", "duration": duration}
    
    return True, {
        "duration": duration,
        "has_video": has_video,
        "has_audio": has_audio,
        "streams": len(streams)
    }


def _extract_video_content(video_obj: Any, out_video_path: str) -> bool:
    """Extract video content from HuggingFace video object to file."""
    if video_obj is None:
        return False
    
    # Check if it's a file path
    if isinstance(video_obj, str):
        if os.path.exists(video_obj):
            if not os.path.exists(out_video_path):
                import shutil
                shutil.copy2(video_obj, out_video_path)
            return True
        return False
    
    # Check if it's a dict with path
    if isinstance(video_obj, dict):
        path = video_obj.get("path")
        if path and os.path.exists(path):
            if not os.path.exists(out_video_path):
                import shutil
                shutil.copy2(path, out_video_path)
            return True
        
        # Check if it's a dict with bytes
        video_bytes = video_obj.get("bytes")
        if video_bytes:
            if not os.path.exists(out_video_path):
                with open(out_video_path, "wb") as f:
                    f.write(video_bytes)
            return True
    
    # Check if it's bytes directly
    if isinstance(video_obj, (bytes, bytearray)):
        if not os.path.exists(out_video_path):
            with open(out_video_path, "wb") as f:
                f.write(bytes(video_obj))
        return True
    
    return False


def load_video_mme_from_hf(
    *,
    repo: str = "lmms-lab/Video-MME",
    split: Optional[str] = None,
    hf_endpoint: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load Video-MME dataset from HuggingFace."""
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    
    try:
        from datasets import load_dataset

        # Try new 'token' parameter first, fallback to 'use_auth_token' for older versions
        if split:
            if hf_token:
                try:
                    ds = load_dataset(repo, split=split, token=hf_token)
                except TypeError:
                    ds = load_dataset(repo, split=split, use_auth_token=hf_token)
            else:
                ds = load_dataset(repo, split=split)
        else:
            if hf_token:
                try:
                    ds = load_dataset(repo, token=hf_token)
                except TypeError:
                    ds = load_dataset(repo, use_auth_token=hf_token)
            else:
                ds = load_dataset(repo)
            if isinstance(ds, dict):
                # Try common splits
                for split_name in ("test", "validation", "train"):
                    if split_name in ds:
                        ds = ds[split_name]
                        break
                else:
                    ds = list(ds.values())[0]
        
        samples = []
        for i, item in enumerate(ds):
            samples.append({
                "index": i,
                "video_id": item.get("video_id", ""),
                "url": item.get("url", ""),
                "videoID": item.get("videoID", ""),
                "question_id": item.get("question_id", ""),
                "domain": item.get("domain", ""),
                "sub_category": item.get("sub_category", ""),
                "task_type": item.get("task_type", ""),
                "question": item.get("question", ""),
                "options": item.get("options", []),
                "answer": item.get("answer", ""),
                "raw_item": item
            })
        
        return samples
        
    except Exception as e:
        raise RuntimeError(f"Failed to load {repo}: {type(e).__name__}: {e}")


def prepare_video_mme(
    *,
    out_root: str,
    max_samples: int = 100,
    seed: int = 42,
    validate: bool = True,
    hf_endpoint: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """Prepare Video-MME dataset."""
    
    print("=== Preparing Video-MME Dataset ===")
    
    # Check dependencies
    deps = _check_dependencies()
    missing_deps = [dep for dep, available in deps.items() if not available]
    if missing_deps:
        error_msg = f"Missing dependencies: {', '.join(missing_deps)}."
        if "ffprobe" in missing_deps or "ffmpeg" in missing_deps:
            error_msg += " Install with: sudo apt install ffmpeg"
        if "yt_dlp" in missing_deps:
            error_msg += " Install with: pip install yt-dlp"
        raise RuntimeError(error_msg)
    
    # Setup output directory
    out_dir = _ensure_dir(os.path.join(out_root, "video_mme"))
    videos_dir = _ensure_dir(os.path.join(out_dir, "videos"))
    
    print(f"Output directory: {out_dir}")
    print(f"Max samples: {max_samples}")
    print(f"Validation: {'enabled' if validate else 'disabled'}")
    
    # Load dataset from HuggingFace
    print("\nLoading Video-MME from HuggingFace...")
    print("⚠️  This may take a while due to the large dataset size (101GB)")
    try:
        hf_samples = load_video_mme_from_hf(hf_endpoint=hf_endpoint, hf_token=hf_token)
        print(f"Loaded {len(hf_samples)} samples")
    except Exception as e:
        raise RuntimeError(f"Failed to load from HuggingFace: {e} (python={sys.executable}, endpoint={os.environ.get('HF_ENDPOINT','')})")
    
    # Sample subset if requested
    if max_samples > 0 and max_samples < len(hf_samples):
        random.seed(seed)
        hf_samples = random.sample(hf_samples, max_samples)
        print(f"Selected {len(hf_samples)} samples")
    
    # Process samples
    prepared_samples: List[VideoMMESample] = []
    stats = {
        "attempted": 0,
        "prepared": 0, 
        "skipped": 0,
        "skip_reasons": {}
    }
    
    for sample in hf_samples:
        stats["attempted"] += 1
        video_id = sample["video_id"]
        url = sample.get("url", "")
        videoID = sample.get("videoID", "")
        question_id = sample["question_id"]
        domain = sample["domain"]
        sub_category = sample["sub_category"]
        task_type = sample["task_type"]
        question = sample["question"]
        options = sample["options"]
        answer = sample["answer"]
        
        if not video_id or not question:
            reason = "missing_data"
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            stats["skipped"] += 1
            continue
        
        # Determine download URL
        dl_url = url.strip() if isinstance(url, str) else ""
        if not dl_url and isinstance(videoID, str) and videoID.strip():
            dl_url = f"https://www.youtube.com/watch?v={videoID.strip()}"
        if not dl_url:
            reason = "missing_video_url"
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            stats["skipped"] += 1
            continue

        # Setup paths: cache by videoID/video_id (shared across multiple Qs)
        cache_key = (videoID or video_id).strip() if isinstance((videoID or video_id), str) else str(video_id)
        cached_video_path = os.path.join(videos_dir, f"{cache_key}.mp4")

        # Each QA still has its own sample_id
        sample_id = f"videomme_{video_id}_{question_id}" if question_id else f"videomme_{video_id}_{stats['attempted']}"

        print(f"\n🎬 Processing {sample_id} (attempt {stats['attempted']})")
        
        ok, msg = download_youtube_video(url=dl_url, out_video_path=cached_video_path, timeout_sec=300)
        if not ok:
            reason = "video_download_failed"
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            stats["skipped"] += 1
            print(f"[WARN] Could not download video for {sample_id}: {msg}")
            continue
        
        # Validate video if requested
        if validate:
            valid, info = validate_video_file(cached_video_path)
            if not valid:
                reason = f"validation_{info.get('reason', 'unknown')}"
                stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                stats["skipped"] += 1
                print(f"[WARN] Validation failed for {sample_id}: {info.get('reason')}")
                # Remove invalid file
                try:
                    os.unlink(cached_video_path)
                except Exception:
                    pass
                continue
        
        # Format question and answer for Video-MME multiple choice
        if isinstance(options, list) and len(options) > 0:
            options_str = "\n".join([str(opt) for opt in options])
            formatted_question = f"{question}\n{options_str}"
        else:
            formatted_question = question
        
        # Add to prepared samples
        prepared_samples.append(VideoMMESample(
            sample_id=sample_id,
            video_id=video_id,
            question_id=question_id,
            domain=domain,
            sub_category=sub_category,
            task_type=task_type,
            question=formatted_question,
            options=options if isinstance(options, list) else [],
            answer=_safe_str(answer),
            video_path=cached_video_path
        ))
        stats["prepared"] += 1
        
        if stats["prepared"] % 10 == 0:
            print(f"✅ Prepared {stats['prepared']} samples...")
    
    # Generate manifest.csv
    if prepared_samples:
        df_data = []
        for sample in prepared_samples:
            df_data.append({
                "sample_id": sample.sample_id,
                "video_path": sample.video_path,
                "audio_path": "",  # Video-MME is video QA
                "question": sample.question,
                "answer": sample.answer,
                "video_id": sample.video_id,
                "question_id": sample.question_id,
                "domain": sample.domain,
                "sub_category": sample.sub_category,
                "task_type": sample.task_type,
                "options": json.dumps(sample.options) if sample.options else "",
                "source_repo": "lmms-lab/Video-MME"
            })
        
        df = pd.DataFrame(df_data)
        manifest_csv = os.path.join(out_dir, "manifest.csv")
        df.to_csv(manifest_csv, index=False)
        
        manifest_json = os.path.join(out_dir, "manifest.json")
        with open(manifest_json, "w", encoding="utf-8") as f:
            json.dump(df_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 Generated manifest: {manifest_csv}")
    
    # Save metadata and statistics
    meta = {
        "dataset": "video_mme",
        "max_samples": max_samples,
        "seed": seed,
        "validation_enabled": validate,
        "n_attempted": stats["attempted"],
        "n_prepared": stats["prepared"],
        "n_skipped": stats["skipped"],
        "skip_reasons": stats["skip_reasons"],
        "dependencies": deps
    }
    
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n✅ Video-MME preparation completed")
    print(f"   Attempted: {stats['attempted']}")
    print(f"   Prepared: {stats['prepared']}")
    print(f"   Skipped: {stats['skipped']}")
    if stats["skip_reasons"]:
        print("   Skip reasons:")
        for reason, count in stats["skip_reasons"].items():
            print(f"     {reason}: {count}")
    
    return out_dir


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare Video-MME dataset for benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/prepare_video_mme.py --max-samples 50
    python tools/prepare_video_mme.py --out-root /tmp/data --validate
        """
    )
    
    parser.add_argument(
        "--out-root",
        type=str,
        default="/root/autodl-tmp/data",
        help="Root directory for output (default: /root/autodl-tmp/data)"
    )
    parser.add_argument(
        "--max-samples",
        type=int, 
        default=100,
        help="Maximum samples to prepare (default: 100, 0 for all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable video validation using ffprobe"
    )
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default="https://huggingface.co",
        help="HuggingFace endpoint (default: https://huggingface.co)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "",
        help="HuggingFace token (or set env HF_TOKEN/HUGGINGFACE_HUB_TOKEN)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.hf_endpoint:
            os.environ["HF_ENDPOINT"] = args.hf_endpoint
        if args.hf_token:
            os.environ["HF_TOKEN"] = args.hf_token
        out_dir = prepare_video_mme(
            out_root=args.out_root,
            max_samples=args.max_samples,
            seed=args.seed,
            validate=args.validate,
            hf_endpoint=args.hf_endpoint,
            hf_token=(args.hf_token or None),
        )
        print(f"\n🎉 Success! Dataset prepared in: {out_dir}")
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
