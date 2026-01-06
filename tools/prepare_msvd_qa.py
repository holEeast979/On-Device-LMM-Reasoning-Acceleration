#!/usr/bin/env python3
"""
Prepare MSVD-QA dataset for benchmark experiments.

MSVD-QA is based on Microsoft Research Video Description Corpus with YouTube videos.
This script downloads QA pairs from HuggingFace, downloads videos from YouTube,
clips them to appropriate segments, validates integrity, and generates manifest.csv.

Usage:
    python tools/prepare_msvd_qa.py --out-root /root/autodl-tmp/data --max-samples 100
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
class MSVDSample:
    sample_id: str
    youtube_id: str
    start_time: float
    end_time: float
    question: str
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
    
    # Check ffmpeg
    code, _, _ = _run_command(["which", "ffmpeg"])
    deps["ffmpeg"] = code == 0
    
    # Check ffprobe  
    code, _, _ = _run_command(["which", "ffprobe"])
    deps["ffprobe"] = code == 0
    
    # Check yt-dlp
    code, _, _ = _run_command(["which", "yt-dlp"])
    deps["yt-dlp"] = code == 0
    
    return deps


def download_video_segment(
    *,
    youtube_id: str,
    start_time: float,
    end_time: float,
    output_path: str,
    max_retries: int = 3,
) -> Tuple[bool, str]:
    """Download and clip YouTube video segment using yt-dlp and ffmpeg."""
    if os.path.exists(output_path):
        return True, "already_exists"
    
    duration = end_time - start_time
    if duration <= 0:
        return False, "invalid_duration"
    
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    
    for attempt in range(max_retries):
        try:
            # Use yt-dlp to get video info first
            cmd_info = [
                "yt-dlp", 
                "--no-download",
                "--print", "duration",
                "--print", "title", 
                url
            ]
            
            code, out, err = _run_command(cmd_info)
            if code != 0:
                if attempt < max_retries - 1:
                    continue
                return False, f"info_failed: {err[:100]}"
            
            lines = out.strip().split('\n')
            if len(lines) < 2:
                return False, "info_parse_failed"
            
            try:
                video_duration = float(lines[0])
                video_title = lines[1]
            except (ValueError, IndexError):
                return False, "info_parse_error"
            
            # Check if requested segment is within video duration
            if end_time > video_duration:
                return False, f"segment_exceeds_duration: {end_time} > {video_duration}"
            
            # Create temp file for full video download
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Download video using yt-dlp (high quality settings for 90GB RAM)
                cmd_download = [
                    "yt-dlp",
                    "--format", "best[height<=1080][ext=mp4]/best[height<=720][ext=mp4]/best[ext=mp4]/best",
                    "--concurrent-fragments", "4",
                    "--socket-timeout", "60",
                    "--retries", "3",
                    "--output", tmp_path,
                    url
                ]
                
                code, _, err = _run_command(cmd_download)
                if code != 0:
                    return False, f"download_failed: {err[:100]}"
                
                if not os.path.exists(tmp_path):
                    return False, "download_file_not_found"
                
                # Clip video segment using ffmpeg
                cmd_clip = [
                    "ffmpeg",
                    "-y",  # Overwrite output
                    "-i", tmp_path,
                    "-ss", str(start_time),
                    "-t", str(duration),
                    "-c", "copy",  # Copy without re-encoding when possible
                    "-avoid_negative_ts", "make_zero",
                    output_path
                ]
                
                code, _, err = _run_command(cmd_clip)
                if code != 0:
                    return False, f"clip_failed: {err[:100]}"
                
                if not os.path.exists(output_path):
                    return False, "clip_output_not_found"
                
                return True, "success"
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                        
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return False, f"exception: {type(e).__name__}: {str(e)[:100]}"
    
    return False, "max_retries_exceeded"


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


def load_msvd_qa_from_hf(
    *,
    repo: str = "morpheushoc/msvd-qa",
    split: Optional[str] = None,
    hf_endpoint: Optional[str] = None,
    hf_token: Optional[str] = None,
    max_items: int = 0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load MSVD-QA dataset from HuggingFace."""
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    
    try:
        from datasets import load_dataset

        # Use streaming to avoid downloading multi-GB parquet shards when only a small subset is needed.
        chosen_split = split or "train"
        
        # Try new 'token' parameter first, fallback to 'use_auth_token' for older versions
        if hf_token:
            try:
                ds = load_dataset(repo, split=chosen_split, streaming=True, token=hf_token)
            except TypeError:
                ds = load_dataset(repo, split=chosen_split, streaming=True, use_auth_token=hf_token)
        else:
            ds = load_dataset(repo, split=chosen_split, streaming=True)

        k = int(max_items) if max_items and int(max_items) > 0 else 0
        rng = random.Random(int(seed))

        samples: List[Dict[str, Any]] = []
        for i, item in enumerate(ds):
            row = {
                "index": i,
                "video_id": item.get("video_id", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "raw_item": item,
            }

            if k <= 0:
                samples.append(row)
                continue

            # Reservoir sampling to get an (approximately) uniform subset without knowing dataset length.
            if len(samples) < k:
                samples.append(row)
            else:
                j = rng.randint(0, i)
                if j < k:
                    samples[j] = row

            # Early stop with larger sample size for 90GB RAM environment
            if i >= max(50_000, k * 500):  # Much larger for comprehensive sampling
                break

        return samples
        
    except Exception as e:
        raise RuntimeError(f"Failed to load {repo}: {type(e).__name__}: {e}")


def parse_msvd_video_id(video_id: str) -> Tuple[str, float, float]:
    """Parse MSVD video ID to extract YouTube ID and timestamps.
    
    Format: {youtube_id}_{start_time}_{end_time}
    Example: "klteYv1Uv9A_33_41" -> ("klteYv1Uv9A", 33.0, 41.0)
    """
    parts = video_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid video ID format: {video_id}")
    
    youtube_id = parts[0]
    try:
        start_time = float(parts[1])
        end_time = float(parts[2])
    except ValueError:
        raise ValueError(f"Invalid timestamps in video ID: {video_id}")
    
    return youtube_id, start_time, end_time


def prepare_msvd_qa(
    *,
    out_root: str,
    max_samples: int = 100,
    seed: int = 42,
    validate: bool = True,
    hf_endpoint: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """Prepare MSVD-QA dataset."""
    
    print("=== Preparing MSVD-QA Dataset ===")
    
    # Check dependencies
    deps = _check_dependencies()
    missing_deps = [dep for dep, available in deps.items() if not available]
    if missing_deps:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing_deps)}. "
                          "Please install: sudo apt install ffmpeg && pip install yt-dlp")
    
    # Setup output directory
    out_dir = _ensure_dir(os.path.join(out_root, "msvd_qa"))
    videos_dir = _ensure_dir(os.path.join(out_dir, "videos"))
    
    print(f"Output directory: {out_dir}")
    print(f"Max samples: {max_samples}")
    print(f"Validation: {'enabled' if validate else 'disabled'}")
    
    # Load QA pairs from HuggingFace
    print("\nLoading QA pairs from HuggingFace (streaming)...")
    try:
        # Load many more candidates for 90GB environment (comprehensive coverage)
        target = int(max_samples) if int(max_samples) > 0 else 0
        candidates = target * 10 if target > 0 else 1000  # High multiplier for better success rate
        hf_samples = load_msvd_qa_from_hf(
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            max_items=candidates,
            seed=seed,
        )
        print(f"Loaded {len(hf_samples)} QA pairs")
    except Exception as e:
        raise RuntimeError(f"Failed to load from HuggingFace: {e} (python={sys.executable}, endpoint={os.environ.get('HF_ENDPOINT','')})")

    # Keep extra candidates (target*3) because some YouTube videos may be unavailable.
    if int(max_samples) > 0:
        print(f"Will try up to {len(hf_samples)} candidates to prepare {int(max_samples)} samples")
    
    # Process samples
    prepared_samples: List[MSVDSample] = []
    stats = {
        "attempted": 0,
        "prepared": 0, 
        "skipped": 0,
        "skip_reasons": {}
    }
    
    for sample in hf_samples:
        if int(max_samples) > 0 and stats["prepared"] >= int(max_samples):
            break
        stats["attempted"] += 1
        video_id = sample["video_id"]
        question = sample["question"] 
        answer = sample["answer"]
        
        print(f"\n🎥 Processing MSVD sample {stats['attempted']}: {video_id}")
        
        if not video_id or not question:
            reason = "missing_data"
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            stats["skipped"] += 1
            continue
        
        # Parse video ID to get YouTube ID and timing
        youtube_id, start_time, end_time = parse_msvd_video_id(video_id)
        if not youtube_id:
            reason = "invalid_video_id"
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            stats["skipped"] += 1
            print(f"  ❌ Invalid video ID format: {video_id}")
            continue
        
        print(f"  📺 YouTube ID: {youtube_id}, segment: {start_time}-{end_time}s")
        
        # Setup video output path
        sample_id = f"msvd_{video_id}"
        video_path = os.path.join(videos_dir, f"{sample_id}.mp4")
        
        # Download video segment
        print(f"Downloading {video_id} (YouTube: {youtube_id}, {start_time}-{end_time}s)...")
        success, status = download_video_segment(
            youtube_id=youtube_id,
            start_time=start_time,
            end_time=end_time,
            output_path=video_path
        )
        
        if not success:
            reason = f"download_{status.split(':')[0]}"
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            stats["skipped"] += 1
            print(f"[WARN] Download failed for {video_id}: {status}")
            continue
        
        # Validate video if requested
        if validate:
            valid, info = validate_video_file(video_path)
            if not valid:
                reason = f"validation_{info.get('reason', 'unknown')}"
                stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                stats["skipped"] += 1
                print(f"[WARN] Validation failed for {video_id}: {info.get('reason')}")
                # Remove invalid file
                try:
                    os.unlink(video_path)
                except Exception:
                    pass
                continue
        
        # Add to prepared samples
        prepared_samples.append(MSVDSample(
            sample_id=sample_id,
            youtube_id=youtube_id,
            start_time=start_time,
            end_time=end_time,
            question=_safe_str(question),
            answer=_safe_str(answer),
            video_path=video_path
        ))
        stats["prepared"] += 1
        
        print(f"✅ Prepared {sample_id}")
    
    # Generate manifest.csv
    if prepared_samples:
        df_data = []
        for sample in prepared_samples:
            df_data.append({
                "sample_id": sample.sample_id,
                "video_path": sample.video_path,
                "audio_path": "",  # MSVD-QA is video QA
                "question": sample.question,
                "answer": sample.answer,
                "youtube_id": sample.youtube_id,
                "start_time": sample.start_time,
                "end_time": sample.end_time,
                "source_repo": "morpheushoc/msvd-qa"
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
        "dataset": "msvd_qa",
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
    print(f"\n✅ MSVD-QA preparation completed")
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
        description="Prepare MSVD-QA dataset for benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/prepare_msvd_qa.py --max-samples 50
    python tools/prepare_msvd_qa.py --out-root /tmp/data --validate
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
        out_dir = prepare_msvd_qa(
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
