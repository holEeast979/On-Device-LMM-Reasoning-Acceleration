#!/usr/bin/env python3
"""
Prepare AudioCaps dataset for benchmark experiments.

AudioCaps contains ~51K audio clips with natural language captions.
This script downloads captions from HuggingFace, downloads audio files using audiocaps-download,
validates integrity, and generates manifest.csv.

Usage:
    python tools/prepare_audiocaps.py --out-root /root/autodl-tmp/data --max-samples 100
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
class AudioCapsSample:
    sample_id: str
    audiocap_id: str
    youtube_id: str
    start_time: float
    caption: str
    audio_path: str


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
    
    # Check if audiocaps-download is installed
    code, _, _ = _run_command([sys.executable, "-c", "import audiocaps_download"])
    deps["audiocaps_download"] = code == 0
    
    # Check ffmpeg (needed by audiocaps-download)
    code, _, _ = _run_command(["which", "ffmpeg"])
    deps["ffmpeg"] = code == 0
    
    # Check if we can import soundfile for validation
    code, _, _ = _run_command([sys.executable, "-c", "import soundfile"])
    deps["soundfile"] = code == 0
    
    return deps


def validate_audio_file(audio_path: str, min_duration_sec: float = 0.1) -> Tuple[bool, Dict[str, Any]]:
    """Validate audio file using soundfile."""
    try:
        import soundfile as sf
        
        if not os.path.exists(audio_path):
            return False, {"reason": "file_not_found"}
        
        info = sf.info(audio_path)
        dur = None
        try:
            dur = float(info.frames) / float(info.samplerate) if info.frames and info.samplerate else None
        except Exception:
            dur = None
            
        if dur is not None and dur < min_duration_sec:
            return False, {"reason": "duration_too_short", "duration": dur}
            
        return True, {
            "duration": dur,
            "samplerate": int(info.samplerate) if info.samplerate else None,
            "channels": int(info.channels) if info.channels else None,
            "frames": int(info.frames) if info.frames else None,
        }
    except Exception as e:
        return False, {"reason": "read_failed", "error": str(e)[:100]}


def load_audiocaps_from_hf(
    *,
    repo: str = "d0rj/audiocaps",
    split: Optional[str] = None,
    hf_endpoint: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load AudioCaps metadata from HuggingFace."""
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
                for split_name in ("train", "validation", "test"):
                    if split_name in ds:
                        ds = ds[split_name]
                        break
                else:
                    ds = list(ds.values())[0]
        
        samples = []
        for i, item in enumerate(ds):
            samples.append({
                "index": i,
                "audiocap_id": item.get("audiocap_id", ""),
                "youtube_id": item.get("youtube_id", ""),
                "start_time": item.get("start_time", 0),
                "caption": item.get("caption", ""),
                "raw_item": item
            })
        
        return samples
        
    except Exception as e:
        raise RuntimeError(f"Failed to load {repo}: {type(e).__name__}: {e}")


def download_audiocaps_files(
    *,
    samples: List[Dict[str, Any]],
    out_dir: str,
    format: str = "wav",
    quality: int = 0,  # Best quality for 90GB RAM
    n_jobs: int = 8,   # High concurrency for 90GB RAM
) -> Tuple[bool, str]:
    """Download AudioCaps audio files using audiocaps-download package (high quality)."""
    
    try:
        # Create a temporary CSV file with the subset of samples we want
        temp_dir = tempfile.mkdtemp()
        temp_csv = os.path.join(temp_dir, "temp_audiocaps.csv")
        
        print(f"Starting high-quality audio download with {n_jobs} parallel jobs...")
        
        # Prepare data for CSV
        csv_data = []
        for sample in samples:
            csv_data.append({
                "audiocap_id": sample["audiocap_id"],
                "youtube_id": sample["youtube_id"], 
                "start_time": sample["start_time"],
                "caption": sample["caption"]
            })
        
        # Write temporary CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(temp_csv, index=False)
        
        # Use Python code to call audiocaps_download directly
        download_script = f"""
import os
import sys
import pandas as pd
from audiocaps_download import Downloader

# Override the dataset loading to use our custom CSV
class CustomDownloader(Downloader):
    def load_dataset(self):
        # Load our custom subset instead of the full dataset
        df = pd.read_csv('{temp_csv}')
        
        # Convert to the expected format
        data = {{
            'train': df.to_dict('records'),
            'validation': [],
            'test': []
        }}
        return data

# Download using custom downloader with high quality settings
try:
    downloader = CustomDownloader(root_path='{out_dir}', n_jobs={n_jobs})
    downloader.download(format='{format}', quality={quality})
    print("SUCCESS: AudioCaps high-quality download completed")
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{e}}")
    sys.exit(1)
"""
        
        # Write and execute the download script
        script_path = os.path.join(temp_dir, "download_script.py")
        with open(script_path, "w") as f:
            f.write(download_script)
        
        code, out, err = _run_command([sys.executable, script_path])
        
        # Clean up temporary files
        try:
            os.unlink(temp_csv)
            os.unlink(script_path)
            os.rmdir(temp_dir)
        except Exception:
            pass
        
        if code != 0:
            return False, f"download_failed: {err[:200]}"
        
        if "SUCCESS" in out:
            return True, "success"
        else:
            return False, f"unexpected_output: {out[:200]}"
            
    except Exception as e:
        return False, f"exception: {type(e).__name__}: {str(e)[:100]}"


def find_downloaded_audio_files(out_dir: str, samples: List[Dict[str, Any]]) -> Dict[str, str]:
    """Find downloaded audio files and map them to sample IDs."""
    audio_files = {}
    
    # AudioCaps download creates files in train/validation/test subdirectories
    for subdir in ["train", "validation", "test"]:
        subdir_path = os.path.join(out_dir, subdir)
        if os.path.exists(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(('.wav', '.ogg', '.mp3', '.flac', '.m4a')):
                    # Extract audiocap_id from filename (usually the filename is the audiocap_id)
                    audiocap_id = os.path.splitext(filename)[0]
                    audio_files[audiocap_id] = os.path.join(subdir_path, filename)
    
    return audio_files


def prepare_audiocaps(
    *,
    out_root: str,
    max_samples: int = 100,
    seed: int = 42,
    validate: bool = True,
    hf_endpoint: Optional[str] = None,
    hf_token: Optional[str] = None,
    audio_format: str = "wav",
    n_jobs: int = 8,
) -> str:
    """Prepare AudioCaps dataset."""
    
    print("=== Preparing AudioCaps Dataset ===")
    
    # Check dependencies
    deps = _check_dependencies()
    missing_deps = [dep for dep, available in deps.items() if not available]
    if missing_deps:
        error_msg = f"Missing dependencies: {', '.join(missing_deps)}. (python={sys.executable})"
        if "audiocaps_download" in missing_deps:
            error_msg += " Install with: pip install audiocaps-download"
        if "ffmpeg" in missing_deps:
            error_msg += " Install with: sudo apt install ffmpeg"
        if "soundfile" in missing_deps:
            error_msg += " Install with: pip install soundfile"
        raise RuntimeError(error_msg)
    
    # Setup output directory
    out_dir = _ensure_dir(os.path.join(out_root, "audiocaps"))
    
    print(f"Output directory: {out_dir}")
    print(f"Max samples: {max_samples}")
    print(f"Audio format: {audio_format}")
    print(f"Validation: {'enabled' if validate else 'disabled'}")
    
    # Load metadata from HuggingFace
    print("\nLoading metadata from HuggingFace...")
    try:
        hf_samples = load_audiocaps_from_hf(hf_endpoint=hf_endpoint, hf_token=hf_token)
        print(f"Loaded {len(hf_samples)} samples")
    except Exception as e:
        raise RuntimeError(f"Failed to load from HuggingFace: {e}")
    
    # Sample subset if requested
    if max_samples > 0 and max_samples < len(hf_samples):
        random.seed(seed)
        hf_samples = random.sample(hf_samples, max_samples)
        print(f"Selected {len(hf_samples)} samples")
    
    # Download audio files using audiocaps-download
    print(f"\n🎧 Downloading audio files with high quality...")
    success, status = download_audiocaps_files(
        samples=hf_samples,
        out_dir=out_dir,
        format=audio_format,
        quality=0,  # Best quality
        n_jobs=n_jobs
    )
    
    if not success:
        print(f"Download failed: {status}")
        return out_dir
    
    print("Audio download completed, locating files...")
    
    # Find downloaded audio files
    audio_files = find_downloaded_audio_files(out_dir, hf_samples)
    print(f"Found {len(audio_files)} downloaded audio files")
    
    # Process samples and validate
    prepared_samples: List[AudioCapsSample] = []
    stats = {
        "attempted": 0,
        "prepared": 0, 
        "skipped": 0,
        "skip_reasons": {}
    }
    
    for sample in hf_samples:
        stats["attempted"] += 1
        audiocap_id = str(sample["audiocap_id"])
        youtube_id = sample["youtube_id"]
        start_time = sample["start_time"]
        caption = sample["caption"]
        
        if not audiocap_id or not caption:
            reason = "missing_data"
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            stats["skipped"] += 1
            continue
        
        # Find corresponding audio file
        if audiocap_id not in audio_files:
            reason = "audio_not_found"
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            stats["skipped"] += 1
            continue
        
        audio_path = audio_files[audiocap_id]
        
        # Validate audio if requested
        if validate:
            valid, info = validate_audio_file(audio_path)
            if not valid:
                reason = f"validation_{info.get('reason', 'unknown')}"
                stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                stats["skipped"] += 1
                print(f"[WARN] Validation failed for {audiocap_id}: {info.get('reason')}")
                continue
        
        # Add to prepared samples
        sample_id = f"audiocaps_{audiocap_id}"
        prepared_samples.append(AudioCapsSample(
            sample_id=sample_id,
            audiocap_id=audiocap_id,
            youtube_id=youtube_id,
            start_time=float(start_time),
            caption=_safe_str(caption),
            audio_path=audio_path
        ))
        stats["prepared"] += 1
    
    # Generate manifest.csv
    if prepared_samples:
        df_data = []
        for sample in prepared_samples:
            df_data.append({
                "sample_id": sample.sample_id,
                "video_path": "",  # AudioCaps is audio captioning
                "audio_path": sample.audio_path,
                "question": "Describe what you hear.",
                "answer": sample.caption,
                "audiocap_id": sample.audiocap_id,
                "youtube_id": sample.youtube_id,
                "start_time": sample.start_time,
                "source_repo": "d0rj/audiocaps"
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
        "dataset": "audiocaps",
        "max_samples": max_samples,
        "seed": seed,
        "audio_format": audio_format,
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
    print(f"\n✅ AudioCaps preparation completed")
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
        description="Prepare AudioCaps dataset for benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/prepare_audiocaps.py --max-samples 50
    python tools/prepare_audiocaps.py --out-root /tmp/data --validate --audio-format mp3
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
        help="Enable audio validation using soundfile"
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
    parser.add_argument(
        "--audio-format",
        type=str,
        default="wav",
        choices=["wav", "mp3", "ogg", "flac", "m4a"],
        help="Audio format for downloads (default: wav)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel download jobs (default: 8 for high-RAM environment)"
    )
    
    args = parser.parse_args()
    
    try:
        out_dir = prepare_audiocaps(
            out_root=args.out_root,
            max_samples=args.max_samples,
            seed=args.seed,
            validate=args.validate,
            hf_endpoint=args.hf_endpoint,
            hf_token=(args.hf_token or None),
            audio_format=args.audio_format,
            n_jobs=args.n_jobs
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
