#!/usr/bin/env python3
"""
Generate manifest.csv files from Hugging Face Arrow datasets.
Run this script once before running experiments.

Usage:
    python scripts/generate_manifests.py
"""

import os
from datasets import load_from_disk
import pandas as pd

# Base paths
DATA_ROOT = "/root/autodl-tmp/data"

def generate_vqav2_manifest():
    """Generate VQAv2 manifest: image_path,question,answer"""
    print("[1/3] Generating VQAv2 manifest...")
    dataset = load_from_disk(os.path.join(DATA_ROOT, "VQAv2_subset"))
    
    rows = []
    for idx, sample in enumerate(dataset):
        # VQAv2 structure: {'image': PIL.Image, 'question': str, 'answers': List[str]}
        # Save image to disk if needed, or use the image object directly
        image_obj = sample.get('image')
        question = sample.get('question', '')
        answers = sample.get('answers', [])
        
        # Take the most common answer (VQAv2 has multiple answers)
        # answers is a list of dicts: [{'answer': 'down', 'answer_confidence': 'yes', 'answer_id': 1}, ...]
        if answers and isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], dict):
                answer = answers[0].get('answer', '')
            else:
                answer = str(answers[0])
        else:
            answer = ''
        
        # Save image to file
        img_path = os.path.join(DATA_ROOT, "VQAv2_subset", "images", f"{idx:06d}.jpg")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        if image_obj:
            image_obj.save(img_path)
        
        rows.append({
            'image_path': img_path,
            'question': question,
            'answer': answer
        })
    
    manifest_path = os.path.join(DATA_ROOT, "VQAv2_subset", "manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"   Saved {len(rows)} samples to {manifest_path}")


def generate_msvd_manifest():
    """Generate MSVD-QA manifest: video_path,question,answer"""
    print("[2/3] Generating MSVD-QA manifest...")
    import numpy as np
    import cv2
    
    dataset = load_from_disk(os.path.join(DATA_ROOT, "MSVD-QA_subset"))
    videos_dir = os.path.join(DATA_ROOT, "MSVD-QA_subset", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    rows = []
    for idx, sample in enumerate(dataset):
        # Extract binary frames and metadata
        binary_frames = sample.get('binary_frames')
        num_frames = sample.get('num_frames', 0)
        height = sample.get('height', 0)
        width = sample.get('width', 0)
        channels = sample.get('channels', 3)
        qa_list = sample.get('qa', [])
        
        if not binary_frames or num_frames == 0:
            continue
        
        # Reconstruct frames from binary data
        try:
            frames_array = np.frombuffer(binary_frames, dtype=np.uint8)
            frames_array = frames_array.reshape((num_frames, height, width, channels))
        except Exception as e:
            print(f"   Warning: Failed to decode video {idx}: {e}")
            continue
        
        # Save as video file
        video_path = os.path.join(videos_dir, f"{idx:06d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        for frame in frames_array:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if channels == 3 else frame
            out.write(frame_bgr)
        out.release()
        
        # Extract QA pairs (take first QA if multiple exist)
        if qa_list and isinstance(qa_list, list) and len(qa_list) > 0:
            qa_pair = qa_list[0]  # 取第一个问答对
            if isinstance(qa_pair, list) and len(qa_pair) >= 2:
                # 格式: ['question', 'answer']
                question = qa_pair[0]
                answer = qa_pair[1]
            elif isinstance(qa_pair, dict):
                # 格式: {'question': '...', 'answer': '...'}
                question = qa_pair.get('question', '')
                answer = qa_pair.get('answer', '')
            else:
                question = ''
                answer = ''
        else:
            question = ''
            answer = ''
        
        rows.append({
            'video_path': video_path,
            'question': question,
            'answer': answer
        })
    
    manifest_path = os.path.join(DATA_ROOT, "MSVD-QA_subset", "manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"   Saved {len(rows)} samples to {manifest_path}")


def generate_audiocaps_manifest():
    """Generate AudioCaps manifest: audio_path,caption"""
    print("[3/3] Generating AudioCaps manifest...")
    dataset = load_from_disk(os.path.join(DATA_ROOT, "AudioCaps_subset"))
    
    rows = []
    for idx, sample in enumerate(dataset):
        # AudioCaps structure: {'audio': audio_array, 'caption': str}
        audio_obj = sample.get('audio')
        caption = sample.get('caption', '')
        
        if not audio_obj:
            continue
        
        # Save audio to disk
        audio_path = os.path.join(DATA_ROOT, "AudioCaps_subset", "audios", f"{idx:06d}.wav")
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        try:
            import soundfile as sf
            # audio_obj is usually a dict with 'array' and 'sampling_rate'
            if isinstance(audio_obj, dict):
                sf.write(audio_path, audio_obj['array'], audio_obj['sampling_rate'])
            else:
                # If not dict, skip this sample
                continue
        except Exception as e:
            print(f"   Warning: Failed to save audio {idx}: {e}")
            continue
        
        rows.append({
            'audio_path': audio_path,
            'caption': caption
        })
    
    manifest_path = os.path.join(DATA_ROOT, "AudioCaps_subset", "manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"   Saved {len(rows)} samples to {manifest_path}")


if __name__ == "__main__":
    print("Generating manifest.csv files from Arrow datasets...")
    print(f"Data root: {DATA_ROOT}\n")
    
    try:
        generate_vqav2_manifest()
    except Exception as e:
        print(f"   [ERROR] VQAv2: {e}")
    
    try:
        generate_msvd_manifest()
    except Exception as e:
        print(f"   [ERROR] MSVD-QA: {e}")
    
    try:
        generate_audiocaps_manifest()
    except Exception as e:
        print(f"   [ERROR] AudioCaps: {e}")
    
    print("\n✅ Done! Check /root/autodl-tmp/data/*/manifest.csv")
