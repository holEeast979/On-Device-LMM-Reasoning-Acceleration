#!/usr/bin/env python3
"""
准备 Video + Audio 配对数据集
将 MSVD 视频和 AudioCaps 音频配对，用于 exp7 实验
"""

import os
import json
import random
import pandas as pd

SEED = 42
N_SAMPLES = 50

# 路径
VIDEO_MANIFEST = "/root/autodl-tmp/data/MSVD-QA_subset/manifest.csv"
VIDEO_DIR = "/root/autodl-tmp/data/MSVD-QA_subset/videos"
AUDIO_MANIFEST = "/root/autodl-tmp/data/AudioCaps_real/manifest.csv"
OUTPUT_DIR = "/root/autodl-tmp/data/VideoAudio_subset"

random.seed(SEED)


def main():
    print("=" * 60)
    print("🎬 准备 Video + Audio 配对数据集")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载视频列表
    print("\n📂 加载视频...")
    video_files = []
    if os.path.exists(VIDEO_MANIFEST):
        video_df = pd.read_csv(VIDEO_MANIFEST)
        for _, row in video_df.iterrows():
            if os.path.exists(row["video_path"]):
                video_files.append({
                    "video_path": row["video_path"],
                    "video_id": row.get("video_id", os.path.basename(row["video_path"])),
                    "question": row.get("question", "Describe this video."),
                })
    else:
        # 直接从目录加载
        for f in os.listdir(VIDEO_DIR):
            if f.endswith(('.mp4', '.avi', '.webm')):
                video_files.append({
                    "video_path": os.path.join(VIDEO_DIR, f),
                    "video_id": os.path.splitext(f)[0],
                    "question": "Describe this video.",
                })
    
    print(f"  找到 {len(video_files)} 个视频")
    
    # 加载音频列表
    print("\n📂 加载音频...")
    audio_df = pd.read_csv(AUDIO_MANIFEST)
    audio_files = []
    for _, row in audio_df.iterrows():
        if os.path.exists(row["audio_path"]):
            audio_files.append({
                "audio_path": row["audio_path"],
                "audio_caption": row.get("caption", ""),
            })
    
    print(f"  找到 {len(audio_files)} 个音频")
    
    # 随机配对
    print(f"\n🔀 随机配对 {N_SAMPLES} 个样本...")
    random.shuffle(video_files)
    random.shuffle(audio_files)
    
    samples = []
    for i in range(min(N_SAMPLES, len(video_files), len(audio_files))):
        samples.append({
            "sample_id": f"va_{i:04d}",
            "video_path": video_files[i]["video_path"],
            "video_id": video_files[i]["video_id"],
            "audio_path": audio_files[i]["audio_path"],
            "audio_caption": audio_files[i]["audio_caption"],
            "question": "Describe what you see and hear in this video.",
        })
    
    # 保存
    print(f"\n💾 保存 manifest...")
    df = pd.DataFrame(samples)
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    df.to_csv(manifest_path, index=False)
    
    # 同时保存 JSON
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"  保存到: {manifest_path}")
    print(f"  样本数: {len(samples)}")
    
    # 验证
    print(f"\n🔍 验证数据...")
    valid_count = 0
    for s in samples:
        if os.path.exists(s["video_path"]) and os.path.exists(s["audio_path"]):
            valid_count += 1
    
    print(f"  有效样本: {valid_count}/{len(samples)}")
    
    print("\n" + "=" * 60)
    print("✅ 数据准备完成！")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
