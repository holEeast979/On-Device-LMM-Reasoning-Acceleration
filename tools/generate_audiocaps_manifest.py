#!/usr/bin/env python3
"""
Generate manifest.csv from local AudioCaps dataset
"""

import os
import json
import csv
from pathlib import Path

def generate_audiocaps_manifest():
    """Generate manifest.csv from AudioCaps local data"""
    
    # Paths
    data_root = Path("/root/autodl-tmp/data/AudioCaps")
    annotations_file = data_root / "annotations" / "audiocaps_test.json"
    audios_dir = data_root / "audios"
    output_file = data_root / "manifest.csv"
    
    print(f"=== 生成 AudioCaps Manifest ===")
    print(f"注解文件: {annotations_file}")
    print(f"音频目录: {audios_dir}")
    print(f"输出文件: {output_file}")
    
    # Load annotations
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"加载了 {len(annotations)} 条注解")
    
    # Get available audio files
    available_audios = set()
    if audios_dir.exists():
        for audio_file in audios_dir.glob("*.wav"):
            # Extract audio ID from filename (remove .wav extension)
            audio_id = audio_file.stem
            available_audios.add(audio_id)
    
    print(f"找到 {len(available_audios)} 个音频文件")
    
    # Generate manifest
    manifest_data = []
    matched_count = 0
    
    for item in annotations:
        audiocap_id = str(item.get("id", ""))
        audio_file = item.get("audio_file", "")
        caption = item.get("caption", "")
        
        # Extract filename without extension
        audio_filename = audio_file.replace(".wav", "") if audio_file.endswith(".wav") else audio_file
        
        # Check if audio file exists
        audio_path = ""
        if audio_filename in available_audios:
            audio_path = f"audios/{audio_file}"
            matched_count += 1
        
        # Use caption as question (audio description task)
        question = "Describe what you hear in this audio."
        answer = caption
        
        manifest_data.append({
            "sample_id": f"audiocaps_{audiocap_id}",
            "video_path": "",  # AudioCaps is audio-only
            "audio_path": audio_path,
            "question": question,
            "answer": answer,
            "extra": json.dumps({
                "audiocap_id": audiocap_id,
                "audio_file": audio_file,
                "has_audio": audio_path != ""
            })
        })
    
    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sample_id', 'video_path', 'audio_path', 'question', 'answer', 'extra']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(manifest_data)
    
    print(f"\n✅ Manifest 生成完成!")
    print(f"总样本数: {len(manifest_data)}")
    print(f"匹配音频数: {matched_count}")
    print(f"匹配率: {matched_count/len(manifest_data)*100:.1f}%")
    print(f"输出文件: {output_file}")
    
    return len(manifest_data), matched_count

if __name__ == "__main__":
    generate_audiocaps_manifest()
