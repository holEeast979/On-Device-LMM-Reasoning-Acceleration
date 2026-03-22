#!/usr/bin/env python3
"""
Generate manifest.csv from local ActivityNet-QA dataset
"""

import os
import json
import csv
from pathlib import Path

def generate_activitynet_manifest():
    """Generate manifest.csv from ActivityNet-QA local data"""
    
    # Paths
    data_root = Path("/root/autodl-tmp/data/ActivityNet-QA")
    annotations_file = data_root / "annotations" / "activitynet_qa_test.json"
    videos_dir = data_root / "videos"
    output_file = data_root / "manifest.csv"
    
    print(f"=== 生成 ActivityNet-QA Manifest ===")
    print(f"注解文件: {annotations_file}")
    print(f"视频目录: {videos_dir}")
    print(f"输出文件: {output_file}")
    
    # Load annotations
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"加载了 {len(annotations)} 条注解")
    
    # Get available video files (check all subdirectories)
    available_videos = set()
    if videos_dir.exists():
        for video_file in videos_dir.rglob("*.mp4"):
            # Extract video ID from filename (remove v_ prefix and .mp4 extension if present)
            video_name = video_file.stem
            if video_name.startswith("v_"):
                video_id = video_name[2:]  # Remove "v_" prefix
            else:
                video_id = video_name
            available_videos.add(video_id)
            # Also add the full filename pattern
            available_videos.add(video_name)
    
    print(f"找到 {len(available_videos)} 个视频文件")
    
    # Generate manifest
    manifest_data = []
    matched_count = 0
    
    for item in annotations:
        video_name = str(item.get("video_name", ""))  # ActivityNet uses video_name field
        question_id = str(item.get("question_id", ""))
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        # Try different video filename patterns
        video_path = ""
        possible_patterns = [
            video_name,         # Direct match
            f"v_{video_name}",  # With v_ prefix
        ]
        
        for pattern in possible_patterns:
            if pattern in available_videos:
                # Find the actual file path
                for video_file in videos_dir.rglob("*.mp4"):
                    if video_file.stem == pattern or video_file.stem == f"v_{pattern}":
                        # Make path relative to data_root
                        video_path = str(video_file.relative_to(data_root))
                        matched_count += 1
                        break
                if video_path:
                    break
        
        manifest_data.append({
            "sample_id": f"activitynet_{video_name}_{question_id}",
            "video_path": video_path,
            "audio_path": "",  # ActivityNet-QA uses video audio
            "question": question,
            "answer": answer,
            "extra": json.dumps({
                "video_name": video_name,
                "question_id": question_id,
                "has_video": video_path != ""
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
    print(f"匹配视频数: {matched_count}")
    print(f"匹配率: {matched_count/len(manifest_data)*100:.1f}%")
    print(f"输出文件: {output_file}")
    
    return len(manifest_data), matched_count

if __name__ == "__main__":
    generate_activitynet_manifest()
