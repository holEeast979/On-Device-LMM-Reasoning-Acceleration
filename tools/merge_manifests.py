#!/usr/bin/env python3
"""
Merge all dataset manifests into unified benchmark manifest
"""

import os
import csv
import json
from pathlib import Path

def merge_manifests():
    """Merge all dataset manifests for benchmark use"""
    
    # Input manifest files
    manifests = {
        "video_mme": "/root/autodl-tmp/data/Video-MME/manifest.csv",
        "audiocaps": "/root/autodl-tmp/data/AudioCaps/manifest.csv", 
        "activitynet": "/root/autodl-tmp/data/ActivityNet-QA/manifest.csv"
    }
    
    # Output unified manifest
    output_file = "/root/autodl-tmp/data/unified_manifest.csv"
    
    print("=== 合并所有数据集 Manifest ===")
    
    all_samples = []
    dataset_stats = {}
    
    for dataset_name, manifest_file in manifests.items():
        if not os.path.exists(manifest_file):
            print(f"⚠️ 跳过不存在的文件: {manifest_file}")
            continue
            
        print(f"\n处理 {dataset_name} manifest...")
        
        with open(manifest_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            dataset_samples = []
            
            for row in reader:
                # Add dataset source to extra field
                extra_data = json.loads(row.get('extra', '{}'))
                extra_data['dataset'] = dataset_name
                row['extra'] = json.dumps(extra_data)
                
                dataset_samples.append(row)
                all_samples.append(row)
            
            # Calculate stats
            has_media = sum(1 for s in dataset_samples 
                           if s.get('video_path') or s.get('audio_path'))
            
            dataset_stats[dataset_name] = {
                'total_samples': len(dataset_samples),
                'has_media': has_media,
                'media_rate': has_media / len(dataset_samples) * 100 if dataset_samples else 0
            }
            
            print(f"  - 总样本: {len(dataset_samples)}")
            print(f"  - 有媒体文件: {has_media}")
            print(f"  - 媒体匹配率: {dataset_stats[dataset_name]['media_rate']:.1f}%")
    
    # Write unified manifest
    print(f"\n写入统一manifest: {output_file}")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if all_samples:
            fieldnames = ['sample_id', 'video_path', 'audio_path', 'question', 'answer', 'extra']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_samples)
    
    # Generate summary report
    summary_file = "/root/autodl-tmp/data/benchmark_summary.json"
    summary = {
        'datasets': dataset_stats,
        'unified_stats': {
            'total_samples': len(all_samples),
            'total_with_media': sum(1 for s in all_samples 
                                  if s.get('video_path') or s.get('audio_path')),
            'by_type': {
                'video_only': sum(1 for s in all_samples 
                                if s.get('video_path') and not s.get('audio_path')),
                'audio_only': sum(1 for s in all_samples 
                                if s.get('audio_path') and not s.get('video_path')),
                'video_audio': sum(1 for s in all_samples 
                                 if s.get('video_path') and s.get('audio_path')),
                'no_media': sum(1 for s in all_samples 
                              if not s.get('video_path') and not s.get('audio_path'))
            }
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 合并完成!")
    print(f"统一manifest: {output_file}")
    print(f"总样本数: {len(all_samples)}")
    print(f"有媒体文件: {summary['unified_stats']['total_with_media']}")
    print(f"详细统计: {summary_file}")
    
    return len(all_samples), summary

if __name__ == "__main__":
    merge_manifests()
