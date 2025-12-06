#!/usr/bin/env python3
"""
下载 MiniCPM-V-2.6 和 Phi-3.5-Vision
带进度条显示
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # 用普通下载，显示进度

from huggingface_hub import snapshot_download
from tqdm import tqdm

models = [
    {
        "repo": "openbmb/MiniCPM-V-2_6",
        "local_dir": "/root/autodl-tmp/MiniCPM-V-2_6",
        "desc": "MiniCPM-V-2.6 (2.6B, ~6GB)"
    },
    {
        "repo": "microsoft/Phi-3.5-vision-instruct",
        "local_dir": "/root/autodl-tmp/Phi-3.5-vision",
        "desc": "Phi-3.5-Vision (4B, ~8GB)"
    },
]

def main():
    print("="*60)
    print("📥 开始下载模型")
    print("="*60)
    
    for i, m in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {m['desc']}")
        print(f"    Repo: {m['repo']}")
        print(f"    目标: {m['local_dir']}")
        print("-"*60)
        
        try:
            snapshot_download(
                m["repo"],
                local_dir=m["local_dir"],
                resume_download=True,
                # 这会显示每个文件的下载进度
            )
            print(f"✅ {m['repo']} 下载完成!")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
    
    print("\n" + "="*60)
    print("🎉 全部完成!")
    print("="*60)
    
    # 检查下载结果
    print("\n📁 下载结果:")
    for m in models:
        if os.path.exists(m["local_dir"]):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, fn in os.walk(m["local_dir"])
                for f in fn
            ) / (1024**3)
            print(f"  ✅ {m['local_dir']}: {size:.2f} GB")
        else:
            print(f"  ❌ {m['local_dir']}: 不存在")


if __name__ == "__main__":
    main()
