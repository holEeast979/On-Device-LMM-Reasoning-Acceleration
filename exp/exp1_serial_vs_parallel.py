#!/usr/bin/env python3
"""
实验：串行 vs 并行推理对比
目标：找出并行输入时哪个阶段影响推理时间

统一使用相同prompt，避免生成长度差异
重点对比：encode、prefill、decode 各阶段时间
"""

from __future__ import annotations
import argparse, os, sys, gc, time, threading
import cv2, numpy as np, pandas as pd, torch
from tqdm import tqdm

# 导入路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import common as C
from transformers.generation.streamers import BaseStreamer

class TokenTimingStreamer(BaseStreamer):
    def __init__(self):
        super().__init__()
        self.first_token_time = None
    
    def put(self, value):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
    
    def end(self):
        pass

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_samples(images_csv, videos_csv, n, n_frames):
    samples = {"image": [], "video": []}
    
    # 加载图片
    for _, row in pd.read_csv(images_csv).head(n).iterrows():
        if os.path.exists(row["image_path"]):
            img = cv2.imread(row["image_path"])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                if max(h,w) > 512:
                    scale = 512/max(h,w)
                    img = cv2.resize(img, (int(w*scale), int(h*scale)))
                samples["image"].append(img)
    
    # 加载视频
    for _, row in pd.read_csv(videos_csv).head(n).iterrows():
        if os.path.exists(row["video_path"]):
            frames, _, _ = C.sample_video_frames(row["video_path"], n_frames, 336)
            if frames:
                samples["video"].append(frames)
    
    return samples

def measure_inference(model, proc, images, test_name, sample_id, max_tokens):
    """统一测量函数 - 针对编码瓶颈分析"""
    clear_gpu()
    
    # 统一prompt - 所有测试使用完全相同的问题
    question = "Describe what you see in one sentence."
    
    # 1. Encode阶段 - 重点监控
    n_imgs = len(images)
    content = [{"type": "image"}] * n_imgs + [{"type": "text", "text": question}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], 
                                   tokenize=False, add_generation_prompt=True)
    
    torch.cuda.synchronize()
    t_encode_start = time.perf_counter()
    inputs = proc(text=text, images=images, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    encode_ms = (time.perf_counter() - t_encode_start) * 1000
    
    input_tokens = inputs.input_ids.shape[1]
    
    # 2. Prefill + Decode 阶段 - 分离测量
    streamer = TokenTimingStreamer()
    outputs_container = {}
    
    def _generate():
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # 固定解码策略
                streamer=streamer,
                return_audio=False,
                pad_token_id=proc.tokenizer.eos_token_id,
            )
        outputs_container["outputs"] = out
    
    torch.cuda.synchronize()
    t_gen_start = time.perf_counter()
    
    worker = threading.Thread(target=_generate)
    worker.start()
    worker.join()
    
    torch.cuda.synchronize()
    t_gen_end = time.perf_counter()
    
    # 计算时间
    total_gen_ms = (t_gen_end - t_gen_start) * 1000
    if streamer.first_token_time is None:
        prefill_ms = total_gen_ms
        decode_ms = 0
    else:
        prefill_ms = (streamer.first_token_time - t_gen_start) * 1000
        decode_ms = (t_gen_end - streamer.first_token_time) * 1000
        decode_ms = max(0, decode_ms)
    
    out = outputs_container.get("outputs")
    n_tokens = out.shape[1] - input_tokens if out is not None else 0
    
    # 计算每token decode时间（排除第一个token）
    per_token_decode_ms = decode_ms / max(1, n_tokens - 1) if n_tokens > 1 else 0
    
    total_ms = encode_ms + total_gen_ms
    
    # 清理
    del inputs
    if out is not None:
        del out
    clear_gpu()
    
    return {
        "test": test_name,
        "sample_id": sample_id,
        "input_tokens": input_tokens,
        "n_tokens": n_tokens,
        "encode_ms": encode_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "per_token_decode_ms": per_token_decode_ms,
        "total_ms": total_ms,
        "n_images": n_imgs,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen25-omni", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--out", default="/root/autodl-tmp/results/exp4_serial_vs_parallel.csv")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=32)  # 控制生成长度
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    print("Loading model...")
    model, proc = C.load_qwen25_omni(args.qwen25_omni, "bf16")
    
    print("Loading samples...")
    samples = load_samples(args.images, args.videos, args.n_samples, args.frames)
    n = min(len(samples["image"]), len(samples["video"]))
    print(f"Using {n} samples")
    
    results = []
    
    # 串行测试 - 使用相同样本对
    print("\n--- Serial: Image Only ---")
    for i in tqdm(range(n), desc="Serial Image"):
        try:
            r = measure_inference(model, proc, [samples["image"][i]], 
                                "serial_image", i, args.max_tokens)
            results.append(r)
        except Exception as e:
            print(f"Serial image {i} error: {e}")
    
    print("--- Serial: Video Only ---")
    for i in tqdm(range(n), desc="Serial Video"):
        try:
            r = measure_inference(model, proc, samples["video"][i], 
                                "serial_video", i, args.max_tokens)
            results.append(r)
        except Exception as e:
            print(f"Serial video {i} error: {e}")
    
    # 并行测试 - 使用相同的样本对进行组合
    print("--- Parallel: Image + Video ---")
    for i in tqdm(range(n), desc="Parallel Both"):
        try:
            all_imgs = [samples["image"][i]] + samples["video"][i]
            r = measure_inference(model, proc, all_imgs, 
                                "parallel_both", i, args.max_tokens)
            results.append(r)
        except Exception as e:
            print(f"Parallel {i} error: {e}")
    
    # 保存结果
    df = pd.DataFrame(results)
    cols = ["test", "sample_id", "input_tokens", "n_tokens", "n_images", 
            "encode_ms", "prefill_ms", "decode_ms", "per_token_decode_ms", "total_ms"]
    df = df[cols]
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")
    
    # ================== 编码瓶颈分析 ==================
    print("\n" + "="*80)
    print("🔍 多模态并行编码瓶颈分析")
    print("="*80)
    
    img_data = df[df["test"] == "serial_image"]
    vid_data = df[df["test"] == "serial_video"] 
    par_data = df[df["test"] == "parallel_both"]
    
    if any(x.empty for x in [img_data, vid_data, par_data]):
        print("❌ 数据不完整，无法分析")
        return
    
    # 1. 各阶段基础统计
    print("\n📊 各阶段基础统计:")
    for test_name, data, icon in [("Image单独", img_data, "🖼️"), 
                                  ("Video单独", vid_data, "🎥"),
                                  ("并行输入", par_data, "🔄")]:
        print(f"\n{icon} {test_name}:")
        print(f"   输入tokens:     {data['input_tokens'].mean():.0f}")
        print(f"   输出tokens:     {data['n_tokens'].mean():.1f}")
        print(f"   Encode:         {data['encode_ms'].mean():.1f}ms")
        print(f"   Prefill:        {data['prefill_ms'].mean():.1f}ms")
        print(f"   每token解码:    {data['per_token_decode_ms'].mean():.1f}ms")
    
    # 2. 关键对比：串行和 vs 并行
    print(f"\n🔥 关键对比：串行逻辑和 vs 并行实际")
    print("-" * 60)
    
    encode_serial = img_data['encode_ms'].mean() + vid_data['encode_ms'].mean()
    encode_parallel = par_data['encode_ms'].mean()
    encode_diff = encode_parallel - encode_serial
    
    prefill_serial = img_data['prefill_ms'].mean() + vid_data['prefill_ms'].mean()
    prefill_parallel = par_data['prefill_ms'].mean()
    prefill_diff = prefill_parallel - prefill_serial
    
    total_serial = img_data['total_ms'].mean() + vid_data['total_ms'].mean()
    total_parallel = par_data['total_ms'].mean()
    
    print(f"Encode阶段:")
    print(f"   串行和:  {encode_serial:.1f}ms")
    print(f"   并行:    {encode_parallel:.1f}ms")
    print(f"   差异:    {encode_diff:+.1f}ms ({encode_diff/encode_serial*100:+.1f}%)")
    
    print(f"\nPrefill阶段:")
    print(f"   串行和:  {prefill_serial:.1f}ms")
    print(f"   并行:    {prefill_parallel:.1f}ms") 
    print(f"   差异:    {prefill_diff:+.1f}ms ({prefill_diff/prefill_serial*100:+.1f}%)")
    
    # 3. 瓶颈识别
    print(f"\n🎯 瓶颈识别:")
    print("-" * 40)
    
    # 计算各阶段占比
    par_encode_pct = par_data['encode_ms'].mean() / par_data['total_ms'].mean() * 100
    par_prefill_pct = par_data['prefill_ms'].mean() / par_data['total_ms'].mean() * 100
    par_decode_pct = par_data['decode_ms'].mean() / par_data['total_ms'].mean() * 100
    
    print(f"并行推理各阶段占比:")
    print(f"   Encode:   {par_encode_pct:.1f}%")
    print(f"   Prefill:  {par_prefill_pct:.1f}%") 
    print(f"   Decode:   {par_decode_pct:.1f}%")
    
    # 找出最大瓶颈
    bottlenecks = [
        ("Encode", abs(encode_diff), par_encode_pct),
        ("Prefill", abs(prefill_diff), par_prefill_pct),
    ]
    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🔴 最影响并行编码时间的因素:")
    print(f"   1. {bottlenecks[0][0]}: 差异{bottlenecks[0][1]:.1f}ms, 占总时间{bottlenecks[0][2]:.1f}%")
    print(f"   2. {bottlenecks[1][0]}: 差异{bottlenecks[1][1]:.1f}ms, 占总时间{bottlenecks[1][2]:.1f}%")
    
    # 4. 总结
    print(f"\n✅ 总结:")
    efficiency = (1 - total_parallel/total_serial) * 100
    if efficiency > 0:
        print(f"   并行比串行快 {efficiency:.1f}%")
    else:
        print(f"   并行比串行慢 {-efficiency:.1f}%")
    print(f"   主要原因: {bottlenecks[0][0]}阶段的{'增加' if encode_diff > 0 else '减少'}")
    
    print("="*80)

if __name__ == "__main__":
    main()
