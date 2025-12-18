#!/usr/bin/env python3
"""
实验5：模块级别耗时分析
用Hook精确测量Qwen2.5-Omni各模块的耗时
重点：多模态特有模块（Vision Encoder, Audio Encoder, 投影层）
"""

from __future__ import annotations
import argparse, os, sys, gc, time
import cv2, numpy as np, pandas as pd, torch
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import common as C


def print_model_structure(model, max_depth=3):
    """打印模型结构，找出关键模块"""
    print("\n" + "="*80)
    print("📦 模型结构探测")
    print("="*80)
    
    modules_info = []
    for name, module in model.named_modules():
        depth = name.count('.')
        if depth <= max_depth and name:  # 只打印前几层
            class_name = module.__class__.__name__
            # 统计参数量
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0 or depth <= 1:
                indent = "  " * depth
                modules_info.append((name, class_name, params, depth))
                if params > 1e6:
                    print(f"{indent}📌 {name}: {class_name} ({params/1e6:.1f}M params)")
                elif depth <= 1:
                    print(f"{indent}📁 {name}: {class_name}")
    
    return modules_info


def find_key_modules(model):
    """自动找出关键的多模态模块"""
    key_modules = {}
    
    # 遍历查找关键模块
    for name, module in model.named_modules():
        name_lower = name.lower()
        
        # Vision相关
        if any(k in name_lower for k in ['visual', 'vision', 'vit', 'image_encoder']):
            if 'merger' in name_lower or 'proj' in name_lower:
                key_modules['vision_projector'] = name
            elif name.count('.') <= 2:  # 顶层vision模块
                key_modules['vision_encoder'] = name
        
        # Audio相关
        if any(k in name_lower for k in ['audio', 'whisper', 'speech']):
            if 'proj' in name_lower:
                key_modules['audio_projector'] = name
            elif name.count('.') <= 2:
                key_modules['audio_encoder'] = name
        
        # LLM相关
        if any(k in name_lower for k in ['language', 'llm', 'lm_head', 'embed_tokens']):
            if name.count('.') <= 2:
                key_modules['llm'] = name
        
        # Thinker/Talker
        if name == 'thinker':
            key_modules['thinker'] = name
        if name == 'talker':
            key_modules['talker'] = name
    
    return key_modules


class ModuleTimer:
    """模块计时器，用Hook记录各模块耗时"""
    
    def __init__(self):
        self.timings = OrderedDict()
        self.start_times = {}
        self.hooks = []
    
    def _make_pre_hook(self, name):
        def hook(module, input):
            torch.cuda.synchronize()
            self.start_times[name] = time.perf_counter()
        return hook
    
    def _make_post_hook(self, name):
        def hook(module, input, output):
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            if name in self.start_times:
                elapsed = (end_time - self.start_times[name]) * 1000
                if name not in self.timings:
                    self.timings[name] = []
                self.timings[name].append(elapsed)
        return hook
    
    def register(self, model, module_names):
        """注册要监控的模块"""
        for name, module in model.named_modules():
            if name in module_names:
                pre_hook = module.register_forward_pre_hook(self._make_pre_hook(name))
                post_hook = module.register_forward_hook(self._make_post_hook(name))
                self.hooks.append(pre_hook)
                self.hooks.append(post_hook)
                print(f"  ✅ 已注册Hook: {name}")
    
    def clear(self):
        """清除计时数据"""
        self.timings.clear()
        self.start_times.clear()
    
    def remove_hooks(self):
        """移除所有Hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_summary(self):
        """获取统计摘要"""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times),
            }
        return summary


def run_inference(model, proc, images, question="Describe what you see."):
    """运行一次推理"""
    gc.collect()
    torch.cuda.empty_cache()
    
    n_imgs = len(images)
    content = [{"type": "image"}] * n_imgs + [{"type": "text", "text": question}]
    text = proc.apply_chat_template(
        [{"role": "user", "content": content}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = proc(text=text, images=images, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            return_audio=False,
        )
    
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen25-omni", required=True, help="模型路径")
    parser.add_argument("--images", required=True, help="图像manifest")
    parser.add_argument("--videos", required=True, help="视频manifest")
    parser.add_argument("--out", default="/root/autodl-tmp/results/exp5_module_timing.csv")
    parser.add_argument("--n-samples", type=int, default=10, help="样本数")
    parser.add_argument("--frames", type=int, default=4, help="视频帧数")
    parser.add_argument("--print-structure", action="store_true", help="只打印模型结构")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # 加载模型
    print("🔄 加载模型...")
    model, proc = C.load_qwen25_omni(args.qwen25_omni, "bf16")
    
    # 打印模型结构
    modules_info = print_model_structure(model, max_depth=2)
    
    # 找出关键模块
    print("\n" + "="*80)
    print("🔍 自动检测到的关键模块")
    print("="*80)
    key_modules = find_key_modules(model)
    for role, name in key_modules.items():
        print(f"  {role}: {name}")
    
    if args.print_structure:
        # 打印更详细的结构
        print("\n" + "="*80)
        print("📋 完整模块列表（前4层）")
        print("="*80)
        for name, module in model.named_modules():
            if name.count('.') <= 3:
                print(f"  {name}: {module.__class__.__name__}")
        return
    
    # 根据探测结果确定的实际模块路径
    # 更细粒度：监控每一层
    modules_to_monitor = [
        # ========== Vision Encoder 细粒度 ==========
        "thinker.visual",                    # 整体
        "thinker.visual.patch_embed",        # Patch Embedding
        "thinker.visual.patch_embed.proj",   # Conv3d
        "thinker.visual.merger",             # 融合层
        "thinker.visual.merger.mlp",         # 融合MLP
    ]
    
    # 添加ViT每一层（32层）
    for i in range(32):
        modules_to_monitor.append(f"thinker.visual.blocks.{i}")
    
    # 如果需要对比LLM（只取几层作为参考）
    modules_to_monitor.extend([
        "thinker.model",                     # 整体LLM
        "thinker.model.embed_tokens",        # Embedding
        "thinker.lm_head",                   # 输出头
        # LLM层采样（不全部监控，太多了）
        "thinker.model.layers.0",            # 第1层
        "thinker.model.layers.13",           # 中间层
        "thinker.model.layers.27",           # 最后一层
    ])
    
    # 过滤出实际存在的模块
    existing_modules = set(name for name, _ in model.named_modules())
    modules_to_monitor = [m for m in modules_to_monitor if m in existing_modules]
    
    print("\n" + "="*80)
    print("⏱️ 注册计时Hook")
    print("="*80)
    timer = ModuleTimer()
    timer.register(model, modules_to_monitor)
    
    # 加载测试数据
    print("\n🔄 加载测试数据...")
    
    # 加载图片
    images_list = []
    img_df = pd.read_csv(args.images).head(args.n_samples)
    for _, row in img_df.iterrows():
        if os.path.exists(row["image_path"]):
            img = cv2.imread(row["image_path"])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                if max(h, w) > 512:
                    scale = 512 / max(h, w)
                    img = cv2.resize(img, (int(w*scale), int(h*scale)))
                images_list.append(img)
    
    # 加载视频
    videos_list = []
    vid_df = pd.read_csv(args.videos).head(args.n_samples)
    for _, row in vid_df.iterrows():
        if os.path.exists(row["video_path"]):
            frames, _, _ = C.sample_video_frames(row["video_path"], args.frames, 336)
            if frames:
                videos_list.append(frames)
    
    n = min(len(images_list), len(videos_list), args.n_samples)
    print(f"  使用 {n} 个样本")
    
    results = []
    
    # 测试1: Image单独
    print("\n" + "="*80)
    print("🖼️ 测试: Image单独")
    print("="*80)
    timer.clear()
    for i in range(n):
        print(f"  样本 {i+1}/{n}", end="\r")
        run_inference(model, proc, [images_list[i]])
    
    img_summary = timer.get_summary()
    for name, stats in img_summary.items():
        results.append({
            "test": "image",
            "module": name,
            "mean_ms": stats['mean'],
            "std_ms": stats['std'],
            "count": stats['count'],
        })
        print(f"  {name}: {stats['mean']:.2f} ± {stats['std']:.2f} ms")
    
    # 测试2: Video单独
    print("\n" + "="*80)
    print("🎥 测试: Video单独")
    print("="*80)
    timer.clear()
    for i in range(n):
        print(f"  样本 {i+1}/{n}", end="\r")
        run_inference(model, proc, videos_list[i])
    
    vid_summary = timer.get_summary()
    for name, stats in vid_summary.items():
        results.append({
            "test": "video",
            "module": name,
            "mean_ms": stats['mean'],
            "std_ms": stats['std'],
            "count": stats['count'],
        })
        print(f"  {name}: {stats['mean']:.2f} ± {stats['std']:.2f} ms")
    
    # 测试3: Image+Video并行
    print("\n" + "="*80)
    print("🔄 测试: Image+Video并行")
    print("="*80)
    timer.clear()
    for i in range(n):
        print(f"  样本 {i+1}/{n}", end="\r")
        all_imgs = [images_list[i]] + videos_list[i]
        run_inference(model, proc, all_imgs)
    
    par_summary = timer.get_summary()
    for name, stats in par_summary.items():
        results.append({
            "test": "parallel",
            "module": name,
            "mean_ms": stats['mean'],
            "std_ms": stats['std'],
            "count": stats['count'],
        })
        print(f"  {name}: {stats['mean']:.2f} ± {stats['std']:.2f} ms")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"\n✅ 结果已保存: {args.out}")
    
    # 分析对比
    print("\n" + "="*80)
    print("📊 模块级别瓶颈分析")
    print("="*80)
    
    # 1. 顶层模块对比
    top_modules = ["thinker.visual", "thinker.visual.patch_embed", 
                   "thinker.visual.merger", "thinker.model"]
    
    for module in top_modules:
        img_time = img_summary.get(module, {}).get('mean', 0)
        vid_time = vid_summary.get(module, {}).get('mean', 0)
        par_time = par_summary.get(module, {}).get('mean', 0)
        serial_sum = img_time + vid_time
        
        if serial_sum > 0:
            diff = par_time - serial_sum
            diff_pct = (diff / serial_sum) * 100
            
            print(f"\n📌 {module}:")
            print(f"   Image:      {img_time:.2f} ms")
            print(f"   Video:      {vid_time:.2f} ms")
            print(f"   串行和:     {serial_sum:.2f} ms")
            print(f"   并行:       {par_time:.2f} ms")
            print(f"   差异:       {diff:+.2f} ms ({diff_pct:+.1f}%)")
    
    # 2. ViT每层详细分析
    print("\n" + "="*80)
    print("🔬 ViT 每层耗时分布（多模态核心）")
    print("="*80)
    print(f"{'Layer':<30} {'Image':>10} {'Video':>10} {'Parallel':>10} {'Serial':>10} {'Diff%':>10}")
    print("-"*80)
    
    vit_layers_img = []
    vit_layers_vid = []
    vit_layers_par = []
    
    for i in range(32):
        layer_name = f"thinker.visual.blocks.{i}"
        img_time = img_summary.get(layer_name, {}).get('mean', 0)
        vid_time = vid_summary.get(layer_name, {}).get('mean', 0)
        par_time = par_summary.get(layer_name, {}).get('mean', 0)
        serial_sum = img_time + vid_time
        
        vit_layers_img.append(img_time)
        vit_layers_vid.append(vid_time)
        vit_layers_par.append(par_time)
        
        if serial_sum > 0:
            diff_pct = (par_time / serial_sum - 1) * 100
            print(f"{layer_name:<30} {img_time:>10.2f} {vid_time:>10.2f} {par_time:>10.2f} {serial_sum:>10.2f} {diff_pct:>+10.1f}%")
    
    # 3. ViT层统计
    print("\n" + "-"*80)
    print("📈 ViT Blocks 统计:")
    if vit_layers_img:
        total_img = sum(vit_layers_img)
        total_vid = sum(vit_layers_vid)
        total_par = sum(vit_layers_par)
        print(f"   Image总计:    {total_img:.2f} ms")
        print(f"   Video总计:    {total_vid:.2f} ms")
        print(f"   Parallel总计: {total_par:.2f} ms")
        print(f"   串行和总计:   {total_img + total_vid:.2f} ms")
        if total_img + total_vid > 0:
            print(f"   并行效率:     {(1 - total_par/(total_img + total_vid))*100:.1f}% 节省")
    
    # 4. 时间占比分析
    print("\n" + "="*80)
    print("📊 Vision Encoder 内部时间占比")
    print("="*80)
    
    for test_name, summary in [("Image", img_summary), ("Video", vid_summary), ("Parallel", par_summary)]:
        visual_total = summary.get("thinker.visual", {}).get('mean', 0)
        patch_embed = summary.get("thinker.visual.patch_embed", {}).get('mean', 0)
        merger = summary.get("thinker.visual.merger", {}).get('mean', 0)
        
        # ViT blocks总时间
        vit_total = sum(summary.get(f"thinker.visual.blocks.{i}", {}).get('mean', 0) for i in range(32))
        
        if visual_total > 0:
            print(f"\n{test_name}:")
            print(f"   Patch Embed:  {patch_embed:>8.2f} ms ({patch_embed/visual_total*100:>5.1f}%)")
            print(f"   ViT Blocks:   {vit_total:>8.2f} ms ({vit_total/visual_total*100:>5.1f}%)")
            print(f"   Merger:       {merger:>8.2f} ms ({merger/visual_total*100:>5.1f}%)")
            print(f"   Total:        {visual_total:>8.2f} ms")
    
    # 清理
    timer.remove_hooks()
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
