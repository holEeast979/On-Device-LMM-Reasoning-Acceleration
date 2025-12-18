import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)


def _now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def set_offline_env():
    # Ensure we never hit network at runtime
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def get_dtype(dtype: str):
    ds = dtype.lower()
    if ds in ("bf16", "bfloat16"):
        return torch.bfloat16
    if ds in ("fp16", "float16"):
        return torch.float16
    return torch.float16


def load_qwen2_vl(model_dir: str, dtype: str = "bf16"):
    set_offline_env()
    from transformers import Qwen2VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, dtype=get_dtype(dtype), device_map="auto", trust_remote_code=True
        ).eval()
    except TypeError:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=get_dtype(dtype), device_map="auto", trust_remote_code=True
        ).eval()
    return model, proc


def load_llava_like(model_dir: str, dtype: str = "bf16"):
    set_offline_env()
    from transformers import LlavaForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, dtype=get_dtype(dtype), device_map="auto", trust_remote_code=True
        ).eval()
    except TypeError:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=get_dtype(dtype), device_map="auto", trust_remote_code=True
        ).eval()
    return model, proc


def load_blip2(model_dir: str, dtype: str = "fp16"):
    set_offline_env()
    from transformers import Blip2ForConditionalGeneration, AutoProcessor as BlipProc

    proc = BlipProc.from_pretrained(model_dir, local_files_only=True)
    try:
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, dtype=get_dtype(dtype), device_map="auto"
        ).eval()
    except TypeError:
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=get_dtype(dtype), device_map="auto"
        ).eval()
    return model, proc


def load_qwen2_audio(model_dir: str, dtype: str = "bf16"):
    set_offline_env()
    from transformers import Qwen2AudioForConditionalGeneration

    proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, dtype=get_dtype(dtype), device_map="auto", trust_remote_code=True
        ).eval()
    except TypeError:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=get_dtype(dtype), device_map="auto", trust_remote_code=True
        ).eval()
    return model, proc


def stream_generate(model, tokenizer_or_proc, gen_kwargs: dict):
    tok = getattr(tokenizer_or_proc, "tokenizer", tokenizer_or_proc)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(gen_kwargs)
    kwargs["streamer"] = streamer
    
    # Force override generation config to ensure deterministic generation and avoid conflicts
    kwargs["do_sample"] = False
    kwargs.pop("temperature", None)
    kwargs.pop("top_p", None)
    kwargs.pop("top_k", None)

    out_tokens: List[str] = []
    ttft: Optional[float] = None

    def _run():
        model.generate(**kwargs)

    start = _now()
    th = torch._C._lazy_init.__self__ if False else None  # placeholder to avoid lints
    import threading

    th = threading.Thread(target=_run)
    th.start()
    n = 0
    for t in streamer:
        if ttft is None:
            ttft = _now() - start
        out_tokens.append(t)
        n += 1
    th.join()
    total = _now() - start
    tok_per_s = 0.0 if total <= 0 else n / total
    return "".join(out_tokens), ttft or total, total, tok_per_s


def sample_video_frames(video_path: str, n: int, short_side: int) -> Tuple[List[np.ndarray], float, float]:
    from decord import VideoReader, cpu
    import cv2

    t0 = _now()
    vr = VideoReader(video_path, ctx=cpu(0))
    if len(vr) == 0:
        return [], 0.0, 0.0
    import numpy as np

    idx = np.linspace(0, max(0, len(vr) - 1), n).astype(int)
    t1 = _now()
    batch = vr.get_batch(idx).asnumpy()  # T,H,W,C
    t_decode = _now() - t1

    t2 = _now()
    frames = []
    for img in batch:
        h, w = img.shape[:2]
        if h < w:
            new_h, new_w = short_side, int(w * (short_side / h))
        else:
            new_w, new_h = short_side, int(h * (short_side / w))
        frames.append(cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR))
    t_pre = _now() - t2
    return frames, t_decode, t_pre


def run_qwen2_vl_image(model, proc, image_np, question: str, max_new_tokens: int):
    # Format question for better short answers
    formatted_question = f"{question} Give a short answer:"
    msgs = [{"role": "user", "content": [{"type": "text", "text": formatted_question}, {"type": "image", "image": image_np}]}]
    pack0 = _now()
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=text, images=[image_np], return_tensors="pt").to(model.device)
    t_pack = _now() - pack0
    out, ttft, total, tok_s = stream_generate(model, proc, {**inputs, "max_new_tokens": max_new_tokens, "do_sample": False})
    return out, ttft, total, tok_s, t_pack


def run_qwen2_vl_video(model, proc, frames: List[np.ndarray], question: str, max_new_tokens: int):
    # Format question for better short answers
    formatted_question = f"{question} Give a short answer:"
    msgs = [{"role": "user", "content": [{"type": "text", "text": formatted_question}]}]
    for f in frames:
        msgs[0]["content"].append({"type": "image", "image": f})
    pack0 = _now()
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=text, images=frames, return_tensors="pt").to(model.device)
    t_pack = _now() - pack0
    out, ttft, total, tok_s = stream_generate(model, proc, {**inputs, "max_new_tokens": max_new_tokens, "do_sample": False})
    return out, ttft, total, tok_s, t_pack


def run_llava_image(model, proc, image_np, question: str, max_new_tokens: int):
    # Format question for better short answers
    formatted_question = f"{question} Give a short answer:"
    msgs = [{"role": "user", "content": [{"type": "text", "text": formatted_question}, {"type": "image", "image": image_np}]}]
    pack0 = _now()
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=text, images=[image_np], return_tensors="pt").to(model.device)
    t_pack = _now() - pack0
    out, ttft, total, tok_s = stream_generate(model, proc, {**inputs, "max_new_tokens": max_new_tokens, "do_sample": False})
    return out, ttft, total, tok_s, t_pack


def run_llava_video(model, proc, frames: List[np.ndarray], question: str, max_new_tokens: int):
    # Format question for better short answers
    formatted_question = f"{question} Give a short answer:"
    msgs = [{"role": "user", "content": [{"type": "text", "text": formatted_question}]}]
    for f in frames:
        msgs[0]["content"].append({"type": "image", "image": f})
    pack0 = _now()
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=text, images=frames, return_tensors="pt").to(model.device)
    t_pack = _now() - pack0
    out, ttft, total, tok_s = stream_generate(model, proc, {**inputs, "max_new_tokens": max_new_tokens, "do_sample": False})
    return out, ttft, total, tok_s, t_pack


def run_blip2_image(model, proc, image_np, question: str, max_new_tokens: int):
    # Format question for better short answers
    formatted_question = f"{question} Give a short answer:"
    pack0 = _now()
    inputs = proc(images=image_np, text=formatted_question, return_tensors="pt").to(model.device)
    t_pack = _now() - pack0
    out, ttft, total, tok_s = stream_generate(model, getattr(proc, "tokenizer", proc), {**inputs, "max_new_tokens": max_new_tokens, "do_sample": False})
    return out, ttft, total, tok_s, t_pack


def run_qwen2_audio(model, proc, wav: np.ndarray, sr: int, question: str, max_new_tokens: int):
    pack0 = _now()
    inputs = proc(text=question, audios=wav, sampling_rate=sr, return_tensors="pt").to(model.device)
    t_pack = _now() - pack0
    out, ttft, total, tok_s = stream_generate(model, proc, {**inputs, "max_new_tokens": max_new_tokens, "do_sample": False})
    return out, ttft, total, tok_s, t_pack







# ============ Qwen2.5-Omni Support (Fixed v2) ============
import gc

def load_qwen25_omni(model_dir: str, dtype: str = "bf16"):
    """Load Qwen2.5-Omni model for true multimodal processing."""
    set_offline_env()
    from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
    import torch
    
    proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=get_dtype(dtype),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"
    ).eval()
    
    return model, proc


def clear_gpu_memory():
    """Clear GPU memory cache."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_qwen25_omni_single(model, proc, content_list: list, question: str, max_new_tokens: int):
    """
    Run Qwen2.5-Omni with flexible multimodal inputs.
    """
    import torch
    
    pack0 = _now()
    clear_gpu_memory()
    
    # Separate content by type
    images_data = []
    video_frames = []
    audio_data = None
    audio_sr = None
    
    for item in content_list:
        if item["type"] == "image":
            images_data.append(item["image"])
        elif item["type"] == "video":
            video_frames.extend(item["frames"])
        elif item["type"] == "audio":
            audio_data = item["audio"][0]
            audio_sr = item["audio"][1]
    
    # Build message content
    content = []
    for _ in images_data:
        content.append({"type": "image"})
    for _ in video_frames:
        content.append({"type": "image"})
    if audio_data is not None:
        content.append({"type": "audio"})
    content.append({"type": "text", "text": question})
    
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    
    all_images = images_data + video_frames if (images_data or video_frames) else None
    
    try:
        if audio_data is not None and all_images:
            inputs = proc(text=text, images=all_images, audios=[audio_data], 
                         sampling_rate=audio_sr, return_tensors="pt").to(model.device)
        elif audio_data is not None:
            inputs = proc(text=text, audios=[audio_data], 
                         sampling_rate=audio_sr, return_tensors="pt").to(model.device)
        elif all_images:
            inputs = proc(text=text, images=all_images, return_tensors="pt").to(model.device)
        else:
            inputs = proc(text=text, return_tensors="pt").to(model.device)
    except Exception as e:
        print(f"Processor error: {e}")
        return "", 0, 0, 0, 0
    
    t_pack = _now() - pack0
    
    try:
        # Use model's native generate (not streaming for simplicity)
        gen0 = _now()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_audio=False,  # Disable audio output to save memory
            )
        total = _now() - gen0
        
        # Decode
        out_text = proc.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        n_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
        tok_s = n_tokens / total if total > 0 else 0
        ttft = total / n_tokens if n_tokens > 0 else total  # Approximate
        
    except torch.cuda.OutOfMemoryError:
        clear_gpu_memory()
        print("OOM - skipping")
        return "", 0, 0, 0, t_pack
    except Exception as e:
        print(f"Generate error: {e}")
        return "", 0, 0, 0, t_pack
    
    del inputs
    clear_gpu_memory()
    
    return out_text, ttft, total, tok_s, t_pack


def load_dataset(manifest_path: str, n_samples: int = 50) -> List[Dict]:
    """加载数据集"""
    import pandas as pd
    df = pd.read_csv(manifest_path)
    
    samples = []
    for i, row in df.head(n_samples).iterrows():
        video_path = row.get("video_path", "")
        if os.path.exists(video_path):
            samples.append({
                "sample_id": row.get("sample_id", f"sample_{i}"),
                "video_path": video_path,
                "question": row.get("question", "Describe what you see and hear."),
            })
    
    return samples
