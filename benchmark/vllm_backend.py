"""
vLLM Backend for Qwen2.5-Omni (Thinker only, video input)

Note: vLLM only supports the Thinker module of Qwen2.5-Omni
      - Supports: text, image, video
      - Does NOT support: audio input/output

Usage:
    backend = VLLMBackend(model_dir="/path/to/Qwen2.5-Omni-7B")
    backend.load()
    result = backend.generate_video(video_path, question, max_tokens=50)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VLLMGenerateResult:
    """vLLM 生成结果"""
    output_text: str
    ttft_ms: float              # Time to first token
    total_ms: float             # Total generation time
    num_tokens: int             # Number of output tokens
    prompt_tokens: int          # Number of input tokens
    tokens_per_sec: float       # Throughput


class VLLMBackend:
    """vLLM 推理后端封装"""
    
    def __init__(
        self,
        model_dir: str,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 32768,
        trust_remote_code: bool = True,
    ):
        self.model_dir = str(model_dir)
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        
        self._llm = None
        self._sampling_params = None
        self._processor = None
    
    def load(self) -> None:
        """加载 vLLM 模型"""
        if self._llm is not None:
            return
        
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Please install with:\n"
                "  pip install vllm>=0.7.0"
            )
        
        print(f"Loading vLLM model from {self.model_dir}...")
        
        self._llm = LLM(
            model=self.model_dir,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            limit_mm_per_prompt={"image": 128, "video": 1},  # Qwen2.5-Omni limits
        )
        
        # Load processor for chat template
        from transformers import AutoProcessor
        self._processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        )
        
        print("vLLM model loaded successfully")
    
    def generate_video(
        self,
        video_path: str,
        question: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        top_p: float = 1.0,
        video_nframes: Optional[int] = None,
    ) -> VLLMGenerateResult:
        """
        使用 vLLM 生成视频问答
        
        Args:
            video_path: 视频文件路径
            question: 问题文本
            max_tokens: 最大生成 token 数
            temperature: 采样温度 (0=greedy)
            top_p: nucleus sampling
            video_nframes: 采样帧数 (None=自动)
        
        Returns:
            VLLMGenerateResult
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        from vllm import SamplingParams
        
        # 构建消息
        video_element = {"type": "video", "video": str(video_path)}
        if video_nframes is not None and int(video_nframes) > 0:
            video_element["nframes"] = int(video_nframes)
        
        messages = [
            {
                "role": "user",
                "content": [
                    video_element,
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # 应用 chat template
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 准备多模态数据
        from qwen_omni_utils import process_mm_info
        
        t_preprocess_start = time.perf_counter()
        _, images, videos = process_mm_info(messages, use_audio_in_video=False)
        t_preprocess = (time.perf_counter() - t_preprocess_start) * 1000
        
        # vLLM 输入
        mm_data = {}
        if videos:
            mm_data["video"] = videos
        
        # Sampling params
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Generate
        t_gen_start = time.perf_counter()
        outputs = self._llm.generate(
            [{"prompt": prompt, "multi_modal_data": mm_data}],
            sampling_params=sampling_params,
        )
        t_gen_total = (time.perf_counter() - t_gen_start) * 1000
        
        if not outputs or not outputs[0].outputs:
            return VLLMGenerateResult(
                output_text="",
                ttft_ms=t_preprocess + t_gen_total,
                total_ms=t_preprocess + t_gen_total,
                num_tokens=0,
                prompt_tokens=0,
                tokens_per_sec=0.0,
            )
        
        output = outputs[0]
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        prompt_tokens = len(output.prompt_token_ids)
        
        # vLLM 不直接提供 TTFT，用总时间估算
        # TTFT ≈ 预处理 + prefill + 1 token decode
        # 这里用 total_time / num_tokens 作为单 token 时间估算
        if num_tokens > 0:
            single_token_ms = t_gen_total / num_tokens
            ttft_estimate = t_preprocess + t_gen_total - (num_tokens - 1) * single_token_ms
        else:
            ttft_estimate = t_preprocess + t_gen_total
        
        tokens_per_sec = num_tokens / (t_gen_total / 1000) if t_gen_total > 0 else 0.0
        
        return VLLMGenerateResult(
            output_text=generated_text,
            ttft_ms=ttft_estimate,
            total_ms=t_preprocess + t_gen_total,
            num_tokens=num_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_sec=tokens_per_sec,
        )
    
    def generate_video_streaming(
        self,
        video_path: str,
        question: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        video_nframes: Optional[int] = None,
    ) -> VLLMGenerateResult:
        """
        使用 vLLM 流式生成，精确测量 TTFT
        
        Note: 需要 vLLM 支持流式输出
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        from vllm import SamplingParams
        
        # 构建消息
        video_element = {"type": "video", "video": str(video_path)}
        if video_nframes is not None and int(video_nframes) > 0:
            video_element["nframes"] = int(video_nframes)
        
        messages = [
            {
                "role": "user",
                "content": [
                    video_element,
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        from qwen_omni_utils import process_mm_info
        
        t_preprocess_start = time.perf_counter()
        _, images, videos = process_mm_info(messages, use_audio_in_video=False)
        t_preprocess = (time.perf_counter() - t_preprocess_start) * 1000
        
        mm_data = {}
        if videos:
            mm_data["video"] = videos
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # 使用流式生成精确测量 TTFT
        t_gen_start = time.perf_counter()
        ttft_ms = None
        output_text = ""
        num_tokens = 0
        prompt_tokens = 0
        
        try:
            # vLLM 流式生成
            for output in self._llm.generate(
                [{"prompt": prompt, "multi_modal_data": mm_data}],
                sampling_params=sampling_params,
                use_tqdm=False,
            ):
                if ttft_ms is None and output.outputs:
                    ttft_ms = (time.perf_counter() - t_gen_start) * 1000 + t_preprocess
                
                if output.outputs:
                    output_text = output.outputs[0].text
                    num_tokens = len(output.outputs[0].token_ids)
                    prompt_tokens = len(output.prompt_token_ids)
        except Exception as e:
            # 如果流式不支持，回退到非流式
            print(f"Streaming not supported, falling back: {e}")
            return self.generate_video(video_path, question, max_tokens, temperature, 1.0, video_nframes)
        
        t_gen_total = (time.perf_counter() - t_gen_start) * 1000
        
        if ttft_ms is None:
            ttft_ms = t_preprocess + t_gen_total
        
        tokens_per_sec = num_tokens / (t_gen_total / 1000) if t_gen_total > 0 else 0.0
        
        return VLLMGenerateResult(
            output_text=output_text,
            ttft_ms=ttft_ms,
            total_ms=t_preprocess + t_gen_total,
            num_tokens=num_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_sec=tokens_per_sec,
        )

    def measure_ttft(self, video_path: str, question: str, video_nframes: Optional[int] = None) -> float:
        """
        精确测量 TTFT：使用 max_tokens=1 直接测量生成第一个 token 的时间。
        
        这比估算更准确，因为不需要假设所有 token 生成时间相同。
        
        Returns:
            TTFT 时间（毫秒），不包含预处理时间
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        from vllm import SamplingParams
        
        # 构建消息
        video_element = {"type": "video", "video": str(video_path)}
        if video_nframes is not None and int(video_nframes) > 0:
            video_element["nframes"] = int(video_nframes)
        
        messages = [
            {
                "role": "user",
                "content": [
                    video_element,
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        from qwen_omni_utils import process_mm_info
        _, images, videos = process_mm_info(messages, use_audio_in_video=False)
        
        mm_data = {}
        if videos:
            mm_data["video"] = videos
        
        # 只生成 1 个 token
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
        )
        
        # 精确测量
        t_gen_start = time.perf_counter()
        outputs = self._llm.generate(
            [{"prompt": prompt, "multi_modal_data": mm_data}],
            sampling_params=sampling_params,
        )
        ttft_ms = (time.perf_counter() - t_gen_start) * 1000
        
        return ttft_ms

    def measure_ttft_with_breakdown(
        self, 
        video_path: str, 
        question: str, 
        video_nframes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        测量 TTFT 并返回阶段分解。
        
        vLLM 是黑盒，无法分解 encoder/prefill，所以只返回：
        - preprocess_ms: 视频提取+处理时间
        - model_ms: vLLM 模型推理时间 (encoder + prefill + 1 token decode)
        - ttft_ms: 总 TTFT
        - prompt_tokens: prompt token 数
        
        Returns:
            Dict with timing breakdown
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        from vllm import SamplingParams
        
        # 构建消息
        video_element = {"type": "video", "video": str(video_path)}
        if video_nframes is not None and int(video_nframes) > 0:
            video_element["nframes"] = int(video_nframes)
        
        messages = [
            {
                "role": "user",
                "content": [
                    video_element,
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # === Preprocess ===
        from qwen_omni_utils import process_mm_info
        t_preprocess_start = time.perf_counter()
        _, images, videos = process_mm_info(messages, use_audio_in_video=False)
        preprocess_ms = (time.perf_counter() - t_preprocess_start) * 1000
        
        mm_data = {}
        if videos:
            mm_data["video"] = videos
        
        # 只生成 1 个 token
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
        )
        
        # === Model inference ===
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t_model_start = time.perf_counter()
        outputs = self._llm.generate(
            [{"prompt": prompt, "multi_modal_data": mm_data}],
            sampling_params=sampling_params,
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_ms = (time.perf_counter() - t_model_start) * 1000
        
        prompt_tokens = len(outputs[0].prompt_token_ids) if outputs else 0
        
        return {
            "preprocess_ms": preprocess_ms,
            "model_ms": model_ms,
            "ttft_ms": preprocess_ms + model_ms,
            "prompt_tokens": prompt_tokens,
        }


class HFBackend:
    """HuggingFace 推理后端封装 (用于公平对比)"""
    
    def __init__(
        self,
        model_dir: str,
        dtype: str = "bf16",
    ):
        self.model_dir = str(model_dir)
        self.dtype = dtype
        
        self._model = None
        self._processor = None
    
    def load(self) -> None:
        """加载 HF 模型"""
        if self._model is not None:
            return
        
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
        
        print(f"Loading HF model from {self.model_dir}...")
        
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        torch_dtype = dtype_map.get(self.dtype.lower(), torch.bfloat16)
        
        self._processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        )
        
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).eval()
        
        print("HF model loaded successfully")
    
    def generate_video(
        self,
        video_path: str,
        question: str,
        max_tokens: int = 50,
        video_nframes: Optional[int] = None,
    ) -> VLLMGenerateResult:
        """
        使用 HF 生成视频问答 (仅视频，无音频，用于与 vLLM 公平对比)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        import torch
        from qwen_omni_utils import process_mm_info
        
        # 构建消息 (无音频)
        video_element = {"type": "video", "video": str(video_path)}
        if video_nframes is not None and int(video_nframes) > 0:
            video_element["nframes"] = int(video_nframes)
        
        messages = [
            {
                "role": "user",
                "content": [
                    video_element,
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # 预处理
        t_preprocess_start = time.perf_counter()
        
        _, images, videos = process_mm_info(messages, use_audio_in_video=False)
        
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self._processor(
            text=prompt,
            videos=videos,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)
        
        t_preprocess = (time.perf_counter() - t_preprocess_start) * 1000
        
        prompt_tokens = inputs.input_ids.shape[-1]
        
        # Generate
        torch.cuda.synchronize()
        t_gen_start = time.perf_counter()
        
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                return_audio=False,
            )
        
        torch.cuda.synchronize()
        t_gen_total = (time.perf_counter() - t_gen_start) * 1000
        
        # Decode
        generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]
        output_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        num_tokens = generated_ids.shape[-1]
        
        # 估算 TTFT (1 token 生成时间)
        if num_tokens > 0:
            single_token_ms = t_gen_total / num_tokens
            ttft_estimate = t_preprocess + t_gen_total - (num_tokens - 1) * single_token_ms
        else:
            ttft_estimate = t_preprocess + t_gen_total
        
        tokens_per_sec = num_tokens / (t_gen_total / 1000) if t_gen_total > 0 else 0.0
        
        return VLLMGenerateResult(
            output_text=output_text,
            ttft_ms=ttft_estimate,
            total_ms=t_preprocess + t_gen_total,
            num_tokens=num_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_sec=tokens_per_sec,
        )

    def measure_ttft(self, video_path: str, question: str, video_nframes: Optional[int] = None, use_cuda_event: bool = True) -> float:
        """
        精确测量 TTFT：使用 max_new_tokens=1 直接测量生成第一个 token 的时间。
        
        Args:
            video_path: 视频路径
            question: 问题文本
            video_nframes: 采样帧数
            use_cuda_event: 是否使用 CUDA Event 计时（更精确）
        
        Returns:
            TTFT 时间（毫秒），不包含预处理时间
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        import torch
        from qwen_omni_utils import process_mm_info
        
        # 构建消息
        video_element = {"type": "video", "video": str(video_path)}
        if video_nframes is not None and int(video_nframes) > 0:
            video_element["nframes"] = int(video_nframes)
        
        messages = [
            {
                "role": "user",
                "content": [
                    video_element,
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        _, images, videos = process_mm_info(messages, use_audio_in_video=False)
        
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self._processor(
            text=prompt,
            videos=videos,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)
        
        # 精确测量 TTFT
        if use_cuda_event and torch.cuda.is_available():
            device = self._model.device
            if device.type == "cuda":
                with torch.cuda.device(device):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    with torch.no_grad():
                        self._model.generate(
                            **inputs,
                            max_new_tokens=1,
                            do_sample=False,
                            return_audio=False,
                        )
                    end_event.record()
                    end_event.synchronize()
                    
                    return float(start_event.elapsed_time(end_event))
        
        # Fallback to wall-clock
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            self._model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_audio=False,
            )
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000


def create_backend(backend_type: str, model_dir: str, **kwargs):
    """工厂函数创建后端"""
    if backend_type.lower() == "vllm":
        return VLLMBackend(model_dir, **kwargs)
    elif backend_type.lower() in ("hf", "huggingface"):
        return HFBackend(model_dir, **kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
