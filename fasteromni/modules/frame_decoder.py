"""
I 帧解码模块

使用 PyAV 只解码选中 GOP 的 I 帧（keyframe），跳过 P/B 帧。
输出格式兼容 Qwen2.5-Omni 的 processor 输入（PIL Image 列表）。

关键优化点：
- 只 seek + decode I 帧，不解码整个视频
- 返回 PIL Image 列表，可直接传入 fetch_video 的 list[Image] 分支
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import av
import numpy as np
from PIL import Image

from .sparse import ScoredGOP


def decode_i_frames(
    video_path: str,
    scored_gops: List[ScoredGOP],
) -> Tuple[List[Image.Image], float]:
    """
    解码选中 GOP 的 I 帧。

    使用 container.decode(stream) 遍历解码帧，遇到 keyframe 时
    检查 pts 是否属于选中的 GOP。

    注意：不能用 packet.decode()，因为 H.264 解码器需要多个 packet
    才能输出一帧，单独解码 keyframe packet 会返回空。

    Args:
        video_path: 视频文件路径
        scored_gops: 打分后的 GOP 列表（含 .selected 标记）

    Returns:
        (frames, decode_ms): PIL Image 列表 + 解码耗时
    """
    t0 = time.perf_counter()

    # 收集选中 GOP 的 I 帧 pts
    selected_pts = set()
    for sg in scored_gops:
        if sg.selected:
            selected_pts.add(sg.gop.i_frame_pts)

    if not selected_pts:
        return [], 0.0

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    video_stream.thread_type = "AUTO"

    frames: List[Image.Image] = []

    # 使用 container.decode() 遍历解码帧
    for frame in container.decode(video_stream):
        if frame.key_frame and frame.pts in selected_pts:
            img = frame.to_image()  # 转为 PIL Image (RGB)
            frames.append(img)
            # 如果所有选中的 I 帧都找到了，提前结束
            if len(frames) >= len(selected_pts):
                break

    container.close()

    decode_ms = (time.perf_counter() - t0) * 1000
    return frames, decode_ms


def decode_i_frames_seek(
    video_path: str,
    scored_gops: List[ScoredGOP],
) -> Tuple[List[Image.Image], float]:
    """
    使用 seek 方式解码 I 帧（适用于 GOP 数量少、视频很长的场景）。

    对于每个选中的 GOP，seek 到 I 帧位置再解码。
    当选中的 GOP 很少时，比遍历全部 packet 更快。

    Args:
        video_path: 视频文件路径
        scored_gops: 打分后的 GOP 列表

    Returns:
        (frames, decode_ms): PIL Image 列表 + 解码耗时
    """
    t0 = time.perf_counter()

    selected = [sg for sg in scored_gops if sg.selected]
    # 按时间排序
    selected.sort(key=lambda sg: sg.gop.start_time_sec or 0)

    if not selected:
        return [], 0.0

    frames: List[Image.Image] = []
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    video_stream.thread_type = "AUTO"
    time_base = float(video_stream.time_base)

    for sg in selected:
        target_pts = sg.gop.i_frame_pts
        # Seek 到目标 pts 附近（backward seek 到最近的 keyframe）
        try:
            container.seek(target_pts, stream=video_stream)
            for frame in container.decode(video_stream):
                if frame.key_frame:
                    img = frame.to_image()
                    frames.append(img)
                    break
        except Exception:
            # Seek 失败时跳过
            continue

    container.close()

    decode_ms = (time.perf_counter() - t0) * 1000
    return frames, decode_ms


def frames_to_tensor(
    frames: List[Image.Image],
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> "torch.Tensor":
    """
    将 PIL Image 列表转为 torch.Tensor (T, C, H, W)。

    如果指定了 target_height/target_width，会先 resize。
    否则保持原始尺寸（所有帧必须尺寸一致）。

    Args:
        frames: PIL Image 列表
        target_height: 目标高度
        target_width: 目标宽度

    Returns:
        torch.Tensor (T, C, H, W) float32, 值域 [0, 255]
    """
    import torch

    if not frames:
        return torch.empty(0, 3, 0, 0)

    tensors = []
    for img in frames:
        if target_height and target_width:
            img = img.resize((target_width, target_height), Image.BICUBIC)
        arr = np.array(img)  # (H, W, C)
        t = torch.from_numpy(arr).permute(2, 0, 1).float()  # (C, H, W)
        tensors.append(t)

    return torch.stack(tensors)  # (T, C, H, W)
