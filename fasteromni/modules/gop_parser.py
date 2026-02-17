"""
GOP (Group of Pictures) 解析模块

使用 PyAV 在 demux 层面解析视频的 GOP 结构，不需要完整解码。
每个 GOP 从一个 I 帧（keyframe）开始，包含后续的 P/B 帧直到下一个 I 帧。

核心输出：
- GOP 列表：每个 GOP 的 I 帧码率（packet.size）、帧数、时间范围
- I 帧码率作为"画面复杂度"的代理指标，用于后续 AV-LRM 打分

使用方式：
    from fasteromni.modules.gop_parser import parse_gops
    gops = parse_gops("/path/to/video.mp4")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import av


@dataclass
class GOPInfo:
    """单个 GOP 的元信息"""
    gop_index: int
    # I 帧信息
    i_frame_pts: int                # I 帧的 presentation timestamp
    i_frame_size: int               # I 帧的 packet 字节大小（码率代理）
    i_frame_dts: Optional[int] = None
    # 时间信息
    start_time_sec: Optional[float] = None   # GOP 起始时间（秒）
    end_time_sec: Optional[float] = None     # GOP 结束时间（秒）
    duration_sec: Optional[float] = None     # GOP 持续时间（秒）
    # 帧统计
    num_frames: int = 0             # GOP 内总帧数
    num_p_frames: int = 0           # P/B 帧数（demux 模式下无法区分 P 和 B）
    # 所有帧的 packet size 之和（用于计算 GOP 平均码率）
    total_packet_size: int = 0


@dataclass
class GOPAnalysis:
    """整个视频的 GOP 分析结果"""
    video_path: str
    video_duration_sec: float
    total_frames: int
    fps: float
    codec: str
    resolution: tuple  # (width, height)
    gops: List[GOPInfo] = field(default_factory=list)

    @property
    def num_gops(self) -> int:
        return len(self.gops)

    @property
    def avg_gop_frames(self) -> float:
        if not self.gops:
            return 0.0
        return sum(g.num_frames for g in self.gops) / len(self.gops)

    @property
    def i_frame_sizes(self) -> List[int]:
        return [g.i_frame_size for g in self.gops]

    @property
    def i_frame_ratio(self) -> float:
        """I 帧占总帧数的比例"""
        if self.total_frames == 0:
            return 0.0
        return len(self.gops) / self.total_frames

    def summary_dict(self) -> dict:
        """输出摘要字典，方便后续汇总分析"""
        sizes = self.i_frame_sizes
        import statistics
        return {
            "video_path": self.video_path,
            "duration_sec": round(self.video_duration_sec, 2),
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2),
            "codec": self.codec,
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "num_gops": self.num_gops,
            "avg_gop_frames": round(self.avg_gop_frames, 1),
            "i_frame_ratio": round(self.i_frame_ratio, 4),
            "i_frame_size_min": min(sizes) if sizes else 0,
            "i_frame_size_max": max(sizes) if sizes else 0,
            "i_frame_size_mean": round(statistics.mean(sizes)) if sizes else 0,
            "i_frame_size_std": round(statistics.stdev(sizes)) if len(sizes) > 1 else 0,
        }


def parse_gops(video_path: str) -> GOPAnalysis:
    """
    解析视频文件的 GOP 结构。

    通过 demux 遍历 packet（不解码），速度快。
    遇到 keyframe 就标记新 GOP 的开始。

    Args:
        video_path: 视频文件路径

    Returns:
        GOPAnalysis: 完整的 GOP 分析结果
    """
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # 视频基本信息
    time_base = float(video_stream.time_base)
    fps = float(video_stream.average_rate) if video_stream.average_rate else 25.0
    duration = float(video_stream.duration * time_base) if video_stream.duration else 0.0
    codec = video_stream.codec_context.name
    width = video_stream.codec_context.width
    height = video_stream.codec_context.height

    analysis = GOPAnalysis(
        video_path=video_path,
        video_duration_sec=duration,
        total_frames=0,
        fps=fps,
        codec=codec,
        resolution=(width, height),
    )

    gop_list: List[GOPInfo] = []
    current_gop: Optional[GOPInfo] = None
    gop_index = 0
    total_frames = 0
    last_pts = 0

    for packet in container.demux(video_stream):
        if packet.size == 0:
            continue

        total_frames += 1
        pts = packet.pts if packet.pts is not None else 0
        pts_sec = pts * time_base

        if packet.is_keyframe:
            # 结束上一个 GOP
            if current_gop is not None:
                current_gop.end_time_sec = pts_sec
                if current_gop.start_time_sec is not None:
                    current_gop.duration_sec = pts_sec - current_gop.start_time_sec
                gop_list.append(current_gop)

            # 开始新 GOP
            current_gop = GOPInfo(
                gop_index=gop_index,
                i_frame_pts=pts,
                i_frame_size=packet.size,
                i_frame_dts=packet.dts,
                start_time_sec=pts_sec,
                num_frames=1,
                num_p_frames=0,
                total_packet_size=packet.size,
            )
            gop_index += 1
        else:
            if current_gop is not None:
                current_gop.num_frames += 1
                current_gop.num_p_frames += 1
                current_gop.total_packet_size += packet.size

        last_pts = pts_sec

    # 处理最后一个 GOP
    if current_gop is not None:
        current_gop.end_time_sec = last_pts
        if current_gop.start_time_sec is not None and current_gop.end_time_sec is not None:
            current_gop.duration_sec = current_gop.end_time_sec - current_gop.start_time_sec
        gop_list.append(current_gop)

    container.close()

    analysis.gops = gop_list
    analysis.total_frames = total_frames

    return analysis


def print_gop_table(analysis: GOPAnalysis, max_rows: int = 30) -> None:
    """以表格形式打印 GOP 信息"""
    print(f"\n{'='*80}")
    print(f"Video: {analysis.video_path}")
    print(f"Duration: {analysis.video_duration_sec:.1f}s | FPS: {analysis.fps:.1f} "
          f"| Codec: {analysis.codec} | Resolution: {analysis.resolution[0]}x{analysis.resolution[1]}")
    print(f"Total frames: {analysis.total_frames} | GOPs: {analysis.num_gops} "
          f"| Avg frames/GOP: {analysis.avg_gop_frames:.1f} "
          f"| I-frame ratio: {analysis.i_frame_ratio:.2%}")
    print(f"{'='*80}")
    print(f"{'GOP':>4} | {'Start(s)':>8} | {'Dur(s)':>7} | {'Frames':>6} | "
          f"{'I-size(KB)':>10} | {'Total(KB)':>10} | {'I/Total':>7}")
    print(f"{'-'*4:>4}-+-{'-'*8:>8}-+-{'-'*7:>7}-+-{'-'*6:>6}-+-"
          f"{'-'*10:>10}-+-{'-'*10:>10}-+-{'-'*7:>7}")

    for g in analysis.gops[:max_rows]:
        start = f"{g.start_time_sec:.2f}" if g.start_time_sec is not None else "N/A"
        dur = f"{g.duration_sec:.2f}" if g.duration_sec is not None else "N/A"
        i_kb = g.i_frame_size / 1024
        total_kb = g.total_packet_size / 1024
        ratio = g.i_frame_size / g.total_packet_size if g.total_packet_size > 0 else 0
        print(f"{g.gop_index:>4} | {start:>8} | {dur:>7} | {g.num_frames:>6} | "
              f"{i_kb:>10.1f} | {total_kb:>10.1f} | {ratio:>7.1%}")

    if len(analysis.gops) > max_rows:
        print(f"  ... ({len(analysis.gops) - max_rows} more GOPs)")
    print()
