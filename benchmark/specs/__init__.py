from __future__ import annotations

from . import multiturn
from . import ttft_breakdown
from . import ttft_10videos
from . import vllm_comparison
from . import gpu_memory_trace
from . import token_scaling


def register_all(subparsers, common_parser) -> None:
    multiturn.register_subcommand(subparsers, common_parser)
    ttft_breakdown.register_subcommand(subparsers, common_parser)
    ttft_10videos.register_subcommand(subparsers, common_parser)
    vllm_comparison.register_subcommand(subparsers, common_parser)
    gpu_memory_trace.register_subcommand(subparsers, common_parser)
    token_scaling.register_subcommand(subparsers, common_parser)
