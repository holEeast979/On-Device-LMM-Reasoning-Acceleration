from __future__ import annotations

from . import token_prefill
from . import audio_padding
from . import multiturn
from . import ttft_breakdown


def register_all(subparsers, common_parser) -> None:
    token_prefill.register_subcommand(subparsers, common_parser)
    audio_padding.register_subcommand(subparsers, common_parser)
    multiturn.register_subcommand(subparsers, common_parser)
    ttft_breakdown.register_subcommand(subparsers, common_parser)
