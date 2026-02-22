#!/usr/bin/env python3
"""
Generate publication-quality figures for FasterOmni analysis.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = "/root/autodl-tmp/results/figures"


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "grid.color": "#999999",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def _annotate_kr(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, kr: List[float],
    offsets: dict = None,
) -> None:
    """Annotate kr values with per-point offset overrides to avoid overlap."""
    default_offset = (4, 8)
    for xi, yi, kri in zip(x, y, kr):
        ofs = (offsets or {}).get(kri, default_offset)
        if ofs == (0, 0):  # skip – handled separately (e.g. bold kr=0.5)
            continue
        ax.annotate(
            f"kr={kri}",
            (xi, yi),
            textcoords="offset points",
            xytext=ofs,
            fontsize=9,
        )


def plot_pareto_curve() -> None:
    # ── data ──
    kr_vals = [0.2, 0.3, 0.5, 0.7, 0.9]
    naive_acc = [68.52, 69.44, 75.93, 71.30, 70.37]
    naive_tok = [2192, 3190, 4939, 6658, 7794]
    naive_spd = [3.6, 2.8, 2.1, 1.6, 1.4]
    sparse_acc = [70.37, 69.44, 69.44, 62.04, 64.81]
    sparse_tok = [2259, 3299, 5041, 6735, 7824]
    baseline_acc = 75.93

    bbox_white = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # ── helper: plot one subplot ──
    def _plot_one(ax, naive_x, sparse_x, xlabel, title, kr05_text):
        # curves
        ax.plot(naive_x, naive_acc, "o-", color="#1f77b4", lw=2.2,
                ms=8, label="naive_iframe", zorder=3)
        ax.plot(sparse_x, sparse_acc, "^--", color="#d62728", lw=2.0,
                ms=7, label="sparse", zorder=3)
        ax.axhline(y=baseline_acc, color="gray", ls="--", lw=1.5,
                   label=f"Baseline {baseline_acc}%", zorder=1)

        # kr labels – place each one manually with white bbox
        # positions: (data_x, data_y) → (text offset dx, dy in points)
        offsets_map = {
            0.2: (0, -18),
            0.3: (0, 10),
            0.5: None,      # special handling below
            0.7: (0, 10),
            0.9: (0, -18),
        }
        for i, kv in enumerate(kr_vals):
            ofs = offsets_map[kv]
            if ofs is None:
                continue
            ax.annotate(
                f"kr={kv}", (naive_x[i], naive_acc[i]),
                textcoords="offset points", xytext=ofs,
                fontsize=8.5, ha="center", va="center",
                bbox=bbox_white, zorder=6,
            )

        # highlight kr=0.5
        i05 = kr_vals.index(0.5)
        ax.scatter([naive_x[i05]], [naive_acc[i05]], s=200,
                   fc="#1f77b4", ec="black", lw=1.5, zorder=7)
        ax.annotate(
            f"kr=0.5\n{kr05_text}",
            (naive_x[i05], naive_acc[i05]),
            textcoords="offset points", xytext=(0, -32),
            fontsize=9.5, fontweight="bold", color="#1f77b4",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4",
                      alpha=0.95, lw=1.2),
            zorder=8,
        )

        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel(xlabel, fontsize=12)

    # ── left: Accuracy vs Visual Tokens ──
    _plot_one(ax1, naive_tok, sparse_tok,
              "Visual Tokens", "Accuracy vs Visual Tokens",
              "zero-loss, −54% tokens")
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.invert_xaxis()

    # ── right: Accuracy vs Speedup ──
    _plot_one(ax2, naive_spd, [naive_spd[i] for i in range(5)],
              "Speedup (×)", "Accuracy vs Speedup",
              "2.1× faster, zero-loss")
    # sparse uses same speedup proxy (kr order)
    ax2.lines[1].set_xdata(naive_spd)
    ax2.lines[1].set_ydata(sparse_acc)

    # ── shared settings ──
    for ax in (ax1, ax2):
        ax.set_ylim(59, 79)
        ax.tick_params(labelsize=11)

    # legend: outside plot, centered at top
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=True, framealpha=0.95, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUTPUT_DIR, "pareto_curve.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "pareto_curve.pdf"), bbox_inches="tight")
    plt.close(fig)


def _short_task_name(name: str) -> str:
    mapping: Dict[str, str] = {
        "counterfactual_inference": "Counterfactual",
        "object_existence": "Object Existence",
        "moving_attribute": "Moving Attribute",
        "moving_direction": "Moving Direction",
        "action_sequence": "Action Sequence",
        "action_prediction": "Action Prediction",
        "moving_count": "Moving Count",
        "object_interaction": "Object Interaction",
        "action_antonym": "Action Antonym",
        "character_order": "Character Order",
        "action_localization": "Action Localization",
        "egocentric_navigation": "Egocentric Nav.",
        "object_shuffle": "Object Shuffle",
        "unexpected_action": "Unexpected Action",
        "state_change": "State Change",
        "scene_transition": "Scene Transition",
        "fine_grained_action": "Fine-grained Action",
        "action_count": "Action Count",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _delta_color(delta: float) -> str:
    if delta < -20:
        return "#d73027"
    if delta < -5:
        return "#fc8d59"
    if delta < 0:
        return "#91bfdb"
    return "#1a9850"


def plot_mvbench_per_task() -> None:
    mvbench_tasks = {
        "task": [
            "counterfactual_inference",
            "object_existence",
            "moving_attribute",
            "moving_direction",
            "action_sequence",
            "action_prediction",
            "moving_count",
            "object_interaction",
            "action_antonym",
            "character_order",
            "action_localization",
            "egocentric_navigation",
            "object_shuffle",
            "unexpected_action",
            "state_change",
            "scene_transition",
            "fine_grained_action",
            "action_count",
        ],
        "baseline": [
            68.0,
            88.5,
            95.0,
            59.5,
            75.4,
            68.0,
            69.0,
            75.8,
            79.5,
            74.1,
            44.3,
            39.5,
            37.9,
            82.1,
            60.7,
            96.5,
            47.5,
            34.6,
        ],
        "naive_iframe": [
            32.0,
            55.5,
            62.5,
            37.0,
            53.0,
            46.0,
            47.0,
            58.5,
            69.3,
            64.0,
            35.0,
            32.0,
            35.0,
            80.0,
            59.5,
            96.5,
            48.7,
            51.0,
        ],
    }

    tasks = np.array(mvbench_tasks["task"])
    baseline = np.array(mvbench_tasks["baseline"])
    naive = np.array(mvbench_tasks["naive_iframe"])
    delta = naive - baseline

    sort_idx = np.argsort(delta)
    delta_sorted = delta[sort_idx]
    task_sorted = tasks[sort_idx]
    labels = [_short_task_name(t) for t in task_sorted]
    colors = [_delta_color(v) for v in delta_sorted]

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(delta_sorted))
    bars = ax.barh(y, delta_sorted, color=colors, edgecolor="black", linewidth=0.6)

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ Accuracy (pp)")
    ax.set_title("MVBench Per-Task Analysis: naive_iframe(kr=0.5) vs Baseline")

    xmin = float(np.min(delta_sorted))
    xmax = float(np.max(delta_sorted))
    xpad = max(1.0, 0.08 * (xmax - xmin))
    ax.set_xlim(xmin - xpad, xmax + xpad)

    label_offset = max(0.4, 0.02 * (xmax - xmin))
    for bar, d in zip(bars, delta_sorted):
        y_pos = bar.get_y() + bar.get_height() / 2
        x_pos = d + label_offset
        ax.text(
            x_pos,
            y_pos,
            f"{d:+.1f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#d73027", label="Delta < -20 (Severe)"),
        plt.Rectangle((0, 0), 1, 1, color="#fc8d59", label="-20 <= Delta < -5 (Moderate)"),
        plt.Rectangle((0, 0), 1, 1, color="#91bfdb", label="-5 <= Delta < 0 (Acceptable)"),
        plt.Rectangle((0, 0), 1, 1, color="#1a9850", label="Delta >= 0 (No-loss / Gain)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "mvbench_per_task.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "mvbench_per_task.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _setup_style()
    plot_pareto_curve()
    plot_mvbench_per_task()
    print(f"Saved figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
