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


def _annotate_kr(ax: plt.Axes, x: np.ndarray, y: np.ndarray, kr: List[float]) -> None:
    for xi, yi, kri in zip(x, y, kr):
        ax.annotate(
            f"kr={kri:.1f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=10,
        )


def plot_pareto_curve() -> None:
    pareto_data = {
        "kr": [0.2, 0.3, 0.5, 0.7, 0.9],
        "accuracy": [68.52, 69.44, 75.93, 71.30, 70.37],
        "visual_tokens": [2192, 3190, 4939, 6658, 7794],
        "speedup": [3.6, 2.8, 2.1, 1.6, 1.4],
    }
    baseline_accuracy = 75.93
    baseline_visual_tokens = 10692
    sparse_data = {
        "kr": [0.2, 0.3, 0.5, 0.7, 0.9],
        "accuracy": [70.37, 69.44, 69.44, 62.04, 64.81],
        "visual_tokens": [2259, 3299, 5041, 6735, 7824],
    }

    kr = np.array(pareto_data["kr"])
    acc = np.array(pareto_data["accuracy"])
    tokens = np.array(pareto_data["visual_tokens"])
    speed = np.array(pareto_data["speedup"])
    sparse_acc = np.array(sparse_data["accuracy"])
    sparse_tokens = np.array(sparse_data["visual_tokens"])

    token_sort_idx = np.argsort(-tokens)
    speed_sort_idx = np.argsort(speed)
    sparse_token_sort_idx = np.argsort(-sparse_tokens)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax1.plot(
        tokens[token_sort_idx],
        acc[token_sort_idx],
        color="#1f77b4",
        marker="o",
        linestyle="-",
        linewidth=2.0,
        label="naive_iframe",
    )
    ax1.plot(
        sparse_tokens[sparse_token_sort_idx],
        sparse_acc[sparse_token_sort_idx],
        color="#d62728",
        marker="^",
        linestyle="--",
        linewidth=2.0,
        label="sparse",
    )
    ax1.axhline(
        y=baseline_accuracy,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Baseline 75.93%",
    )
    ax1.text(
        baseline_visual_tokens * 0.98,
        baseline_accuracy + 0.2,
        "Baseline 75.93%",
        color="gray",
        fontsize=10,
        ha="right",
    )
    _annotate_kr(ax1, tokens[token_sort_idx], acc[token_sort_idx], kr[token_sort_idx].tolist())

    idx_05 = int(np.where(np.isclose(kr, 0.5))[0][0])
    ax1.scatter(
        [tokens[idx_05]],
        [acc[idx_05]],
        s=170,
        facecolor="#1f77b4",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
    )
    ax1.annotate(
        "kr=0.5 (zero-loss)",
        (tokens[idx_05], acc[idx_05]),
        textcoords="offset points",
        xytext=(10, -24),
        fontsize=10,
        fontweight="bold",
    )
    token_reduction = (1.0 - tokens[idx_05] / baseline_visual_tokens) * 100.0
    ax1.annotate(
        f"-{token_reduction:.0f}% tokens",
        (tokens[idx_05], acc[idx_05]),
        textcoords="offset points",
        xytext=(10, -40),
        fontsize=10,
        color="#1f77b4",
    )

    ax1.set_title("Accuracy vs Visual Tokens")
    ax1.set_xlabel("Visual Tokens")
    ax1.set_ylabel("Accuracy (%)")
    ax1.invert_xaxis()
    ax1.legend(loc="lower left", frameon=True)

    ax2.plot(
        speed[speed_sort_idx],
        acc[speed_sort_idx],
        color="#1f77b4",
        marker="o",
        linestyle="-",
        linewidth=2.0,
        label="naive_iframe",
    )
    ax2.plot(
        speed[speed_sort_idx],
        sparse_acc[speed_sort_idx],
        color="#d62728",
        marker="^",
        linestyle="--",
        linewidth=2.0,
        label="sparse",
    )
    ax2.axhline(
        y=baseline_accuracy,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Baseline 75.93%",
    )
    ax2.text(
        np.max(speed) * 0.98,
        baseline_accuracy + 0.2,
        "Baseline 75.93%",
        color="gray",
        fontsize=10,
        ha="right",
    )
    _annotate_kr(ax2, speed[speed_sort_idx], acc[speed_sort_idx], kr[speed_sort_idx].tolist())
    ax2.scatter(
        [speed[idx_05]],
        [acc[idx_05]],
        s=170,
        facecolor="#1f77b4",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
    )
    ax2.annotate(
        "2.1x faster, zero-loss",
        (speed[idx_05], acc[idx_05]),
        textcoords="offset points",
        xytext=(10, -24),
        fontsize=10,
        fontweight="bold",
    )
    ax2.set_title("Accuracy vs Speedup")
    ax2.set_xlabel("Speedup (x)")
    ax2.legend(loc="lower left", frameon=True)

    for ax in (ax1, ax2):
        ax.set_ylim(60, 78)

    fig.tight_layout()
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
