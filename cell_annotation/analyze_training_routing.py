#!/usr/bin/env python3
"""Summarize routed GRPO/SDPO training dynamics from a verl console log.

The log can contain several resumed attempts.  For each optimizer step we retain
the last complete training record, which corresponds to the latest successful
continuation written to the append-only log.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+global_seqlen/")

METRICS = (
    "routing/sdpo_sample_count",
    "routing/grpo_sample_count",
    "routing/sdpo_token_fraction",
    "routing/grpo_token_fraction",
    "routing/strict_success_fraction",
    "routing/fallback_grpo_fraction",
    "routing/feedback_fallback_fraction",
    "routing/all_success_group_fraction",
    "routing/all_failure_group_fraction",
    "routing/mixed_group_fraction",
    "critic/score/mean",
    "response_length/mean",
    "response_length/clip_ratio",
)


def metric_value(line: str, name: str) -> float:
    match = re.search(rf"(?:^|\s){re.escape(name)}:([^\s]+)", line)
    if match is None:
        raise ValueError(f"missing metric {name!r}")
    return float(match.group(1))


def parse_latest_steps(log_path: Path) -> dict[int, dict[str, float]]:
    latest: dict[int, dict[str, float]] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            step_match = STEP_RE.search(line)
            if step_match is None or "routing/sdpo_sample_count:" not in line:
                continue
            try:
                record = {name: metric_value(line, name) for name in METRICS}
            except ValueError:
                continue
            latest[int(step_match.group(1))] = record
    if not latest:
        raise RuntimeError(f"no routed training records found in {log_path}")
    return latest


def aggregate_epochs(
    steps: dict[int, dict[str, float]], steps_per_epoch: int, total_epochs: int
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    expected_last_step = steps_per_epoch * total_epochs
    missing = [step for step in range(1, expected_last_step + 1) if step not in steps]
    if missing:
        preview = ", ".join(map(str, missing[:12]))
        raise RuntimeError(f"missing {len(missing)} optimizer steps; first missing: {preview}")

    for epoch_idx in range(total_epochs):
        start = epoch_idx * steps_per_epoch + 1
        stop = (epoch_idx + 1) * steps_per_epoch
        epoch_records = [steps[step] for step in range(start, stop + 1)]

        totals = np.asarray(
            [r["routing/sdpo_sample_count"] + r["routing/grpo_sample_count"] for r in epoch_records]
        )
        success = np.asarray(
            [r["routing/strict_success_fraction"] * n for r, n in zip(epoch_records, totals)]
        )
        fallback_grpo = np.asarray(
            [r["routing/fallback_grpo_fraction"] * n for r, n in zip(epoch_records, totals)]
        )
        o1_sdpo = np.asarray(
            [r["routing/feedback_fallback_fraction"] * n for r, n in zip(epoch_records, totals)]
        )
        sdpo = np.asarray([r["routing/sdpo_sample_count"] for r in epoch_records])
        sibling_sdpo = sdpo - o1_sdpo

        route_counts = np.asarray(
            [success.sum(), sibling_sdpo.sum(), o1_sdpo.sum(), fallback_grpo.sum()]
        )
        route_shares = route_counts / totals.sum()
        if not np.isclose(route_shares.sum(), 1.0, atol=1e-5):
            raise RuntimeError(
                f"epoch {epoch_idx + 1} route shares sum to {route_shares.sum():.8f}, not 1"
            )

        def weighted_mean(name: str) -> float:
            values = np.asarray([r[name] for r in epoch_records])
            return float(np.average(values, weights=totals))

        response_tokens = np.asarray(
            [r["response_length/mean"] * n for r, n in zip(epoch_records, totals)]
        )
        sdpo_token_share = float(
            sum(
                r["routing/sdpo_token_fraction"] * token_count
                for r, token_count in zip(epoch_records, response_tokens)
            )
            / response_tokens.sum()
        )

        rows.append(
            {
                "epoch": epoch_idx + 1,
                "start_step": start,
                "end_step": stop,
                "num_rollouts": float(totals.sum()),
                "grpo_success": float(route_shares[0]),
                "sibling_sdpo": float(route_shares[1]),
                "o1_sdpo": float(route_shares[2]),
                "grpo_fallback": float(route_shares[3]),
                "grpo_total": float(route_shares[0] + route_shares[3]),
                "sdpo_total": float(route_shares[1] + route_shares[2]),
                "grpo_token_total": 1.0 - sdpo_token_share,
                "sdpo_token_total": sdpo_token_share,
                "mean_dense_reward": weighted_mean("critic/score/mean"),
                "all_failure_groups": weighted_mean("routing/all_failure_group_fraction"),
                "mixed_groups": weighted_mean("routing/mixed_group_fraction"),
                "all_success_groups": weighted_mean("routing/all_success_group_fraction"),
                "mean_response_tokens": weighted_mean("response_length/mean"),
                "response_clip_ratio": weighted_mean("response_length/clip_ratio"),
            }
        )
    return rows


def save_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float]], output_dir: Path) -> None:
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    route_series = [
        ("Qualified GRPO", "grpo_success", "#3B82F6"),
        ("Sibling SDPO", "sibling_sdpo", "#F59E0B"),
        ("O1-fallback SDPO", "o1_sdpo", "#8B5CF6"),
        ("Fallback GRPO", "grpo_fallback", "#9CA3AF"),
    ]
    group_series = [
        ("All-failure groups", "all_failure_groups", "#A78BFA"),
        ("Mixed groups", "mixed_groups", "#FBBF24"),
        ("All-success groups", "all_success_groups", "#34D399"),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    bottom = np.zeros(len(rows))
    for label, key, color in route_series:
        values = np.asarray([row[key] for row in rows]) * 100.0
        axes[0].bar(epochs, values, bottom=bottom, width=0.68, label=label, color=color)
        for x, low, value in zip(epochs, bottom, values):
            if value >= 5.0:
                axes[0].text(
                    x,
                    low + value / 2,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if color != "#F59E0B" else "#3F2D00",
                    fontweight="bold",
                )
        bottom += values
    for x, row in zip(epochs, rows):
        axes[0].text(
            x,
            103.0,
            f"reward={row['mean_dense_reward']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_title("(a) Rollout-level routing by epoch")
    axes[0].set_xlabel("Training epoch")
    axes[0].set_ylabel("Fraction of rollouts (%)")
    axes[0].set_xticks(epochs)
    axes[0].set_ylim(0, 112)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=2, frameon=False)
    axes[0].grid(axis="y", alpha=0.18)

    bottom = np.zeros(len(rows))
    for label, key, color in group_series:
        values = np.asarray([row[key] for row in rows]) * 100.0
        axes[1].bar(epochs, values, bottom=bottom, width=0.68, label=label, color=color)
        for x, low, value in zip(epochs, bottom, values):
            if value >= 5.0:
                axes[1].text(
                    x,
                    low + value / 2,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#1F2937",
                    fontweight="bold",
                )
        bottom += values
    axes[1].set_title("(b) Composition of K=8 rollout groups")
    axes[1].set_xlabel("Training epoch")
    axes[1].set_ylabel("Fraction of prompt groups (%)")
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=2, frameon=False)
    axes[1].grid(axis="y", alpha=0.18)

    fig.suptitle(
        "Training-time Routing Dynamics: Dense GRPO + Sibling/O1 SDPO + Dynamic Weighting",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"training_time_routing_dynamics.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps-per-epoch", type=int, default=108)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps = parse_latest_steps(args.log)
    rows = aggregate_epochs(steps, args.steps_per_epoch, args.epochs)
    save_csv(rows, args.output_dir / "routing_dynamics_by_epoch.csv")
    plot(rows, args.output_dir)
    print(f"parsed latest records for {len(steps)} distinct optimizer steps")
    for row in rows:
        print(
            f"epoch {int(row['epoch'])}: reward={row['mean_dense_reward']:.4f}, "
            f"GRPO-success={row['grpo_success']:.4%}, "
            f"Sibling-SDPO={row['sibling_sdpo']:.4%}, "
            f"O1-SDPO={row['o1_sdpo']:.4%}, "
            f"GRPO-fallback={row['grpo_fallback']:.4%}"
        )
    print(f"wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
