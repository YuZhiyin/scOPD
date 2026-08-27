#!/usr/bin/env python3
"""Plot the DGPR sensitivity analysis for the rollout count M."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


M_VALUES = np.array([2, 4, 6, 8, 10])
CELL_LEVEL_ACCURACY = np.array([0.8106, 0.8159, 0.8131, 0.8171, 0.8189])
BATCH_LEVEL_ACCURACY = np.array([0.4252, 0.4437, 0.4405, 0.4449, 0.4458])


def plot_m_sensitivity(output_stem: Path) -> None:
    """Save the sensitivity figure as both PNG and PDF."""

    # Keep the source metrics in [0, 1] and convert only their presentation to
    # percentages. Thus, for example, 0.8106 is displayed as 81.06%.
    cell_percent = 100.0 * CELL_LEVEL_ACCURACY
    batch_percent = 100.0 * BATCH_LEVEL_ACCURACY

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 22,
            # Match the typography used by the disagreement bar chart.
            "axes.labelsize": 42,
            "xtick.labelsize": 32,
            "ytick.labelsize": 32,
            "legend.fontsize": 26,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    cell_color = "#E64B35CC"
    batch_color = "#4DBBD5CC"
    fig, cell_ax = plt.subplots(figsize=(11, 8), dpi=600)
    batch_ax = cell_ax.twinx()

    cell_line = cell_ax.plot(
        M_VALUES,
        cell_percent,
        color=cell_color,
        marker="o",
        linestyle="-",
        linewidth=3,
        markersize=14,
        clip_on=False,
        label="Cell-level",
    )[0]
    batch_line = batch_ax.plot(
        M_VALUES,
        batch_percent,
        color=batch_color,
        marker="^",
        linestyle="-",
        linewidth=3,
        markersize=15,
        clip_on=False,
        label="Batch-level",
    )[0]

    cell_ax.set_xlabel("Number of Rollouts ($M$)", labelpad=14)
    cell_ax.set_ylabel("Cell-level Acc (%)", color="black", labelpad=14)
    batch_ax.set_ylabel("Batch-level Acc (%)", color="black", labelpad=14)
    cell_ax.set_xticks(M_VALUES)
    cell_ax.set_xticklabels([str(value) for value in M_VALUES])
    # Give both y-axes the same numerical range and tick spacing. This aligns
    # every horizontal grid line and puts both lowest ticks on the bottom spine.
    cell_ax.set_ylim(80.0, 83.0)
    batch_ax.set_ylim(42.0, 45.0)
    cell_ax.set_yticks(np.arange(80.0, 83.0 + 0.1, 1.0))
    batch_ax.set_yticks(np.arange(42.0, 45.0 + 0.1, 1.0))
    cell_ax.grid(axis="both", linestyle="--", linewidth=1.0, alpha=0.55)

    cell_ax.tick_params(
        axis="x",
        bottom=False,
        top=False,
        labelcolor="black",
    )
    cell_ax.tick_params(
        axis="y",
        left=False,
        right=False,
        labelcolor="black",
    )
    batch_ax.tick_params(
        axis="y",
        left=False,
        right=False,
        labelcolor="black",
    )
    for ax in (cell_ax, batch_ax):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
    cell_ax.spines["left"].set_color("black")
    batch_ax.spines["right"].set_color("black")

    cell_ax.legend(
        handles=[cell_line, batch_line],
        labels=["Cell-level", "Batch-level"],
        loc="lower right",
        frameon=False,
        ncol=1,
    )

    fig.subplots_adjust(left=0.16, right=0.84, bottom=0.16, top=0.95)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "dgpr_m_sensitivity",
        help="Output path without an extension (default: next to this script).",
    )
    args = parser.parse_args()
    plot_m_sensitivity(args.output)
    print(f"Saved {args.output.with_suffix('.png')}")
    print(f"Saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
