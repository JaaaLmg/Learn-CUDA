#!/usr/bin/env python3
"""Plot SGEMM benchmark results.

Reads every result_*.txt file in ./results, where each file holds one
implementation version's benchmark (columns: p, gflops, diff), and draws a
line chart of GFLOPS vs. matrix size (p), one colored line per version.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "results", "gflops_comparison.png")

# Validated categorical palette (light mode), assigned in fixed order.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# Chart chrome / ink (light mode).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"


def load_results(path):
    """Return (sizes, gflops) parsed from a result file, skipping the header."""
    sizes, gflops = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                p = int(parts[0])
                g = float(parts[1])
            except ValueError:
                continue  # header or malformed line
            sizes.append(p)
            gflops.append(g)
    return sizes, gflops


def version_label(path):
    """result_v0.txt -> v0"""
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    return name[len("result_"):] if name.startswith("result_") else name


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*.txt")))
    if not files:
        print(f"No result files found in {RESULTS_DIR}")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, path in enumerate(files):
        sizes, gflops = load_results(path)
        if not sizes:
            continue
        color = PALETTE[i % len(PALETTE)]
        ax.plot(
            sizes,
            gflops,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            label=version_label(path),
        )

    ax.set_xlabel("Matrix size (p)", color=INK_PRIMARY)
    ax.set_ylabel("Performance (GFLOPS)", color=INK_PRIMARY)
    ax.set_title("SGEMM Performance by Version", color=INK_PRIMARY)

    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.legend(frameon=False, labelcolor=INK_PRIMARY)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, facecolor=SURFACE)
    print(f"Saved chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
