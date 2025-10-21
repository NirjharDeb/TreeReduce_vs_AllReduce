#!/usr/bin/env python3
"""
Plot min/avg/max elapsed times vs # of PEs from multiple CSV files.

CSV format (header required):
    pes,min,avg,max  (order can vary; names matter)

Usage:
    python benchmark.py impl1.csv impl2.csv ...

Outputs:
    min_elapsed.png, avg_elapsed.png, max_elapsed.png
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main(argv):
    if len(argv) < 2:
        print("Usage: python benchmark.py impl1.csv impl2.csv ...")
        sys.exit(1)

    csv_paths = [Path(p) for p in argv[1:]]
    for p in csv_paths:
        if not p.exists():
            print(f"Error: file not found: {p}")
            sys.exit(2)

    metrics = ["min", "avg", "max"]
    figs_axes = {}
    for metric in metrics:
        fig, ax = plt.subplots()
        figs_axes[metric] = (fig, ax)

    required_cols = {"pes", "min", "avg", "max"}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        # Validate columns (order-agnostic)
        if not required_cols.issubset(df.columns):
            missing = sorted(list(required_cols - set(df.columns)))
            print(f"Error: {csv_path} missing columns: {missing}")
            sys.exit(3)

        # Coerce numeric, drop NA, and sort by PEs so lines connect correctly
        df = df.copy()
        for c in ["pes", "min", "avg", "max"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["pes", "min", "avg", "max"]).sort_values("pes")

        x = df["pes"].values
        label = csv_path.stem

        for metric in metrics:
            fig, ax = figs_axes[metric]
            ax.plot(x, df[metric].values, marker="o", label=label)

    titles = {
        "min": "Min Elapsed Time for Different Global Done Implementations",
        "avg": "Avg Elapsed Time for Different Global Done Implementations",
        "max": "Max Elapsed Time for Different Global Done Implementations",
    }

    for metric in metrics:
        fig, ax = figs_axes[metric]
        ax.set_title(titles[metric])
        ax.set_xlabel("Number of PEs")
        ax.set_ylabel("Elapsed Time (ms)")
        ax.set_ylim(bottom=0)  # y-axis starts at 0 ms
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="CSV File", fontsize="small")
        fig.tight_layout()
        fig.savefig(f"{metric}_elapsed.png", dpi=150)

    # Optional interactive display
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
