#!/usr/bin/env python3
"""
Plot termination latency vs # of PEs from multiple CSV files.

Expected CSV format (header required):
    pes,termination_latency_ms

Usage:
    python benchmark_termination.py impl1.csv impl2.csv ...

Each CSV is treated as one implementation/config; its file stem
(name without extension) is used as the legend label.

Outputs:
    termination_latency.png
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main(argv):
    if len(argv) < 2:
        print("Usage: python benchmark_termination.py impl1.csv impl2.csv ...")
        sys.exit(1)

    csv_paths = [Path(p) for p in argv[1:]]
    for p in csv_paths:
        if not p.exists():
            print(f"Error: file not found: {p}")
            sys.exit(2)

    required_cols = {"pes", "termination_latency_ms"}

    fig, ax = plt.subplots()

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        # Validate columns (order-agnostic)
        if not required_cols.issubset(df.columns):
            missing = sorted(list(required_cols - set(df.columns)))
            print(f"Error: {csv_path} missing columns: {missing}")
            sys.exit(3)

        # Coerce numeric, drop NA, and sort by PEs so lines connect correctly
        df = df.copy()
        df["pes"] = pd.to_numeric(df["pes"], errors="coerce")
        df["termination_latency_ms"] = pd.to_numeric(
            df["termination_latency_ms"], errors="coerce"
        )
        df = df.dropna(subset=["pes", "termination_latency_ms"]).sort_values("pes")

        x = df["pes"].values
        y = df["termination_latency_ms"].values
        label = csv_path.stem

        ax.plot(x, y, marker="o", label=label)

    ax.set_title("Termination Latency for Different Global Done Implementations")
    ax.set_xlabel("Number of PEs")
    ax.set_ylabel("Termination Latency (ms)")
    ax.set_ylim(bottom=0)  # y-axis starts at 0 ms
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Implementation", fontsize="small")

    fig.tight_layout()
    fig.savefig("termination_latency.png", dpi=150)

    # Optional interactive display
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
