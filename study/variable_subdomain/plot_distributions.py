#!/usr/bin/env python3
"""Plot lower/upper-bound distributions produced by variable_subdomain_study.

Reads the raw float64 arrays main.cu writes to results/N{N}_lb.f64 and
results/N{N}_ub.f64 (one file per refinement factor N, no header -- element
count is the file size / 8) and draws one histogram per N, lower and upper
bound overlaid.

Usage: .venv/bin/python plot_distributions.py [results_dir] [--out FILE]

With no --out, the image is named distributions_<results_dir name>.png, so
e.g. results/rosenbrock produces distributions_rosenbrock.png.
"""

import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np


def discover_n_values(results_dir: pathlib.Path) -> list[int]:
    n_values = set()
    for path in results_dir.glob("N*_lb.f64"):
        m = re.match(r"N(\d+)_lb\.f64$", path.name)
        if m:
            n_values.add(int(m.group(1)))
    return sorted(n_values)


def load(results_dir: pathlib.Path, n: int, bound: str) -> np.ndarray:
    return np.fromfile(results_dir / f"N{n}_{bound}.f64", dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default="results", type=pathlib.Path,
                         help="directory containing N{N}_lb.f64 / N{N}_ub.f64 (default: results)")
    parser.add_argument("--out", default=None,
                         help="output image path (default: distributions_<results_dir name>.png)")
    args = parser.parse_args()
    if args.out is None:
        args.out = f"distributions_{args.results_dir.name}.png"

    n_values = discover_n_values(args.results_dir)
    if not n_values:
        raise SystemExit(f"no N*_lb.f64 files found in {args.results_dir}")

    fig, axes = plt.subplots(len(n_values), 1, figsize=(8, 2.5 * len(n_values)), squeeze=False)

    for ax, n in zip(axes[:, 0], n_values):
        lb = load(args.results_dir, n, "lb")
        ub = load(args.results_dir, n, "ub")
        bins = min(200, max(10, lb.size // 20))

        ax.hist(lb, bins=bins, alpha=0.6, label="lower bound", color="tab:blue")
        ax.hist(ub, bins=bins, alpha=0.6, label="upper bound", color="tab:orange")
        ax.set_title(f"N={n}  ({lb.size} subregions)")
        ax.set_xlabel("objective bound")
        ax.set_ylabel("count")
        ax.legend()

        # Pin the axis to the data's actual extremes and force ticks at
        # those endpoints, so the full range -- including the leftmost and
        # rightmost values -- is always labelled, not just whatever
        # matplotlib's default tick spacing happens to land on.
        lo, hi = min(lb.min(), ub.min()), max(lb.max(), ub.max())
        ax.set_xlim(lo, hi)
        ticks = ax.get_xticks()
        ticks = ticks[(ticks > lo) & (ticks < hi)]
        ax.set_xticks(np.concatenate([[lo], ticks, [hi]]))
        ax.tick_params(axis="x", labelrotation=30)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
