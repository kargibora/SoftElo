#!/usr/bin/env python
"""Run the minimal Hard-Elo / Soft-Elo experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from softelo_minimal import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_annotations.csv")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--bootstrap-resamples", type=int, default=20)
    parser.add_argument("--summary-bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--beta-stability-repeats", type=int, default=20)
    parser.add_argument("--split-repeats", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data = Path(args.data)
    out = Path(args.out)
    if not data.is_absolute():
        data = root / data
    if not out.is_absolute():
        out = root / out

    result = run_experiment(
        data,
        out,
        regularization=args.regularization,
        bootstrap_resamples=args.bootstrap_resamples,
        summary_bootstrap_repeats=args.summary_bootstrap_repeats,
        beta_stability_repeats=args.beta_stability_repeats,
        split_repeats=args.split_repeats,
        alpha=args.alpha,
        seed=args.seed,
    )

    print("\nLOO Elo summary")
    print(result["summary"].to_string(index=False))
    print("\nSplit-conformal summary")
    print(result["conformal"].to_string(index=False))
    print("\nBootstrap summary over held-out models")
    print(result["bootstrap_summary"].to_string(index=False))
    print("\nScore-signal summary")
    print(result["signal"].to_string(index=False))
    print(f"\nWrote outputs to: {out}")


if __name__ == "__main__":
    main()
