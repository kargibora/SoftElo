#!/usr/bin/env python
"""Create simple diagnostic plots from the minimal experiment outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {"hard": "#2563eb", "soft": "#db2777"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    root = ROOT
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out

    detail = pd.read_csv(out / "heldout_models.csv")
    calib = pd.read_csv(out / "calibration_bins.csv")
    beta = pd.read_csv(out / "beta_stability_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    ax = axes[0]
    for method, sub in detail.groupby("method"):
        ax.scatter(
            sub["human_elo"],
            sub["judge_elo"],
            s=28,
            alpha=0.8,
            label=method,
            color=COLORS.get(method),
        )
    lo = min(detail["human_elo"].min(), detail["judge_elo"].min())
    hi = max(detail["human_elo"].max(), detail["judge_elo"].max())
    ax.plot([lo, hi], [lo, hi], color="0.35", lw=1, ls="--")
    ax.set_title("Held-out Elo")
    ax.set_xlabel("Human Elo")
    ax.set_ylabel("Judge Elo")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(
        calib["mean_abs_score_gap"],
        calib["judge_human_agreement"],
        marker="o",
        label="empirical",
        color="#111827",
    )
    ax.plot(
        calib["mean_abs_score_gap"],
        calib["mean_pred_confidence"],
        marker="s",
        label="sigmoid(beta |s|)",
        color=COLORS["soft"],
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Score Calibration")
    ax.set_xlabel("Mean |score gap|")
    ax.set_ylabel("Agreement / confidence")
    ax.legend(frameon=False)

    ax = axes[2]
    ax.errorbar(
        beta["budget_rows"],
        beta["beta_mean"],
        yerr=beta["beta_sd"].fillna(0.0),
        marker="o",
        color=COLORS["soft"],
        capsize=3,
    )
    ax.set_title("Beta Stability")
    ax.set_xlabel("Calibration rows")
    ax.set_ylabel("Fitted beta")

    fig.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=200)
    print(f"Wrote {out / 'diagnostics.png'}")


if __name__ == "__main__":
    main()
