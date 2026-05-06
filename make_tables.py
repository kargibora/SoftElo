#!/usr/bin/env python
"""Create compact Markdown and LaTeX tables from experiment outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import pandas as pd


def fmt_float(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def paired_summary(loo: pd.DataFrame) -> pd.DataFrame:
    hard = loo.loc[loo["method"] == "hard"].iloc[0]
    soft = loo.loc[loo["method"] == "soft"].iloc[0]
    mae_red = 100.0 * (hard["mae"] - soft["mae"]) / hard["mae"]
    return pd.DataFrame(
        [
            {
                "Hard MAE": fmt_float(hard["mae"]),
                "Soft MAE": fmt_float(soft["mae"]),
                "MAE reduction": f"{mae_red:.1f}%",
                "Hard rho": fmt_float(hard["spearman_rho"], 3),
                "Soft rho": fmt_float(soft["spearman_rho"], 3),
                "Mean beta": fmt_float(soft["mean_beta"], 3),
            }
        ]
    )


def conformal_summary(conf: pd.DataFrame) -> pd.DataFrame:
    hard = conf.loc[conf["method"] == "hard"].iloc[0]
    soft = conf.loc[conf["method"] == "soft"].iloc[0]
    width_red = 100.0 * (hard["median_width"] - soft["median_width"]) / hard["median_width"]
    return pd.DataFrame(
        [
            {
                "Hard coverage": fmt_float(hard["coverage"], 3),
                "Soft coverage": fmt_float(soft["coverage"], 3),
                "Hard width": fmt_float(hard["median_width"], 1),
                "Soft width": fmt_float(soft["median_width"], 1),
                "Width reduction": f"{width_red:.1f}%",
            }
        ]
    )


def to_markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    loo = pd.read_csv(out / "loo_summary.csv")
    conf = pd.read_csv(out / "conformal_summary.csv")
    config_path = out / "run_config.json"
    run_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    bootstrap_resamples = int(run_config.get("bootstrap_resamples", 0) or 0)

    tables = {
        "elo_summary": paired_summary(loo),
        "conformal_summary": conformal_summary(conf),
    }

    md_path = out / "tables.md"
    tex_dir = out / "tables_tex"
    tex_dir.mkdir(exist_ok=True)

    with md_path.open("w") as f:
        if bootstrap_resamples and bootstrap_resamples < 20:
            f.write(
                "> Warning: this run used "
                f"`bootstrap_resamples={bootstrap_resamples}`. "
                "Conformal widths are unstable with very small bootstrap counts; "
                "use at least `20` for paper-style runs.\n\n"
            )
        for name, table in tables.items():
            f.write(f"## {name}\n\n")
            f.write(to_markdown(table))
            f.write("\n\n")
            table.to_latex(tex_dir / f"{name}.tex", index=False, escape=False)

    print(f"Wrote {md_path}")
    print(f"Wrote LaTeX tables to {tex_dir}")


if __name__ == "__main__":
    main()
