"""Recompute explicitly selected paper protocols from released annotations.

This is a source-preserving reference pipeline. Rebuilding a table from cached
measurements and rerunning its fits are separate operations; neither silently
substitutes the main fold-specific protocol for a pooled-temperature ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import historical_core as core


def canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the original quarter-to-tie rule to both human and judge labels."""
    return core.collapse_preference_frame(
        core.clean_scores(frame), collapse_policy="quarter_to_tie",
        collapse_threshold=0.26,
    ).reset_index(drop=True)


def run_point_estimates(frame: pd.DataFrame, *, pooled: bool = False):
    frame = canonical_frame(frame)
    dims = core.infer_score_dimensions(frame)
    beta = core.fit_temperature_mle(frame, dims, mle_mode="decisive")["temperature"] if pooled else None
    return core.run_temperature_holdout(
        frame, dims, core.model_universe(frame), reg=0.01,
        tie_strategy="half", mle_mode="decisive", threshold=0.05,
        soft_tie_strategy="half", precomputed_beta=beta,
    )


def run_smoothing(frame: pd.DataFrame):
    """Historical E.1: pooled beta, with a retrospective constant sweep."""
    frame = canonical_frame(frame)
    baseline_summary, baseline_detail = run_point_estimates(frame, pooled=True)
    models = core.model_universe(frame)
    human = baseline_detail[baseline_detail.method == "hard"].set_index("model")["human_bt_elo"].to_dict()
    details = []
    summaries = []
    for c in (0.55, 0.65, 0.75, 0.85, 0.95):
        estimates = {}
        for target in models:
            anchors = [m for m in models if m != target]
            anchor_frame = frame[frame.model_a.isin(anchors) & frame.model_b.isin(anchors)]
            matches, labels, weights = core._build_constant_matches(anchor_frame, anchors, "judge_pref", c)
            strengths = core.fit_softlabel_bt(anchors, matches, labels, regularization=0.01, sample_weight=weights)
            target_frame = core.target_vs_anchor_rows(frame, target, allowed_anchors=set(anchors))
            y, theta, weights = core._encode_constant_target_for_held_out(target_frame, target, c, strengths)
            estimate = core.ELO_BASE + core.BT_TO_ELO * core.solve_target_theta(y, theta, regularization=0.01, sample_weight=weights)
            estimates[target] = estimate
            details.append({"model": target, "method": f"const_{c:.2f}", "judge_bt_elo": estimate,
                            "human_bt_elo": human[target], "bt_elo_residual": human[target] - estimate,
                            "abs_bt_elo_residual": abs(human[target] - estimate)})
        summaries.append({"method": f"const_{c:.2f}", **core.eval_elo(estimates, human)})
    baseline_summary = baseline_summary.replace({"method": {"temperature_mle": "soft_beta"}})
    baseline_detail = baseline_detail.replace({"method": {"temperature_mle": "soft_beta"}})
    return (pd.concat([baseline_summary, pd.DataFrame(summaries)], ignore_index=True),
            pd.concat([baseline_detail, pd.DataFrame(details)], ignore_index=True))


def run_historical_conformal(frame: pd.DataFrame):
    """Historical Table 3 protocol, including its original quantile operation.

    This deliberately preserves the original higher-interpolation quantile,
    pooled beta, and bootstrap/split seeds. It is not an exact-order-statistic
    implementation intended for new analyses.
    """
    frame = core.prepare_shared_frame(canonical_frame(frame), tie_strategy="half")
    dims = core.infer_score_dimensions(frame)
    beta = core.fit_temperature_mle(frame, dims, mle_mode="decisive")["temperature"]
    detail_frames, repeat_frames, summaries = [], [], []
    for method in ("hard", "soft"):
        detail, _ = core.build_heldout_table(
            frame, judge_mode=method, min_matches=25, regularization=0.01,
            # The original hard pipeline resampled anchors; soft held them fixed.
            bootstrap_resamples=20, seed=42, bootstrap_anchor_pool=(method == "hard"),
            dims=dims, beta=beta, tie_strategy="half", progress_every=0,
        )
        repeats, _ = core.repeated_split_conformal(
            detail, alpha=0.1, use_normalized=True, seed=42,
            n_repeats=5, progress_every=0,
        )
        summary = core.build_alpha_summary(repeats, [])
        for obj in (detail, repeats, summary):
            obj["method"] = method
        detail_frames.append(detail)
        repeat_frames.append(repeats)
        summaries.append(summary)
    return pd.concat(summaries), pd.concat(detail_frames), pd.concat(repeat_frames)


def summarize_cached_main(paths):
    """Recalculate Table 2 metrics from archived model-level measurements."""
    rows = []
    for path in paths:
        detail = pd.read_csv(path)
        detail = detail[detail.method.isin(["hard", "temperature_mle"])]
        for method, group in detail.groupby("method", sort=False):
            if group.model.duplicated().any():
                raise ValueError(f"Duplicate model rows for {method}: {path}")
            metrics = core.eval_elo(group.set_index("model").judge_bt_elo.to_dict(), group.set_index("model").human_bt_elo.to_dict())
            rows.append({"label": group.label.iloc[0], "method": method, **metrics,
                         "temperature_fit_mean": group.temperature.mean()})
    return pd.DataFrame(rows)


def _check_number(checks, name, actual, expected, *, atol=1e-9):
    actual, expected = float(actual), float(expected)
    passed = bool(np.isclose(actual, expected, rtol=0, atol=atol, equal_nan=True))
    checks.append({"name": name, "actual": actual if np.isfinite(actual) else None,
                   "expected": expected if np.isfinite(expected) else None, "passed": passed})
    if not passed:
        raise ValueError(f"Cached result inconsistency: {name}: {actual} != {expected}")


def _markdown_table(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}" if np.isfinite(value) else ""
        return str(value).replace("|", "\\|")
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def build_cached_tables(root, out):
    """Validate and aggregate archived measurements without running any fits.

    Expected layout: main91/JUDGE/{heldout_models_decisive,summary_decisive}.csv;
    conformal61/{hard,soft}/JUDGE/quarter_tie/{split_repeats.csv,condition_summary.json};
    conformal61/mode_comparison.csv; smoothing/per_judge.csv.
    """
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checks, sources = [], []
    def record(path):
        sources.append({"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {"operation": "aggregate_saved_measurements_no_fits", "checks": checks, "sources": sources}
    try:
        main_paths = sorted((root / "main91").glob("*/heldout_models_decisive.csv"))
        if not main_paths:
            raise ValueError("No main91 model-level result files found")
        main_table = summarize_cached_main(main_paths)
        for path in main_paths:
            record(path)
            summary_path = path.with_name("summary_decisive.csv")
            record(summary_path)
            expected = pd.read_csv(summary_path)
            label = expected.label.iloc[0]
            for method in ("hard", "temperature_mle"):
                actual = main_table[(main_table.label == label) & (main_table.method == method)].iloc[0]
                target = expected[expected.method == method].iloc[0]
                for metric in ("mae", "spearman_rho", "kendall_tau", "n_models", "temperature_fit_mean"):
                    _check_number(checks, f"main91/{label}/{method}/{metric}", actual[metric], target[metric])
        main_table.to_csv(out / "main_results.csv", index=False)
        mode_path = root / "conformal61" / "mode_comparison.csv"
        record(mode_path)
        modes = pd.read_csv(mode_path)
        conformal_rows = []
        for path in sorted((root / "conformal61").glob("*/*/quarter_tie/split_repeats.csv")):
            method = path.parents[2].name
            if method not in ("hard", "soft"):
                continue
            record(path)
            condition_path = path.with_name("condition_summary.json")
            record(condition_path)
            label = json.loads(condition_path.read_text())["label"]
            repeats = pd.read_csv(path)
            summary = core.build_alpha_summary(repeats, [])
            for _, row in summary.iterrows():
                expected = modes[(modes.label == label) & (modes.judge_mode == method)
                                 & np.isclose(modes.operational_alpha, row.alpha)]
                if len(expected) != 1:
                    raise ValueError(f"Expected one conformal summary for {label}/{method}/{row.alpha}")
                for metric in ("split_coverage_point", "split_coverage_lo", "split_coverage_hi",
                               "split_median_width_point", "split_median_width_lo", "split_median_width_hi"):
                    _check_number(checks, f"conformal61/{label}/{method}/{metric}", row[metric], expected.iloc[0][metric])
                conformal_rows.append({"label": label, "method": method, "alpha": row.alpha,
                    "coverage_mean": row.split_coverage_point, "width_mean": row.split_median_width_point,
                    "width_p025": row.split_median_width_lo, "width_p975": row.split_median_width_hi,
                    "split_repeats": len(repeats[np.isclose(repeats.alpha, row.alpha)])})
        if not conformal_rows:
            raise ValueError("No conformal61 split-repeat files found")
        conformal_table = pd.DataFrame(conformal_rows)
        conformal_table.to_csv(out / "conformal_results.csv", index=False)
        smoothing_path = root / "smoothing" / "per_judge.csv"
        record(smoothing_path)
        smoothing = pd.read_csv(smoothing_path)
        judge_col = "judge" if "judge" in smoothing.columns else "label"
        if smoothing.duplicated([judge_col, "method"]).any() or not np.isfinite(smoothing.mae).all():
            raise ValueError("Invalid smoothing table: duplicate or nonfinite results")
        smoothing_table = smoothing.pivot(index=judge_col, columns="method", values="mae")
        smoothing_table.loc["Mean"] = smoothing_table.mean(axis=0)
        smoothing_table.to_csv(out / "label_smoothing_results.csv")
        report["passed"] = True
        (out / "tables.md").write_text(
            "# Archived paper measurements\n\nThese tables aggregate saved measurements; no model fitting was rerun.\n\n"
            "## Main Table 2: fold-specific temperature\n\n" + _markdown_table(main_table) +
            "\n\n## Table 3: historical pooled-temperature conformal protocol\n\n"
            "Widths are mean medians across five splits; p025/p975 are percentiles across splits, not min–max.\n\n" +
            _markdown_table(conformal_table) +
            "\n\n## Label smoothing: pooled-temperature ablation\n\n"
            "Means use unrounded judge-level MAEs. Constants were compared retrospectively using human Elo.\n\n" +
            _markdown_table(smoothing_table.reset_index()) + "\n")
    except Exception as exc:
        report["passed"] = False
        report["error"] = str(exc)
        raise
    finally:
        (out / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", help="Local dataset root or Hugging Face dataset ID")
    inputs.add_argument("--cached-results", type=Path, help="Aggregate and validate saved measurements without fitting")
    parser.add_argument("--corpus", choices=["lmarena100k", "lmarena140k", "comparia"])
    parser.add_argument("--judge", help="Exact released judge identifier")
    parser.add_argument("--revision", help="Dataset commit, required for a reproducible remote load")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", choices=["main", "pooled", "smoothing", "conformal-historical"], default="main")
    args = parser.parse_args()
    if args.cached_results:
        report = build_cached_tables(args.cached_results, args.out)
        print(f"Verified {len(report['checks'])} numerical comparisons; tables written to {args.out}")
        return
    if not args.corpus or not args.judge:
        parser.error("--data requires --corpus and --judge")
    from .data import load_annotations, resolve_release
    root = resolve_release(args.data, args.revision)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    spec = next(r for r in manifest["annotations"] if r["corpus"] == args.corpus and r["judge"] == args.judge)
    annotation_sha = hashlib.sha256((root / spec["file"]).read_bytes()).hexdigest()
    if annotation_sha != spec["sha256"]:
        raise ValueError("Annotation file differs from the release manifest")
    frame = load_annotations(root, corpus=args.corpus, judge=args.judge)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.protocol == "conformal-historical":
        summary, detail, repeats = run_historical_conformal(frame)
        repeats.to_csv(args.out / "split_repeats.csv", index=False)
    elif args.protocol == "smoothing":
        summary, detail = run_smoothing(frame)
    else:
        summary, detail = run_point_estimates(frame, pooled=args.protocol == "pooled")
    summary.to_csv(args.out / "summary.csv", index=False)
    detail.to_csv(args.out / "heldout_models.csv", index=False)
    config = vars(args).copy()
    config["out"] = str(config["out"])
    config["rows"] = len(frame)
    config["annotation_sha256"] = annotation_sha
    config["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config["regularization"] = 0.01
    config["core_sha256"] = hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    config["note"] = "Recomputed outputs; compare with archived measurements before claiming numerical reproduction."
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
