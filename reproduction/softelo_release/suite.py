"""Fit the paper protocols and compare newly computed results with references."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import load_annotations, resolve_release
from .reproduce import run_point_estimates, run_smoothing, run_historical_conformal

DATASET_ID = "kbora/SoftElo"
DATASET_REVISION = "6223cb4116d76a1bae376c0730df4d968d7172be"


def compare_frames(actual, expected, keys, columns, *, atol=1e-5):
    """Require identical keys and compare every requested numeric cell."""
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f"Duplicate comparison keys: {keys}")
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Computed and reference row identities differ")
    checks = []
    for column in columns:
        x, y = left[column].to_numpy(float), right[column].to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        delta = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
        checks.append({"column": column, "rows": len(x), "max_abs_difference": delta,
                       "passed": bool(np.allclose(x, y, rtol=0, atol=atol, equal_nan=True))})
    return checks


def run_task(task):
    started = time.monotonic()
    root, out = Path(task["data"]), Path(task["out"])
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    spec = next(s for s in manifest["annotations"]
                if s["corpus"] == task["corpus"] and s["judge"] == task["judge"])
    source = root / spec["file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Annotation hash differs: {source}")
    config = dict(task, annotation_sha256=spec["sha256"],
                  manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
                  numerical_core_sha256=hashlib.sha256(Path(__file__).with_name("historical_core.py").read_bytes()).hexdigest(),
                  protocol_code_sha256=hashlib.sha256(Path(__file__).with_name("reproduce.py").read_bytes()).hexdigest(),
                  regularization=0.01, beta_fit="per_target" if task["protocol"] == "main" else "pooled")
    if task["protocol"] == "conformal-historical":
        config.update(bootstrap_resamples=20, seed=42, split_repeats=5, alpha=0.1,
                      bootstrap_anchor_pool={"hard": True, "soft": False})
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"event": "start", "corpus": task["corpus"], "judge": task["judge"],
                      "protocol": task["protocol"]}), flush=True)
    frame = load_annotations(root, corpus=task["corpus"], judge=task["judge"])
    slug = Path(spec["file"]).stem.removeprefix("annotations_")
    # Historical result directories used the older Gemma display name.
    if slug == "gemma4_26b_a4b":
        slug = "gemma4_27b"
    checks = []
    def save_predictions(summary, detail):
        summary.to_csv(out / "summary.csv", index=False)
        detail.to_csv(out / "heldout_models.csv", index=False)

    if task["protocol"] == "main":
        summary, detail = run_point_estimates(frame)
        save_predictions(summary, detail)
        if task["corpus"] == "lmarena100k":
            ref = root / "results/main91" / slug / "heldout_models_decisive.csv"
            expected = pd.read_csv(ref)
            expected = expected[expected.method.isin(["hard", "temperature_mle"])]
            checks += compare_frames(detail, expected, ["method", "model"],
                                     ["judge_bt_elo", "human_bt_elo", "temperature"])
        else:
            ref = Path(task["references"]) / "lmarena140k" / slug / "heldout_summary.csv"
            expected = pd.read_csv(ref).replace({"method": {"soft": "temperature_mle"}})
            checks += compare_frames(summary, expected, ["method"],
                                     ["mae", "spearman_rho", "kendall_tau", "spread",
                                      "n_models", "temperature_fit_mean", "temperature_fit_std"])
        references = [ref]
    elif task["protocol"] == "smoothing":
        summary, detail = run_smoothing(frame)
        save_predictions(summary, detail)
        ref = root / "results/smoothing/per_judge.csv"
        expected = pd.read_csv(ref)
        expected = expected[expected.judge == task["judge"]]
        checks += compare_frames(summary, expected, ["method"], ["mae"])
        references = [ref]
    else:
        summary, detail, repeats = run_historical_conformal(frame)
        save_predictions(summary, detail)
        repeats.to_csv(out / "split_repeats.csv", index=False)
        references = []
        for method in ("hard", "soft"):
            base = root / "results/conformal61" / method / slug / "quarter_tie"
            ref = base / "heldout_models.csv"
            checks += compare_frames(detail[detail.method == method], pd.read_csv(ref), ["model"],
                                     ["judge_bt_elo", "human_bt_elo", "judge_bt_elo_se", "human_bt_elo_se"])
            references.append(ref)
            ref = base / "split_repeats.csv"
            checks += compare_frames(repeats[repeats.method == method], pd.read_csv(ref), ["repeat", "alpha"],
                                     ["seed", "qhat", "empirical_coverage", "median_interval_width",
                                      "n_total", "n_calibration", "n_test"])
            references.append(ref)
    report = {"passed": all(c["passed"] for c in checks), "operation": "fresh_annotation_fits",
              "corpus": task["corpus"], "judge": task["judge"], "protocol": task["protocol"],
              "annotation_rows": len(frame), "heldout_rows": len(detail), "checks": checks,
              "elapsed_seconds": time.monotonic() - started,
              "references": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in references]}
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "finished", **{k: v for k, v in report.items() if k not in ["checks", "references"]},
                      "max_abs_difference": max(c["max_abs_difference"] for c in checks)}), flush=True)
    return report


def run_suite(root, out, references, *, suite="all", judge=None, jobs=2):
    root, out, references = Path(root).resolve(), Path(out).resolve(), Path(references).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    tasks = []
    for spec in manifest["annotations"]:
        if judge and spec["judge"] != judge:
            continue
        corpus = spec["corpus"]
        if corpus not in {"lmarena100k", "lmarena140k"}:
            continue
        protocols = (["main", "smoothing", "conformal-historical"] if corpus == "lmarena100k" else ["main"])
        for protocol in protocols:
            category = "lmarena140k" if corpus == "lmarena140k" else protocol
            if suite != "all" and suite != category:
                continue
            slug = Path(spec["file"]).stem.removeprefix("annotations_")
            tasks.append({"data": str(root), "out": str(out / corpus / protocol / slug),
                          "references": str(references), "corpus": corpus, "judge": spec["judge"], "protocol": protocol})
    if not tasks:
        raise ValueError("No experiments match the requested suite and judge")
    reports = []
    errors = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        pending = {pool.submit(run_task, t): t for t in tasks}
        for future in as_completed(pending):
            try:
                reports.append(future.result())
            except Exception as exc:
                failure = {"task": pending[future], "error_type": type(exc).__name__, "error": str(exc)}
                errors.append(failure)
                print(json.dumps({"event": "error", **failure}), flush=True)
            status = {"passed": not errors and all(r["passed"] for r in reports) and len(reports) == len(tasks),
                      "expected_tasks": len(tasks), "completed_tasks": len(reports), "errors": errors, "results": reports}
            (out / "validation.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({k: v for k, v in status.items() if k != "results"}), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATASET_ID, help="Local release directory or Hugging Face dataset ID")
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--suite", choices=["main", "smoothing", "conformal-historical", "lmarena140k", "all"], default="main")
    parser.add_argument("--judge", help="Omit to run every judge in the selected suite")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("outputs/reproduction"))
    installed_references = Path(__file__).resolve().parent / "references"
    default_references = (installed_references if installed_references.exists()
                          else Path(__file__).resolve().parents[1] / "references")
    parser.add_argument("--references", type=Path, default=default_references)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    root = resolve_release(args.data, revision=args.revision)
    result = run_suite(root, args.out, args.references, suite=args.suite, judge=args.judge, jobs=args.jobs)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
