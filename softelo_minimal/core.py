"""Self-contained Hard-Elo and Soft-Elo reference implementation.

Input convention
----------------
The CSV must contain:

``model_a, model_b, human_pref, judge_pref, scores_a, scores_b``.

Preference labels follow the convention used in the paper pipeline:
``0`` means model A wins, ``1`` means model B wins, and ``0.5`` means tie.
The Bradley--Terry target is internally converted to
``P(A beats B)``: 1, 0, or 0.5.

Soft-Elo fits a scalar temperature beta on human non-tie rows and uses
``sigmoid(beta * (score_a - score_b))`` as the BT target.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

BT_TO_ELO = 400.0 / math.log(10.0)
ELO_BASE = 1500.0
TIE_EPS = 0.05


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def parse_scores(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            raw = ast.literal_eval(value)
    else:
        return {}
    out: dict[str, float] = {}
    for key, val in raw.items():
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fval):
            out[str(key)] = fval
    return out


def load_annotations(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model_a", "model_b", "human_pref", "judge_pref", "scores_a", "scores_b"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["scores_a"] = df["scores_a"].apply(parse_scores)
    df["scores_b"] = df["scores_b"].apply(parse_scores)
    df["human_pref"] = pd.to_numeric(df["human_pref"], errors="coerce")
    df["judge_pref"] = pd.to_numeric(df["judge_pref"], errors="coerce")
    return df.dropna(subset=["model_a", "model_b"]).reset_index(drop=True)


def model_universe(df: pd.DataFrame) -> list[str]:
    return sorted(set(df["model_a"].astype(str)) | set(df["model_b"].astype(str)))


def score_dimensions(df: pd.DataFrame) -> list[str]:
    dims: set[str] = set()
    for col in ["scores_a", "scores_b"]:
        for scores in df[col]:
            dims.update(scores.keys())
    return sorted(dims)


def score_gap(row: pd.Series, dims: Iterable[str]) -> float:
    vals = []
    for dim in dims:
        if dim in row["scores_a"] and dim in row["scores_b"]:
            vals.append(float(row["scores_a"][dim]) - float(row["scores_b"][dim]))
    return float(np.mean(vals)) if vals else float("nan")


def pref_to_a_target(pref: float) -> float | None:
    if not np.isfinite(pref):
        return None
    if abs(float(pref) - 0.5) <= TIE_EPS:
        return 0.5
    return 1.0 if float(pref) < 0.5 else 0.0


def build_design(
    df: pd.DataFrame,
    models: list[str],
    *,
    target_col: str | None = None,
    soft_beta: float | None = None,
    dims: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = {m: i for i, m in enumerate(models)}
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    w_rows: list[float] = []

    for _, row in df.iterrows():
        ma, mb = str(row["model_a"]), str(row["model_b"])
        if ma not in idx or mb not in idx:
            continue
        x = np.zeros(len(models), dtype=float)
        x[idx[ma]] = 1.0
        x[idx[mb]] = -1.0

        if soft_beta is not None:
            gap = score_gap(row, dims or [])
            if not np.isfinite(gap):
                continue
            y = float(sigmoid(float(soft_beta) * gap))
        else:
            assert target_col is not None
            y = pref_to_a_target(float(row[target_col]))
            if y is None:
                continue

        x_rows.append(x)
        y_rows.append(float(np.clip(y, 1e-6, 1.0 - 1e-6)))
        w_rows.append(1.0)

    if not x_rows:
        return np.empty((0, len(models))), np.empty(0), np.empty(0)
    return np.vstack(x_rows), np.asarray(y_rows), np.asarray(w_rows)


def fit_bt(
    df: pd.DataFrame,
    models: list[str],
    *,
    target_col: str | None = None,
    soft_beta: float | None = None,
    dims: list[str] | None = None,
    regularization: float = 0.01,
) -> dict[str, float]:
    x, y, w = build_design(df, models, target_col=target_col, soft_beta=soft_beta, dims=dims)
    if len(y) == 0:
        return {m: float("nan") for m in models}

    def safe_theta(theta: np.ndarray) -> np.ndarray:
        return np.clip(
            np.nan_to_num(theta, nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0,
            20.0,
        ).astype(float, copy=False)

    def logits(theta: np.ndarray) -> np.ndarray:
        theta = safe_theta(theta)
        return np.clip(np.sum(x * theta[None, :], axis=1), -60.0, 60.0)

    def objective(theta: np.ndarray) -> float:
        theta = safe_theta(theta)
        p = np.clip(sigmoid(logits(theta)), 1e-9, 1.0 - 1e-9)
        nll = -np.sum(w * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        return float(nll + 0.5 * regularization * np.dot(theta, theta))

    def gradient(theta: np.ndarray) -> np.ndarray:
        theta = safe_theta(theta)
        p = sigmoid(logits(theta))
        return np.sum(x * (w * (p - y))[:, None], axis=0) + regularization * theta

    res = minimize(
        objective,
        np.zeros(len(models), dtype=float),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0)] * len(models),
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    theta = np.asarray(res.x if res.success else np.zeros(len(models)), dtype=float)
    theta -= float(np.mean(theta))
    return {m: float(theta[i]) for i, m in enumerate(models)}


def theta_to_elo(theta: dict[str, float]) -> dict[str, float]:
    return {m: ELO_BASE + BT_TO_ELO * v for m, v in theta.items()}


def fit_beta_mle(df: pd.DataFrame, dims: list[str]) -> dict[str, float]:
    gaps: list[float] = []
    targets: list[float] = []
    for _, row in df.iterrows():
        y = pref_to_a_target(float(row["human_pref"]))
        if y is None or abs(y - 0.5) <= 1e-9:
            continue
        gap = score_gap(row, dims)
        if np.isfinite(gap):
            gaps.append(gap)
            targets.append(y)

    if len(gaps) < 8:
        return {"beta": float("nan"), "n_fit_rows": len(gaps), "nll": float("nan")}

    gap_arr = np.asarray(gaps, dtype=float)
    y_arr = np.asarray(targets, dtype=float)

    def nll(log_beta: float) -> float:
        beta = float(np.exp(log_beta))
        p = np.clip(sigmoid(beta * gap_arr), 1e-9, 1.0 - 1e-9)
        return float(-np.sum(y_arr * np.log(p) + (1.0 - y_arr) * np.log(1.0 - p)))

    opt = minimize_scalar(
        nll,
        bounds=(math.log(1e-3), math.log(50.0)),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 500},
    )
    beta = float(np.exp(opt.x)) if opt.success else float("nan")
    return {"beta": beta, "n_fit_rows": len(gaps), "nll": float(opt.fun)}


def target_rows(df: pd.DataFrame, target: str, anchors: set[str]) -> pd.DataFrame:
    rows = df[(df["model_a"] == target) | (df["model_b"] == target)].copy()
    other = np.where(rows["model_a"] == target, rows["model_b"], rows["model_a"])
    return rows[pd.Series(other, index=rows.index).isin(anchors)].reset_index(drop=True)


def encode_target(
    rows: pd.DataFrame,
    target: str,
    anchor_theta: dict[str, float],
    *,
    target_col: str | None = None,
    soft_beta: float | None = None,
    dims: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y_rows: list[float] = []
    theta_rows: list[float] = []
    for _, row in rows.iterrows():
        target_is_a = str(row["model_a"]) == target
        anchor = str(row["model_b"]) if target_is_a else str(row["model_a"])
        if anchor not in anchor_theta:
            continue

        if soft_beta is not None:
            gap = score_gap(row, dims or [])
            if not np.isfinite(gap):
                continue
            y_a = float(sigmoid(float(soft_beta) * gap))
        else:
            assert target_col is not None
            y_a = pref_to_a_target(float(row[target_col]))
            if y_a is None:
                continue

        y_target = y_a if target_is_a else 1.0 - y_a
        y_rows.append(float(np.clip(y_target, 1e-6, 1.0 - 1e-6)))
        theta_rows.append(float(anchor_theta[anchor]))

    return np.asarray(y_rows), np.asarray(theta_rows)


def solve_target_theta(
    y: np.ndarray,
    theta_anchor: np.ndarray,
    *,
    regularization: float = 0.01,
    max_iter: int = 100,
) -> float:
    if len(y) < 2:
        return float("nan")
    theta = 0.0
    for _ in range(max_iter):
        p = sigmoid(theta - theta_anchor)
        grad = float(np.sum(y - p) - regularization * theta)
        hess = float(-np.sum(p * (1.0 - p)) - regularization)
        if abs(hess) < 1e-12:
            break
        step = grad / hess
        theta -= step
        if abs(step) < 1e-8:
            break
    return float(theta)


def bootstrap_target_se(
    y: np.ndarray,
    theta_anchor: np.ndarray,
    *,
    regularization: float,
    rng: np.random.Generator,
    n_resamples: int,
) -> float:
    if n_resamples <= 1 or len(y) < 3:
        return float("nan")
    estimates = []
    n = len(y)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        t = solve_target_theta(y[idx], theta_anchor[idx], regularization=regularization)
        if np.isfinite(t):
            estimates.append(ELO_BASE + BT_TO_ELO * t)
    if len(estimates) < 2:
        return float("nan")
    return float(np.std(estimates, ddof=1))


def loo_evaluation(
    df: pd.DataFrame,
    *,
    regularization: float = 0.01,
    bootstrap_resamples: int = 20,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dims = score_dimensions(df)
    models = model_universe(df)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str | int]] = []

    for target in models:
        anchors = [m for m in models if m != target]
        anchor_set = set(anchors)
        anchor_df = df[df["model_a"].isin(anchor_set) & df["model_b"].isin(anchor_set)]
        tgt_df = target_rows(df, target, anchor_set)

        beta_fit = fit_beta_mle(anchor_df, dims)
        beta = float(beta_fit["beta"])

        human_theta = fit_bt(anchor_df, anchors, target_col="human_pref", regularization=regularization)
        hard_theta = fit_bt(anchor_df, anchors, target_col="judge_pref", regularization=regularization)
        soft_theta = fit_bt(anchor_df, anchors, soft_beta=beta, dims=dims, regularization=regularization)

        y_human, th_human = encode_target(tgt_df, target, human_theta, target_col="human_pref")
        y_hard, th_hard = encode_target(tgt_df, target, hard_theta, target_col="judge_pref")
        y_soft, th_soft = encode_target(tgt_df, target, soft_theta, soft_beta=beta, dims=dims)

        human_elo = ELO_BASE + BT_TO_ELO * solve_target_theta(
            y_human, th_human, regularization=regularization,
        )

        for method, y, th in [("hard", y_hard, th_hard), ("soft", y_soft, th_soft)]:
            theta = solve_target_theta(y, th, regularization=regularization)
            judge_elo = ELO_BASE + BT_TO_ELO * theta if np.isfinite(theta) else float("nan")
            se = bootstrap_target_se(
                y,
                th,
                regularization=regularization,
                rng=rng,
                n_resamples=bootstrap_resamples,
            )
            rows.append({
                "model": target,
                "method": method,
                "human_elo": human_elo,
                "judge_elo": judge_elo,
                "residual": human_elo - judge_elo,
                "abs_residual": abs(human_elo - judge_elo),
                "judge_elo_se": se,
                "beta": beta if method == "soft" else float("nan"),
                "n_target_rows": int(len(y)),
                "n_beta_fit_rows": int(beta_fit["n_fit_rows"]) if method == "soft" else 0,
            })

    detail = pd.DataFrame(rows)
    summary_rows = []
    for method, sub in detail.groupby("method"):
        valid = sub.dropna(subset=["human_elo", "judge_elo"])
        if len(valid) < 3:
            continue
        summary_rows.append({
            "method": method,
            "n_models": int(len(valid)),
            "mae": float(np.mean(np.abs(valid["residual"]))),
            "spearman_rho": float(spearmanr(valid["human_elo"], valid["judge_elo"]).statistic),
            "kendall_tau": float(kendalltau(valid["human_elo"], valid["judge_elo"]).statistic),
            "mean_beta": float(valid["beta"].mean()) if method == "soft" else float("nan"),
        })
    return pd.DataFrame(summary_rows), detail


def conformal_summary(
    detail: pd.DataFrame,
    *,
    alpha: float = 0.10,
    n_split_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for method, sub in detail.groupby("method"):
        valid = sub.dropna(subset=["human_elo", "judge_elo", "judge_elo_se"]).copy()
        valid = valid[valid["judge_elo_se"] > 1e-6].reset_index(drop=True)
        if len(valid) < 6:
            continue
        repeat_rows = []
        for repeat in range(int(n_split_repeats)):
            order = rng.permutation(len(valid))
            n_cal = len(valid) // 2
            cal = valid.iloc[order[:n_cal]].copy()
            test = valid.iloc[order[n_cal:]].copy()
            scores = np.abs(cal["residual"].to_numpy()) / cal["judge_elo_se"].to_numpy()
            q_index = int(np.ceil((len(scores) + 1) * (1.0 - alpha))) - 1
            q_index = int(np.clip(q_index, 0, len(scores) - 1))
            qhat = float(np.sort(scores)[q_index])
            half_width = qhat * test["judge_elo_se"].to_numpy()
            lo = test["judge_elo"].to_numpy() - half_width
            hi = test["judge_elo"].to_numpy() + half_width
            covered = (test["human_elo"].to_numpy() >= lo) & (test["human_elo"].to_numpy() <= hi)
            repeat_rows.append({
                "qhat": qhat,
                "coverage": float(np.mean(covered)),
                "median_width": float(np.median(hi - lo)),
                "n_calibration_models": int(len(cal)),
                "n_test_models": int(len(test)),
            })
        repeat_df = pd.DataFrame(repeat_rows)
        rows.append({
            "method": method,
            "alpha": alpha,
            "n_split_repeats": int(n_split_repeats),
            "n_calibration_models": int(repeat_df["n_calibration_models"].median()),
            "n_test_models": int(repeat_df["n_test_models"].median()),
            "qhat": float(repeat_df["qhat"].mean()),
            "coverage": float(repeat_df["coverage"].mean()),
            "median_width": float(repeat_df["median_width"].mean()),
            "coverage_min": float(repeat_df["coverage"].min()),
            "coverage_max": float(repeat_df["coverage"].max()),
            "median_width_min": float(repeat_df["median_width"].min()),
            "median_width_max": float(repeat_df["median_width"].max()),
        })
    return pd.DataFrame(rows)


def score_gap_frame(df: pd.DataFrame) -> pd.DataFrame:
    dims = score_dimensions(df)
    rows = []
    for _, row in df.iterrows():
        gap = score_gap(row, dims)
        h = pref_to_a_target(float(row["human_pref"]))
        j = pref_to_a_target(float(row["judge_pref"]))
        if h is None or j is None or not np.isfinite(gap):
            continue
        rows.append({
            "score_gap": gap,
            "abs_score_gap": abs(gap),
            "human_a_wins": h,
            "judge_a_wins": j,
            "human_decisive": bool(abs(h - 0.5) > 1e-9),
            "judge_decisive": bool(abs(j - 0.5) > 1e-9),
            "judge_human_agree": bool(abs(h - j) < 1e-9),
        })
    return pd.DataFrame(rows)


def calibration_bins(
    df: pd.DataFrame,
    *,
    beta: float | None = None,
    n_bins: int = 8,
) -> pd.DataFrame:
    """Reliability-style bins for score-margin confidence.

    Rows are restricted to non-tie human labels and non-tie judge labels, which
    matches the beta fitting contract in the paper.
    """
    sg = score_gap_frame(df)
    sg = sg[sg["human_decisive"] & sg["judge_decisive"]].copy()
    if sg.empty:
        return pd.DataFrame()
    if beta is None or not np.isfinite(float(beta)):
        beta = float(fit_beta_mle(df, score_dimensions(df))["beta"])
    sg["pred_confidence"] = sigmoid(float(beta) * sg["abs_score_gap"])
    sg["bin"] = pd.qcut(sg["abs_score_gap"], q=min(n_bins, len(sg)), duplicates="drop")
    rows = []
    for interval, sub in sg.groupby("bin", observed=True):
        rows.append({
            "bin": str(interval),
            "n": int(len(sub)),
            "mean_abs_score_gap": float(sub["abs_score_gap"].mean()),
            "judge_human_agreement": float(sub["judge_human_agree"].mean()),
            "mean_pred_confidence": float(sub["pred_confidence"].mean()),
        })
    return pd.DataFrame(rows)


def beta_stability(
    df: pd.DataFrame,
    *,
    budgets: list[int] | None = None,
    repeats: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Subsample beta fits to show sample-efficiency / stability."""
    dims = score_dimensions(df)
    decisive = df.dropna(subset=["human_pref"]).copy()
    decisive = decisive[np.abs(decisive["human_pref"].astype(float) - 0.5) > TIE_EPS]
    if budgets is None:
        budgets = [50, 100, 250, 500, 1000]
    rng = np.random.default_rng(seed)
    rows = []
    for budget in budgets:
        n = min(int(budget), len(decisive))
        if n < 8:
            continue
        for rep in range(int(repeats)):
            sample = decisive.sample(n=n, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
            fit = fit_beta_mle(sample, dims)
            rows.append({
                "budget_rows": n,
                "repeat": rep,
                "beta": fit["beta"],
                "n_fit_rows": fit["n_fit_rows"],
            })
    return pd.DataFrame(rows)


def bootstrap_method_summary(
    detail: pd.DataFrame,
    *,
    repeats: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap uncertainty over held-out models for MAE and rank metrics."""
    rng = np.random.default_rng(seed)
    rows = []
    for method, sub in detail.groupby("method"):
        valid = sub.dropna(subset=["human_elo", "judge_elo", "residual"]).reset_index(drop=True)
        if len(valid) < 3:
            continue
        draws = []
        n = len(valid)
        for _ in range(int(repeats)):
            idx = rng.integers(0, n, size=n)
            boot = valid.iloc[idx]
            draws.append({
                "mae": float(np.mean(np.abs(boot["residual"]))),
                "spearman_rho": float(spearmanr(boot["human_elo"], boot["judge_elo"]).statistic),
            })
        d = pd.DataFrame(draws)
        rows.append({
            "method": method,
            "n_models": int(n),
            "mae_mean": float(d["mae"].mean()),
            "mae_lo": float(d["mae"].quantile(0.025)),
            "mae_hi": float(d["mae"].quantile(0.975)),
            "spearman_mean": float(d["spearman_rho"].mean()),
            "spearman_lo": float(d["spearman_rho"].quantile(0.025)),
            "spearman_hi": float(d["spearman_rho"].quantile(0.975)),
        })
    return pd.DataFrame(rows)


def score_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One-row diagnostics for whether score gaps carry useful signal."""
    sg = score_gap_frame(df)
    dec = sg[sg["human_decisive"] & sg["judge_decisive"]].copy()
    if dec.empty:
        return pd.DataFrame()
    y = dec["judge_human_agree"].astype(int).to_numpy()
    x = dec["abs_score_gap"].to_numpy()
    try:
        auc = float(roc_auc_score(y, x)) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return pd.DataFrame([{
        "n_decisive_rows": int(len(dec)),
        "agreement_rate": float(np.mean(y)),
        "mean_abs_score_gap": float(np.mean(x)),
        "score_gap_auc_for_agreement": auc,
    }])


def run_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    regularization: float = 0.01,
    bootstrap_resamples: int = 20,
    summary_bootstrap_repeats: int = 1000,
    beta_stability_repeats: int = 20,
    split_repeats: int = 5,
    alpha: float = 0.10,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_annotations(data_path)
    summary, detail = loo_evaluation(
        df,
        regularization=regularization,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    conformal = conformal_summary(detail, alpha=alpha, n_split_repeats=split_repeats, seed=seed)
    boot_summary = bootstrap_method_summary(
        detail,
        repeats=summary_bootstrap_repeats,
        seed=seed,
    )
    beta_draws = beta_stability(
        df,
        repeats=beta_stability_repeats,
        seed=seed,
    )
    beta_summary = (
        beta_draws.groupby("budget_rows", as_index=False)
        .agg(beta_mean=("beta", "mean"), beta_sd=("beta", "std"), n=("beta", "count"))
        if not beta_draws.empty else pd.DataFrame()
    )
    beta_full = fit_beta_mle(df, score_dimensions(df))
    calib = calibration_bins(df, beta=float(beta_full["beta"]))
    signal = score_signal_summary(df)
    summary.to_csv(out / "loo_summary.csv", index=False)
    detail.to_csv(out / "heldout_models.csv", index=False)
    conformal.to_csv(out / "conformal_summary.csv", index=False)
    boot_summary.to_csv(out / "bootstrap_summary.csv", index=False)
    beta_draws.to_csv(out / "beta_stability_draws.csv", index=False)
    beta_summary.to_csv(out / "beta_stability_summary.csv", index=False)
    calib.to_csv(out / "calibration_bins.csv", index=False)
    signal.to_csv(out / "score_signal_summary.csv", index=False)
    pd.DataFrame([beta_full]).to_csv(out / "beta_full_fit.csv", index=False)
    with (out / "run_config.json").open("w") as f:
        json.dump(
            {
                "data_path": str(data_path),
                "regularization": regularization,
                "bootstrap_resamples": bootstrap_resamples,
                "summary_bootstrap_repeats": summary_bootstrap_repeats,
                "beta_stability_repeats": beta_stability_repeats,
                "split_repeats": split_repeats,
                "alpha": alpha,
                "seed": seed,
                "note": (
                    "Conformal widths are sensitive to bootstrap_resamples because "
                    "the normalized score divides residuals by bootstrap SE."
                ),
            },
            f,
            indent=2,
        )
    return {
        "summary": summary,
        "detail": detail,
        "conformal": conformal,
        "bootstrap_summary": boot_summary,
        "beta_stability": beta_summary,
        "calibration": calib,
        "signal": signal,
        "beta_full": pd.DataFrame([beta_full]),
    }
