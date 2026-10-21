"""Numerical implementation of the Soft-Elo paper experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
BT_TO_ELO = 400.0 / math.log(10.0)
ELO_BASE = 1500.0
TIE_EPS_QUARTER = 0.26
EXACT_TIE_EPS = 0.05
PREFERENCE_COLUMNS = [
    "human_pref", "judge_pref", "preference", "human_preference",
    "preference_original", "preference_swapped",
]


# Source: OpenJury/openjury/arena/config.py:703
@dataclass
class MatchResult:
    """Result of a single pairwise matchup.

    ``preference``: 0.0 = model_a wins, 0.5 = tie, 1.0 = model_b wins.
    """

    model_a: str
    model_b: str
    instruction_index: int
    scores_a: dict[str, float]
    scores_b: dict[str, float]
    preference: float
    instruction: str = ""
    instruction_id: str = ""
    instruction_metadata: dict[str, Any] = field(default_factory=dict)
    completion_a: str = ""
    completion_b: str = ""
    raw_judge_output: str = ""
    raw_judge_output_swapped: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "instruction_index": self.instruction_index,
            "scores_a": self.scores_a,
            "scores_b": self.scores_b,
            "preference": self.preference,
        }
        if self.instruction:
            d["instruction"] = self.instruction
        if self.instruction_id:
            d["instruction_id"] = self.instruction_id
        if self.instruction_metadata:
            d["instruction_metadata"] = self.instruction_metadata
        if self.completion_a:
            d["completion_a"] = self.completion_a
        if self.completion_b:
            d["completion_b"] = self.completion_b
        if self.raw_judge_output:
            d["raw_judge_output"] = self.raw_judge_output
        if self.raw_judge_output_swapped is not None:
            d["raw_judge_output_swapped"] = self.raw_judge_output_swapped
        return d

# Source: OpenJury/openjury/arena/ratings.py:481
def _logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# Source: OpenJury/openjury/arena/ratings.py:204
def fit_multi_bt(
    models: list[str],
    matches: list[MatchResult],
    regularization: float = 0.01,
    tie_strategy: str = "quarter",
) -> dict[str, float]:
    """Fit multi-player BT strengths θ_m for each model.

    Uses the formulation: P(i ≻ j) = σ(θ_i − θ_j).
    Each match becomes one training sample with a one-hot feature vector:
    +1 for model_a, −1 for model_b.  Logistic regression on this matrix
    yields the strength parameters.

    Args:
        models: List of model names.
        matches: Pairwise match results.
        regularization: L2 penalty.
        tie_strategy:
            ``"drop"``    – remove ties (|pref − 0.5| ≤ 0.05) before fitting.
            ``"half"``    – ties (|pref − 0.5| ≤ 0.05) become half-win /
                           half-loss evidence.
            ``"quarter"`` – like ``"half"`` but the tie zone is wider:
                           |pref − 0.5| ≤ 0.26, so 0.25 and 0.75 are
                           also treated as ties.

    Returns:
        ``{model_name: θ}`` — higher is better.
    """
    _VALID = {"drop", "half", "quarter"}
    if tie_strategy not in _VALID:
        raise ValueError(
            f"Unsupported tie_strategy={tie_strategy!r}. "
            f"Expected one of {_VALID}."
        )
    tie_eps = 0.26 if tie_strategy == "quarter" else 0.05
    keep_ties = tie_strategy in {"half", "quarter"}

    n_models = len(models)
    model_to_idx = {m: i for i, m in enumerate(models)}

    # Build feature matrix: one row per match, one column per model
    rows_X: list[list[float]] = []
    rows_y: list[float] = []
    rows_w: list[float] = []
    n_ties = 0
    n_decisive = 0

    for m in matches:
        if m.model_a not in model_to_idx or m.model_b not in model_to_idx:
            continue
        row = [0.0] * n_models
        row[model_to_idx[m.model_a]] = 1.0
        row[model_to_idx[m.model_b]] = -1.0
        is_tie = abs(m.preference - 0.5) <= tie_eps

        if is_tie:
            n_ties += 1
            if not keep_ties:
                continue
            rows_X.append(row.copy())
            rows_y.append(1.0)
            rows_w.append(0.5)
            rows_X.append(row.copy())
            rows_y.append(0.0)
            rows_w.append(0.5)
            continue

        n_decisive += 1
        rows_X.append(row)
        # y = 1 means model_a wins (preference < 0.5)
        rows_y.append(1.0 if m.preference < 0.5 else 0.0)
        rows_w.append(1.0)

    if len(rows_y) < 2:
        logger.warning("Too few decisive matches (%d) to fit BT strengths", len(rows_y))
        return {m: 0.0 for m in models}

    X = np.array(rows_X)
    y = np.array(rows_y)
    sample_weight = np.array(rows_w, dtype=float)

    C = 1.0 / max(regularization, 1e-12)
    lr = LogisticRegression(
        C=C,
        fit_intercept=False,  # strengths are relative
        max_iter=1000,
        solver="lbfgs",
    )
    lr.fit(X, y, sample_weight=sample_weight)

    strengths = lr.coef_[0]
    # Center so mean = 0
    strengths = strengths - strengths.mean()

    result = {m: float(strengths[i]) for i, m in enumerate(models)}

    logger.info(
        "BT strengths fitted on %d training rows (%d decisive matches, %d ties, tie_strategy=%s):",
        len(rows_y),
        n_decisive,
        n_ties,
        tie_strategy,
    )
    for m, s in sorted(result.items(), key=lambda x: -x[1]):
        logger.info("  %s: %+.4f", m.rsplit("/", 1)[-1], s)

    return result

# Source: OpenJury/openjury/arena/ratings.py:486
def score_gap_array(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    dimension_names: list[str],
    dim_weights: dict[str, float] | None = None,
) -> np.ndarray:
    r"""Weighted score gap per comparison.

    .. math::

        \Delta_i = \sum_d w_d \bigl(s^A_{i,d} - s^B_{i,d}\bigr)

    Args:
        scores_a: ``(n, d)`` DataFrame of model-A rubric scores.
        scores_b: ``(n, d)`` DataFrame of model-B rubric scores.
        dimension_names: Ordered list of rubric dimension column names.
        dim_weights: ``{dim: weight}``.  ``None`` → uniform.

    Returns:
        ``(n,)`` array of score gaps (positive = A scored higher).
    """
    delta = scores_a[dimension_names].values - scores_b[dimension_names].values
    if dim_weights is not None:
        w = np.array([dim_weights.get(d, 0.0) for d in dimension_names])
        w = w / (w.sum() + 1e-12)
    else:
        w = np.ones(len(dimension_names)) / len(dimension_names)
    gap = (delta * w[None, :]).sum(axis=1)
    return np.nan_to_num(gap, nan=0.0)

# Source: OpenJury/openjury/arena/ratings.py:517
def compute_soft_labels(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    dimension_names: list[str],
    beta: float,
    dim_weights: dict[str, float] | None = None,
) -> np.ndarray:
    r"""Convert rubric scores into continuous soft preference labels.

    .. math::

        p_i = \sigma\!\bigl(\beta \cdot \Delta_i\bigr)

    where :math:`\Delta_i` is from :func:`score_gap_array`.  Values near
    0.5 indicate an uncertain comparison; values near 0 or 1 a decisive
    one.  Convention: *lower* p → model A preferred (matching the arena
    0.0 = A-wins convention).

    Args:
        scores_a: ``(n, d)`` rubric scores for model A.
        scores_b: ``(n, d)`` rubric scores for model B.
        dimension_names: Ordered rubric dimension names.
        beta: Temperature — controls how sharply score gaps map to
            preferences.  0 → all labels ≈ 0.5;  ∞ → hard labels.
        dim_weights: Per-dimension weights; ``None`` → uniform.

    Returns:
        ``(n,)`` array of soft labels in (0, 1), clipped to
        ``[1e-6, 1-1e-6]`` for numerical safety.
    """
    gap = score_gap_array(scores_a, scores_b, dimension_names, dim_weights)
    # Negate gap: positive gap means A is better → preference < 0.5 (A wins)
    # but σ(β·gap) > 0.5 when gap > 0.  The BT solver interprets
    # p < 0.5 as "A preferred", so we must negate to stay consistent
    # with the 0 = A, 1 = B convention used elsewhere.
    soft = _logistic(beta * gap)
    return np.clip(soft, 1e-6, 1 - 1e-6)

# Source: OpenJury/openjury/arena/ratings.py:555
def fit_softlabel_bt(
    models: list[str],
    matches: list[MatchResult],
    soft_labels: np.ndarray,
    regularization: float = 0.01,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    r"""Fit Bradley-Terry with continuous soft labels via L-BFGS-B.

    Instead of hard binary outcomes, each comparison carries a continuous
    label :math:`p_i \in (0, 1)`.  The negative log-likelihood is the
    weighted cross-entropy between :math:`p_i` and
    :math:`\sigma(\theta_A - \theta_B)`, plus an L2 penalty:

    .. math::

        \mathcal{L}(\theta) =
            - \sum_i w_i \bigl[ p_i \log\sigma(\Delta\theta_i)
              + (1-p_i)\log(1-\sigma(\Delta\theta_i)) \bigr]
            + \tfrac{\lambda}{2}\|\theta\|^2

    Args:
        models: Ordered list of model names.
        matches: Pairwise comparisons (only ``model_a`` / ``model_b`` used).
        soft_labels: ``(n,)`` continuous preferences aligned with *matches*.
        regularization: L2 penalty λ.
        sample_weight: ``(n,)`` per-sample weights; ``None`` → uniform 1.

    Returns:
        ``{model: θ}`` — centred BT strength parameters.
    """
    n_models = len(models)
    idx_map = {m: i for i, m in enumerate(models)}

    rows_X: list[list[float]] = []
    rows_p: list[float] = []
    rows_w: list[float] = []
    for mi, m in enumerate(matches):
        if m.model_a not in idx_map or m.model_b not in idx_map:
            continue
        row = [0.0] * n_models
        row[idx_map[m.model_a]] = 1.0
        row[idx_map[m.model_b]] = -1.0
        rows_X.append(row)
        rows_p.append(float(soft_labels[mi]))
        rows_w.append(float(sample_weight[mi]) if sample_weight is not None else 1.0)

    if len(rows_p) < 2:
        logger.warning("Too few matches (%d) for softlabel BT", len(rows_p))
        return {m: 0.0 for m in models}

    X = np.array(rows_X)
    p = np.clip(np.array(rows_p), 1e-6, 1 - 1e-6)
    w = np.array(rows_w, dtype=float)
    lam = regularization

    def _objective(theta: np.ndarray) -> float:
        logits = X @ theta
        sig = np.clip(_logistic(logits), 1e-12, 1 - 1e-12)
        nll = -float(np.sum(w * (p * np.log(sig) + (1.0 - p) * np.log(1.0 - sig))))
        return nll + 0.5 * lam * float(np.dot(theta, theta))

    def _gradient(theta: np.ndarray) -> np.ndarray:
        sig = _logistic(X @ theta)
        return X.T @ (w * (sig - p)) + lam * theta

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        res = minimize(
            _objective, np.zeros(n_models), jac=_gradient,
            method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10},
        )
    theta = res.x
    theta -= theta.mean()

    result = {m: float(theta[i]) for i, m in enumerate(models)}
    logger.info(
        "Soft-label BT fitted on %d comparisons (regularization=%.4f):",
        len(rows_p), regularization,
    )
    return result

# Source: experiments/bt.py:68
def model_universe(frame: pd.DataFrame) -> list[str]:
    """Sorted union of all model identifiers in ``model_a`` / ``model_b``."""
    return sorted(set(frame["model_a"]).union(set(frame["model_b"])))

# Source: experiments/bt.py:73
def infer_score_dimensions(frame: pd.DataFrame) -> list[str]:
    """Return sorted dimension names from the first non-empty ``scores_a`` dict."""
    for value in frame.get("scores_a", []):
        if isinstance(value, dict) and value:
            return sorted(value.keys())
    return []

# Source: experiments/bt.py:81
def clean_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any score value is None or NaN."""
    def _ok(s: Any) -> bool:
        if not isinstance(s, dict):
            return False
        for v in s.values():
            if v is None:
                return False
            try:
                if pd.isna(v):
                    return False
            except (TypeError, ValueError):
                pass
        return True
    mask = df["scores_a"].apply(_ok) & df["scores_b"].apply(_ok)
    dropped = int((~mask).sum())
    if dropped:
        print(f"  [clean] dropped {dropped}/{len(df)} rows with None scores")
    return df[mask].reset_index(drop=True)

# Source: experiments/bt.py:181
def valid_decisive(frame: pd.DataFrame, pref_col: str) -> pd.DataFrame:
    """Rows with decisive (non-tie) preferences."""
    out = frame[frame[pref_col].notna()].copy()
    out = out[np.abs(out[pref_col].astype(float) - 0.5) > 0.05]
    return out.reset_index(drop=True)

# Source: experiments/bt.py:188
def score_matrices(
    frame: pd.DataFrame,
    dimension_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract (scores_a, scores_b) as aligned DataFrames."""
    scores_a = pd.DataFrame(
        [{dim: (row or {}).get(dim, np.nan) for dim in dimension_names}
         for row in frame["scores_a"]],
        columns=dimension_names,
    )
    scores_b = pd.DataFrame(
        [{dim: (row or {}).get(dim, np.nan) for dim in dimension_names}
         for row in frame["scores_b"]],
        columns=dimension_names,
    )
    return scores_a, scores_b

# Source: experiments/bt.py:230
def build_matches(
    frame: pd.DataFrame,
    pref_col: str,
    *,
    decisive_only: bool = False,
) -> list[MatchResult]:
    """Build a list of ``MatchResult`` from a DataFrame."""
    src = (valid_decisive(frame, pref_col) if decisive_only
           else frame.dropna(subset=[pref_col]).copy())
    matches: list[MatchResult] = []
    for idx, row in src.iterrows():
        matches.append(
            MatchResult(
                model_a=str(row["model_a"]),
                model_b=str(row["model_b"]),
                instruction_index=int(idx),
                scores_a=row.get("scores_a", {}) or {},
                scores_b=row.get("scores_b", {}) or {},
                preference=float(row[pref_col]),
                instruction=row.get("instruction", "") or "",
                instruction_id=row.get("instruction_id", "") or "",
                instruction_metadata={},
                completion_a="",
                completion_b="",
                raw_judge_output="",
                raw_judge_output_swapped=None,
            )
        )
    return matches

# Source: experiments/bt.py:265
def theta_to_elo(theta: dict[str, float]) -> dict[str, float]:
    """Convert raw BT θ values to the Elo scale."""
    return {m: ELO_BASE + BT_TO_ELO * v for m, v in theta.items()}

# Source: experiments/bt.py:290
def eval_elo(
    judge_elo: dict[str, float],
    human_elo: dict[str, float],
) -> dict[str, float]:
    """Compute MAE / ρ / τ / spread between two Elo dicts (NaN-safe)."""
    common = sorted(set(judge_elo) & set(human_elo))
    j = np.array([judge_elo[m] for m in common])
    h = np.array([human_elo[m] for m in common])
    mask = np.isfinite(j) & np.isfinite(h)
    j, h = j[mask], h[mask]
    if len(j) < 3:
        return {
            "mae": float("nan"), "spearman_rho": float("nan"),
            "kendall_tau": float("nan"), "spread": float("nan"),
            "n_models": int(mask.sum()),
        }
    return {
        "mae": float(np.mean(np.abs(j - h))),
        "spearman_rho": float(spearmanr(j, h).statistic),
        "kendall_tau": float(kendalltau(j, h).statistic),
        "spread": float(j.std() / h.std()) if h.std() > 0 else float("nan"),
        "n_models": int(mask.sum()),
    }

# Source: experiments/preferences.py:48
def collapse_preference_value(
    value: Any,
    *,
    collapse_policy: str,
    collapse_threshold: float,
) -> Any:
    if value is None:
        return value
    try:
        val = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(val):
        return value
    if collapse_policy == "quarter_to_tie" and abs(val - 0.5) <= collapse_threshold:
        return 0.5
    return val

# Source: experiments/preferences.py:67
def collapse_preference_frame(
    frame: pd.DataFrame,
    *,
    collapse_policy: str,
    collapse_threshold: float,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if collapse_policy == "none":
        return frame
    use_cols = list(columns) if columns is not None else PREFERENCE_COLUMNS
    out = frame.copy()
    for col in use_cols:
        if col not in out.columns:
            continue
        out[col] = out[col].map(
            lambda v: collapse_preference_value(
                v,
                collapse_policy=collapse_policy,
                collapse_threshold=collapse_threshold,
            )
        )
    return out

# Source: experiments/softlabel.py:118
def target_vs_anchor_rows(
    frame: pd.DataFrame,
    target_model: str,
    *,
    allowed_anchors: set[str] | None = None,
) -> pd.DataFrame:
    """Filter *frame* to rows involving *target_model*."""
    mask = (frame["model_a"] == target_model) | (frame["model_b"] == target_model)
    rows = frame.loc[mask].copy()
    if allowed_anchors is not None:
        opp = np.where(
            rows["model_a"] == target_model, rows["model_b"], rows["model_a"],
        )
        rows = rows.loc[pd.Series(opp).isin(allowed_anchors).values].copy()
    return rows.reset_index(drop=True)

# Source: experiments/softlabel.py:135
def _encoded_outcome_for_target(
    preference: float,
    *,
    target_is_model_a: bool,
    tie_strategy: str,
    threshold: float,
) -> float | None:
    is_tie = abs(float(preference) - 0.5) <= threshold
    if is_tie:
        return 0.5 if tie_strategy in {"half", "quarter"} else None
    if target_is_model_a:
        return 1.0 if float(preference) < 0.5 else 0.0
    return 1.0 if float(preference) > 0.5 else 0.0

# Source: experiments/softlabel.py:150
def encode_target_outcomes(
    rows: pd.DataFrame,
    target_model: str,
    pref_col: str,
    anchor_theta: dict[str, float],
    *,
    tie_strategy: str,
    threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode pairwise outcomes for a held-out target against anchors."""
    outcomes: list[float] = []
    anchor_strengths: list[float] = []
    for _, row in rows.iterrows():
        target_is_a = str(row["model_a"]) == target_model
        opp = str(row["model_b"]) if target_is_a else str(row["model_a"])
        if opp not in anchor_theta:
            continue
        encoded = _encoded_outcome_for_target(
            float(row[pref_col]),
            target_is_model_a=target_is_a,
            tie_strategy=tie_strategy,
            threshold=threshold,
        )
        if encoded is None:
            continue
        outcomes.append(float(encoded))
        anchor_strengths.append(float(anchor_theta[opp]))
    return np.asarray(outcomes, dtype=float), np.asarray(anchor_strengths, dtype=float)

# Source: experiments/softlabel.py:180
def solve_target_theta(
    y: np.ndarray,
    theta_anchor: np.ndarray,
    *,
    regularization: float,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """Newton\u2013Raphson for a single held-out model's BT strength."""
    if len(y) < 2:
        return float("nan")
    w = sample_weight if sample_weight is not None else np.ones_like(y)
    theta = 0.0
    for _ in range(max_iter):
        z = theta - theta_anchor
        p = _logistic(z)
        grad = float(np.sum(w * (y - p)) - regularization * theta)
        hess = float(-np.sum(w * p * (1.0 - p)) - regularization)
        if abs(hess) < 1e-12:
            break
        new_theta = theta - grad / hess
        if abs(new_theta - theta) < tol:
            theta = new_theta
            break
        theta = new_theta
    return float(theta)

# Source: experiments/softlabel.py:209
def fit_pool_bt(
    frame: pd.DataFrame,
    pref_col: str,
    models: list[str],
    *,
    regularization: float,
    tie_strategy: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit BT on an anchor pool.  Returns ``(\u03b8, Elo)``."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.")
        theta = fit_multi_bt(
            models=models,
            matches=build_matches(frame, pref_col),
            regularization=regularization,
            tie_strategy=tie_strategy,
        )
    return theta, theta_to_elo(theta)

# Source: experiments/softlabel.py:229
def _row_soft_label(
    scores_a: dict | None,
    scores_b: dict | None,
    dims: list[str],
    beta: float,
    dim_weights: dict[str, float] | None = None,
) -> float:
    """Compute \u03c3(\u03b2 \u00b7 weighted-gap) \u2208 (0, 1) from model-A's perspective."""
    sa = scores_a or {}
    sb = scores_b or {}
    w = dim_weights or {}
    total_w = 0.0
    weighted_gap = 0.0
    for d in dims:
        va = float(sa.get(d, 0.0))
        vb = float(sb.get(d, 0.0))
        dw = float(w.get(d, 1.0))
        weighted_gap += dw * (va - vb)
        total_w += dw
    gap = weighted_gap / total_w if total_w > 0 else 0.0
    x = max(min(beta * gap, 60.0), -60.0)
    return float(1.0 / (1.0 + np.exp(-x)))

# Source: experiments/softlabel.py:253
def encode_target_soft_outcomes(
    rows: pd.DataFrame,
    target_model: str,
    dims: list[str],
    beta: float,
    anchor_theta: dict[str, float],
    dim_weights: dict[str, float] | None = None,
    soft_tie_strategy: str | None = None,
    adaptive_beta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode soft labels for a held-out target against anchors.

    Returns ``(y, theta_anc, weights)`` — three aligned arrays.
    When *soft_tie_strategy* is ``"half"`` or ``"quarter"``, ties emit two
    half-weighted rows (y=1, w=0.5) and (y=0, w=0.5) instead of the soft
    label. ``"quarter"`` uses a wider tie zone; ``"half"`` expects exact ties.
    """
    tie_eps = 0.26 if soft_tie_strategy == "quarter" else 0.05
    outcomes: list[float] = []
    anchor_strengths: list[float] = []
    weights: list[float] = []
    adaptive_enabled = bool((adaptive_beta or {}).get("enabled", False))
    amb_scale = float((adaptive_beta or {}).get("amb_scale", 0.5))
    amb_weight = float((adaptive_beta or {}).get("amb_weight", 0.6))
    tie_eps_adapt = float((adaptive_beta or {}).get("tie_eps", 0.26))

    for _, row in rows.iterrows():
        target_is_a = str(row["model_a"]) == target_model
        opp = str(row["model_b"]) if target_is_a else str(row["model_a"])
        if opp not in anchor_theta:
            continue

        pref = row.get("judge_pref")
        is_tie = (soft_tie_strategy in {"half", "quarter"}
                  and pref is not None
                  and abs(float(pref) - 0.5) <= tie_eps)

        if is_tie:
            anc_val = float(anchor_theta[opp])
            # Two half-weighted rows
            outcomes.append(1.0)
            anchor_strengths.append(anc_val)
            weights.append(0.5)
            outcomes.append(0.0)
            anchor_strengths.append(anc_val)
            weights.append(0.5)
        elif adaptive_enabled:
            pref_f = float(pref) if pref is not None else 0.5
            beta_row = float(beta) if abs(pref_f - 0.5) > tie_eps_adapt else float(beta) * amb_scale
            soft_a = _row_soft_label(
                row.get("scores_a"), row.get("scores_b"),
                dims, beta_row, dim_weights,
            )
            y = soft_a if target_is_a else (1.0 - soft_a)
            y = max(min(y, 1.0 - 1e-6), 1e-6)
            outcomes.append(y)
            anchor_strengths.append(float(anchor_theta[opp]))
            weights.append(1.0 if abs(pref_f - 0.5) > tie_eps_adapt else amb_weight)
        else:
            soft_a = _row_soft_label(
                row.get("scores_a"), row.get("scores_b"),
                dims, beta, dim_weights,
            )
            y = soft_a if target_is_a else (1.0 - soft_a)
            y = max(min(y, 1.0 - 1e-6), 1e-6)
            outcomes.append(y)
            anchor_strengths.append(float(anchor_theta[opp]))
            weights.append(1.0)
    return (np.asarray(outcomes, dtype=float),
            np.asarray(anchor_strengths, dtype=float),
            np.asarray(weights, dtype=float))

# Source: experiments/softlabel.py:326
def fit_pool_bt_soft(
    frame: pd.DataFrame,
    judge_col: str,
    models: list[str],
    dims: list[str],
    beta: float,
    *,
    regularization: float,
    dim_weights: dict[str, float] | None = None,
    soft_tie_strategy: str | None = None,
    adaptive_beta: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit anchor-pool BT with soft labels.  Returns ``(θ, Elo)``.

    If *soft_tie_strategy* is ``"half"`` or ``"quarter"``, ties use two
    half-weighted rows instead of the soft label σ(β·gap). ``"quarter"`` uses
    a wider tie zone; ``"half"`` expects exact ties.
    """
    valid = frame.dropna(subset=[judge_col]).copy().reset_index(drop=True)
    sa, sb = score_matrices(valid, dims)
    matches = build_matches(valid, judge_col)
    adaptive_enabled = bool((adaptive_beta or {}).get("enabled", False))
    amb_scale = float((adaptive_beta or {}).get("amb_scale", 0.5))
    amb_weight = float((adaptive_beta or {}).get("amb_weight", 0.6))
    tie_eps_adapt = float((adaptive_beta or {}).get("tie_eps", 0.26))

    soft = compute_soft_labels(sa, sb, dims, beta, dim_weights)

    if adaptive_enabled:
        prefs = valid[judge_col].astype(float).values
        tie_mask = np.abs(prefs - 0.5) <= tie_eps_adapt
        soft_amb = compute_soft_labels(sa, sb, dims, beta * amb_scale, dim_weights)
        soft_final = np.where(tie_mask, soft_amb, soft)
        weights = np.where(tie_mask, amb_weight, 1.0)
        theta = fit_softlabel_bt(
            models,
            matches,
            soft_labels=np.array(soft_final, dtype=float),
            regularization=regularization,
            sample_weight=np.array(weights, dtype=float),
        )
        return theta, theta_to_elo(theta)

    if soft_tie_strategy in {"half", "quarter"}:
        # Build hybrid labels + weights: exact-tie/quarter-tie for ties, soft otherwise.
        tie_eps = 0.26 if soft_tie_strategy == "quarter" else 0.05
        prefs = valid[judge_col].values
        hybrid_labels = []
        hybrid_weights = []
        hybrid_matches = []
        for i, m in enumerate(matches):
            pref = float(prefs[i])
            is_tie = abs(pref - 0.5) <= tie_eps
            if is_tie:
                # Two half-weighted rows: y=1 (A wins) and y=0 (B wins)
                hybrid_matches.append(m)
                hybrid_labels.append(1.0)
                hybrid_weights.append(0.5)
                hybrid_matches.append(m)
                hybrid_labels.append(0.0)
                hybrid_weights.append(0.5)
            else:
                hybrid_matches.append(m)
                hybrid_labels.append(float(soft[i]))
                hybrid_weights.append(1.0)
        theta = fit_softlabel_bt(
            models, hybrid_matches,
            soft_labels=np.array(hybrid_labels),
            regularization=regularization,
            sample_weight=np.array(hybrid_weights),
        )
    else:
        theta = fit_softlabel_bt(
            models, matches, soft_labels=soft, regularization=regularization,
        )
    return theta, theta_to_elo(theta)

# Source: experiments/temperature.py:26
def _pref_to_model_a_prob(pref: float, *, threshold: float = 0.05) -> float | None:
    pref_f = float(pref)
    if not np.isfinite(pref_f):
        return None
    if abs(pref_f - 0.5) <= threshold:
        return 0.5
    return 1.0 if pref_f < 0.5 else 0.0

# Source: experiments/temperature.py:35
def fit_temperature_mle(
    frame: pd.DataFrame,
    dims: list[str],
    *,
    pref_col: str = "human_pref",
    mle_mode: str = "all",
    threshold: float = 0.05,
    t_min: float = 1e-3,
    t_max: float = 50.0,
) -> dict[str, float]:
    """Fit a single temperature from rubric score gaps to preferences.

    ``mle_mode='all'`` keeps ties as soft labels (y=0.5), while
    ``mle_mode='decisive'`` drops near-tie rows.
    """
    base = frame.dropna(subset=[pref_col]).copy().reset_index(drop=True)
    if base.empty:
        return {
            "temperature": float("nan"),
            "n_rows": 0,
            "n_decisive": 0,
            "n_ties": 0,
            "nll": float("nan"),
        }

    sa, sb = score_matrices(base, dims)
    gaps = (sa.mean(axis=1) - sb.mean(axis=1)).to_numpy(dtype=float)
    y = np.array([
        _pref_to_model_a_prob(v, threshold=threshold)
        for v in base[pref_col].to_numpy(dtype=float)
    ], dtype=float)

    valid = np.isfinite(gaps) & np.isfinite(y)
    if mle_mode == "decisive":
        valid &= np.abs(base[pref_col].to_numpy(dtype=float) - 0.5) > threshold

    gaps = gaps[valid]
    y = y[valid]
    n_rows = int(len(gaps))
    if n_rows < 8 or np.allclose(gaps, gaps[0]):
        return {
            "temperature": float("nan"),
            "n_rows": n_rows,
            "n_decisive": int(np.sum(np.abs(y - 0.5) > 1e-9)),
            "n_ties": int(np.sum(np.abs(y - 0.5) <= 1e-9)),
            "nll": float("nan"),
        }

    def _nll(log_t: float) -> float:
        temp = float(np.exp(log_t))
        logits = np.clip(temp * gaps, -60.0, 60.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    opt = minimize_scalar(
        _nll,
        bounds=(float(np.log(t_min)), float(np.log(t_max))),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 500},
    )
    temp = float(np.exp(opt.x)) if opt.success else float("nan")
    return {
        "temperature": temp,
        "n_rows": n_rows,
        "n_decisive": int(np.sum(np.abs(y - 0.5) > 1e-9)),
        "n_ties": int(np.sum(np.abs(y - 0.5) <= 1e-9)),
        "nll": float(opt.fun) if opt.success else float("nan"),
    }

# Source: experiments/temperature.py:106
def run_temperature_holdout(
    frame: pd.DataFrame,
    dims: list[str],
    models: list[str],
    *,
    reg: float,
    tie_strategy: str,
    mle_mode: str = "all",
    threshold: float = 0.05,
    compare_beta: float | None = None,
    soft_tie_strategy: str | None = None,
    precomputed_beta: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate temperature-MLE Soft-Elo under proper held-out-model fitting.

    For each target model, removes it from the anchor pool, then estimates the
    target Elo using a temperature scaling. By default the temperature is
    refit per fold on the anchor-only rows (strict LOO purification). When
    ``precomputed_beta`` is provided, that single value is used for every
    fold instead — matching a "fit β globally on all decisive battles, then
    use it for every held-out target" policy. Returns ``(summary_df, detail_df)``.
    """
    targets = list(models)
    detail_rows: list[dict[str, Any]] = []

    human_loo: dict[str, float] = {}
    hard_loo: dict[str, float] = {}
    temp_loo: dict[str, float] = {}
    beta_loo: dict[str, float] = {}

    use_global_beta = (
        precomputed_beta is not None and np.isfinite(float(precomputed_beta))
    )
    global_beta_value = float(precomputed_beta) if use_global_beta else float("nan")

    # When using a global beta, compute the temperature-fit diagnostics once
    # on the full (decisive) frame so detail rows still carry meaningful counts.
    if use_global_beta:
        global_temp_fit = fit_temperature_mle(
            frame,
            dims,
            pref_col="human_pref",
            mle_mode=mle_mode,
            threshold=threshold,
        )
    else:
        global_temp_fit = None

    for target in targets:
        anchors = [m for m in models if m != target]
        anchor_set = set(anchors)
        anchor_frame = frame[
            frame["model_a"].isin(anchor_set) & frame["model_b"].isin(anchor_set)
        ].copy()
        target_rows = target_vs_anchor_rows(frame, target, allowed_anchors=anchor_set)

        human_anchor_theta, _ = fit_pool_bt(
            anchor_frame,
            "human_pref",
            anchors,
            regularization=reg,
            tie_strategy=tie_strategy,
        )
        y_human, theta_human = encode_target_outcomes(
            target_rows,
            target,
            "human_pref",
            human_anchor_theta,
            tie_strategy=tie_strategy,
            threshold=threshold,
        )
        t_human = solve_target_theta(y_human, theta_human, regularization=reg)
        human_loo[target] = (
            float(ELO_BASE + BT_TO_ELO * t_human) if np.isfinite(t_human) else float("nan")
        )

        judge_anchor_theta_hard, _ = fit_pool_bt(
            anchor_frame,
            "judge_pref",
            anchors,
            regularization=reg,
            tie_strategy=tie_strategy,
        )
        y_hard, theta_hard = encode_target_outcomes(
            target_rows,
            target,
            "judge_pref",
            judge_anchor_theta_hard,
            tie_strategy=tie_strategy,
            threshold=threshold,
        )
        t_hard = solve_target_theta(y_hard, theta_hard, regularization=reg)
        hard_loo[target] = (
            float(ELO_BASE + BT_TO_ELO * t_hard) if np.isfinite(t_hard) else float("nan")
        )

        if use_global_beta:
            # Skip per-fold temperature MLE; reuse the global value.
            temp_fit = global_temp_fit
            temp_value = global_beta_value
        else:
            temp_fit = fit_temperature_mle(
                anchor_frame,
                dims,
                pref_col="human_pref",
                mle_mode=mle_mode,
                threshold=threshold,
            )
            temp_value = float(temp_fit["temperature"])
        if np.isfinite(temp_value):
            judge_anchor_theta_temp, _ = fit_pool_bt_soft(
                anchor_frame,
                "judge_pref",
                anchors,
                dims,
                temp_value,
                regularization=reg,
                soft_tie_strategy=soft_tie_strategy,
            )
            y_temp, theta_temp, sw_temp = encode_target_soft_outcomes(
                target_rows,
                target,
                dims,
                temp_value,
                judge_anchor_theta_temp,
                soft_tie_strategy=soft_tie_strategy,
            )
            t_temp = solve_target_theta(
                y_temp,
                theta_temp,
                regularization=reg,
                sample_weight=sw_temp,
            )
            temp_loo[target] = (
                float(ELO_BASE + BT_TO_ELO * t_temp) if np.isfinite(t_temp) else float("nan")
            )
        else:
            temp_loo[target] = float("nan")

        if compare_beta is not None and np.isfinite(float(compare_beta)):
            judge_anchor_theta_beta, _ = fit_pool_bt_soft(
                anchor_frame,
                "judge_pref",
                anchors,
                dims,
                float(compare_beta),
                regularization=reg,
                soft_tie_strategy=soft_tie_strategy,
            )
            y_beta, theta_beta, sw_beta = encode_target_soft_outcomes(
                target_rows,
                target,
                dims,
                float(compare_beta),
                judge_anchor_theta_beta,
                soft_tie_strategy=soft_tie_strategy,
            )
            t_beta = solve_target_theta(
                y_beta,
                theta_beta,
                regularization=reg,
                sample_weight=sw_beta,
            )
            beta_loo[target] = (
                float(ELO_BASE + BT_TO_ELO * t_beta) if np.isfinite(t_beta) else float("nan")
            )
        else:
            beta_loo[target] = float("nan")

        detail_rows.extend([
            {
                "model": target,
                "method": "hard",
                "judge_bt_elo": hard_loo[target],
                "human_bt_elo": human_loo[target],
                "bt_elo_residual": human_loo[target] - hard_loo[target],
                "abs_bt_elo_residual": abs(human_loo[target] - hard_loo[target]),
                "temperature": float("nan"),
                "mle_mode": mle_mode,
                "temperature_fit_rows": int(temp_fit["n_rows"]),
                "temperature_fit_decisive": int(temp_fit["n_decisive"]),
                "temperature_fit_ties": int(temp_fit["n_ties"]),
            },
            {
                "model": target,
                "method": "temperature_mle",
                "judge_bt_elo": temp_loo[target],
                "human_bt_elo": human_loo[target],
                "bt_elo_residual": human_loo[target] - temp_loo[target],
                "abs_bt_elo_residual": abs(human_loo[target] - temp_loo[target]),
                "temperature": temp_value,
                "mle_mode": mle_mode,
                "temperature_fit_rows": int(temp_fit["n_rows"]),
                "temperature_fit_decisive": int(temp_fit["n_decisive"]),
                "temperature_fit_ties": int(temp_fit["n_ties"]),
            },
        ])
        if compare_beta is not None and np.isfinite(float(compare_beta)):
            detail_rows.append({
                "model": target,
                "method": "soft_beta_selected",
                "judge_bt_elo": beta_loo[target],
                "human_bt_elo": human_loo[target],
                "bt_elo_residual": human_loo[target] - beta_loo[target],
                "abs_bt_elo_residual": abs(human_loo[target] - beta_loo[target]),
                "temperature": float(compare_beta),
                "mle_mode": mle_mode,
                "temperature_fit_rows": int(temp_fit["n_rows"]),
                "temperature_fit_decisive": int(temp_fit["n_decisive"]),
                "temperature_fit_ties": int(temp_fit["n_ties"]),
            })

    summary_rows: list[dict[str, Any]] = []
    method_to_elo = {
        "hard": hard_loo,
        "temperature_mle": temp_loo,
    }
    if compare_beta is not None and np.isfinite(float(compare_beta)):
        method_to_elo["soft_beta_selected"] = beta_loo

    temp_values = np.array([
        row["temperature"] for row in detail_rows if row["method"] == "temperature_mle"
    ], dtype=float)

    for method, judge_elo in method_to_elo.items():
        ev = eval_elo(judge_elo, human_loo)
        summary_rows.append({
            "method": method,
            "mle_mode": mle_mode,
            "beta_reference": float(compare_beta) if compare_beta is not None else float("nan"),
            "temperature_fit_mean": float(np.nanmean(temp_values)) if method == "temperature_mle" else float("nan"),
            "temperature_fit_std": float(np.nanstd(temp_values, ddof=1)) if method == "temperature_mle" and np.sum(np.isfinite(temp_values)) >= 2 else float("nan"),
            **ev,
        })

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, detail_df

# Source: experiments/conformal.py:49
def prepare_shared_frame(
    frame: pd.DataFrame,
    *,
    human_col: str = "human_pref",
    judge_col: str = "judge_pref",
    tie_strategy: str = "quarter",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Restrict to rows usable by both human and judge BT fits."""
    if tie_strategy not in {"drop", "half", "quarter"}:
        raise ValueError(f"Unsupported tie_strategy={tie_strategy!r}.")
    out = frame.dropna(subset=[human_col, judge_col]).copy()
    if tie_strategy == "drop":
        human_decisive = (out[human_col].astype(float) - 0.5).abs() > threshold
        judge_decisive = (out[judge_col].astype(float) - 0.5).abs() > threshold
        out = out.loc[human_decisive & judge_decisive].copy()
    return out.reset_index(drop=True)

# Source: experiments/conformal.py:105
def _count_target_outcomes(
    rows: pd.DataFrame,
    target_model: str,
    pref_col: str,
    *,
    tie_strategy: str,
    threshold: float = 0.05,
) -> int:
    count = 0
    for _, row in rows.iterrows():
        target_is_a = str(row["model_a"]) == target_model
        encoded = _encoded_outcome_for_target(
            float(row[pref_col]),
            target_is_model_a=target_is_a,
            tie_strategy=tie_strategy,
            threshold=threshold,
        )
        if encoded is not None:
            count += 1
    return count

# Source: experiments/conformal.py:127
def _eligible_target_models(
    frame: pd.DataFrame,
    *,
    min_matches: int,
    human_col: str = "human_pref",
    judge_col: str = "judge_pref",
    tie_strategy: str = "quarter",
    threshold: float = 0.05,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in model_universe(frame):
        target_rows = target_vs_anchor_rows(frame, model)
        human_count = _count_target_outcomes(
            target_rows, model, human_col,
            tie_strategy=tie_strategy, threshold=threshold,
        )
        judge_count = _count_target_outcomes(
            target_rows, model, judge_col,
            tie_strategy=tie_strategy, threshold=threshold,
        )
        rows.append({
            "model": model,
            "n_human_pairs": human_count,
            "n_judge_pairs": judge_count,
            "eligible": min(human_count, judge_count) >= min_matches,
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["eligible", "n_judge_pairs"], ascending=[False, False])
        .reset_index(drop=True)
    )

# Source: experiments/conformal.py:160
def _estimate_target_theta(
    rows: pd.DataFrame,
    target_model: str,
    pref_col: str,
    anchor_theta: dict[str, float],
    *,
    regularization: float,
    tie_strategy: str,
    threshold: float = 0.05,
) -> tuple[float, int]:
    y, theta_anchor = encode_target_outcomes(
        rows, target_model, pref_col, anchor_theta,
        tie_strategy=tie_strategy, threshold=threshold,
    )
    n = len(y)
    if n < 2:
        return float("nan"), n
    theta = solve_target_theta(y, theta_anchor, regularization=regularization)
    return theta, n

# Source: experiments/conformal.py:181
def _bootstrap_target_theta(
    rows: pd.DataFrame,
    target_model: str,
    pref_col: str,
    anchor_theta: dict[str, float],
    *,
    tie_strategy: str,
    threshold: float,
    anchor_frame: pd.DataFrame | None = None,
    anchor_models: list[str] | None = None,
    bootstrap_anchor_pool: bool = False,
    n_bootstrap: int,
    seed: int,
    regularization: float,
) -> pd.DataFrame:
    encoded_y, _ = encode_target_outcomes(
        rows, target_model, pref_col, anchor_theta,
        tie_strategy=tie_strategy, threshold=threshold,
    )
    if len(encoded_y) < 6:
        return pd.DataFrame(columns=["bootstrap_id", "theta", "elo"])

    if bootstrap_anchor_pool and (anchor_frame is None or not anchor_models):
        raise ValueError("bootstrap_anchor_pool=True requires anchor_frame and anchor_models.")

    rng = np.random.default_rng(seed)
    samples: list[dict[str, float]] = []

    for boot_id in range(n_bootstrap):
        row_idx = rng.choice(np.arange(len(rows)), size=len(rows), replace=True)
        rows_boot = rows.iloc[row_idx].reset_index(drop=True)

        anchor_theta_boot = anchor_theta
        if bootstrap_anchor_pool:
            anchor_idx = rng.choice(
                np.arange(len(anchor_frame)), size=len(anchor_frame), replace=True,
            )
            anchor_frame_boot = anchor_frame.iloc[anchor_idx].reset_index(drop=True)
            anchor_theta_boot, _ = fit_pool_bt(
                anchor_frame_boot, pref_col, anchor_models,
                regularization=regularization, tie_strategy=tie_strategy,
            )

        y_boot, theta_anchor_boot = encode_target_outcomes(
            rows_boot, target_model, pref_col, anchor_theta_boot,
            tie_strategy=tie_strategy, threshold=threshold,
        )
        if len(y_boot) < 2:
            continue

        theta = solve_target_theta(y_boot, theta_anchor_boot, regularization=regularization)
        samples.append({
            "bootstrap_id": boot_id,
            "theta": float(theta),
            "elo": float(ELO_BASE + BT_TO_ELO * theta),
        })

    return pd.DataFrame(samples)

# Source: experiments/conformal.py:241
def _bootstrap_target_theta_soft(
    rows: pd.DataFrame,
    target_model: str,
    dims: list[str],
    beta: float,
    anchor_theta: dict[str, float],
    *,
    dim_weights: dict[str, float] | None = None,
    n_bootstrap: int,
    seed: int,
    regularization: float,
) -> pd.DataFrame:
    y_full, _, _w = encode_target_soft_outcomes(
        rows, target_model, dims, beta, anchor_theta, dim_weights,
    )
    if len(y_full) < 6:
        return pd.DataFrame(columns=["bootstrap_id", "theta", "elo"])

    rng = np.random.default_rng(seed)
    samples: list[dict[str, float]] = []

    for boot_id in range(n_bootstrap):
        idx = rng.choice(np.arange(len(rows)), size=len(rows), replace=True)
        rows_boot = rows.iloc[idx].reset_index(drop=True)
        y_boot, theta_anc_boot, _w = encode_target_soft_outcomes(
            rows_boot, target_model, dims, beta, anchor_theta, dim_weights,
        )
        if len(y_boot) < 2:
            continue
        theta = solve_target_theta(y_boot, theta_anc_boot, regularization=regularization)
        samples.append({
            "bootstrap_id": boot_id,
            "theta": float(theta),
            "elo": float(ELO_BASE + BT_TO_ELO * theta),
        })

    return pd.DataFrame(samples)

# Source: experiments/conformal.py:285
def build_heldout_table(
    frame: pd.DataFrame,
    *,
    judge_mode: str = "hard",
    min_matches: int,
    regularization: float,
    bootstrap_resamples: int,
    seed: int,
    # hard-label specific
    bootstrap_anchor_pool: bool = False,
    # soft-label specific
    dims: list[str] | None = None,
    beta: float | None = None,
    dim_weights: dict[str, float] | None = None,
    # common
    human_col: str = "human_pref",
    judge_col: str = "judge_pref",
    tie_strategy: str = "quarter",
    threshold: float = 0.05,
    partial_save_path: Path | None = None,
    progress_every: int = 1,
    human_full_fit_elo: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build held-out calibration table (hard-label or soft-label judge)."""
    if judge_mode == "soft":
        if not dims:
            raise ValueError("soft judge_mode requires dims.")
        if beta is None or not np.isfinite(float(beta)):
            raise ValueError("soft judge_mode requires a finite beta.")

    inventory = _eligible_target_models(
        frame, min_matches=min_matches,
        human_col=human_col, judge_col=judge_col,
        tie_strategy=tie_strategy, threshold=threshold,
    )
    targets = inventory.loc[inventory["eligible"], "model"].tolist()
    all_models = model_universe(frame)
    rows: list[dict[str, Any]] = []

    if not targets:
        print(
            f"  ⚠ Skipping held-out calibration: no eligible target models "
            f"(need at least {min_matches} target-vs-anchor comparisons)."
        )
        return pd.DataFrame(), inventory

    tag = "[heldout-soft]" if judge_mode == "soft" else "[heldout]"
    for idx, target_model in enumerate(targets):
        if progress_every > 0 and (idx == 0 or (idx + 1) % progress_every == 0):
            print(f"{tag} target {idx + 1}/{len(targets)}: {target_model}", flush=True)

        anchors = [m for m in all_models if m != target_model]
        anchor_set = set(anchors)
        anchor_frame = frame[
            frame["model_a"].isin(anchor_set) & frame["model_b"].isin(anchor_set)
        ].copy()

        # Human side (always hard-label)
        human_anchor_theta, _ = fit_pool_bt(
            anchor_frame, human_col, anchors,
            regularization=regularization, tie_strategy=tie_strategy,
        )

        # Judge side
        if judge_mode == "soft":
            judge_anchor_theta, _ = fit_pool_bt_soft(
                anchor_frame, judge_col, anchors, dims, beta,
                regularization=regularization, dim_weights=dim_weights,
            )
        else:
            judge_anchor_theta, _ = fit_pool_bt(
                anchor_frame, judge_col, anchors,
                regularization=regularization, tie_strategy=tie_strategy,
            )

        target_rows = target_vs_anchor_rows(
            frame, target_model, allowed_anchors=anchor_set,
        )

        # Human target θ
        human_theta, n_human = _estimate_target_theta(
            target_rows, target_model, human_col, human_anchor_theta,
            regularization=regularization,
            tie_strategy=tie_strategy, threshold=threshold,
        )

        # Judge target θ
        if judge_mode == "soft":
            y_soft, theta_anc, _w = encode_target_soft_outcomes(
                target_rows, target_model, dims, beta,
                judge_anchor_theta, dim_weights,
            )
            n_judge = len(y_soft)
            judge_theta = (
                solve_target_theta(y_soft, theta_anc, regularization=regularization)
                if n_judge >= 2 else float("nan")
            )
        else:
            judge_theta, n_judge = _estimate_target_theta(
                target_rows, target_model, judge_col, judge_anchor_theta,
                regularization=regularization,
                tie_strategy=tie_strategy, threshold=threshold,
            )

        # Bootstrap SE
        if judge_mode == "soft":
            judge_boot = _bootstrap_target_theta_soft(
                target_rows, target_model, dims, beta, judge_anchor_theta,
                dim_weights=dim_weights,
                n_bootstrap=bootstrap_resamples,
                seed=seed + idx * 17,
                regularization=regularization,
            )
        else:
            judge_boot = _bootstrap_target_theta(
                target_rows, target_model, judge_col, judge_anchor_theta,
                tie_strategy=tie_strategy, threshold=threshold,
                anchor_frame=anchor_frame, anchor_models=anchors,
                bootstrap_anchor_pool=bootstrap_anchor_pool,
                n_bootstrap=bootstrap_resamples,
                seed=seed + idx * 17,
                regularization=regularization,
            )

        human_boot = _bootstrap_target_theta(
            target_rows, target_model, human_col, human_anchor_theta,
            tie_strategy=tie_strategy, threshold=threshold,
            n_bootstrap=min(60, bootstrap_resamples),
            seed=seed + idx * 23,
            regularization=regularization,
        )

        judge_elo = ELO_BASE + BT_TO_ELO * judge_theta if np.isfinite(judge_theta) else float("nan")
        human_elo_loo = ELO_BASE + BT_TO_ELO * human_theta if np.isfinite(human_theta) else float("nan")
        human_elo = (
            human_full_fit_elo.get(target_model, human_elo_loo)
            if human_full_fit_elo is not None
            else human_elo_loo
        )

        rows.append({
            "model": target_model,
            "n_human_pairs": int(n_human),
            "n_judge_pairs": int(n_judge),
            "n_pairs": int(min(n_human, n_judge)),
            "human_bt_theta": float(human_theta),
            "judge_bt_theta": float(judge_theta),
            "human_bt_elo": float(human_elo),
            "human_bt_elo_loo": float(human_elo_loo),
            "judge_bt_elo": float(judge_elo),
            "bt_elo_residual": float(human_elo - judge_elo),
            "abs_bt_elo_residual": float(abs(human_elo - judge_elo)),
            "judge_bt_elo_se": (
                float(judge_boot["elo"].std(ddof=1)) if len(judge_boot) >= 5 else float("nan")
            ),
            "human_bt_elo_se": (
                float(human_boot["elo"].std(ddof=1)) if len(human_boot) >= 5 else float("nan")
            ),
            "judge_bt_elo_p05": (
                float(np.percentile(judge_boot["elo"], 5)) if len(judge_boot) >= 5 else float("nan")
            ),
            "judge_bt_elo_p95": (
                float(np.percentile(judge_boot["elo"], 95)) if len(judge_boot) >= 5 else float("nan")
            ),
        })

        if partial_save_path is not None:
            pd.DataFrame(rows).to_csv(partial_save_path, index=False)

    heldout = pd.DataFrame(rows).sort_values("abs_bt_elo_residual", ascending=False).reset_index(drop=True)
    heldout["judge_bt_elo_se"] = heldout["judge_bt_elo_se"].replace(0.0, np.nan)
    heldout["normalized_residual"] = heldout["bt_elo_residual"] / heldout["judge_bt_elo_se"]
    return heldout, inventory

# Source: experiments/conformal.py:692
def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No valid conformal scores.")
    level = min(np.ceil((len(values) + 1) * (1 - alpha)) / len(values), 1.0)
    return float(np.quantile(values, level, method="higher"))

# Source: experiments/conformal.py:744
def split_conformal_model_calibration(
    heldout_table: pd.DataFrame,
    *,
    alpha: float,
    use_normalized: bool,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    table = heldout_table.copy()
    if use_normalized:
        table = table[
            table["judge_bt_elo_se"].notna()
            & np.isfinite(table["judge_bt_elo_se"])
            & (table["judge_bt_elo_se"] > 0)
        ].copy()
    else:
        table = table[np.isfinite(table["bt_elo_residual"])].copy()

    shuffled = table.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    if n < 8:
        raise ValueError("Need at least 8 eligible held-out models for split conformal.")

    n_cal = max(4, n // 2)
    calib = shuffled.iloc[:n_cal].copy()
    test = shuffled.iloc[n_cal:].copy()

    if use_normalized:
        calib_scores = calib["abs_bt_elo_residual"] / calib["judge_bt_elo_se"]
        qhat = _conformal_quantile(calib_scores.to_numpy(dtype=float), alpha)
        test["interval_lo"] = test["judge_bt_elo"] - qhat * test["judge_bt_elo_se"]
        test["interval_hi"] = test["judge_bt_elo"] + qhat * test["judge_bt_elo_se"]
        mode = "normalized"
    else:
        qhat = _conformal_quantile(calib["abs_bt_elo_residual"].to_numpy(dtype=float), alpha)
        test["interval_lo"] = test["judge_bt_elo"] - qhat
        test["interval_hi"] = test["judge_bt_elo"] + qhat
        mode = "absolute"

    test["covered"] = (
        (test["human_bt_elo"] >= test["interval_lo"])
        & (test["human_bt_elo"] <= test["interval_hi"])
    )
    test["interval_width"] = test["interval_hi"] - test["interval_lo"]

    meta: dict[str, float | int | str] = {
        "alpha": float(alpha), "mode": mode,
        "n_total": int(n), "n_calibration": int(len(calib)), "n_test": int(len(test)),
        "qhat": float(qhat),
        "empirical_coverage": float(test["covered"].mean()),
        "median_interval_width": float(test["interval_width"].median()),
    }
    return test.sort_values("abs_bt_elo_residual", ascending=False).reset_index(drop=True), meta

# Source: experiments/conformal.py:798
def repeated_split_conformal(
    heldout_table: pd.DataFrame,
    *,
    alpha: float,
    use_normalized: bool,
    seed: int,
    n_repeats: int,
    partial_save_path: Path | None = None,
    progress_every: int = 1,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    rows: list[dict[str, float | int]] = []
    for repeat in range(n_repeats):
        if progress_every > 0 and (repeat == 0 or (repeat + 1) % progress_every == 0):
            print(f"[conformal] repeat {repeat + 1}/{n_repeats} @ alpha={alpha:.3f}", flush=True)
        split_seed = seed + repeat * 9973
        _, meta = split_conformal_model_calibration(
            heldout_table, alpha=alpha, use_normalized=use_normalized, seed=split_seed,
        )
        rows.append({
            "repeat": int(repeat), "seed": int(split_seed), "alpha": float(alpha),
            "qhat": float(meta["qhat"]),
            "empirical_coverage": float(meta["empirical_coverage"]),
            "median_interval_width": float(meta["median_interval_width"]),
            "n_total": int(meta["n_total"]),
            "n_calibration": int(meta["n_calibration"]),
            "n_test": int(meta["n_test"]),
        })
        if partial_save_path is not None:
            pd.DataFrame(rows).to_csv(partial_save_path, index=False)

    df = pd.DataFrame(rows).sort_values(["alpha", "repeat"]).reset_index(drop=True)
    coverage = df["empirical_coverage"].to_numpy(dtype=float)
    qhats = df["qhat"].to_numpy(dtype=float)
    widths = df["median_interval_width"].to_numpy(dtype=float)
    summary: dict[str, float | int] = {
        "n_repeats": int(n_repeats),
        "coverage_mean": float(np.mean(coverage)),
        "coverage_std": float(np.std(coverage, ddof=1)) if len(df) > 1 else 0.0,
        "coverage_min": float(np.min(coverage)),
        "coverage_max": float(np.max(coverage)),
        "qhat_mean": float(np.mean(qhats)),
        "qhat_std": float(np.std(qhats, ddof=1)) if len(df) > 1 else 0.0,
        "interval_width_median_of_medians": float(np.median(widths)),
        "interval_width_mean": float(np.mean(widths)),
        "interval_width_std": float(np.std(widths, ddof=1)) if len(df) > 1 else 0.0,
    }
    return df, summary

# Source: experiments/conformal.py:1082
def build_alpha_summary(
    repeat_df: pd.DataFrame,
    operational_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a summary table across alpha values."""
    rows: list[dict[str, Any]] = []
    operational_lookup = {float(row["alpha"]): row for row in operational_rows}
    for alpha, sub in repeat_df.groupby("alpha", sort=True):
        coverage = sub["empirical_coverage"].to_numpy(dtype=float)
        widths = sub["median_interval_width"].to_numpy(dtype=float)
        qhats = sub["qhat"].to_numpy(dtype=float)
        op = operational_lookup.get(float(alpha), {})
        rows.append({
            "alpha": float(alpha),
            "split_coverage_point": float(np.mean(coverage)),
            "split_coverage_lo": float(np.percentile(coverage, 2.5)),
            "split_coverage_hi": float(np.percentile(coverage, 97.5)),
            "split_median_width_point": float(np.mean(widths)),
            "split_median_width_lo": float(np.percentile(widths, 2.5)),
            "split_median_width_hi": float(np.percentile(widths, 97.5)),
            "qhat_point": float(np.mean(qhats)),
            "qhat_lo": float(np.percentile(qhats, 2.5)),
            "qhat_hi": float(np.percentile(qhats, 97.5)),
            "operational_coverage": float(op.get("coverage", float("nan"))),
            "operational_median_width": float(op.get("median_width", float("nan"))),
            "operational_qhat": float(op.get("qhat", float("nan"))),
            "n_models": float(op.get(
                "n_models",
                sub["n_total"].iloc[0] if len(sub) else float("nan"),
            )),
        })
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)

# Source: experiments/adhoc_constant_smoothing.py:70
def _build_constant_matches(
    anchor_frame: pd.DataFrame,
    anchors: list[str],
    pref_col: str,
    c: float,
    tie_eps: float = TIE_EPS_QUARTER,
) -> tuple[list[MatchResult], np.ndarray, np.ndarray]:
    """Build canonical-format soft labels for the constant-smoothing target.

    For each anchor battle:
    * |pref - 0.5| <= tie_eps: emit two half-weighted rows (y=1, w=0.5) and
      (y=0, w=0.5) — matches `fit_pool_bt_soft` quarter-tie expansion.
    * pref < 0.5 (A wins, OpenJury convention): y = c, w = 1.
    * pref > 0.5 (B wins): y = 1 - c, w = 1.
    """
    matches: list[MatchResult] = []
    labels: list[float] = []
    weights: list[float] = []
    for idx, row in anchor_frame.reset_index(drop=True).iterrows():
        pref = float(row[pref_col])
        if abs(pref - 0.5) <= tie_eps:
            for y, w in [(1.0, 0.5), (0.0, 0.5)]:
                matches.append(_make_match(row, idx))
                labels.append(y)
                weights.append(w)
        else:
            # fit_softlabel_bt uses logit = θ_A − θ_B, so p→1 means A wins.
            # A wins (pref<0.5) must therefore map to the HIGH target c,
            # matching compute_soft_labels (σ(β·(score_A−score_B))).
            y = c if pref < 0.5 else 1.0 - c
            matches.append(_make_match(row, idx))
            labels.append(y)
            weights.append(1.0)
    return matches, np.array(labels, dtype=float), np.array(weights, dtype=float)

# Source: experiments/adhoc_constant_smoothing.py:106
def _make_match(row: pd.Series, idx: int) -> MatchResult:
    return MatchResult(
        model_a=str(row["model_a"]),
        model_b=str(row["model_b"]),
        instruction_index=int(idx),
        scores_a=row.get("scores_a", {}) or {},
        scores_b=row.get("scores_b", {}) or {},
        preference=float(row.get("judge_pref", 0.5)),
        instruction=row.get("instruction", "") or "",
        instruction_id=row.get("instruction_id", "") or "",
        instruction_metadata={},
        completion_a="",
        completion_b="",
        raw_judge_output="",
        raw_judge_output_swapped=None,
    )

# Source: experiments/adhoc_constant_smoothing.py:124
def _encode_constant_target_for_held_out(
    target_rows: pd.DataFrame,
    target: str,
    c: float,
    anchor_theta: dict[str, float],
    tie_eps: float = TIE_EPS_QUARTER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quarter-tie expansion for placing the held-out target's strength."""
    y_list, w_list, anc_list = [], [], []
    for _, row in target_rows.iterrows():
        target_is_a = str(row["model_a"]) == target
        opp = str(row["model_b"]) if target_is_a else str(row["model_a"])
        if opp not in anchor_theta:
            continue
        pref = float(row["judge_pref"])
        anc = float(anchor_theta[opp])
        if abs(pref - 0.5) <= tie_eps:
            y_list += [1.0, 0.0]
            w_list += [0.5, 0.5]
            anc_list += [anc, anc]
        else:
            # In target_rows we map back to the target's perspective.
            target_won = (pref < 0.5) if target_is_a else (pref > 0.5)
            y = c if target_won else (1.0 - c)
            y_list.append(y)
            w_list.append(1.0)
            anc_list.append(anc)
    return (
        np.asarray(y_list, dtype=float),
        np.asarray(anc_list, dtype=float),
        np.asarray(w_list, dtype=float),
    )
