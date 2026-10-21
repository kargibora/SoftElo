"""Load the release locally or from a pinned Hugging Face dataset revision."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SOURCES = {
    "lmarena100k": ("lmarena-ai/arena-human-preference-100k", "question_id"),
    "lmarena140k": ("lmarena-ai/arena-human-preference-140k", "id"),
    "comparia": ("ministere-culture/comparia-votes", "id"),
}
SCORE_COLUMNS = ["scores_a", "scores_b", "scores_a_original", "scores_b_original",
                 "scores_a_swapped", "scores_b_swapped"]
PREF_COLUMNS = ["human_pref", "judge_pref", "preference_original", "preference_swapped"]


def battle_id(corpus, source_id, model_a, model_b):
    """A source record plus unordered model pair, independent of judge/order."""
    value = [corpus, str(source_id), *sorted([str(model_a), str(model_b)])]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def resolve_release(path_or_repo, revision=None):
    path = Path(path_or_repo).expanduser()
    if path.is_dir():
        return path
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pass a full 40-character Hugging Face commit as revision for remote reproduction")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=str(path_or_repo), repo_type="dataset", revision=revision))


def load_annotations(path_or_repo, *, corpus, judge=None, subset="paper_valid", revision=None,
                     collapse_preferences=True):
    """Return rows in original per-judge order; scores are Python dictionaries.

    subset: all / paper_valid / shared_valid. Shared valid means the range-checked
    intersection across the released judges for this corpus, with matching human
    preferences. It is distinct from the per-judge Table 2 filtering.
    """
    if corpus not in SOURCES or subset not in {"all", "paper_valid", "shared_valid"}:
        raise ValueError("Unknown corpus or subset")
    root = resolve_release(path_or_repo, revision)
    files = sorted((root / "data" / corpus).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No annotation files for {corpus} under {root}")
    frames = []
    for file in files:
        table = pq.read_table(file)
        df = table.to_pandas()
        if judge is not None:
            df = df[df.judge == judge]
        if len(df):
            frames.append(df)
    if not frames:
        raise ValueError(f"No annotations for judge {judge!r} in {corpus}")
    df = pd.concat(frames, ignore_index=True)
    if subset == "paper_valid":
        df = df[df.paper_valid]
    elif subset == "shared_valid":
        ids = set(pq.read_table(root / "subsets" / f"{corpus}_shared_valid.parquet")["battle_id"].to_pylist())
        df = df[df.shared_eligible & df.battle_id.isin(ids)]
    for col in SCORE_COLUMNS:
        df[col] = df[col].map(lambda value: json.loads(value) if value is not None else None)
    if collapse_preferences:
        for col in PREF_COLUMNS:
            df[col] = df[col].map(lambda v: .5 if pd.notna(v) and abs(v - .5) <= .26 else v)
    # Canonical experiment loader uses these names. No prompt text is required.
    df["human_preference"] = df.human_pref
    df["preference"] = df.judge_pref
    return df.sort_values(["judge", "row_index"], kind="stable").reset_index(drop=True)


def load_source_rows(corpus, source_ids, *, revision):
    """Retrieve original rows through upstream access controls (including ComparIA).

    Original text is downloaded locally, never republished by this function.
    Authenticate with Hugging Face and accept the upstream terms if required.
    """
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Upstream source retrieval requires a full commit hash")
    from huggingface_hub import snapshot_download
    repo, id_col = SOURCES[corpus]
    root = Path(snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                                  allow_patterns="*.parquet"))
    wanted = set(map(str, source_ids))
    found = []
    for file in sorted(root.rglob("*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
            found.extend(row for row in batch.to_pylist() if str(row[id_col]) in wanted)
    missing = wanted - {str(row[id_col]) for row in found}
    if missing:
        raise ValueError(f"{len(missing)} source IDs absent from pinned upstream snapshot")
    return found
