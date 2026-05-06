#!/usr/bin/env python
"""Export sanitized annotation CSVs for the Soft-Elo release package.

The raw annotation JSON files contain prompts and completions. This exporter
keeps only the fields needed by the reference implementation: model IDs,
human/judge preference labels, judge score dictionaries, lightweight metadata,
and deterministic hashed row/instruction identifiers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PAPER_JUDGES = {
    "lmarena100k": {
        "config": "experiments/config.json",
        "labels": {
            "DeepSeek-V3.2": "DeepSeek-V3.2",
            "Gemma4-27B": "Gemma4-26B-A4B",
            "Qwen3.5-27B": "Qwen3.5-27B",
            "GPT-OSS-120B": "GPT-OSS-120B",
            "Gemma4-E4B": "Gemma4-E4B",
            "Llama-3.3-70B": "Llama-3.3-70B",
            "GPT-OSS-20B": "GPT-OSS-20B",
            "Qwen3-32B": "Qwen3-32B",
        },
    },
    "lmarena140k": {
        "config": "experiments/config_lmsys140k.json",
        "labels": {
            "Gemma4-E4B": "Gemma4-E4B",
            "Gemma4-27B": "Gemma4-26B-A4B",
            "DeepSeek-V3.2": "DeepSeek-V3.2",
            "Qwen3.5-27B": "Qwen3.5-27B",
            "GPT-OSS-20B": "GPT-OSS-20B",
            "GPT-OSS-120B": "GPT-OSS-120B",
            "Llama-3.3-70B": "Llama-3.3-70B",
        },
    },
    "comparia": {
        "config": "experiments/config_comparia.json",
        "labels": {
            "Gemma4-27B": "Gemma4-26B-A4B",
            "Gemma4-E4B": "Gemma4-E4B",
            "GPT-OSS-120B": "GPT-OSS-120B",
            "GPT-OSS-20B": "GPT-OSS-20B",
            "Llama-3.3-70B": "Llama-3.3-70B",
            "Qwen3.5-27B": "Qwen3.5-27B",
            "Qwen3-32B": "Qwen3-32B",
        },
    },
}


EXPORT_COLUMNS = [
    "battle_id",
    "source_dataset",
    "judge",
    "row_index",
    "instruction_id_hash",
    "original_question_id_hash",
    "instruction_index",
    "model_a",
    "model_b",
    "human_pref",
    "human_label",
    "judge_pref",
    "judge_label",
    "scores_a",
    "scores_b",
    "len_a",
    "len_b",
    "language",
    "source",
    "metadata_model_a",
    "metadata_model_b",
    "parse_error",
    "agree",
]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def stable_hash(value: Any) -> str:
    if value is None or value == "":
        return ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def as_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    metadata = row.get("instruction_metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def as_json_dict(value: Any) -> str:
    if not isinstance(value, dict):
        return "{}"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def as_pref(row: dict[str, Any], primary: str, fallback: str) -> Any:
    value = row.get(primary)
    if value is None:
        value = row.get(fallback)
    return "" if value is None else value


def sanitize_row(
    *,
    row: dict[str, Any],
    row_index: int,
    source_dataset: str,
    judge: str,
) -> dict[str, Any]:
    metadata = as_metadata(row)
    instruction_id = row.get("instruction_id")
    original_question_id = metadata.get("original_question_id")
    battle_id = f"{slugify(source_dataset)}-{slugify(judge)}-{row_index:06d}"

    return {
        "battle_id": battle_id,
        "source_dataset": source_dataset,
        "judge": judge,
        "row_index": row_index,
        "instruction_id_hash": stable_hash(instruction_id),
        "original_question_id_hash": stable_hash(original_question_id),
        "instruction_index": "" if row.get("instruction_index") is None else row.get("instruction_index"),
        "model_a": row.get("model_a", ""),
        "model_b": row.get("model_b", ""),
        "human_pref": as_pref(row, "human_pref", "human_preference"),
        "human_label": row.get("human_label", ""),
        "judge_pref": as_pref(row, "judge_pref", "preference"),
        "judge_label": row.get("judge_label", ""),
        "scores_a": as_json_dict(row.get("scores_a")),
        "scores_b": as_json_dict(row.get("scores_b")),
        "len_a": "" if row.get("len_a") is None else row.get("len_a"),
        "len_b": "" if row.get("len_b") is None else row.get("len_b"),
        "language": metadata.get("lang", metadata.get("language", "")) or "",
        "source": metadata.get("source", source_dataset) or source_dataset,
        "metadata_model_a": metadata.get("model_a", ""),
        "metadata_model_b": metadata.get("model_b", ""),
        "parse_error": row.get("parse_error", False),
        "agree": "" if row.get("agree") is None else row.get("agree"),
    }


def read_runs(repo_root: Path, config_path: str) -> dict[str, dict[str, Any]]:
    with (repo_root / config_path).open() as f:
        config = json.load(f)
    return {run["label"]: run for run in config["runs"]}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    rows = data.get("per_sample", data if isinstance(data, list) else [])
    if not isinstance(rows, list):
        raise ValueError(f"No per-sample rows found in {path}")
    return rows


def export_dataset(repo_root: Path, out_root: Path, dataset: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    runs = read_runs(repo_root, spec["config"])
    dataset_dir = out_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    for config_label, paper_label in spec["labels"].items():
        if config_label not in runs:
            raise KeyError(f"{config_label} not found in {spec['config']}")
        source_json = repo_root / runs[config_label]["path"]
        rows = read_rows(source_json)
        out_csv = dataset_dir / f"annotations_{slugify(paper_label)}.csv"

        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS)
            writer.writeheader()
            for row_index, row in enumerate(rows):
                writer.writerow(
                    sanitize_row(
                        row=row,
                        row_index=row_index,
                        source_dataset=dataset,
                        judge=paper_label,
                    )
                )

        model_ids = {
            str(row.get("model_a"))
            for row in rows
            if row.get("model_a") not in (None, "")
        } | {
            str(row.get("model_b"))
            for row in rows
            if row.get("model_b") not in (None, "")
        }
        manifest_rows.append(
            {
                "source_dataset": dataset,
                "judge": paper_label,
                "csv_path": str(out_csv.relative_to(out_root.parent)),
                "source_config": spec["config"],
                "source_json": runs[config_label]["path"],
                "n_rows": len(rows),
                "n_models": len(model_ids),
                "contains_prompts": "false",
                "contains_completions": "false",
                "id_policy": "synthetic battle_id plus 16-character SHA256 hashes",
            }
        )

    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="..")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent
    repo_root = (package_root / args.repo_root).resolve()
    out_root = (package_root / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for dataset, spec in PAPER_JUDGES.items():
        manifest_rows.extend(export_dataset(repo_root, out_root, dataset, spec))

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        fieldnames = [
            "source_dataset",
            "judge",
            "csv_path",
            "source_config",
            "source_json",
            "n_rows",
            "n_models",
            "contains_prompts",
            "contains_completions",
            "id_policy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} sanitized annotation CSVs")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
