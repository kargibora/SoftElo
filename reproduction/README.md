# Paper reproduction implementation

This directory contains the numerical reference implementation, its locked
Python environment, tests, and LMArena140K summary references. Start with the
[repository README](../README.md) for runnable examples.

From the repository root:

```bash
uv sync --project reproduction --locked --python 3.12
uv run --project reproduction python reproduce.py --suite main --judge GPT-OSS-120B
```

`softelo_release.suite` downloads the pinned Hugging Face release, fits the
selected protocol, saves predictions, and compares them with archived results.
An existing local release can be supplied with `--data /path/to/dataset`.

## Protocols

| Suite | Temperature | Reference |
| --- | --- | --- |
| `main` | Refit without the held-out model | LMArena100K main Table 2, section91 model-level results |
| `smoothing` | Pooled | E.1 per-judge MAE for five fixed constants and Hard/Soft baselines |
| `conformal-historical` | Pooled | Table3 ratings, bootstrap standard errors and five split outcomes |
| `lmarena140k` | Refit without the held-out model | Saved section51 held-out summaries |

The historical conformal code uses 20 bootstrap replicates, five model splits,
seed42 and target coverage90%. The original Hard-Elo bootstrap resamples anchor
battles; Soft-Elo keeps anchors fixed. Its quantile uses the original higher
interpolation rule. These choices are preserved for numerical reproduction.

Constants in the smoothing sweep were compared retrospectively using human Elo.
Their objective differs from fitting beta by preference likelihood.

## Provenance and verification

[core_provenance.json](softelo_release/core_provenance.json) identifies the
extracted numerical functions. [references/lmarena140k/provenance.json](references/lmarena140k/provenance.json)
identifies the additional saved summaries. The dataset itself contains the main,
smoothing and conformal reference measurements.

[VERIFICATION.md](VERIFICATION.md) records the executed checks and their limits.
The package includes Apache-2.0 [LICENSE](LICENSE) and [NOTICE](NOTICE).
Dataset terms are available in the
[Hugging Face data license notes](https://huggingface.co/datasets/kbora/SoftElo/blob/main/DATA_LICENSES.md).
