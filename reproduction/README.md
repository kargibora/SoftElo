# Reproduction package

See the [repository README](../README.md) for installation and commands.

| Suite | Experiment | Temperature fitting |
| --- | --- | --- |
| `main` | LMArena 100K Hard-Elo and Soft-Elo | Excludes each held-out model |
| `smoothing` | Five constant smoothing values and Hard/Soft baselines | Pooled for Soft-Elo |
| `conformal-historical` | Conformal intervals | Pooled |
| `lmarena140k` | LMArena 140K Hard-Elo and Soft-Elo | Excludes each held-out model |

Run the tests from the repository root:

```bash
uv run --project reproduction python -m unittest discover -s reproduction/tests
```

Code is licensed under [Apache-2.0](LICENSE). Dataset licenses are listed on
[Hugging Face](https://huggingface.co/datasets/kbora/SoftElo/blob/main/DATA_LICENSES.md).
