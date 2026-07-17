# Getting started

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- no network access for benchmark construction or validation
- optional local Hugging Face checkpoint for model calibration

## Install

Clone the repository and create the locked environment:

```bash
git clone https://github.com/l1ghtsource/steering-research.git
cd steering-research
uv sync --extra dev
```

Check the CLI:

```bash
uv run pib --help
```

## Validate the checked-in benchmark

```bash
uv run pib validate
```

The command exits nonzero on an executable structural failure. It prints a JSON report with
inventory, solver results, split checks, shortcut baselines, runtime replays, release gates,
and limitations.

To refresh the checked-in JSON and Markdown reports:

```bash
uv run pib validate --write-artifacts
```

## Rebuild from source

```bash
uv run pib build
uv run pib validate --write-artifacts
```

Construction is deterministic. Rebuilding must reproduce the JSONL file hashes and dataset
hash recorded in `data/manifests/candidate_manifest.json`.

## Run development checks

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
```

The tests include an independent portable build under a temporary directory, schema
validation, solver reproducibility, authorization counterfactuals, secret rendering, and
input/key separation.

## Build the documentation

```bash
uv sync --extra docs
uv run zensical build --clean --strict
```

For local live preview:

```bash
uv run zensical serve
```

## Install model dependencies

Model calibration is optional and isolated from the benchmark runtime:

```bash
uv sync --extra model
```

Use a local model directory when the machine has no internet access:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run pib calibrate /models/Qwen3.5-2B-Base \
  --report runs/qwen35_2b_base.json \
  --calibration-label qwen35_2b_base_diagnostic \
  --device cuda \
  --dtype bfloat16
```

`model_path` must contain the complete Transformers checkpoint, tokenizer, configuration,
and custom model files required by that checkpoint. The calibration code does not download
benchmark data.

## Next step

Read [Dataset format](dataset.md) before writing a loader. Read
[Evaluation protocol](evaluation.md) before opening validation or choosing a layer, feature,
steering strength, parser, or prompt template.
