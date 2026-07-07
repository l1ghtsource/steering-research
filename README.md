# steering-research

Activation steering research harness for LatentBehaviorBench, Qwen3.5, and
Qwen-Scope sparse autoencoders.

## Setup

```bash
uv sync --extra dev --extra model --extra training --extra docs
uv run steering validate-data
uv run ruff check
uv run ty check
uv run pytest
uv run zensical build --clean --strict
```

The benchmark is mounted at `external/LatentBehaviorBench` as a git submodule.

## Experiments

```bash
uv run steering e001 --config configs/experiments/e001_mean_direction.yaml --backend qwen
uv run steering e002 --config configs/experiments/e002_activation_monitor.yaml --backend qwen
uv run steering e003 --config configs/experiments/e003_sae_delta.yaml --backend qwen
uv run steering e004 --config configs/experiments/e004_steering_eval.yaml --backend qwen
uv run steering e005 --config configs/experiments/e005_sae_feature_steering.yaml --backend qwen
uv run steering e006-lora-sft --config configs/experiments/e006_lora_sft.yaml
```

Each run writes `manifest.json`, `metrics.jsonl`, `summary.json`, `report.md`,
`run.log`, and dashboard inputs under the configured output directory.

## Artifacts

```bash
uv run steering verify-runs --runs-root runs
uv run steering dashboard --runs-root runs
```

## Docs

Docs are built with Zensical and deployed to GitHub Pages by CI:

```bash
uv run zensical build --clean --strict
```
