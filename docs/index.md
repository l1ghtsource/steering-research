---
icon: lucide/radar
---

# Steering Research

This repository is a reproducible experiment harness for studying latent
behavioral features in Qwen3.5 models with LatentBehaviorBench and Qwen-Scope
sparse autoencoders.

!!! warning "Research status"

    The benchmark is usable for activation extraction and steering smoke tests.
    It is not a basis for strong safety, OOD, or capability-preservation claims
    without larger runs and manual review of source-specific caveats.

## What the repository does

The codebase implements the complete path from benchmark records to activation
directions, sparse feature deltas, steering interventions, run artifacts, and
static dashboards:

```mermaid
flowchart LR
  A["LatentBehaviorBench submodule"] --> B["Clean split loader"]
  B --> C["Qwen3.5 forward pass"]
  C --> D["CAA / probes / SAE delta"]
  D --> E["Residual or SAE decoder steering"]
  E --> F["Metrics, reports, logs"]
  F --> G["Static HTML dashboard"]
```

## Current local target

The default local model target is:

- Base model: `Qwen/Qwen3.5-2B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50`
- Hook point: residual stream
- SAE layers: `0..23`
- SAE format: `layer{n}.sae.pt` with `W_enc`, `W_dec`, `b_enc`, `b_dec`

## Commands that must stay green

```bash
uv sync --extra dev --extra model --extra training --extra docs
uv run steering validate-data
uv run steering smoke --backend fake
uv run steering smoke-real --examples 10
uv run steering verify-runs --runs-root runs
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
uv run zensical build --clean --strict
```

## Experiment set

| ID | Name | Mode | Purpose |
| --- | --- | --- | --- |
| E001 | Mean direction | training-free | CAA behavior direction maps |
| E002 | Activation monitor | training-free | AUROC detector over projections |
| E003 | SAE delta | training-free | Qwen-Scope feature ranking |
| E004 | CAA steering | training-free | Residual steering dose response |
| E005 | SAE feature steering | training-free | Decoder-vector steering |
| E006 | LoRA SFT | training | Good-side contrast supervised training |

