---
icon: lucide/flask-conical
---

# Experiments

Every experiment has a YAML config under `configs/experiments/`. Smoke configs
for local Qwen3.5-2B runs live under `configs/experiments/smoke/`.

## E001 mean direction

Build CAA directions from extraction pairs and evaluate pairwise projection
separation.

Outputs:

- `metrics.jsonl`: one row per behavior/origin/layer/view;
- `summary.json`: best row metadata;
- `report.md`: table of the first metric rows.

## E002 activation monitor

Use the CAA direction as a white-box detector over individual positive and
negative examples.

Metrics:

- AUROC;
- mean positive projection;
- mean negative projection;
- score gap;
- number of positive and negative examples.

## E003 SAE delta

Run Qwen3.5 activations through Qwen-Scope SAE and rank features by
positive-minus-negative activation deltas.

Outputs:

- feature index;
- delta;
- positive mean;
- negative mean.

## E004 CAA steering

Build a dense CAA direction, generate responses over an eval subset at multiple
alpha values, and score generated text with transparent heuristic probes.

The smoke run is intentionally short and uses small `max_new_tokens` so it can
run on the local Mac.

## E005 SAE feature steering

Rank Qwen-Scope features, select the strongest feature, and steer with its
decoder vector. This checks that the Qwen-Scope path is not just an analysis
path; it is also wired into inference-time steering.

## E006 LoRA SFT

Build a supervised dataset from the good side of contrast pairs and train a
PEFT LoRA adapter. The smoke config uses a tiny number of steps. H200 configs can
scale `limit_pairs`, batch size, gradient accumulation, and epochs.

!!! warning "Training smoke is not a research result"

    The local E006 smoke only verifies that the training codepath, dataset
    construction, adapter saving, and report writing work.

