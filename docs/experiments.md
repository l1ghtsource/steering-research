---
icon: lucide/flask-conical
---

# Experiments

Every experiment is driven by a YAML config under `configs/experiments/`.
The config defines the model, benchmark source, behavior axis, split policy,
activation view, layer set, limits, output directory, and steering schedule.

For full H200 runs, create dedicated configs under a separate directory such as:

```text
configs/experiments/h200/
```

The important fields to set deliberately are:

| Field | Purpose |
| --- | --- |
| `model` | Qwen/Qwen-Scope model config |
| `dataset` | LatentBehaviorBench data config |
| `behavior` or `behaviors` | behavior axis under study |
| `origin` | source-backed or synthetic contrast source |
| `layers` or `layer` | residual stream intervention/extraction layer |
| `activation_view` | token/span representation view |
| `output_dir` | run artifact root |

## E001 mean direction

Build CAA directions from extraction pairs and evaluate pairwise projection
separation on held-out contrast pairs.

Outputs:

- `metrics.jsonl`: one row per behavior/origin/layer/view;
- `summary.json`: best row metadata;
- `report.md`: compact ranked table.

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

Build a dense CAA direction, generate responses over the evaluation split at
multiple alpha values, and score generated text with transparent heuristic
probes.

The primary readout is the dose-response curve: the target behavior should move
monotonically with alpha while response length and unrelated control behaviors
remain stable.

## E005 SAE feature steering

Rank Qwen-Scope features, select the strongest feature, and steer with its
decoder vector. This checks that the Qwen-Scope path is not just an analysis
path; it is also wired into inference-time steering.

Interpret E005 together with E003. A convincing sparse feature should have a
large signed delta, appear in neighboring validation slices, and produce a
behavioral movement that is not explained only by verbosity or refusal style.

## E006 LoRA SFT

Build a supervised dataset from the good side of contrast pairs and train a
PEFT LoRA adapter. This is the training baseline against training-free steering.

Key knobs:

- `limit_pairs`;
- `lora_rank`;
- `lora_alpha`;
- `learning_rate`;
- `per_device_train_batch_size`;
- `gradient_accumulation_steps`;
- `num_train_epochs`;
- `max_steps`.

Interpret LoRA against E004/E005 on the same held-out buckets. The adapter must
improve the target behavior without degrading capability controls, safety
controls, or calibration on unrelated behavior axes.
