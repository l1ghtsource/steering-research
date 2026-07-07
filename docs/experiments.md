---
icon: lucide/flask-conical
---

# Experiments

Every experiment is driven by a YAML config under `configs/experiments/`.
The config defines the model, benchmark source, behavior axis, split policy,
activation view, layer set, limits, output directory, and steering schedule.

The six experiment families form one research ladder:

| ID | Question | Output |
| --- | --- | --- |
| E001 | Is the behavior linearly separable in residual activations? | dense directions and layer/view ranking |
| E002 | Can the direction act as a detector on individual examples? | AUROC monitor |
| E003 | Which Qwen-Scope sparse features track the behavior? | ranked SAE feature deltas |
| E004 | Does dense residual steering change generated behavior? | CAA alpha sweep |
| E005 | Does sparse feature steering change generated behavior? | SAE decoder-vector alpha sweep |
| E006 | Does supervised parameter training solve the same task? | LoRA SFT baseline |

## Campaign Structure

For full H200 runs, create a campaign-specific config directory:

```text
configs/experiments/h200/qwen35_9b/
```

Use a dedicated output root:

```yaml
output_dir: runs_h200_qwen35_9b
```

Each experiment should specify the same dataset config and model config so
results can be compared directly:

```yaml
dataset: configs/data/latent_behavior_bench.yaml
model: configs/models/qwen35_9b_h200_offline.yaml
```

Primary campaigns should be stratified by:

- behavior axis;
- source-backed versus synthetic origin;
- layer;
- activation view;
- steering alpha;
- model scale.

## Shared Config Fields

| Field | Meaning |
| --- | --- |
| `dataset` | LatentBehaviorBench config |
| `model` | Qwen/Qwen-Scope model config |
| `backend` | execution backend, usually `qwen` |
| `behavior` | one behavior axis for single-axis experiments |
| `behaviors` | multiple behavior axes for sweeps |
| `origin` | one contrast origin |
| `origins` | multiple contrast origins for sweeps |
| `layer` | one residual layer |
| `layers` | multiple residual layers |
| `activation_view` | one activation extraction view |
| `activation_views` | multiple activation extraction views |
| `eval_bucket` | held-out clean evaluation bucket |
| `alphas` | steering coefficients |
| `output_dir` | root for run artifacts |

## E001 Mean Direction

### Research Question

E001 asks whether a behavior axis has a stable mean-difference direction in
model activations:

```text
v = mean(h_positive - h_negative)
u = v / ||v||
```

If the direction is real, held-out positive examples should project above
held-out negative examples along `u`.

### Inputs

- behavior axes such as hallucination, sycophancy, premature refusal, deception,
  unsafe planning, and overconfidence;
- source-backed and synthetic origins, kept separate;
- layer sweep;
- activation view sweep.

### Procedure

1. Load contrast pairs from the selected origin.
2. Split pairs into discovery and held-out evaluation subsets according to the
   configured `train_fraction`.
3. Extract Qwen3.5 residual activations for both sides of each pair.
4. Build a CAA direction from discovery pairs.
5. Score held-out pairs by projection difference.
6. Log one metric row per behavior, origin, layer, and activation view.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `direction_accuracy` | fraction of held-out pairs with positive projection gap |
| `mean_projection_gap` | average signed separation |
| `median_projection_gap` | robust signed separation |
| `n_train_pairs` | discovery pair count |
| `n_eval_pairs` | held-out pair count |

### Artifacts

- `metrics.jsonl`: full sweep table;
- `summary.json`: best direction metadata;
- `report.md`: ranked human-readable summary.

### Interpretation

Strong E001 evidence means direction accuracy is clearly above 0.5, the signed
gap has the expected direction, and the result is not isolated to one accidental
layer or one templated origin. Weak E001 evidence means later steering is
unlikely to be behavior-specific without a better representation view or more
data.

## E002 Activation Monitor

### Research Question

E002 asks whether an E001-style direction can detect the behavior at the level
of individual examples, not just paired differences.

```text
score(x) = <h(x), u>
```

### Inputs

- one behavior axis;
- one origin;
- one layer;
- one activation view;
- discovery and evaluation limits.

### Procedure

1. Build a CAA direction from discovery contrast pairs.
2. Extract activations for individual positive and negative examples.
3. Score each example by projection onto the unit direction.
4. Compute threshold-free detector quality with AUROC.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `auroc` | probability that a positive example scores above a negative example |
| `mean_positive_score` | average projection for undesirable behavior |
| `mean_negative_score` | average projection for desirable behavior |
| `score_gap` | signed detector margin |
| `n_positive`, `n_negative` | evaluation sample sizes |

### Artifacts

- `metrics.jsonl`: one row per scored example or aggregate event;
- `summary.json`: AUROC and score gap;
- `report.md`: monitor summary.

### Interpretation

E002 is the bridge from representation discovery to monitoring. A high E001
score with a weak E002 AUROC means the direction may only work in pairwise
comparison and may not be suitable for standalone detection. A strong E002
result suggests the behavior is directly measurable during inference.

## E003 SAE Delta

### Research Question

E003 asks whether Qwen-Scope sparse autoencoders expose interpretable candidate
features for the same behavior signal.

For each feature `j`:

```text
delta_j = E[act_j | positive] - E[act_j | negative]
```

### Inputs

- behavior axis;
- origin;
- layer;
- activation view;
- top feature count.

### Procedure

1. Extract residual activations from Qwen3.5.
2. Encode each activation with the matching Qwen-Scope SAE for that layer.
3. Compute positive and negative mean activation for each SAE feature.
4. Rank features by signed or absolute delta.
5. Store the top features for inspection and steering.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `feature_index` | Qwen-Scope feature id |
| `delta` | positive-minus-negative activation gap |
| `positive_mean` | average feature activation on undesirable side |
| `negative_mean` | average feature activation on desirable side |
| `rank` | feature rank within the run |

### Artifacts

- `metrics.jsonl`: ranked feature rows;
- `top_sae_features.json`: machine-readable top feature table when used by E005;
- `summary.json`: row count and run metadata;
- `report.md`: top feature report.

### Interpretation

A useful SAE feature should have a stable signed delta and should remain
plausible under nearby layers, resampling, and source-backed evaluation. A
single high-delta feature is a candidate, not a conclusion. It becomes more
convincing if E005 shows that steering with its decoder vector changes the
target behavior.

## E004 CAA Steering

### Research Question

E004 asks whether a dense CAA direction can causally move model generations.
Detection is not enough: the intervention should change behavior under an
alpha sweep.

```text
h_layer = h_layer + alpha * u
```

With the benchmark convention `positive = undesirable`, negative alpha usually
attempts to suppress the target behavior.

### Inputs

- behavior axis;
- discovery origin;
- held-out `eval_bucket`;
- layer;
- activation view;
- alpha schedule;
- generation length.

### Procedure

1. Build a dense CAA direction from discovery pairs.
2. Generate from held-out prompts at each alpha.
3. Apply the residual stream hook at the configured layer.
4. Score generated text with transparent heuristic probes.
5. Aggregate behavior markers, refusal markers, and length statistics by alpha.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `alpha` | steering coefficient |
| `agreement_marker` | sycophancy-oriented marker |
| `refusal_marker` | refusal-oriented marker |
| `unsafe_marker` | unsafe planning marker when applicable |
| `uncertainty_marker` | calibration marker when applicable |
| `length_tokens` | verbosity control |

### Artifacts

- `metrics.jsonl`: generation-level rows;
- `tables/generations.csv`: text generations for manual review;
- `aggregate.json`: alpha-level summary;
- `summary.json`: run metadata;
- `report.md`: dose-response table.

### Interpretation

The strongest result is monotonic: behavior markers move in the expected
direction as alpha changes, while length and unrelated markers remain stable.
If only length changes, the intervention is not behavior-specific. If refusal
spikes, the direction may be inducing generic refusal rather than correcting the
target behavior.

## E005 SAE Feature Steering

### Research Question

E005 asks whether a sparse feature found by E003 can be used as an intervention
vector.

```text
h_layer = h_layer + alpha * normalize(W_dec[:, feature_j])
```

This tests whether the sparse feature is merely diagnostic or has causal
leverage over generation.

### Inputs

- behavior axis;
- origin;
- held-out `eval_bucket`;
- layer;
- activation view;
- top feature count;
- alpha schedule.

### Procedure

1. Rank Qwen-Scope feature deltas using E003 logic.
2. Select the strongest candidate feature.
3. Retrieve the decoder vector for that feature.
4. Generate held-out responses under an alpha sweep.
5. Log generations and selected feature metadata.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `selected_feature` | SAE feature used for steering |
| `selected_feature_delta` | discovery strength of the feature |
| `alpha` | steering coefficient |
| generation markers | behavior movement under intervention |
| `fallback_caa_norm` | dense direction magnitude for comparison |

### Artifacts

- `top_sae_features.json`: ranked sparse candidates;
- `metrics.jsonl`: generation rows;
- `summary.json`: selected feature and run metadata;
- `report.md`: feature steering report.

### Interpretation

E005 is most compelling when its selected feature is also high-ranking in E003,
the alpha sweep moves behavior in the expected direction, and the effect is
cleaner than dense CAA steering. If the selected feature changes only style,
verbosity, or refusal rate, it should be treated as an artifact rather than a
behavior-specific feature.

## E006 LoRA SFT

### Research Question

E006 asks how inference-time steering compares with supervised parameter
training. It builds training rows from the desirable side of contrast pairs and
fits a PEFT LoRA adapter.

### Inputs

- behavior axis;
- origin;
- pair limit;
- base model config;
- LoRA rank and alpha;
- learning rate, batch size, accumulation, epochs, and max steps.

### Procedure

1. Load contrast pairs for the selected behavior and origin.
2. Convert the negative side into supervised prompt-answer rows.
3. Tokenize rows with the target Qwen tokenizer.
4. Attach LoRA modules to attention and MLP projection layers.
5. Train with causal language modeling.
6. Save the adapter and tokenizer.
7. Log training metrics and write run artifacts.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `rows` | supervised row count |
| `train_loss` | optimization signal |
| `train_runtime` | training cost |
| `epoch` | training progress |
| `max_steps` | explicit training cap when configured |

### Artifacts

- LoRA adapter directory under `output_dir`;
- `metrics.jsonl`: dataset and training events;
- `summary.json`: training metrics and adapter path;
- `report.md`: training summary.

### Interpretation

E006 is not evaluated by training loss alone. A useful adapter must be evaluated
on the same held-out buckets as E004 and E005. It should improve the target
behavior while preserving capability controls, safety controls, calibration, and
unrelated behavior axes. If it beats steering only by becoming more generally
refusal-heavy, the result is not a clean behavioral improvement.

## Cross-Experiment Reading

The experiments should be interpreted together:

| Pattern | Meaning |
| --- | --- |
| E001 strong, E002 strong, E004 strong | dense behavior direction is detectable and steerable |
| E003 strong, E005 strong | Qwen-Scope sparse feature has causal leverage |
| E001 strong, E004 weak | direction is diagnostic but not causal at the chosen layer or alpha |
| E003 strong, E005 weak | sparse feature is correlational or wrong intervention sign |
| E006 strong, E004/E005 weak | training changes behavior more reliably than steering |
| E004/E005 strong, E006 weak | training-free intervention may be more efficient for that behavior |
| Source-backed weak, synthetic strong | likely templating or data-origin artifact |

## Campaign Report Checklist

Every campaign summary should state:

- model scale and exact model config;
- matching Qwen-Scope SAE config;
- behavior axes;
- origins;
- layers and activation views;
- held-out evaluation buckets;
- alpha schedule;
- run artifact root;
- best E001/E002 layers;
- top E003 features;
- E004/E005 dose-response interpretation;
- E006 adapter path and held-out comparison;
- known caveats and failed controls.
