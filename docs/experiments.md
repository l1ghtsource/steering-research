---
icon: lucide/flask-conical
---

# Experiments

Every experiment is driven by a YAML config under `configs/experiments/`.
The config defines the model, benchmark source, behavior axis, split policy,
activation view, layer set, limits, output directory, and steering schedule.

The experiment families form one research ladder. E001-E006 establish the
baseline discovery, steering, and training comparison path. E007-E015 extend
that path with controls, transfer tests, sparse feature sweeps, dynamic
interventions, and layer-localization ablations.

| ID | Question | Output |
| --- | --- | --- |
| E001 | Is the behavior linearly separable in residual activations? | dense directions and layer/view ranking |
| E002 | Can the direction act as a detector on individual examples? | AUROC monitor |
| E003 | Which Qwen-Scope sparse features track the behavior? | ranked SAE feature deltas |
| E004 | Does dense residual steering change generated behavior? | CAA alpha sweep |
| E005 | Does sparse feature steering change generated behavior? | SAE decoder-vector alpha sweep |
| E006 | Does supervised parameter training solve the same task? | LoRA SFT baseline |
| E007 | Do best E001/E002 layers remain causal under generation? | best-layer CAA sweep |
| E008 | Are discovered directions behavior-specific or shared across axes? | specificity matrix |
| E009 | Does the causal effect survive sign, random, shuffled, and unrelated controls? | causal control panel |
| E010 | Which top Qwen-Scope features have causal leverage? | SAE feature intervention sweep |
| E011 | Does removing known nuisance directions improve steering specificity? | orthogonalized CAA steering |
| E012 | Do directions transfer between source-backed and synthetic origins? | origin transfer matrix |
| E013 | Can steering be gated by an activation monitor instead of applied always? | dynamic steering comparison |
| E014 | Is steering stronger when applied over a layer window? | multi-layer steering sweep |
| E015 | How stable is the direction across source and target layers? | layer-transfer matrix |
| E016 | Does steering change preference between paired desirable and undesirable answers? | forced-choice logprob margin |
| E017 | Can alpha be calibrated across behaviors and layers? | calibrated coefficient sweep |
| E018 | Which token positions make steering causal? | position-mode forced-choice ablation |

## Campaign Structure

For full H200 runs, create a campaign-specific config directory. The repository
ships workstation-ready 2B campaign configs under:

```text
configs/experiments/full/qwen35_2b/
configs/experiments/phase2/qwen35_2b/
```

For 9B or 27B H200 runs, mirror the same layout:

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
| `entries` | named best-layer steering entries for E007 |
| `sources` | source behavior directions for specificity tests |
| `target_behaviors` | target axes scored in specificity tests |
| `target_origins` | target origins scored in specificity tests |
| `control_behaviors` | nuisance axes removed in orthogonalized steering |
| `train_origins` | origins used to build directions in transfer tests |
| `eval_origins` | origins used to evaluate transfer tests |
| `threshold_quantile` | activation-monitor gate used by dynamic steering |
| `groups` | named layer groups for multi-layer steering |
| `source_layers` | layers used to build transfer directions |
| `target_layers` | layers used to evaluate transfer directions |
| `alpha_coefficients` | calibrated alpha multipliers for E017 |
| `scale_method` | calibration scale definition for E017 |
| `scale_fraction` | residual-norm fraction used by E017 when applicable |
| `position_modes` | token-position hook modes for E018 |
| `logprob_batch_size` | batch size for forced-choice conditional logprob scoring |
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

## E007 Best-Layer CAA Sweep

### Research Question

E007 asks whether the strongest representation-level layer choices from E001
and E002 remain useful when used as actual generation-time interventions.

The experiment is a targeted CAA steering pass over a small set of named
entries:

```yaml
entries:
  - name: sycophancy_best
    behavior: sycophancy
    origin: source_backed_contrasts
    layer: 18
    activation_view: assistant_answer_mean
```

Each entry builds its own direction, evaluates the same held-out prompt bucket,
and writes alpha-level aggregates.

### Inputs

- named behavior-origin-layer-view entries;
- held-out evaluation bucket;
- dense CAA train limit and evaluation limit;
- alpha schedule;
- generation length and batch size.

### Procedure

1. For each entry, load the configured contrast pairs.
2. Build a dense mean-difference direction at the selected layer and activation
   view.
3. Generate held-out prompts at every alpha.
4. Score generations with transparent behavior markers and style controls.
5. Aggregate metrics by entry and alpha.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `entry` | named best-layer candidate |
| `alpha` | steering coefficient |
| `mean_agreement_marker` | sycophancy-style agreement marker |
| `mean_refusal_marker` | refusal-style marker |
| `mean_unsafe_planning_marker` | unsafe planning marker |
| `mean_uncertainty_marker` | uncertainty/calibration marker |
| `mean_length_tokens` | verbosity shift |
| `mean_repetition_proxy` | degeneration or repetition proxy |

### Artifacts

- `metrics.jsonl`: generation-level rows;
- `aggregate.json`: entry-by-alpha table;
- `tables/generations.csv`: generated text for review;
- `summary.json` and `report.md`: run metadata and compact report.

### Interpretation

E007 is the first check that a representation-level "best layer" is also
causal. A strong result has a clear dose-response on the target marker while
length, repetition, and unrelated markers remain stable. If the target marker is
flat but repetition or length moves, the direction is diagnostic or stylistic
rather than a clean behavior controller.

## E008 Specificity Matrix

### Research Question

E008 asks whether a direction discovered for one behavior is specific to that
behavior or also separates other behavior axes. This matters because a broad
"bad answer" or "instruction conflict" direction can look strong while being
too generic for mechanistic claims.

### Inputs

- source behavior directions with explicit origin, layer, and activation view;
- target behavior list;
- target origin list;
- train and evaluation limits.

### Procedure

1. Build one dense direction for each configured source.
2. For every target behavior and target origin, load target contrast pairs.
3. Score target positive and negative activations along the source direction.
4. Log one matrix row per source-target-origin combination.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `source_behavior`, `source_origin` | direction being tested |
| `target_behavior`, `target_origin` | contrast bucket being scored |
| `direction_accuracy` | fraction of target pairs separated by the source direction |
| `mean_projection_gap` | signed target separation |
| `mean_abs_margin` | unsigned separation magnitude |
| `n_eval_pairs` | target evaluation pair count |

### Artifacts

- `metrics.jsonl`: full specificity matrix;
- `summary.json`: row count and backend metadata;
- `report.md`: top matrix rows for manual reading.

### Interpretation

The diagonal should be strong. Off-diagonal strength is not automatically bad:
some behaviors may share a latent safety/helpfulness axis. It becomes a problem
when a direction used for a causal claim also strongly activates unrelated
targets and the generation-level intervention changes generic refusal, length,
or style more than the target behavior.

## E009 Causal Controls

### Research Question

E009 asks whether the generation-level effect of a CAA direction survives
basic causal controls. It compares the intended direction against:

- the opposite sign;
- a norm-matched random vector;
- a shuffled-label direction;
- an unrelated behavior direction;
- a same-behavior synthetic-origin direction.

### Inputs

- primary behavior and origin;
- synthetic origin for same-behavior comparison;
- unrelated behavior and origin;
- held-out evaluation bucket;
- layer, activation view, alpha schedule, and generation settings.

### Procedure

1. Build the primary dense direction.
2. Build control directions with matched layer and norm where possible.
3. Generate the same held-out prompts under every variant and alpha.
4. Score all outputs with the same marker stack.
5. Aggregate by variant and alpha.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `variant` | `base`, `opposite_sign`, `random_norm_matched`, `shuffled_labels`, `unrelated_*`, or synthetic control |
| `alpha` | steering coefficient |
| generation markers | target and nuisance response changes |
| `mean_length_tokens` | verbosity control |
| `mean_repetition_proxy` | degeneration control |

### Artifacts

- `metrics.jsonl`: generation-level rows for all variants;
- `aggregate.json`: variant-by-alpha summary;
- `tables/generations.csv`: generated text and variant labels;
- `summary.json` and `report.md`: campaign-readable output.

### Interpretation

A credible causal effect should be stronger and more directionally coherent for
the intended `base` variant than for random, shuffled, or unrelated controls.
If controls move the target marker by a similar amount, the effect should be
reported as weak or confounded.

## E010 SAE Feature Sweep

### Research Question

E010 asks whether several top Qwen-Scope features have causal leverage, instead
of selecting a single best feature once and overfitting the interpretation.

### Inputs

- behavior and origin;
- layer and activation view;
- `top_features`;
- held-out evaluation bucket;
- alpha schedule and generation settings.

### Procedure

1. Encode positive and negative contrast activations with the matching
   Qwen-Scope SAE.
2. Rank features by signed behavior delta.
3. Retrieve each selected feature decoder vector.
4. Generate held-out prompts under each feature and alpha.
5. Aggregate by feature rank, feature index, and alpha.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `feature_rank` | rank by absolute or signed delta |
| `feature_index` | Qwen-Scope feature id |
| `alpha` | decoder-vector steering coefficient |
| generation markers | causal response to sparse feature intervention |
| `mean_repetition_proxy` | sparse steering degeneration check |

### Artifacts

- `metrics.jsonl`: generation rows;
- `aggregate.json`: feature-by-alpha table;
- `top_sae_features.json` when produced by upstream feature ranking;
- `tables/generations.csv`;
- `summary.json` and `report.md`.

### Interpretation

E010 is stronger than a single-feature result because it exposes whether the
effect is concentrated, redundant, or unstable across nearby top-ranked
features. A candidate feature is useful only if its intervention changes the
target marker more than style, length, refusal, or repetition.

## E011 Orthogonalized Steering

### Research Question

E011 asks whether a target behavior direction becomes cleaner after removing
components aligned with nuisance or control behavior directions.

For target direction `u` and control directions `c_i`, the intervention uses:

```text
u_clean = normalize(u - projection_span(C)(u))
```

### Inputs

- target behavior and origin;
- control behavior list;
- common layer and activation view;
- held-out evaluation bucket;
- alpha schedule and generation settings.

### Procedure

1. Build the raw target CAA direction.
2. Build CAA directions for every control behavior.
3. Orthogonalize the target direction against the control span.
4. Generate held-out prompts with raw and orthogonalized variants.
5. Compare target markers and nuisance/style controls.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `variant` | `raw` or `orthogonalized` |
| `alpha` | steering coefficient |
| generation markers | target and nuisance changes |
| `n_controls_used` | number of control axes removed |

### Artifacts

- `metrics.jsonl`;
- `aggregate.json`: raw versus orthogonalized response table;
- `tables/generations.csv`;
- `summary.json` and `report.md`.

### Interpretation

Orthogonalization is useful when it preserves the target movement while reducing
refusal, unsafe, uncertainty, length, or repetition side effects. If raw and
orthogonalized curves are identical, the selected controls did not explain the
failure mode. If the target effect disappears, the target direction was mostly
shared with the control axes.

## E012 Origin Transfer

### Research Question

E012 asks whether directions learned from one data origin transfer to another:
source-backed to synthetic, synthetic to source-backed, and within-origin.

This is a representation-level transfer test, not a generation intervention.

### Inputs

- behavior list;
- train origin list;
- evaluation origin list;
- layer and activation view;
- train and evaluation limits.

### Procedure

1. For each behavior and train origin, build a dense direction.
2. For each evaluation origin, score held-out contrast pairs along that
   direction.
3. Log one row per behavior, train origin, and evaluation origin.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `behavior` | target behavior axis |
| `train_origin` | origin used to build the direction |
| `eval_origin` | origin used to score held-out pairs |
| `direction_accuracy` | transfer separation quality |
| `mean_projection_gap` | signed transfer margin |
| `n_eval_pairs` | evaluation pair count |

### Artifacts

- `metrics.jsonl`: origin transfer matrix;
- `summary.json`;
- `report.md`.

### Interpretation

Source-backed to source-backed is the primary grounded evidence. Synthetic to
synthetic can be high because of templating or construction artifacts. The most
important stress test is cross-origin transfer. A behavior is more credible when
the sign and margin remain stable across origins.

## E013 Dynamic Steering

### Research Question

E013 asks whether steering should be applied only when an activation monitor
predicts risk, instead of applying the intervention to every prompt.

The monitor threshold is configured by a quantile over discovery scores:

```yaml
threshold_quantile: 0.75
```

### Inputs

- target behavior and origin;
- layer and activation view;
- held-out evaluation bucket;
- threshold quantile;
- alpha schedule and generation settings.

### Procedure

1. Build a dense target direction.
2. Estimate a monitor threshold from discovery-set projections.
3. Score each held-out prompt before generation.
4. Apply steering only when the monitor crosses the threshold.
5. Compare dynamic steering against always-on steering.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `variant` | `always` or `dynamic` |
| `alpha` | steering coefficient |
| `monitor_score` | pre-generation activation score when logged |
| `applied_steering` | whether the gate fired |
| generation markers | behavioral and side-effect response |

### Artifacts

- `metrics.jsonl`;
- `aggregate.json`: always-on versus dynamic response;
- `tables/generations.csv`;
- `summary.json` with the selected threshold;
- `report.md`.

### Interpretation

Dynamic steering is useful only if it applies to a meaningful subset of prompts
and preserves or improves the target effect while reducing unnecessary side
effects. A zero application rate means the threshold or monitor view is wrong
for the evaluation bucket. A near-one application rate means the dynamic policy
has collapsed to always-on steering.

## E014 Multi-Layer Steering

### Research Question

E014 asks whether steering should be applied at one layer or distributed across
a small window of neighboring layers.

Each group defines a named intervention plan:

```yaml
groups:
  - name: window_17_19
    layers: [17, 18, 19]
    divide_alpha: true
```

When `divide_alpha` is true, the configured alpha is split across hooks so the
total intervention scale remains comparable.

### Inputs

- behavior and origin;
- candidate layer list;
- named layer groups;
- activation view;
- held-out bucket;
- alpha schedule and generation settings.

### Procedure

1. Build one dense direction per configured layer.
2. For each group, install steering hooks at all group layers.
3. Generate held-out prompts for every group and alpha.
4. Aggregate behavior and side-effect metrics by group.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `variant` | group name such as `single_18` or `window_17_19` |
| `n_hooks` | number of installed steering hooks |
| `alpha` | total or per-hook steering scale according to `divide_alpha` |
| generation markers | target and side-effect response |

### Artifacts

- `metrics.jsonl`;
- `aggregate.json`: group-by-alpha table;
- `tables/generations.csv`;
- `summary.json` with group definitions;
- `report.md`.

### Interpretation

Multi-layer steering is useful when it increases target movement without
increasing repetition, refusal, or generic style shifts. If a layer window does
not beat the best single layer, the simpler single-layer intervention should
remain the primary protocol.

## E015 Layer Transfer

### Research Question

E015 asks whether a direction discovered at one layer is still aligned with
contrast separation at another layer. It is a representation-level localization
test that helps decide whether a behavior is layer-specific or broadly present
through the residual stream.

### Inputs

- behavior and origin;
- activation view;
- source layer list;
- target layer list;
- train and evaluation limits.

### Procedure

1. Build a direction at each source layer.
2. Extract target-layer activations for held-out contrast pairs.
3. Score each target layer along each source-layer direction.
4. Log a source-layer by target-layer matrix.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `source_layer` | layer used to build the direction |
| `target_layer` | layer used for held-out scoring |
| `direction_accuracy` | cross-layer separation quality |
| `mean_projection_gap` | signed cross-layer margin |
| `n_eval_pairs` | held-out contrast count |

### Artifacts

- `metrics.jsonl`: layer transfer matrix;
- `summary.json`: best source-target layer pair;
- `report.md`.

### Interpretation

A strong diagonal means the behavior is visible at individual layers. Strong
off-diagonal transfer means the direction is stable across the residual stream.
Large late-layer gaps may be useful for monitoring, while earlier transferable
directions may be preferable for steering if they avoid output-style artifacts.

## E016 Forced-Choice CAA

### Research Question

E016 asks whether steering changes the model's preference between the benchmark
paired answers without relying on free-generation sampling or keyword markers.

For each contrast pair, the model scores:

```text
margin = log P(desirable_answer | prompt, steering)
       - log P(undesirable_answer | prompt, steering)
```

The positive side of LatentBehaviorBench is treated as the undesirable answer
and the negative side as the desirable answer.

### Inputs

- behavior/origin/layer entries;
- train/eval split over contrast pairs;
- activation view;
- raw alpha schedule;
- forced-choice logprob batch size.

### Procedure

1. Build a CAA direction from the discovery split.
2. For each held-out contrast pair, reconstruct the shared prompt and the two
   assistant completions.
3. Score both completions under every alpha.
4. Aggregate preference accuracy and preference margin by entry and alpha.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `mean_preference_accuracy` | fraction where desirable answer has higher mean logprob |
| `mean_preference_margin` | normalized desirable-minus-undesirable logprob gap |
| `delta_margin_vs_alpha0` | causal shift relative to no steering |
| `mean_same_prompt` | whether paired answers share the same prompt |

### Interpretation

E016 is the first choice when free-generation markers are noisy. A strong
result means the steering direction moves the paired-answer likelihood margin
toward the desirable answer. If E016 is weak, free-generation changes should be
treated as style, length, or sampling artifacts until proven otherwise.

## E017 Calibrated Alpha

### Research Question

E017 asks whether raw alpha schedules are comparable across layers and
behaviors. A raw alpha of `2.0` may be tiny for one layer and excessive for
another, so this experiment scores forced-choice margins under calibrated
coefficients.

### Inputs

- behavior/origin/layer entries;
- `scale_method`, such as `mean_residual_norm`;
- `scale_fraction` for residual-norm calibration;
- `alpha_coefficients`;
- forced-choice scoring settings.

### Procedure

1. Build the target CAA direction.
2. Estimate a calibration scale on the discovery split.
3. Convert each coefficient into a raw alpha.
4. Run the same forced-choice scoring protocol as E016.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `alpha_value` | calibrated coefficient |
| `calibration_scale` | raw-alpha scale for the entry |
| `delta_margin_vs_alpha0` | forced-choice causal shift |
| `mean_preference_accuracy` | desirable-answer preference rate |

### Interpretation

E017 separates real steering effects from arbitrary raw-alpha scale choices.
If calibrated coefficients preserve the best E016 effects, the intervention is
more robust. If the effect disappears after calibration, the original result may
depend on layer-specific norm artifacts.

## E018 Position Steering

### Research Question

E018 asks which token positions need the steering hook. The supported modes are:

- `all`: every sequence position;
- `prompt`: prompt tokens only;
- `last_prompt`: the final prompt token;
- `answer`: completion tokens only;
- `first_answer`: the first completion token.

### Inputs

- one or more behavior/origin/layer entries;
- raw alpha schedule;
- position mode list;
- forced-choice scoring settings.

### Procedure

1. Build the target CAA direction.
2. Score held-out paired answers under every alpha and position mode.
3. Compare preference-margin deltas across modes.

### Metrics

| Metric | Interpretation |
| --- | --- |
| `position_mode` | hook placement policy |
| `alpha` | raw steering coefficient |
| `delta_margin_vs_alpha0` | causal shift for the position mode |
| `mean_preference_accuracy` | desirable-answer preference rate |

### Interpretation

E018 tells whether the effect comes from prompt-state steering, answer-state
steering, or boundary-token perturbation. If only `all` works, the intervention
may be diffuse. If `last_prompt` or `first_answer` works, the effect is more
localized and easier to test on larger models.

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
| E007 strong, E009 controls weak | best-layer steering has credible causal evidence |
| E008 diagonal strong, off-diagonal weak | behavior direction is specific |
| E008 off-diagonal strong | direction may reflect a shared latent axis or confound |
| E010 feature strong, E003 delta strong | sparse feature is both diagnostic and causal |
| E011 improves side effects | nuisance axes explain part of the raw direction |
| E012 cross-origin weak | origin-specific artifacts or non-transferable behavior signature |
| E013 dynamic better than always-on | monitor-gated steering reduces unnecessary intervention |
| E014 window weak | single-layer steering is sufficient |
| E015 late-layer transfer strong | behavior signal is stable in later residual stream layers |
| E016 strong, E004/E007 weak | behavior changes likelihood before it changes free generations |
| E017 strong where E016 strong | effect is not raw-alpha scale artifact |
| E018 localized mode strong | steering can be narrowed to specific token positions |
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
- E007 best-layer causal confirmation;
- E008 specificity matrix and strongest off-diagonal confounds;
- E009 control comparison;
- E010 sparse feature intervention winners and failures;
- E011 raw versus orthogonalized comparison;
- E012 source-backed/synthetic transfer matrix;
- E013 dynamic gate firing rate and effect;
- E014 single-layer versus multi-layer comparison;
- E015 source-target layer transfer matrix;
- E016 forced-choice preference-margin shift;
- E017 calibration scale and coefficient response;
- E018 position-mode localization;
- known caveats and failed controls.
