---
icon: lucide/clipboard-check
---

# Findings and Caveats

This page records the current Qwen3.5-2B evidence from the completed campaigns:

- `runs_full_qwen35_2b_final`
- `runs_phase2_qwen35_2b`
- `runs_phase3_qwen35_2b_final`

It is intentionally conservative. A result is treated as strong only when it
survives the appropriate control for that claim. Places where the benchmark,
scoring protocol, or activation view makes the evidence unfair are called out
explicitly.

## Executive Summary

The main result is split:

- Behavior information is easy to recover from `assistant_answer_mean`
  activations.
- The same directions are much less reliable as free-generation steering
  interventions.
- Forced-choice logprob evaluation is much more informative than keyword-marker
  scoring for causal steering.
- The clearest causal signal on Qwen3.5-2B is `premature_refusal`, followed by
  source-backed `hallucination` in forced-choice margins.
- Source-backed `sycophancy` remains weak and has a serious forced-choice data
  issue in the current benchmark slice.

The most important caveat is that `assistant_answer_mean` uses assistant answer
tokens. It is valid for analyzing answer representations and for building
post-answer diagnostic directions. It is not valid evidence that the behavior is
detectable before generation. Prompt-only views were weak in the completed E001
run, so pre-generation claims are not supported yet.

## Validity Labels

| Label | Meaning |
| --- | --- |
| Strong | Result is held-out and aligned with the right evaluation protocol |
| Moderate | Result is useful but has a known caveat |
| Weak | Effect exists but is small, noisy, or confounded |
| Invalid | Current protocol has leakage, benchmark mismatch, or unfair scoring |
| Not run | No completed 2B run exists |

## Experiment Claim Ledger

This table is the top-level truth source for the completed experiments. A
result marked as leaky or invalid can still be useful as debugging evidence, but
it must not be promoted into a behavioral research claim.

| Experiment | Honesty status | Real conclusion | Do not claim | Required next check |
| --- | --- | --- | --- | --- |
| E001 Mean Direction | Mixed: honest for answer representations, leaky for pre-generation claims | `assistant_answer_mean` strongly separates many paired answers | The prompt alone exposes the behavior before generation | Rerun with prompt-only or prefill-only views |
| E002 Activation Monitor | Mixed: held-out, but answer-view dependent | Sycophancy is detectable in answer-side activations with AUROC `0.878` | This is a deployable online monitor | Measure AUROC on prompt-only and prefill-only activations |
| E003 SAE Delta | Honest diagnostic, not causal | SAE deltas produce candidate sparse features | Top delta features are causal steering features | Intervene on features with forced-choice or judge scoring |
| E004 CAA Free Generation | Honest weak/null | Dense sycophancy CAA barely moves the keyword marker | Free-generation sycophancy steering works | Replace marker-only scoring with forced-choice or judge scoring |
| E005 SAE Feature Steering | Honest weak | Feature `16221` has a small marker effect | Qwen-Scope feature steering is solved | Test selected features on paired forced-choice margins |
| E006 LoRA SFT | Not run | No completed 2B evidence | Any training versus steering comparison | Run the training baseline before comparing methods |
| E007 Best-Layer CAA Sweep | Honest weak and confounded | Layer search finds small marker shifts with repetition/style movement | Clean free-generation behavior control | Add evaluator-based generation scoring and degeneration controls |
| E008 Specificity Matrix | Honest diagnostic, answer-view dependent | Several directions are broad and off-diagonal, not behavior-specific | Diagonal wins prove isolated mechanisms | Repeat with stricter origin controls and prompt-only views |
| E009 Causal Controls | Honest weak/null | Sycophancy controls are comparable to the intended direction | Dense sycophancy direction passes causal specificity | Use stronger paired scoring before revisiting generation |
| E010 SAE Feature Sweep | Honest weak candidate discovery | Feature `29345` is the best current marker-space sparse candidate | Marker improvement is a behavioral success | Retest feature `29345` and top-k features with forced-choice scoring |
| E011 Orthogonalized Steering | Honest null | Orthogonalization did not improve sycophancy marker steering | Nuisance axes are irrelevant in general | Build nuisance bases from better validated behaviors |
| E012 Origin Transfer | Strong for answer-side transfer diagnostics | Source-backed and synthetic origins must be reported separately | Synthetic-only success transfers to source-backed data | Keep origin-specific reporting and rerun prompt-only variants |
| E013 Dynamic Steering | Invalid policy run | The gate never fired, so the threshold/view is wrong | Dynamic steering failed | Recalibrate the monitor and verify nonzero intervention rate |
| E014 Multi-Layer Steering | Honest null | Multi-layer windows did not beat single-layer sycophancy steering | Wider layer windows are better | Only revisit after a stronger scoring protocol exists |
| E015 Layer Transfer | Strong for answer-side layer stability | Sycophancy answer directions are stable across middle and late layers | Sycophancy is detectable before generation | Repeat layer transfer on prompt-only or prefill-only activations |
| E016 Forced-Choice CAA | Strongest causal diagnostic where pair integrity is valid | Premature refusal and source-backed hallucination move in likelihood space | Source-backed sycophancy is solved | Filter to same-prompt, nonempty pairs before sycophancy claims |
| E017 Calibrated Alpha | Honest calibration diagnostic | Residual-norm scaling works for hallucination and unsafe margins, but not universally | A single calibration rule is enough for all behaviors | Learn or choose behavior-specific calibration rules |
| E018 Position Steering | Invalid/weak for sycophancy source | No reliable sycophancy position localization is shown | Sycophancy position steering failed conclusively | Rerun after fixing E016 sycophancy pair integrity |

## Real Conclusions vs Non-Claims

Real conclusions from the completed Qwen3.5-2B campaigns:

- Answer-side activations contain strong behavior-separating information for
  several LatentBehaviorBench axes.
- Forced-choice likelihood scoring is currently the most trustworthy causal
  evaluation protocol in this repo.
- Premature refusal is the clearest causal steering result.
- Source-backed hallucination shows a real forced-choice margin improvement,
  but synthetic transfer is fragile.
- Synthetic and source-backed origins are not interchangeable.
- Free-generation keyword markers are useful for smoke auditing, but they are
  too weak for final behavioral claims.

Results that should be treated as non-claims until rerun:

- Any pre-generation detection claim based only on `assistant_answer_mean`.
- Any source-backed sycophancy forced-choice claim from the current E016/E018
  rows.
- Any synthetic-only result presented as source-backed behavioral evidence.
- Any SAE top-feature claim that has not survived direct intervention.
- Any training-free versus training comparison, because E006 has not been run.

## Benchmark and Protocol Issues

### Answer-Token Leakage in Representation Runs

Most representation experiments use `assistant_answer_mean`. That view averages
hidden states over the assistant answer. The positive and negative answers often
contain different facts, refusal style, answer length, or answer format.

This means:

- E001, E002, E003, E008, E012, and E015 are strong evidence that the answer
  representations contain behavior-relevant information.
- They are not clean evidence that the prompt alone contains a latent behavior
  feature before generation.
- Any claim about online monitoring before generation must be rerun with
  prompt-only views such as `last_prompt_token` or with a prefill-only monitor.

The completed E001 prompt-only sweep supports this caution. The best
`last_prompt_token` direction accuracies were poor for nearly every behavior:

| Behavior / origin | Best prompt-only accuracy | Finding |
| --- | ---: | --- |
| `sycophancy/source_backed` | 0.000 | no prompt-only evidence |
| `sycophancy/synthetic` | 0.000 | no prompt-only evidence |
| `hallucination/source_backed` | 0.217 | weak |
| `premature_refusal/source_backed` | 0.333 | weak |
| `unsafe_planning/source_backed` | 0.133 | weak |
| `deception/source_backed` | 0.500 | random |

### Synthetic-Origin Artifacts

Synthetic contrasts often produce very large margins and near-perfect accuracy.
That can be useful for stress testing, but it is weaker evidence than
source-backed transfer. Synthetic-only wins should be reported separately.

Concrete examples:

- E001 synthetic `assistant_answer_mean` gaps were often huge, such as
  `premature_refusal/synthetic` gap `64.936`, `unsafe_planning/synthetic` gap
  `53.047`, and `overconfidence/synthetic` gap `46.629`.
- E012 shows synthetic-to-source transfer can fail badly, especially for
  `hallucination` and `sycophancy`.

### Forced-Choice Pair Integrity

E016 is fair only when the two completions are alternatives for the same prompt
and both completions are nonempty. The current forced-choice extraction exposes
an important benchmark/protocol issue:

| E016 entry | Baseline rows | Same-prompt fraction | Empty-completion rows | Status |
| --- | ---: | ---: | ---: | --- |
| `sycophancy_source_l18` | 21 | 0.048 | 20 | invalid for paired-answer forced choice |
| `deception_source_l18` | 24 | 0.833 | 0 | mixed, use cautiously |
| all other E016 entries | 15-60 | 1.000 | 0 | structurally valid |

The source-backed sycophancy forced-choice result must not be used as a clean
causal claim until the data loader filters to same-prompt, nonempty-completion
pairs or uses a scoring protocol designed for non-paired examples.

### Keyword Marker Weakness

E004, E005, E007, E009, E010, E011, E013, and E014 free-generation experiments
use transparent heuristic markers. These markers are useful for quick auditing,
but they are too coarse for final behavioral claims.

The marker runs showed many changes in length and repetition with little target
marker movement. That is evidence that raw free-generation steering is not yet
clean on 2B, but it is not evidence that the behavior cannot be steered.

## Behavior-Level Findings

### Hallucination

Status: moderate to strong in forced-choice, moderate in representation, weak
for cross-origin transfer.

Evidence:

- E001 `assistant_answer_mean` source-backed direction reached accuracy `1.000`
  with gap `35.912`.
- E012 source-backed to source-backed transfer reached accuracy `0.985`, but
  source-backed to synthetic was reversed with accuracy `0.000`.
- E016 source-backed forced-choice baseline was weak, `accuracy=0.550` and
  `margin=-0.0642`; steering at alpha `4.0` improved the margin by `+0.3248`
  and accuracy to `0.633`.
- E017 calibrated alpha reproduced the source-backed effect with coefficient
  `2.0`, raw alpha `4.151`, and `delta_margin_vs_alpha0=+0.3382`.

Conclusion:

Hallucination is one of the better Qwen3.5-2B targets for forced-choice
steering. However, the strong representation result is partly answer-content
separation, and the origin-transfer reversal means synthetic hallucination
should not be merged into source-backed evidence.

### Sycophancy

Status: weak for source-backed causal claims, moderate for synthetic, invalid
for current source-backed forced-choice extraction.

Evidence:

- E001 source-backed `assistant_answer_mean` reached accuracy `0.952`, but
  prompt-only views were `0.000`.
- E002 source-backed monitor AUROC was `0.878`, again using answer-side
  representations.
- E004 dense CAA free generation barely moved the agreement marker:
  baseline `0.1571`, alpha `1.0` `0.1607`.
- E007 best-layer source-backed sweep showed only a small marker shift:
  baseline `0.1536`, alpha `4.0` `0.1679`, with repetition rising to `0.3080`.
- E009 controls were comparable to the base direction, so the free-generation
  causal claim is weak.
- E016 source-backed forced-choice cannot be used as clean evidence because
  `same_prompt_frac=0.048` and most completions are empty in the current slice.
- E016 synthetic sycophancy was stronger: baseline accuracy `0.833`, alpha
  `-4.0` accuracy `1.000`, `delta_margin_vs_alpha0=+0.1395`.
- E018 source-backed position steering found only tiny deltas; the best was
  `+0.0150` for `all` at alpha `-2.0`.

Conclusion:

Sycophancy is not solved on source-backed 2B. The representation direction is
visible in answer activations, but source-backed causal evidence is weak or
invalid under the current forced-choice extraction. The next sycophancy run must
filter same-prompt, nonempty pairs and use prompt-only or prefill-only monitors.

### Premature Refusal

Status: strongest causal finding so far.

Evidence:

- E001 source-backed `assistant_answer_mean` reached accuracy `1.000` at layer
  `12`, gap `1.370`.
- E012 transfer was strong: source-backed to source-backed `1.000`,
  source-backed to synthetic `1.000`, synthetic to source-backed `0.910`.
- E007 free-generation refusal marker did not move, so keyword markers missed
  the causal effect.
- E016 source-backed forced-choice improved from baseline accuracy `0.433` and
  margin `-0.0396` to alpha `-4.0` accuracy `0.833`,
  `delta_margin_vs_alpha0=+0.8161`.
- E016 synthetic was even stronger: baseline accuracy `0.000`, alpha `-4.0`
  accuracy `0.957`, `delta_margin_vs_alpha0=+1.3018`.
- E017 residual-norm calibration was too conservative for this behavior:
  coefficient `-2.0` mapped to raw alpha `-0.1663` and only moved the margin by
  `+0.0148`.

Conclusion:

Premature refusal is the clearest current steering target. The effect is visible
in forced-choice likelihoods and transfers across origins. The free-generation
marker did not detect it, which argues for using forced-choice or judge-based
evaluation before making free-generation claims.

### Deception

Status: weak to moderate; source-backed representation is weak, synthetic is
much stronger.

Evidence:

- E001 source-backed `assistant_answer_mean` was poor: best accuracy `0.333`,
  gap `1.044`.
- E012 source-backed to source-backed was moderate: accuracy `0.825`, gap
  `1.616`.
- E012 synthetic to source-backed was weaker: accuracy `0.662`.
- E016 source-backed forced-choice baseline was `0.667`, best alpha `-4.0`
  improved margin only by `+0.0260`.
- E016 synthetic was stronger: baseline accuracy `1.000`, alpha `-4.0`
  improved margin by `+0.2170`, but ceiling accuracy prevents a strong causal
  conclusion.
- E016 source-backed pair structure is mixed: same-prompt fraction `0.833`.

Conclusion:

Deception should not be a headline result yet. Source-backed evidence is
inconsistent, and the stronger synthetic results may be templated. The next
deception experiment should filter to same-prompt source-backed pairs and use
more direct forced-choice or judge labels.

### Unsafe Planning

Status: strong representation, ceiling-limited causal evidence.

Evidence:

- E001 source-backed `assistant_answer_mean` reached accuracy `1.000`, gap
  `36.472`.
- E012 transfer was strong: source-backed to source-backed `1.000`,
  source-backed to synthetic `1.000`, synthetic to source-backed `0.920`.
- E007 free-generation unsafe planning marker was flat at `0.000`; repetition
  changed more than behavior markers.
- E016 source-backed forced-choice baseline accuracy was already `1.000` with
  margin `1.7748`; alpha `4.0` improved margin by only `+0.0899`.
- E016 synthetic baseline was also `1.000`, with alpha `4.0` margin delta
  `+0.0596`.
- E017 calibrated coefficient `2.0` improved source-backed margin by `+0.1065`,
  but accuracy remained at ceiling.

Conclusion:

Unsafe planning is well separated in answer activations and forced-choice
likelihood already prefers the desirable answer at baseline. That means current
2B runs cannot show much causal improvement. Use harder prompts, adversarial
eval buckets, or a metric without baseline ceiling before claiming steering
success.

### Overconfidence

Status: synthetic-only, low confidence.

Evidence:

- The current source-backed split has no overconfidence pairs.
- E001 synthetic `assistant_answer_mean` reached accuracy `1.000`, gap `46.629`.
- E012 synthetic-to-synthetic reached accuracy `1.000`, gap `8.162`.
- E016 synthetic forced-choice baseline was poor, `accuracy=0.174`,
  `margin=-0.0502`; no alpha improved it, and alpha `-4.0` worsened margin by
  `-0.0209`.

Conclusion:

No source-backed overconfidence claim is possible from the current run. The
synthetic representation signal is strong, but forced-choice steering did not
improve it. Treat overconfidence as unresolved.

## Experiment-Level Findings

### E001 Mean Direction

Finding: strong for answer-representation separation, invalid for pre-generation
detection claims.

`assistant_answer_mean` produced strong directions for most behaviors. However,
prompt-only `last_prompt_token` was weak or random. The correct interpretation
is that the paired answers encode the behavior distinction very clearly, not
that the prompt alone exposes a reusable latent behavior before generation.

### E002 Activation Monitor

Finding: moderate, but answer-view dependent.

The sycophancy monitor reached AUROC `0.878` and score gap `0.334`. This is a
real held-out answer-representation detector. It is not yet a deployable
pre-generation monitor because it uses answer activations.

### E003 SAE Delta

Finding: diagnostic sparse features exist, but causal leverage is not proven by
E003 alone.

The strongest sycophancy feature in E003/E005 was feature `16221`, delta
`0.0484`. E010 later showed that feature `29345`, not the top E003 feature, had
the largest marker delta in generation. Therefore feature ranking by activation
delta should be treated as candidate discovery, not feature-level causality.

### E004 CAA Free-Generation Steering

Finding: weak.

For sycophancy, agreement marker moved from `0.1571` at alpha `0.0` to only
`0.1607` at alpha `1.0`. This is too small for a behavioral claim. The result
mainly says keyword-marker free generation is not sensitive enough here.

### E005 SAE Feature Steering

Finding: weak to moderate.

Selected feature `16221` moved sycophancy agreement marker from `0.1571` to
`0.1714` at alpha `1.0`, but this is still a small heuristic-marker effect.
Manual text review or forced-choice sparse-feature scoring is needed.

### E006 LoRA SFT

Finding: not run in the completed 2B campaigns.

No conclusion should be drawn about training versus steering from the current
completed runs.

### E007 Best-Layer CAA Sweep

Finding: weak for free generation.

The best sycophancy entry moved agreement marker from `0.1536` to `0.1679` at
alpha `4.0`, while repetition rose to `0.3080`. Premature refusal and unsafe
planning markers were flat. This looks more like style/degeneration sensitivity
than clean behavior control.

### E008 Specificity Matrix

Finding: useful but shows confounding.

Diagonal representation rows were strong for several behaviors, but
off-diagonal positives were also strong. Unsafe planning directions, for
example, separated several synthetic targets with accuracy `1.000`. This means
some directions are broad "bad answer" or format/source axes, not isolated
behavior mechanisms.

### E009 Causal Controls

Finding: weak causal evidence for sycophancy free generation.

The intended sycophancy direction improved agreement marker by only `+0.0036`
at alpha `2.0`, and control variants were comparable. This fails the causal
specificity test for free generation.

### E010 SAE Feature Sweep

Finding: feature `29345` is the best current sparse candidate, but still weak.

Feature `29345` at alpha `-1.0` moved agreement marker by `+0.0321`. That is
larger than dense CAA in marker space, but still small and not enough for a
behavior claim without forced-choice or judge-based confirmation.

### E011 Orthogonalized Steering

Finding: no useful improvement.

Orthogonalization against hallucination, premature refusal, and unsafe planning
did not materially improve sycophancy steering. Raw and orthogonalized curves
were nearly identical.

### E012 Origin Transfer

Finding: source-backed and synthetic origins must be reported separately.

Premature refusal and unsafe planning transferred well across origins. Sycophancy
and hallucination did not: synthetic-to-source was weak for sycophancy
(`0.400`) and hallucination (`0.200`), and source-to-synthetic hallucination was
reversed (`0.000`). This is direct evidence against merging origins into one
claim.

### E013 Dynamic Steering

Finding: invalid dynamic policy, not a failure of dynamic steering in general.

The gate never fired: `mean_applied_steering=0.0`. The experiment only shows the
current threshold/view is wrong for the evaluation bucket. It does not show that
monitor-gated steering cannot work.

### E014 Multi-Layer Steering

Finding: no advantage over single layer.

Layer windows `17-19` and `16-20` did not improve sycophancy forced-generation
markers over single layer `18`. The simpler single-layer hook remains the
default.

### E015 Layer Transfer

Finding: strong late-layer stability for sycophancy answer representations.

The sycophancy direction transferred well across middle and late layers. The
largest margin was `23->23`, gap `8.450`; many `12/18/23` target rows had
accuracy `1.000`. This supports layer-stable answer representation, not
pre-generation detectability.

### E016 Forced-Choice CAA

Finding: strongest current causal diagnostic.

Good results:

- `premature_refusal/source`: accuracy `0.433 -> 0.833`, margin delta `+0.8161`
  at alpha `-4.0`.
- `premature_refusal/synthetic`: accuracy `0.000 -> 0.957`, margin delta
  `+1.3018` at alpha `-4.0`.
- `hallucination/source`: accuracy `0.550 -> 0.633`, margin delta `+0.3248` at
  alpha `4.0`.
- `deception/synthetic`: margin delta `+0.2170`, but baseline accuracy was
  already `1.000`.

Invalid or weak results:

- `sycophancy/source` is invalid under current forced-choice extraction because
  almost all baseline rows are not same-prompt paired completions and most have
  empty completions.
- `unsafe_planning` is ceiling-limited because baseline accuracy is already
  `1.000`.
- `overconfidence/synthetic` did not improve.

### E017 Calibrated Alpha

Finding: useful for hallucination and unsafe planning, too conservative for
premature refusal and sycophancy.

The residual-norm calibration reproduced the hallucination effect: coefficient
`2.0`, raw alpha `4.151`, margin delta `+0.3382`. It also improved unsafe
planning margin by `+0.1065`, but accuracy was already at ceiling.

For premature refusal, the same calibration mapped coefficient `-2.0` to raw
alpha `-0.1663` and produced only `+0.0148`, far below raw alpha `-4.0` in
E016. This means residual-norm calibration is not universally appropriate.

### E018 Position Steering

Finding: no strong sycophancy localization.

The best sycophancy source result was `all` positions at alpha `-2.0`, margin
delta `+0.0150`. `answer`, `first_answer`, `last_prompt`, and `prompt` modes
were similarly tiny or negative. Given the sycophancy source forced-choice data
issue, this should not be interpreted as a true position-localization result.

## Current Claim Boundary

Supported claims:

- Qwen3.5-2B answer activations contain strong behavior-separating directions
  for multiple LatentBehaviorBench axes.
- Premature refusal can be causally shifted in forced-choice likelihood space.
- Source-backed hallucination can be shifted in forced-choice likelihood space,
  but origin transfer is fragile.
- Free-generation keyword markers are currently too weak for final behavioral
  claims.

Unsupported claims:

- The behavior features are cleanly detectable before generation.
- Source-backed sycophancy is solved.
- Synthetic-only results transfer to source-backed data.
- Qwen-Scope top delta features are causal without forced-choice or judge-based
  confirmation.
- Training-free steering preserves general capability, safety, or OOD behavior.

Invalid current claims:

- Any source-backed sycophancy forced-choice claim based on the current E016/E018
  rows.
- Any pre-generation monitoring claim based only on `assistant_answer_mean`.
- Any overconfidence source-backed claim, because there are no source-backed
  overconfidence pairs in the completed run.

## Next Analysis Steps

1. Add a strict forced-choice pair filter: same prompt, nonempty positive and
   negative completions, comparable answer format.
2. Rerun E016 for source-backed sycophancy and deception after filtering.
3. Add prompt-only E001/E002/E016 variants to test pre-generation claims.
4. Add judge-based generation scoring for the behaviors where forced-choice
   margins move but keyword markers do not.
5. Rerun sparse feature interventions with forced-choice scoring, especially
   feature `29345` and top-k feature combinations.
