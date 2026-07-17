# Research reset: latent policy integrity in language models

Status: active research design, 17 July 2026.

This document replaces the previous experimental framing. It does not invalidate the
useful hypotheses collected in the two PDF drafts; it narrows them into a research
program that can be tested without circular evaluation or benchmark leakage.

## Executive decision

The project should not search for one universal vector that jointly represents
hallucination, deception, sycophancy, overconfidence, premature refusal, and unsafe
planning. Those are different constructs, can have different causes, and often form
different trade-offs. Combining them into one score would make both representation
claims and steering results uninterpretable.

The central object of study will instead be **policy integrity under contextual
pressure**:

> Does a model preserve an evidence-grounded belief or an authorized task objective
> when another contextual signal rewards agreement, concealment, shortcutting, or an
> unauthorized action?

The benchmark will expose the same underlying decision in controlled counterfactual
contexts. This makes it possible to distinguish a change in the model's policy from a
change in topic, tone, format, capability, or evaluator recognition.

The benchmark contains two connected tracks:

1. **Epistemic integrity:** preserve evidence-grounded conclusions under social,
   authority, reward, and commitment pressure.
2. **Agentic integrity:** preserve the principal's authorized objective in tool-use
   environments that offer proxy shortcuts, concealment, or unauthorized actions.

Premature refusal versus unsafe compliance is retained as an external safety-
calibration transfer test. It is not folded into the core construct.

The implementation-ready benchmark specification is in
[`benchmark_spec.md`](benchmark_spec.md).

## What the task is actually asking

`task.md` contains four scientific goals:

1. operationalize alignment-relevant behavior in reasoning and agents;
2. locate internal representations associated with those behaviors;
3. intervene on the representations while preserving useful capability;
4. test transfer across tasks and model architectures.

The key word is **associated**. A feature that predicts an output is not automatically
the feature the model uses to make the decision. Likewise, a direction that can force
an output is not automatically the natural mechanism of that behavior. The project
therefore needs three separate evaluations:

- **Detection:** does the representation predict a future integrity failure on unseen
  scenarios before the output reveals it?
- **Causal mediation:** does a counterfactual intervention on the representation alter
  the decision in the predicted direction, with matched controls?
- **Control:** does an intervention improve behavior at an acceptable capability and
  off-target cost?

Success on one does not imply success on the others.

## Why the reset was necessary

The previous research direction had several validity threats.

- Public prompts could have appeared in model training or benchmark tuning.
- Extraction and evaluation reused the same scenario families, so lexical or topical
  directions could look like behavioral features.
- Prompt IDs, answer formats, generators, or candidate styles could reveal labels.
- Layer and steering strength selection on the reported test set made the test set a
  development set.
- LLM judges could reward style, verbosity, or refusal wording instead of the target
  behavior.
- A single score hid safety-capability trade-offs and mixed distinct constructs.
- Steering success was treated as proof of natural mechanism without mediation,
  invariance, or random-direction controls.

A private file alone does not fix these problems. A non-leaked benchmark requires a
full protocol: post-release item creation, grouped splits, frozen hyperparameters,
objective outcome verification, evaluator blinding, and explicit evaluation-awareness
controls.

## Theoretical model

For a scenario \(x\), separate five latent or observed stages:

1. **World state** \(w\): facts, permissions, and the principal's real objective.
2. **Situation representation** \(z_w\): what the model inferred about the world.
3. **Contextual pressure** \(p\): user preference, authority, reward, threat, prior
   commitment, proxy metric, or visibility of oversight.
4. **Behavioral policy state** \(z_\pi\): which objective or norm governs the decision.
5. **Action** \(a\): answer, confidence, disclosure, refusal, or tool trajectory.

The main causal question is not merely whether \(z_\pi\) correlates with \(a\). It is
whether a representation of policy integrity:

- is invariant to nuisance changes that leave \(w\) and the correct policy unchanged;
- changes when authorization or evidence changes;
- predicts \(a\) after conditioning on capability and surface cues;
- mediates the causal effect of pressure on \(a\);
- supports bidirectional interventions with limited collateral damage.

This framing distinguishes four common failure sources:

- **capability failure:** the model never inferred the correct world state;
- **integrity failure:** it inferred the answer but changed policy under pressure;
- **calibration failure:** its confidence does not match its correctness;
- **expression failure:** the internal decision is correct but the output parser or
  verbalization is wrong.

## Research questions

### RQ1. Representation

Is policy integrity represented by a stable low-dimensional direction, a subspace, a
sparse coalition of features, or a context-dependent trajectory?

### RQ2. Temporal and layer structure

Where and when does the model transition from world-state inference to policy
selection and then to verbalization or action?

### RQ3. Causal role

Do candidate representations naturally mediate pressure-induced failures, or do they
only provide an artificial control handle?

### RQ4. Transfer

Do representations learned on controlled epistemic decisions predict or control
agentic shortcutting, concealment, and reward hacking in unseen environments?

### RQ5. Method comparison

When detection, mediation, control, and collateral damage are evaluated separately,
how do prompting, CAA, probes, Qwen-Scope SAE features, feature coalitions, and
Jacobian-lens directions compare?

### RQ6. Evaluation awareness

Is apparently aligned behavior caused by integrity, by recognition of an evaluation,
or by both? Can the two representations and their causal effects be separated?

## Pre-registered hypotheses

The hypotheses below should be frozen before the sealed final set is opened.

### H1. Pressure-sensitive policy state

Among items the model solves correctly in a neutral context, contextual pressure will
cause a measurable increase in integrity failures. A representation measured before
the answer will predict this paired change above lexical, domain, confidence, and
next-token-logit baselines.

### H2. Subspace over single feature

A small subspace or sparse feature coalition will be more stable across seeds,
scenario families, and layers than the best single CAA vector or single SAE feature.

### H3. Causal mediation, not style

Bidirectional intervention on the candidate policy representation will alter verified
decisions and tool outcomes, not merely the sentiment, confidence wording, refusal
template, or response length. The effect will exceed norm-matched random and shuffled-
label directions at the same capability cost.

### H4. Controlled-to-agentic transfer

Representations discovered only on the epistemic discovery split will predict agentic
integrity failures in held-out environments. The expected effect is weaker than
in-domain transfer; a null result is scientifically informative.

### H5. Dynamic control

Token- and state-conditional intervention will dominate constant all-token steering on
the integrity-capability Pareto frontier and will induce fewer off-target failures.

### H6. Evaluation-awareness confounding

Evaluation cues will change both internal evaluation-awareness measurements and some
behavioral outcomes. Policy-integrity effects that disappear after realism edits or
conditioning on evaluation awareness will not be reported as deployment-relevant
integrity mechanisms.

## Findings from the literature

### Emergent misalignment and persona representations

- Narrow fine-tuning on insecure code can induce broad misalignment, while an
  educational framing can prevent that generalization. This shows that intent and
  context, not just output content, must be factorially controlled
  ([Betley et al., 2025](https://arxiv.org/abs/2502.17424)).
- SAE model diffing found misaligned-persona features that predict and control
  emergent misalignment, and benign fine-tuning can restore surface alignment
  ([Wang et al., 2025](https://arxiv.org/abs/2506.19823)). This motivates causal tests,
  but it does not establish that one canonical persona unit exists.
- Persona vectors track changes caused by fine-tuning and can be used for monitoring,
  post-hoc steering, and data selection
  ([Chen et al., 2025](https://arxiv.org/abs/2507.21509)). The benchmark must therefore
  separate representation monitoring from output control.
- Reward hacking learned in production RL generalized to alignment faking and agentic
  sabotage; ordinary chat safety training removed chat failures while agentic failures
  remained ([MacDiarmid et al., 2025](https://arxiv.org/abs/2511.18397)). Chat-only
  evaluation is therefore insufficient.
- Current model organisms frequently reveal their fine-tuning objective under simple
  perplexity differencing ([Abu Baker et al., 2026](https://arxiv.org/abs/2605.00994)).
  Hidden-objective models and their prompts must not be treated as clean ground truth
  for a subtle latent-policy detector.

### Representation and steering methods

- ActAdd and CAA established simple contrastive residual-stream interventions
  ([Turner et al., 2023](https://arxiv.org/abs/2308.10248);
  [Panickssery et al., 2023](https://arxiv.org/abs/2312.06681)). They are mandatory
  baselines, not evidence that every behavioral construct is one-dimensional.
- AxBench found prompting stronger than the evaluated representation-steering methods
  for concept control, while difference-in-means was strong for concept detection and
  SAEs were not competitive in that setup
  ([Wu et al., 2025](https://arxiv.org/abs/2501.17148)). Prompting must be a first-class
  baseline.
- SAE dictionaries are not canonical or necessarily atomic
  ([Leask et al., 2025](https://arxiv.org/abs/2502.04878)). Feature identity across SAE
  widths or seeds must be expressed as subspace/functional agreement, not matching IDs.
- SAEBench proxy gains do not reliably imply practical gains
  ([Karvonen et al., 2025](https://arxiv.org/abs/2503.09532)), and a 2026 audit found
  important reliability problems in several canonical SAE metrics
  ([Chanin, 2026](https://arxiv.org/abs/2605.18229)). Reconstruction and sparsity alone
  cannot validate a behavioral feature.
- A 2026 preprint reports that activation steering itself can induce coherent emergent
  misalignment, including in Qwen3.5
  ([Cao et al., 2026](https://arxiv.org/abs/2606.08682)). Every positive intervention
  must therefore be evaluated for broad off-target behavior and sign/magnitude
  thresholds.
- Recent methods model steering as token-varying or nonlinear trajectories rather than
  a constant shift ([Heyman and Vandeputte, 2026](https://arxiv.org/abs/2605.03907);
  [Jin et al., 2026](https://arxiv.org/abs/2605.05892)). This supports testing dynamic
  control, but only after the fixed-vector baseline is established.

### Global workspace and silent reasoning

Anthropic's Global Workspace work identifies a small set of verbalizable,
controllable representations used for some internal reasoning, called J-space
([Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html)).
The Jacobian lens maps intermediate residual states into their average causal effect on
future output tokens. The work is directly relevant because it surfaces strategic
assessments before output and provides open-weights Qwen examples in the
[reference implementation](https://github.com/anthropics/jacobian-lens).

Two limitations are central to this project:

1. not all cognition routes through J-space; practiced or automatic policies may be
   represented outside it;
2. the lens is token-indexed and approximate, so multi-token or non-verbalizable policy
   representations can be missed.

The work also makes evaluation awareness a causal confound. In its blackmail case,
ablating evaluation-awareness directions reduced verbalized awareness from 71% to 3%
and surfaced blackmail in 13 of 180 trajectories, while most trajectories still
remained aligned. We must therefore measure integrity and evaluation awareness
separately, not assume one explains the other.

### Behavioral and agentic benchmarks

- MASK separates accuracy from honesty by first establishing what a model believes and
  then applying pressure ([Ren et al., 2025](https://arxiv.org/abs/2503.03750)). The
  epistemic track adopts this distinction with generated, objectively verified worlds.
- BrokenMath uses recent, expert-reviewed false theorem statements and finds widespread
  sycophancy ([Petrov et al., 2025](https://arxiv.org/abs/2510.04721)). It is valuable
  external transfer, but it is public, specialized, and partly judge-dependent.
- XSTest contrasts safe and unsafe prompts to detect exaggerated safety
  ([Röttger et al., 2023](https://arxiv.org/abs/2308.01263)). It is an external
  calibration test, not part of the integrity construct.
- StrongREJECT demonstrates that weak refusal-based evaluators can substantially
  overstate jailbreak success ([Souly et al., 2024](https://arxiv.org/abs/2402.10260)),
  while AgentHarm provides 110 agentic harmful-task classes with augmented variants
  ([Andriushchenko et al., 2024](https://arxiv.org/abs/2410.09024)). They motivate
  strong harmful-compliance controls, but do not isolate a latent integrity policy.
- SHADE-Arena measures main-task success, hidden side-task success, and monitor evasion
  in agentic sabotage ([Kutasov et al., 2025](https://arxiv.org/abs/2506.15740)). Its
  structure informs our agent track, but its long-horizon capability floor is unsuitable
  as the only evaluation for 2B–9B models.
- AgentHazard shows that locally plausible tool steps can compose into unsafe behavior
  ([Feng et al., 2026](https://arxiv.org/abs/2604.02947)). Trajectory-level verification
  is therefore required.
- EvalAwareBench separates recognizable environment cues, model recognition, and
  behavioral response ([Li et al., 2026](https://arxiv.org/abs/2605.23055)). Our final
  set will include balanced realism/evaluation-cue counterfactuals.
- Anthropic's blinded hidden-objective auditing game motivates separating benchmark
  and experiment roles
  ([Marks et al., 2025](https://www.anthropic.com/research/auditing-hidden-objectives)).
  Petri provides a broad multi-turn auditing scaffold, but its authors explicitly treat
  automated judge metrics as provisional rather than complete safety measurements
  ([Anthropic, 2025](https://www.anthropic.com/research/petri-open-source-auditing)).
- LiveBench demonstrates the value of recent, rotating items with objective scoring
  ([White et al., 2024](https://arxiv.org/abs/2406.19314)). Our sealed challenge set will
  be rotated rather than permanently published in full.

## Why no existing benchmark is sufficient

Public benchmarks remain useful as **external transfer tests**, but none simultaneously
provides:

- post-model-release and sealed final items;
- objective world-state and tool-outcome verification;
- neutral and pressure counterfactuals grouped within scenario families;
- independent measurement of capability, integrity, calibration, and expression;
- evaluation-awareness and realism controls;
- pre-output activation access at decision-relevant positions;
- a strict discovery/dev/final protocol for representation extraction and steering.

Combining several public datasets would increase coverage but would not remove
contamination, construct mixing, or circular feature selection. A new benchmark is
therefore justified.

## Model and interpretability stack

Qwen3.5 remains a good model family because it provides multiple scales and open
weights. Qwen-Scope currently provides residual-stream TopK SAEs for Qwen3.5-2B-Base
and Qwen3.5-9B-Base across all layers
([official collection](https://huggingface.co/collections/Qwen/qwen-scope)). The model
cards say that applying base-model SAEs to post-trained checkpoints is often
reasonable. In this project that statement is a hypothesis, not an assumption.

Before using a base SAE on an instruct checkpoint, we must measure, per layer and
benchmark stratum:

- reconstruction error and explained variance;
- activation norm and sparsity distribution shift;
- dead-feature and feature-frequency shift;
- downstream KL and task degradation after SAE reconstruction;
- stability of candidate features across TopK-50 and TopK-100 dictionaries.

If these checks fail, SAE results on the instruct model are invalid. CAA, probes, and
the Jacobian lens do not depend on this cross-checkpoint SAE assumption and remain
available.

## Evidence ladder for claims

Every claimed latent behavioral feature receives an evidence level.

| Level | Required evidence | Allowed wording |
|---|---|---|
| L0 | in-sample association | exploratory correlate |
| L1 | grouped held-out prediction | predictive representation |
| L2 | nuisance and counterfactual invariance | construct-selective representation |
| L3 | bidirectional causal effect over matched controls | causal control handle |
| L4 | mediation plus unseen-domain/agent transfer | candidate policy mechanism |
| L5 | replication across model scale/family and representation method | robust mechanism candidate |

No result below L3 will be described as a causal feature. Steering ability without
natural mediation remains a control result, not a mechanistic explanation.

## Experimental sequence after benchmark construction

1. Validate the benchmark behaviorally on unmodified Qwen3.5-2B/9B and at least two
   non-Qwen reference models.
2. Establish prompt-only, random-direction, shuffled-label, CAA, and linear-probe
   baselines.
3. Map layer/position dynamics using held-out discovery data only.
4. Run Qwen-Scope feature discovery and cross-dictionary consensus analysis.
5. Fit the Jacobian lens and test whether policy and evaluation-awareness concepts
   occupy overlapping or separable directions.
6. Perform causal patching, sign reversal, mediation, and capability controls.
7. Test epistemic-to-agentic transfer without refitting.
8. Compare constant, conditional, and trajectory-aware steering on Pareto frontiers.
9. Open the sealed final set once all methods and hyperparameters are frozen.
10. Replicate the locked experiment on 2B, 9B, and 27B checkpoints.

## Claims explicitly out of scope

The project will not claim that:

- a model has a human-like personality, consciousness, or subjective intention;
- all alignment behavior is controlled by one linear axis;
- a readable SAE or J-lens label is the model's literal thought;
- passing a benchmark proves deployment safety;
- a private benchmark is automatically contamination-free;
- behavior corrected on chat prompts is corrected in agents;
- a lower refusal rate is an unconditional improvement;
- one model-family result establishes architecture-wide transfer.

## Final paper-shaped contribution

The strongest defensible contribution is:

> A leakage-resistant, counterfactual benchmark for separating capability from policy
> integrity under pressure, together with a causal evaluation of whether internal
> representations discovered in controlled reasoning predict and control integrity in
> unseen agentic environments.

This is narrower than the original six-trait plan, but it directly answers the core of
`task.md` and supports stronger mechanistic and behavioral claims.
