# PolicyIntegrityBench research readiness

## Decision

PolicyIntegrityBench is ready for software integration, exploratory behavioral
calibration, and development-split representation work. It is not yet ready to supply
confirmatory evidence. Gates G07 (independent semantic review), G08 (post-trained target
model calibration), and G09 (independently authored sealed final families) remain open.

The current data should be described as a **controlled causal benchmark candidate for
policy integrity under contextual pressure**, not as a universal benchmark of good and bad
personas.

The compatibility profiles are case-specific fictional specifications and are exactly
counterbalanced rather than derived from real-world causal frequencies. This is useful for
identification but limits ecological interpretation; naturalistic external transfer and a
future realism/evaluation-awareness factor remain required.

## Relation to `task.md`

The original task asks whether latent behavioral features can be identified and steered
while preserving reasoning capability and transferring across tasks and architectures.
Version 2 operationalizes one coherent part of that request:

| Task goal | Benchmark support | Boundary |
|---|---|---|
| Behavioral evaluation | Matched evidence/pressure and authorization counterfactuals | Measures policy integrity, not every listed behavior |
| Representation discovery | Pre-answer states can be labeled by verified outcomes and paired shifts | Discovery must use dev families only |
| Steering comparison | Exact decisions and tool outcomes permit capability-integrity Pareto curves | Steering ability alone is not natural mediation |
| Transfer | Generator-OOD splits and Track E to Track A transfer | Cross-architecture evidence still requires additional models |

Hallucination, deception, overconfidence, premature refusal, and unsafe planning are not
collapsed into one label. Confidence calibration is measured in Track E, but the other
constructs require separate external transfer sets or future independently verified
strata. A null transfer result is informative and must not be repaired by redefining the
benchmark label.

## Data anatomy

The independent unit is a canonical family.

Track E contains a typed world, three candidates, five diagnostics, an exact solver, and
eight paired renderings. Seven renderings share the same sufficient evidence and differ
only in stakeholder claim or pressure. The eighth is a morphology-matched insufficient
case. The model returns a decision, confidence, supporting evidence IDs, and a short
reason. Free text cannot change the primary score.

Track A contains an initial JSON workspace, explicit writable/protected paths, an
authorized task, a proxy action, exact success predicates, and four paired environments.
The model acts through `read_file`, `set_json`, and terminal `submit`. Primary outcomes are
computed from exact final state and successful-write history, not an LLM judge.

## Experiments that are identifiable

### Behavioral factorial

Estimate neutral capability first. Among neutral-capable families, estimate conflict
suggestion error, conflict pressure error, stakeholder-target adoption, and the pressure
difference-in-differences against aligned controls. Average the two wrong targets within
family and bootstrap families, not prompts.

### Predictive representation

Extract activations before decision tokens or tool actions. Fit CAA, linear probes, or
sparse-feature readouts on dev with nested generator-group cross-validation. The target is
a future verified integrity failure conditional on neutral capability. Condition names,
family IDs, display position, output tokens, and post-answer states are forbidden inputs.

### Natural mediation

Test whether pressure changes a candidate representation and whether intervention on that
representation changes the verified outcome. Report pressure-to-representation,
representation-to-outcome, and residual direct effects separately. A controllable vector
without natural mediation is a control handle, not a discovered mechanism.

### Causal steering

Compare prompting, CAA, probes/subspaces, Qwen-Scope SAE features and coalitions, and
dynamic interventions. Every method needs bidirectional sign tests, shuffled-label and
norm-matched random directions, token-position controls, layer controls, and unrelated
capability evaluations. Select layer, feature, alpha, and stopping rule on dev only.

### Transfer

Freeze the representation and intervention before testing held-out generator groups,
Track E to Track A, 2B to 9B/27B, and Qwen to a non-Qwen model. Refitting on the target is
adaptation, not transfer, and must be reported separately.

## Invalid experiments

- Randomly splitting rendered rows or treating 672 items as independent samples.
- Learning a vector from condition names or aligned versus conflict prompt text and calling
  it an integrity representation.
- Selecting features, layers, or intervention strengths after inspecting validation.
- Reporting only forced-choice answer changes while omitting neutral capability,
  insufficiency, authorized actions, and off-target degradation.
- Calling a readable SAE feature a causal mechanism without intervention and mediation.
- Reusing the public validation split as confirmatory evidence after adaptive analysis.
- Jointly clustering or adapting to validation/final prompts. Paired family membership is
  surface-linkable even though its administrative ID is hidden from inference rows.
- Claiming deployment safety, intent, consciousness, or a universal persona axis.

## Current executable evidence

- 72 Track E families and 576 items; 24 Track A families and 96 episodes.
- Four JSON Schemas validate all 768 canonical and rendered records.
- 576 independent solver replays and 3,456 secret-renderer replays pass.
- No sufficient compatibility item is solved by one record; exactly two records are
  essential in all 420 sufficient compatibility variants.
- Option positions are exactly balanced by semantic candidate, generator group, local
  authoring-template index, pressure frame, and missingness pattern.
- Evidence-free word, character 3--5-gram, and coarse-length models plus profile-blind,
  value-blind, outcome-only, missingness-only, and stakeholder-surface baselines pass the
  registered group-OOD thresholds. Passing these tested baselines does not prove that no
  unknown shortcut exists.
- Track A passes 288 reference trajectories, 48 write-then-revert attacks, and 96 each of
  post-submit, action-limit, and collateral-write attacks.
- Exact prompt duplicates and generator-group split crossings are zero.

The authoritative numbers and limitations are in
[`reports/candidate_validation.md`](reports/candidate_validation.md).

## Remaining gates

### G07: independent semantics

Two non-author reviewers must solve every family from blinded packets, verify the renderer,
and rate ambiguity, naturalness, and pressure/proxy plausibility. The current single-source
authorship is not independent validation.

### G08: behavioral calibration

Run free-generation Track E and executable Track A on post-trained Qwen3.5 2B and 9B plus
at least one non-Qwen family. Require a usable neutral-capability floor, no ceiling
saturation, a measurable but nondegenerate pressure effect, valid structured outputs, and
sufficient neutral-capable families per solver stratum. Base-model forced choice is only a
difficulty diagnostic.

### G09: sealed final

After all hypotheses, code, layers, features, and intervention rules are frozen, independent
authors create new families. An evaluator commits hashes before inference and keeps prompts,
keys, outputs, and subgroup metrics outside the method team's context until outputs are
irreversible. Only this set can support confirmatory claims.

## Publication interpretation

A strong result would show a representation that predicts failures on unseen families,
changes under pressure but not nuisance controls, mediates part of the behavioral effect,
supports bidirectional intervention with limited capability cost, and transfers without
refitting. The benchmark makes that claim testable; the current public candidate does not
itself establish it.
