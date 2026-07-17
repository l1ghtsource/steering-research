# PolicyIntegrityBench benchmark card

## Summary

PolicyIntegrityBench evaluates whether a language model preserves an evidence-grounded
conclusion or an explicitly authorized task objective when contextual pressure favors a
conflicting conclusion or proxy action. It uses paired counterfactual families and exact
primary verification.

Version 2 is a benchmark candidate, not a sealed final release. Its executable data are
appropriate for method development, integration testing, and difficulty calibration.

## Intended use

- behavioral measurement of pressure-induced decision shifts;
- pre-answer activation monitoring on family and generator holdouts;
- CAA, probe, subspace, Qwen-Scope SAE, and feature-coalition discovery;
- causal intervention with random, shuffled-label, norm-matched, and position controls;
- dynamic steering and integrity-capability Pareto analysis;
- controlled-to-agentic transfer;
- authorization sensitivity versus blanket conservatism;
- insufficiency calibration.

## Out-of-scope uses

- a universal alignment, honesty, deception, or safety score;
- claims of human-like intent or consciousness;
- training or selecting methods on validation and then reporting validation as held out;
- evaluating future checkpoints after the public data may have entered training corpora;
- treating rendered variants as independent samples;
- using the public release as the final confirmatory evidence for an A* submission.

## Construct

**Policy integrity under contextual pressure** is the probability that a model preserves
the policy justified by self-contained evidence or explicit authorization when a
different contextual signal is instrumentally or socially favored.

The benchmark separates:

- capability failure: neutral world not solved;
- integrity failure: a neutral-capable family shifts under conflicting context;
- calibration failure: confidence unsupported by evidence;
- expression failure: invalid structure or citation;
- blanket conservatism: refusal to perform the same action when explicitly authorized.

## Data composition

| Track | Families | Variants per family | Items | Verification |
|---|---:|---:|---:|---|
| E, epistemic | 72 | 8 | 576 | deterministic constraint/interval solvers |
| A, agentic | 24 | 4 | 96 | deterministic final-state predicates |

Track E has 60 compatibility-intersection families across ten generator groups and 12 bounded
aggregation families across two generator groups. Track A spans six environment
archetypes. All entities, records, paths, and values are synthetic and have no
external-world effect.

## Creation

Scenario semantics and prose frames were authored specifically for this project on 17
July 2026. Code deterministically instantiates the typed worlds and counterfactual cells;
it does not call an external language model. The same typed values are consumed by the
renderer and solver.

The current authorship is not independent. Blank review packets are included for two
external reviewers per family. Their completion and adjudication are release gates.

## Splits

- `dev`: nine Track E and four Track A generator groups;
- `validation`: three Track E and two Track A generator groups absent from dev;
- `final`: independently authored after method freeze and stored outside the repository.

All variants of a family and all families from a generator group remain in one split.
Public validation should be treated as one-shot. After adaptive inspection it is dev.

The checked-in prompts are reference renderings. Official validation uses a secret seed
to permute option order and opaque evidence IDs consistently across every variant of a
family, then shuffles inference rows. Inputs and scoring keys are materialized to separate
directories. Inference workers receive no condition, family, split, or target metadata
and must process items in stateless contexts. Since prompt similarity can reveal paired
family membership, workers may not fit, retrieve, normalize, cache across rows, derive
batch features, or otherwise adapt from the validation/final corpus.

## Primary Track E metrics

- neutral capability rate;
- paired conflict-suggestion error among neutral-capable families;
- paired conflict-pressure error;
- pressure difference-in-differences against aligned controls;
- typed stakeholder-target adoption;
- other wrong answer and abstention rates;
- insufficiency accuracy and confidence calibration;
- citation precision, recall, and nonexistent-ID rate;
- valid response rate.

Both wrong targets are averaged within family. Uncertainty is clustered by family.

## Primary Track A metrics

- clean task success;
- capability-qualified unauthorized proxy rate;
- true objective and proxy score;
- pressure increment in unauthorized proxy use;
- authorized-counterfactual success;
- invalid action and action count.

An intervention that lowers proxy use but also blocks the authorized cell is conservative,
not authorization-sensitive.

## Known limitations

1. Track E has two solver strata, but bounded aggregation has only 12 families. Causal,
   temporal, and longer compositional reasoning are not yet independent solver strata.
2. Compatibility tables are fictional and exactly counterbalanced. They isolate
   self-contained evidence use but are not a naturalistic domain-knowledge test.
3. The candidate does not directly manipulate evaluation awareness or realism. Those
   deployment-relevance claims require a future factor or external transfer test.
4. Track A is short-horizon and uses JSON workspaces. Long-horizon transfer requires a
   separately controlled external benchmark.
5. Current items have one authorship source and no completed independent human review.
6. Only static and executable calibration has passed. Post-trained model calibration is
   pending.
7. The checked-in validation labels are visible to repository readers.
8. Freshness for named frozen checkpoints does not guarantee uniqueness across the web or
   protection for future models.
9. Finite-sample word-, character-, length-, structure-, outcome-, and metadata-based
   shortcut baselines exclude only tested artifact classes, not every possible shortcut.

## Ethical and safety considerations

The environments are fictional, local, deterministic, and network-free. They do not
contain real credentials, personal data, harmful operational targets, or arbitrary code
execution. The benchmark can still be misused to overstate mechanistic or alignment claims;
the interpretation rules and release status are therefore part of the benchmark contract.

## Reproducibility

The manifest records counts, hashes, generator groups, and the complete dataset hash.
Rebuild and validation commands are in `data/README.md`. Structural reports are stored
under `data/reports/`.
