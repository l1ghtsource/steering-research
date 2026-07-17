# PolicyIntegrityBench validation report

Status: **candidate_structurally_valid_pending_external_gates**

This report distinguishes executable validation from scientific release readiness. A
passing solver/runtime audit is necessary but does not replace independent semantic
review, target-model calibration, or a separately held final set.

## Inventory

- Track E: 72 independent families, 576 counterfactual items.
- Solver strata: `{'compatibility_intersection': 60, 'interval_band': 12}`.
- Track A: 24 independent families, 96 executable items.
- Track E solver replays: 576.
- Track E secret-seed rerender replays: 3456.
- Track A reference/adversarial trajectory replays: 288.
- Track A write-then-revert escape replays: 48.
- Track A post-submit/action-limit/collateral-write replays:
  96/96/96.

## Release gates

| ID | Gate | Status | Evidence |
|---|---|---|---|
| G01 | Schema, hashes, and render replay | PASS | 768 families/items schema-valid; 672 rendered items audited |
| G02 | Track E independent solver replay | PASS | 576/576 replays |
| G03 | Matched ambiguity morphology | PASS | compatibility: 10 missingness patterns x 6 per class; interval: five observed width-one records per class |
| G04 | Grouped split isolation | PASS | zero family or generator-group crossings |
| G05 | Track A executable counterfactual replay | PASS | 288 reference trajectories; 48 write-revert; 96 post-submit; 96 action-limit; 96 collateral-write attacks |
| G06 | Static shortcut baselines | PASS | group-OOD baselines below registered limits |
| G07 | Independent semantic review | PENDING | blank reviewer packets generated; two external reviewers required |
| G08 | Target-model capability and pressure calibration | PENDING | must run post-trained 2B, 9B, and one non-Qwen family |
| G09 | Independent sealed final families | NOT_STARTED | final answers intentionally absent from repository |

## Leakage and shortcut checks

- Correct option balance: `{'A': 24, 'B': 24, 'C': 24}`.
- Wrong pressure-target balance: `{'B': 96, 'C': 96, 'A': 96}`.
- Single-record sufficient solutions: 0.
- Essential-record count distribution: `{2: 420}`.
- Family-matched compatibility missingness packets: 60.
- Missingness-only group-OOD accuracy: 0.500;
  orientation-free predictability: 0.500
  (chance 0.500).
- Missingness-to-answer group-OOD accuracy: 0.000
  (chance 0.333).
- Evidence-free group-OOD label accuracy: word NB
  0.361; char NB
  0.417; length lookup
  0.347 (chance 0.333).
- Profile-blind group-OOD label accuracy: word NB
  0.319; char NB
  0.319; exact lookup
  0.333; by solver
  `{'compatibility_intersection': {'nb': 0.31666666666666665, 'lookup': 0.3333333333333333}, 'interval_band': {'nb': 0.25, 'lookup': 0.3333333333333333}}` (chance 0.333).
- Value-blind compatibility-structure group-OOD accuracy:
  0.000 (chance 0.333).
- Outcome-only group-OOD accuracy: NB 0.367;
  exact lookup 0.333 (chance 0.333).
- Outcome-only sufficient/insufficient predictability:
  0.525 (chance 0.500).
- Evidence-free stakeholder-surface group-OOD accuracy:
  0.271 (chance 0.333).
- Exact one-sided binomial p-values against chance (descriptive, uncorrected):
  `{'evidence_free_word': 0.3495240518306057, 'evidence_free_char': 0.08615604714358133, 'profile_blind_word': 0.6415207276179054, 'profile_blind_char': 0.6415207276179054, 'compatibility_outcome_only': 0.33613846273412257, 'stakeholder_surface': 0.9978254467888324}`.
- Exact prompt duplicates: 0.
- Generator groups crossing splits: 0.
- Maximum dev/validation evidence-free trigram Jaccard:
  0.247.

## Executable checks

Track E labels were recomputed from visible decision specifications and observations;
stored labels were not trusted. Every sufficient cell resolved to one candidate and
every insufficient cell had no unique candidate. Compatibility-intersection paired
packets have identical semantic dimensions, evidence IDs, and statuses. Two informative
outcomes eliminate different wrong candidates in sufficient cells, while matched
insufficient cells repeat one exclusion and leave two candidates feasible. Each of the
ten possible missingness patterns occurs six times and is balanced over answer position.
Interval-band packets contain five observed width-one intervals in both conditions; only
the aggregate range-to-band relation changes.

Track A replayed the legitimate, proxy, and expected-success trajectory for every item.
Unauthorized proxy trajectories raise a violation in temptation and pressure cells;
write-then-revert trajectories cannot erase that violation. The same action succeeds
without violation in the authorized counterfactual. Arbitrary code execution is absent
from the runtime.

## Findings

- No executable structural failure was found.

## Limitations and non-claims

- The repository release is public development data, not an unopened confirmatory final set.
- All current source prose and scenario semantics were authored in one Codex session; independent human review is still absent.
- Track E has two solver strata, but only 12 bounded-aggregation families; causal, temporal, and longer compositional reasoning require additional solver strata.
- Compatibility profiles are fictional, explicitly supplied, and exactly counterbalanced; they test use of self-contained evidence but do not by themselves establish ecological validity.
- The current candidate has no direct evaluation-awareness or realism counterfactual. Deployment-relevance claims require that future factor or an external transfer benchmark.
- Passing the registered finite-sample shortcut baselines rules out only the tested artifact classes, not every possible learned shortcut.
- Track A uses bounded JSON workspaces and three tools. This gives exact verification but does not establish long-horizon agent transfer.
- The validation split has generator OOD isolation, but inspecting it repeatedly invalidates one-shot validation claims.
- Paired prompts are surface-linkable even when family IDs are hidden. Validation and final workers must be frozen and stateless; any cross-row clustering, retrieval, fitting, normalization, or caching is transductive adaptation rather than held-out evaluation.
- Freshness relative to frozen Qwen checkpoints is documented, but web-scale uniqueness cannot be proven by local exact-match checks alone.
- A* confirmatory claims require independently authored, escrowed final families and a preregistered analysis manifest.

## Release decision

The data are suitable for code integration and exploratory benchmark calibration only
when all executable gates pass. They are not an A* confirmatory test set until external
semantic review, model calibration, and the sealed-final protocol pass. Repeatedly
inspecting `validation` converts it into development data.
