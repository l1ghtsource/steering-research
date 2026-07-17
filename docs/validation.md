# Validation

PolicyIntegrityBench uses executable validation as a release requirement. A successful run
means the checked-in data satisfy the implemented structural contract. It does not certify
ecological validity or readiness for confirmatory publication.

## Run the audit

```bash
uv run pib validate --write-artifacts
```

The machine-readable and Markdown reports are written to `data/reports/`.

## Release gates

| Gate | Automated | Purpose |
|---|---|---|
| Schemas, hashes, render replay | yes | Detect malformed or stale generated artifacts |
| Independent Track E solver replay | yes | Recompute every label from visible typed facts |
| Matched ambiguity morphology | yes | Block evidence-count and missingness shortcuts |
| Grouped split isolation | yes | Prevent family and generator leakage across splits |
| Track A executable replay | yes | Verify success, violation, and adversarial semantics |
| Static shortcut baselines | yes | Detect tested text, position, structure, and metadata artifacts |
| Independent semantic review | no | Verify ambiguity, naturalness, authorization, and plausibility |
| Target-model calibration | no | Establish usable capability and pressure range |
| Independently authored sealed final | no | Support confirmatory claims after method freeze |

## Solver and renderer checks

The validator checks every stored family hash, item prompt hash, canonical render, and 3,456
secret-seed rerenders. It independently solves all 576 Track E items and verifies unique or
ambiguous feasible sets rather than trusting `target` fields.

For compatibility families, no single record may solve a sufficient item. Exactly two
records must be essential, and sufficient/insufficient conditions must share statuses,
evidence IDs, dimensions, and missingness. For interval families, both conditions contain
five observed width-one intervals and differ only in aggregate relation to the decision
bands.

## Shortcut baselines

All classifier baselines use leave-one-generator-group-out prediction. Registered checks
include:

- evidence-free word multinomial Naive Bayes;
- evidence-free character 3--5-gram Naive Bayes;
- coarse text-length lookup;
- profile-blind word, character, and exact-pattern models;
- value-blind structural lookup;
- observed-outcome-only models;
- stakeholder-surface text models;
- missingness-only condition and answer models.

The three-way global ceiling is `0.45`; binary morphology predictability is capped at `0.60`.
Exact one-sided binomial tail probabilities are reported descriptively. These checks rule out
only tested artifact classes.

## Agent runtime attacks

In addition to three reference trajectories per item, validation tests:

- write a protected proxy value and revert it;
- act after terminal `submit`;
- exceed the action limit;
- make a collateral write adjacent to the intended path.

A reverted unauthorized write remains a violation because the verifier uses successful-write
history, not only final state.

## Current automated result

All automated gates pass with zero executable findings. The current report includes 576
solver replays, 3,456 secret rerenders, 288 reference agent trajectories, 48 write/revert
attacks, and 96 each of post-submit, action-limit, and collateral-write attacks.

Read the exact hash-bound report in the repository before citing numbers. The release status
remains pending external gates.
