---
icon: lucide/database
---

# Dataset

The benchmark is mounted as a git submodule:

```text
external/LatentBehaviorBench
```

The repository uses only processed benchmark artifacts for experiments. It does
not mutate the benchmark submodule during normal runs.

## Role in the Research Design

LatentBehaviorBench is used as a source of contrastive evidence rather than as a
generic prompt collection. The key object is a pair of examples that share a
behavior axis and often share much of their prompt context, but differ in
whether the undesirable behavior is present.

This contrastive structure lets the repository estimate behavior-specific
activation differences:

```text
h_positive - h_negative
```

The same pairs support three different uses:

- CAA direction construction;
- Qwen-Scope feature-delta ranking;
- supervised rows for the LoRA SFT baseline.

Evaluation must then happen on held-out clean buckets rather than on the same
pairs used for discovery.

## Files consumed

| File | Purpose |
| --- | --- |
| `processed/examples.jsonl` | Atomic prompts, labels, references, metadata |
| `processed/contrasts.jsonl` | Positive/negative contrast groups |
| `processed/eval_splits_clean.json` | Clean extraction/eval/control split map |
| `processed/stats.json` | Dataset-level counts |

## Positive and negative convention

For every behavior axis:

```text
positive = undesirable behavior present
negative = undesirable behavior absent, corrected, calibrated, or safely refused
```

CAA and monitor directions are therefore built as:

```text
v_behavior = mean(h_positive - h_negative)
```

Suppressing an undesirable behavior normally means using a negative steering
coefficient for this direction.

## Behavior Axes

| Axis | What the positive side captures | What the negative side captures |
| --- | --- | --- |
| `hallucination` | unsupported, contradicted, or fabricated answer content | source-backed or abstaining answer |
| `sycophancy` | agreement with a false or leading user premise | calibrated correction or disagreement |
| `premature_refusal` | refusal despite benign answerability | helpful answer within policy |
| `deception` | misleading, concealed, or strategically false response | honest answer |
| `unsafe_planning` | concrete unsafe plan or facilitation | safe refusal, redirection, or high-level safety framing |
| `overconfidence` | excessive certainty under uncertainty | calibrated uncertainty or qualification |

## Clean split policy

Use `processed/eval_splits_clean.json` for all claims. The loader treats this
file as first-class and exposes:

- extraction source-backed contrasts;
- extraction synthetic contrasts;
- clean held-out eval buckets;
- clean capability controls;
- clean safety controls.

!!! danger "Do not mix extraction and held-out evidence"

    Extraction contrasts are for feature discovery and steering vector
    construction. Clean held-out examples are for evaluation. Non-held-out
    buckets are diagnostics only.

## Origins

The `origin` field separates evidence sources:

| Origin | Use |
| --- | --- |
| `source_backed_contrasts` | primary evidence for grounded behavior claims |
| `synthetic_contrasts` | coverage expansion, diagnostics, and stress testing |

Primary reports should keep these origins separate. Combining them can improve
sample size, but it can also hide a method that works on templated synthetic
examples and fails source-backed examples.

## Evaluation Buckets

The clean split file defines the buckets used by experiments. The practical
discipline is:

- extraction buckets build directions, sparse feature rankings, and adapters;
- external clean buckets evaluate behavior transfer;
- capability controls check whether general answer quality is harmed;
- safety controls check whether reducing one behavior worsens another.

## Current benchmark caveats

The repository records these limitations directly in reports:

- overconfidence extraction is synthetic-heavy;
- unsafe planning lacks full real tool-use trajectories;
- source-prompt synthetic completions are deterministic templates;
- multilingual coverage is currently absent;
- license status should be checked before public release.
