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

## Current benchmark caveats

The repository records these limitations directly in reports:

- overconfidence extraction is synthetic-heavy;
- unsafe planning lacks full real tool-use trajectories;
- source-prompt synthetic completions are deterministic templates;
- multilingual coverage is currently absent;
- license status should be checked before public release.

