---
icon: lucide/file-chart-column
---

# Run Artifacts

Each experiment creates a timestamped run directory under `runs/`.

Run artifacts are part of the research protocol. A run is only useful if it can
be audited after the fact: exact config, benchmark counts, metrics, generated
text, training metadata, and campaign dashboard must remain available together.

## Required files

| File | Required | Description |
| --- | --- | --- |
| `manifest.json` | yes | Config, backend, benchmark validation counts |
| `metrics.jsonl` | yes | Append-only metrics/events table |
| `summary.json` | yes | Small machine-readable summary |
| `report.md` | yes | Human-readable report |
| `run.log` | yes | Timestamped run log |

`steering verify-runs` checks these files and fails if any run is incomplete.

## Dashboard

The dashboard is static HTML only:

```text
runs/dashboard.html
```

It is rebuilt from `runs/*/summary.json` by:

```bash
uv run steering dashboard
```

No Streamlit or server process is required.

The dashboard is an overview, not the source of truth. Use it to find candidate
runs, then inspect `summary.json`, `metrics.jsonl`, `aggregate.json`, generation
tables, and `report.md` for the experiment-specific evidence.

## Metrics format

`metrics.jsonl` is deliberately plain JSONL so it can be consumed by:

- shell tools;
- Python notebooks;
- DuckDB;
- dashboard builders;
- artifact verifiers.

## Reading Results

Use the artifacts differently for each experiment family:

| Experiment | Primary readout | Manual review target |
| --- | --- | --- |
| E001 | best layer/view direction accuracy and projection gap | whether source-backed and synthetic results agree |
| E002 | AUROC and score gap | score distributions and class balance |
| E003 | top SAE feature deltas | feature stability across resamples or neighboring layers |
| E004 | alpha-level behavior markers | `tables/generations.csv` |
| E005 | selected feature and alpha response | generated text and top feature list |
| E006 | adapter path and training metrics | held-out behavior eval after loading adapter |
| E007 | entry-by-alpha best-layer response | generated text, length, and repetition shifts |
| E008 | source-target specificity matrix | strongest off-diagonal behavior confounds |
| E009 | variant-by-alpha causal controls | whether controls match the intended effect |
| E010 | feature-by-alpha causal sweep | top feature outputs and repetition side effects |
| E011 | raw versus orthogonalized steering | nuisance marker and style reduction |
| E012 | train-origin by eval-origin matrix | cross-origin sign and margin stability |
| E013 | always-on versus dynamic steering | gate firing rate and missed-risk examples |
| E014 | layer-group by alpha response | single-layer versus windowed hook side effects |
| E015 | source-layer by target-layer matrix | diagonal strength and late-layer transfer |
| E016 | alpha-level forced-choice preference margin | paired desirable versus undesirable answer scores |
| E017 | calibrated coefficient response | calibration scale and raw-alpha reconstruction |
| E018 | position-mode by alpha response | whether prompt, answer, or boundary positions matter |

## Campaign Summary

After a campaign, the written interpretation should include:

- exact model and SAE paths;
- behavior axes and origins;
- held-out bucket names;
- best dense layers and activation views;
- top sparse features;
- steering alpha ranges;
- whether behavior changes are monotonic;
- whether refusal, length, or generic style explains the result;
- comparison against LoRA SFT;
- specificity, control, origin-transfer, and layer-transfer conclusions;
- unresolved caveats.
