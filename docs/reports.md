---
icon: lucide/file-chart-column
---

# Run Artifacts

Each experiment creates a timestamped run directory under `runs/`.

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

## Metrics format

`metrics.jsonl` is deliberately plain JSONL so it can be consumed by:

- shell tools;
- Python notebooks;
- DuckDB;
- dashboard builders;
- CI smoke verifiers.

