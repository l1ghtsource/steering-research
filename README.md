# PolicyIntegrityBench

[Full documentation](https://l1ghtsource.github.io/steering-research/) | [Benchmark card](data/benchmark_card.md) | [Validation report](data/reports/candidate_validation.md) | [Research protocol](ideas/benchmark_spec.md)

PolicyIntegrityBench is a solver-backed benchmark for measuring whether a language model
preserves an evidence-grounded or explicitly authorized policy when contextual pressure
favors another answer or action.

The benchmark separates three questions that are often conflated:

- **Capability:** can the model solve the neutral task?
- **Integrity:** does a neutral-capable model change its answer or action under conflicting
  suggestion, social pressure, or proxy incentives?
- **Authorization sensitivity:** does an intervention suppress only unauthorized behavior,
  while preserving the same action when it is explicitly authorized?

Primary outcomes are computed by deterministic solvers and executable state predicates.
No LLM judge determines benchmark labels.

## Dataset

| Track | Unit | Families | Variants | Items | Verification |
|---|---|---:|---:|---:|---|
| Track E | Evidence-grounded decision | 72 | 8 | 576 | Constraint and interval solvers |
| Track A | Agentic workspace task | 24 | 4 | 96 | Executable final-state predicates |

Track E contains neutral, insufficient-evidence, aligned, and conflicting stakeholder
conditions. Track A contains clean, temptation, proxy-pressure, and explicitly authorized
counterfactual environments. Every variant from one family remains in the same split.

The checked-in dataset is under [`data/`](data/):

```text
data/
├── dev/                 # method development and feature selection
├── validation/          # generator-group-held-out evaluation
├── schema/              # JSON Schemas for families and items
├── manifests/           # dataset hashes and preregistration template
├── reports/             # executable validation and calibration reports
├── reviewer_packets/    # blinded external semantic-review forms
└── authoring/            # deterministic source catalog
```

## Quick start

Install the benchmark and developer tooling with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pib --help
uv run pib validate
```

Rebuild every generated family, item, manifest, and hash:

```bash
uv run pib build
uv run pib validate --write-artifacts
```

Run repository checks:

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
```

## Leak-resistant evaluation

The public JSONL files are intended for development and transparent audit. For an actual
held-out run, create a secret-seed rendering with physically separate inference and scoring
directories:

```bash
uv run pib materialize \
  --split validation \
  --seed "$SECRET_RENDER_SEED" \
  --input-output /inference-only/pib-validation \
  --key-output /scoring-only/pib-validation-key
```

Inference rows contain only `eval_id`, `prompt`, and `prompt_hash`. The key contains family,
condition, solver, and target metadata. Workers must be frozen before corpus access and
stateless across `eval_id`s. Cross-row fitting, retrieval, clustering, normalization, or
caching is transductive adaptation and invalidates a held-out claim.

## Optional model calibration

Install model dependencies and run the forced-choice diagnostic on a local Hugging Face
checkpoint:

```bash
uv sync --extra model
uv run pib calibrate /models/Qwen3.5-2B-Base \
  --report runs/qwen35_2b_base.json \
  --calibration-label qwen35_2b_base_diagnostic \
  --device auto \
  --dtype auto
```

This diagnostic checks task difficulty and pressure sensitivity. It does not replace the
registered free-generation and executable-agent calibration protocol.

## Scientific status

The current public release passes schema validation, deterministic solver replay, split
isolation, static shortcut baselines, and adversarial Track A runtime checks. It is suitable
for software integration and exploratory research.

It is not a sealed confirmatory test set. Independent semantic review, post-trained model
calibration, and independently authored final families remain required. See
[`data/research_readiness.md`](data/research_readiness.md) for the exact release decision and
open gates.

## Development

```bash
uv sync --extra dev --extra docs
uv run pre-commit install
uv run zensical serve
```

CI runs Ruff, ty, and a strict Zensical build. Tests remain a local release check because
the benchmark includes generated data and executable validation beyond ordinary unit tests.

## License

MIT.
