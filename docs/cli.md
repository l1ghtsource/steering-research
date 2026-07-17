# CLI reference

The `pib` executable is installed by the project package.

```bash
uv run pib --help
```

## `pib build`

Deterministically constructs canonical families, rendered conditions, source catalog,
manifest, preregistration template, and content hashes.

```bash
uv run pib build
uv run pib build --output /tmp/pib-build
```

The default output is `data/`. A custom output receives copies of the normative schemas and
can be validated independently.

## `pib validate`

Runs schemas, item/family hashes, solver replay, renderer replay, split isolation, shortcut
baselines, agent-runtime trajectories, and release gates.

```bash
uv run pib validate
uv run pib validate --data /tmp/pib-build
uv run pib validate --write-artifacts
```

`--write-artifacts` refreshes `reports/candidate_validation.{json,md}` and blinded reviewer
packets.

## `pib materialize`

Creates a secret-seed Track E view. Both output directories must not exist and must be
different, non-nested paths.

```bash
uv run pib materialize \
  --data data \
  --split validation \
  --seed 123456789 \
  --input-output /tmp/pib-inputs \
  --key-output /tmp/pib-key
```

The command prints input/key hashes and a hash of the render seed. It does not print the
seed into model inputs.

## `pib calibrate`

Runs an optional forced-choice next-token diagnostic over Track E using a local Transformers
checkpoint.

```bash
uv run pib calibrate MODEL_PATH \
  --report runs/calibration.json \
  --calibration-label LABEL \
  [--splits dev validation] \
  [--conditions neutral ...] \
  [--limit-families N] \
  [--batch-size N] \
  [--device auto|cpu|mps|cuda] \
  [--dtype auto|float32|bfloat16|float16]
```

The report includes per-item label probabilities, condition metrics, solver strata,
neutral-capable families, stakeholder adoption, and paired estimands. It is an exploratory
difficulty check, not the registered post-trained free-generation protocol.

## Python API

```python
from pathlib import Path

from policy_integrity_bench import build, run_audit

manifest = build(Path("/tmp/pib-build"))
report = run_audit(Path("/tmp/pib-build"))
assert not report["findings"]
```
