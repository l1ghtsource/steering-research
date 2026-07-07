---
icon: lucide/wrench
---

# Setup

## Clone

```bash
git clone --recurse-submodules https://github.com/l1ghtsource/steering-research.git
cd steering-research
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Local development

```bash
uv sync --extra dev --extra model --extra training --extra docs
```

This installs:

- project package;
- static checks;
- Qwen model dependencies;
- PEFT training dependencies;
- Zensical docs builder.

## Smoke commands

```bash
uv run steering validate-data
uv run steering smoke --backend fake
uv run steering smoke-real --examples 10
uv run steering verify-runs --runs-root runs
uv run zensical build --clean --strict
```

## Static gates

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
```

CI intentionally does not run tests. Tests are local smoke/development gates.

