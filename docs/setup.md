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

## Environment

```bash
uv sync --extra dev --extra model --extra training --extra docs
```

This installs:

- project package;
- static checks;
- Qwen model dependencies;
- PEFT training dependencies;
- Zensical docs builder.

For an offline H200 server, install through the internal PyPI mirror and keep
Hugging Face loaders in offline mode:

```bash
export HF_HOME=/data/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Dataset check

```bash
uv run steering validate-data
```

This confirms that the LatentBehaviorBench submodule exposes the processed
examples, contrast pairs, and clean split file required by the experiments.

## Quality gates

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
uv run zensical build --clean --strict
```

CI runs the static gates and builds the Zensical site. Model execution and
training runs are launched explicitly from experiment configs.
