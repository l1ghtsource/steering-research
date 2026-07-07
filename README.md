# steering-research

Activation steering research harness for LatentBehaviorBench, Qwen3.5, and
Qwen-Scope sparse autoencoders.

## Smoke

```bash
uv sync --extra dev --extra model --extra training --extra docs
uv run steering smoke --backend fake
uv run steering smoke-real --examples 10
uv run steering verify-runs --runs-root runs
uv run ruff check
uv run ty check
uv run pytest
uv run zensical build --clean --strict
```

## Real Qwen3.5 2B

```bash
uv sync --extra dev --extra model --extra training --extra docs
uv run steering smoke-real --examples 10
```

The benchmark is mounted at `external/LatentBehaviorBench` as a git submodule.

## Training

```bash
uv sync --extra model --extra training
uv run steering e006-lora-sft
```

## Docs

Docs are built with Zensical and deployed to GitHub Pages by CI:

```bash
uv run zensical build --clean --strict
```
