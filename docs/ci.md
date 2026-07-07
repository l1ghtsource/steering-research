---
icon: lucide/git-pull-request
---

# CI and GitHub Pages

CI has two jobs:

1. `static`: install dev dependencies and run `ruff` + `ty`.
2. `pages`: build the Zensical site and deploy it to GitHub Pages.

Experiment execution is intentionally not part of CI because Qwen and
Qwen-Scope runs depend on large model artifacts, GPU placement, and campaign
configs. CI stays focused on static correctness and documentation publishing.

## Pages build

The workflow configures Pages with `enablement: true`, so the first successful
push to `main` can create the Pages site and set its source to GitHub Actions.
If the repository or organization blocks workflow-managed Pages setup, enable it
once in GitHub: `Settings -> Pages -> Source -> GitHub Actions`.

The docs job runs:

```bash
uv sync --extra docs
uv run zensical build --clean --strict
```

The generated site is uploaded from:

```text
site/
```
