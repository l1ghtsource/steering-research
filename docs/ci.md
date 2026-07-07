---
icon: lucide/git-pull-request
---

# CI and GitHub Pages

CI has two jobs:

1. `static`: install dev dependencies and run `ruff` + `ty`.
2. `pages`: build the Zensical site and deploy it to GitHub Pages.

Tests are intentionally not run in CI because local Qwen/SAE smoke tests are
resource-sensitive and are run manually.

## Pages build

The docs job runs:

```bash
uv sync --extra docs
uv run zensical build --clean --strict
```

The generated site is uploaded from:

```text
site/
```

