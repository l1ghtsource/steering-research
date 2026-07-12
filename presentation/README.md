# Pre-defense presentation

The presentation is a self-contained Beamer deck covering the research problem,
validated Qwen3.5-2B causal results, reproducibility stack, and H200 scale-up
plan.

## Build

From the repository root:

```bash
tectonic -X compile presentation/predefense.tex --outdir output/pdf
```

The expected artifact is:

```text
output/pdf/predefense.pdf
```

Before presenting, update `\PresenterName` and `\Affiliation` near the top of
`predefense.tex`.

## Content policy

All quantitative claims are sourced from committed Qwen3.5-2B forced-choice and
calibration artifacts. The main deck reports only source-backed series that
satisfy the same-prompt, nonempty-completion evaluation contract.
