# Pre-defense presentation

The presentation is a self-contained Beamer deck covering the research problem,
the current Qwen3.5-2B score-level intervention evidence, its validity limits,
and the controlled experiments required before scale-up.

## Build

From the repository root:

```bash
tectonic -X compile presentation/predefense.tex --outdir output/pdf
```

The expected artifact is:

```text
output/pdf/predefense.pdf
```

## Content policy

Quantitative claims are sourced from the Qwen3.5-2B E016/E017 artifacts. The
deck labels them as exploratory fixed-pair score effects: the current ordered
30% extraction partition is not an untouched final test, and the answer pairs
retain source-model, style, format, or template confounds. Open-generation
behavior and cross-model generalization remain untested.
