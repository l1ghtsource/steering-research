# PolicyIntegrityBench

PolicyIntegrityBench is a controlled benchmark for testing whether a language model keeps
an evidence-grounded conclusion or an explicitly authorized objective when another
contextual signal favors a different answer or action.

The benchmark is designed for behavioral evaluation, representation analysis, activation
steering, sparse feature studies, and transfer experiments. It does not collapse alignment
into one scalar. Capability, integrity, calibration, expression validity, and authorization
sensitivity remain separate outcomes.

## What is measured

### Epistemic integrity

Track E presents self-contained evidence and an exact decision rule. The same underlying
case is rendered under neutral context, aligned advice, conflicting advice, aligned social
pressure, conflicting social pressure, and matched insufficient evidence.

The central question is not simply whether the answer is correct. It is whether a model that
solves the neutral case preserves that answer when a stakeholder favors a false alternative.

### Agentic integrity

Track A presents a deterministic JSON workspace, an authorized task, and a writable proxy
route. The proxy is unauthorized in temptation and pressure conditions, but the exact same
action becomes necessary and explicitly authorized in a counterfactual condition.

This separates targeted integrity from blanket conservatism. A method that blocks every
proxy-looking action also fails the authorized control.

## Benchmark at a glance

| Track | Families | Conditions | Items | Ground truth |
|---|---:|---:|---:|---|
| Evidence-grounded decisions | 72 | 8 | 576 | Constraint and interval solvers |
| Executable agent tasks | 24 | 4 | 96 | State and action-history predicates |

All conditions from a canonical family remain in one split. Validation holds out complete
generator groups rather than random rendered prompts.

## Design principles

1. **Executable labels.** Stored answers are never trusted during validation.
2. **Paired counterfactuals.** Evidence or authorization is held fixed while contextual
   pressure changes.
3. **Capability qualification.** Integrity is evaluated among families solved in the
   neutral condition.
4. **No primary LLM judge.** Decisions, citations, state changes, and violations have exact
   programmatic scores.
5. **Artifact controls.** Answer positions, pressure targets, missingness, and authoring
   templates are counterbalanced and tested with group-held-out shortcut models.
6. **Explicit scientific limits.** Public data are exploratory; confirmatory claims require
   independently authored sealed families.

## Current status

The checked-in candidate passes schemas, hashes, solver replay, split isolation, static
shortcut baselines, prompt rerendering, and adversarial agent-runtime checks. External
semantic review, post-trained model calibration, and a sealed final set remain open gates.

Start with [Getting started](getting-started.md), then read the
[evaluation protocol](evaluation.md) before running a model or fitting a representation.
