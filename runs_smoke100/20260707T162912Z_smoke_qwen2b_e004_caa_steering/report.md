# E004 Steering Eval

## Summary

- `experiment`: smoke_qwen2b_e004_caa_steering
- `backend`: qwen35_2b_base
- `behavior`: sycophancy
- `rows`: 200
- `alphas`: [-0.5, 0.0]

## Metrics

| alpha | n | mean_refusal_marker | mean_agreement_marker | mean_length_tokens |
| --- | --- | --- | --- | --- |
| -0.5 | 100 | 0.0 | 0.41 | 147.08 |
| 0.0 | 100 | 0.0 | 0.41 | 147.13 |
