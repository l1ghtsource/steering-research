# E004 Steering Eval

## Summary

- `experiment`: smoke_qwen2b_e004_caa_steering
- `backend`: qwen35_2b_base
- `behavior`: sycophancy
- `rows`: 20
- `alphas`: [-0.5, 0.0]

## Metrics

| alpha | n | mean_refusal_marker | mean_agreement_marker | mean_length_tokens |
| --- | --- | --- | --- | --- |
| -0.5 | 10 | 0.0 | 0.4 | 147.9 |
| 0.0 | 10 | 0.0 | 0.4 | 148.0 |
