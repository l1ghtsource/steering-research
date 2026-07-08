# E013 Dynamic Steering

## Summary

- `experiment`: phase2_qwen35_2b_e013_dynamic_steering
- `backend`: qwen35_2b_base
- `behavior`: sycophancy
- `rows`: 1680
- `threshold`: -1.1201097461764848

## Metrics

| variant | alpha | n | mean_refusal_marker | mean_agreement_marker | mean_uncertainty_marker | mean_unsafe_planning_marker | mean_length_tokens | mean_repetition_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always | -2.0 | 280 | 0.0 | 0.14642857142857144 | 0.0 | 0.0 | 74.52857142857142 | 0.15565453931312373 |
| always | 0.0 | 280 | 0.0 | 0.15357142857142858 | 0.0 | 0.0 | 75.10357142857143 | 0.17355835948593473 |
| always | 2.0 | 280 | 0.0 | 0.15714285714285714 | 0.0 | 0.0 | 76.325 | 0.20313697009664564 |
| dynamic | -2.0 | 280 | 0.0 | 0.15357142857142858 | 0.0 | 0.0 | 75.10357142857143 | 0.17355835948593473 |
| dynamic | 0.0 | 280 | 0.0 | 0.15357142857142858 | 0.0 | 0.0 | 75.10357142857143 | 0.17355835948593473 |
| dynamic | 2.0 | 280 | 0.0 | 0.15357142857142858 | 0.0 | 0.0 | 75.10357142857143 | 0.17355835948593473 |
