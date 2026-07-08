# E011 Orthogonalized Steering

## Summary

- `experiment`: phase2_qwen35_2b_e011_orthogonalized_steering
- `backend`: qwen35_2b_base
- `behavior`: sycophancy
- `rows`: 2800
- `control_behaviors`: ['hallucination', 'premature_refusal', 'unsafe_planning']
- `n_controls_used`: 3

## Metrics

| variant | alpha | n | mean_refusal_marker | mean_agreement_marker | mean_uncertainty_marker | mean_unsafe_planning_marker | mean_length_tokens | mean_repetition_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orthogonalized | -1.0 | 280 | 0.0 | 0.14642857142857144 | 0.0 | 0.0 | 74.65357142857142 | 0.16392517514775398 |
| orthogonalized | -2.0 | 280 | 0.0 | 0.14642857142857144 | 0.0 | 0.0 | 74.51071428571429 | 0.1555832363945767 |
| orthogonalized | 0.0 | 280 | 0.0 | 0.15357142857142858 | 0.0 | 0.0 | 75.10357142857143 | 0.17355835948593473 |
| orthogonalized | 1.0 | 280 | 0.0 | 0.15 | 0.0 | 0.0 | 75.65 | 0.18840828613815347 |
| orthogonalized | 2.0 | 280 | 0.0 | 0.16071428571428573 | 0.0 | 0.0 | 76.38928571428572 | 0.20717168491150068 |
| raw | -1.0 | 280 | 0.0 | 0.14642857142857144 | 0.0 | 0.0 | 74.61428571428571 | 0.16123216963495426 |
| raw | -2.0 | 280 | 0.0 | 0.14642857142857144 | 0.0 | 0.0 | 74.52857142857142 | 0.15565453931312373 |
| raw | 0.0 | 280 | 0.0 | 0.15357142857142858 | 0.0 | 0.0 | 75.10357142857143 | 0.17355835948593473 |
| raw | 1.0 | 280 | 0.0 | 0.15 | 0.0 | 0.0 | 75.61785714285715 | 0.19095849118687336 |
| raw | 2.0 | 280 | 0.0 | 0.15714285714285714 | 0.0 | 0.0 | 76.325 | 0.20313697009664564 |
