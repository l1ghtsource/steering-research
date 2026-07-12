# E007 Best-Layer CAA Sweep

## Summary

- `experiment`: phase2_qwen35_2b_e007_best_layer_caa
- `backend`: qwen35_2b_base
- `rows`: 3080
- `entries`: 3
- `alphas`: [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

## Metrics

| entry | source_behavior | layer | activation_view | alpha | n | mean_refusal_marker | mean_agreement_marker | mean_uncertainty_marker | mean_unsafe_planning_marker | mean_length_tokens | mean_repetition_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| premature_refusal_best | premature_refusal | 12 | assistant_answer_mean | -1.0 | 90 | 0.0 | 0.0 | 0.0 | 0.0 | 30.988888888888887 | 0.16167249412975754 |
| premature_refusal_best | premature_refusal | 12 | assistant_answer_mean | -2.0 | 90 | 0.0 | 0.0 | 0.0 | 0.0 | 32.81111111111111 | 0.20377152261346124 |
| premature_refusal_best | premature_refusal | 12 | assistant_answer_mean | -4.0 | 90 | 0.0 | 0.0 | 0.0 | 0.0 | 35.955555555555556 | 0.22109937555870193 |
| premature_refusal_best | premature_refusal | 12 | assistant_answer_mean | 0.0 | 90 | 0.0 | 0.0 | 0.0 | 0.0 | 31.07777777777778 | 0.12524839451149125 |
| premature_refusal_best | premature_refusal | 12 | assistant_answer_mean | 1.0 | 90 | 0.0 | 0.0 | 0.0 | 0.0 | 35.111111111111114 | 0.15127398161613914 |
| premature_refusal_best | premature_refusal | 12 | assistant_answer_mean | 2.0 | 90 | 0.0 | 0.0 | 0.0 | 0.0 | 38.44444444444444 | 0.23418079071852982 |
| premature_refusal_best | premature_refusal | 12 | assistant_answer_mean | 4.0 | 90 | 0.0 | 0.0 | 0.0 | 0.0 | 46.98888888888889 | 0.2897430702247924 |
| sycophancy_best | sycophancy | 18 | assistant_answer_mean | -1.0 | 280 | 0.0 | 0.14642857142857144 | 0.0 | 0.0 | 74.61428571428571 | 0.16123216963495426 |
| sycophancy_best | sycophancy | 18 | assistant_answer_mean | -2.0 | 280 | 0.0 | 0.14642857142857144 | 0.0 | 0.0 | 74.52857142857142 | 0.15565453931312373 |
| sycophancy_best | sycophancy | 18 | assistant_answer_mean | -4.0 | 280 | 0.0 | 0.15 | 0.0 | 0.0 | 74.66785714285714 | 0.1533870626194805 |
| sycophancy_best | sycophancy | 18 | assistant_answer_mean | 0.0 | 280 | 0.0 | 0.15357142857142858 | 0.0 | 0.0 | 75.10357142857143 | 0.17355835948593473 |
| sycophancy_best | sycophancy | 18 | assistant_answer_mean | 1.0 | 280 | 0.0 | 0.15 | 0.0 | 0.0 | 75.61785714285715 | 0.19095849118687336 |
| sycophancy_best | sycophancy | 18 | assistant_answer_mean | 2.0 | 280 | 0.0 | 0.15714285714285714 | 0.0 | 0.0 | 76.325 | 0.20313697009664564 |
| sycophancy_best | sycophancy | 18 | assistant_answer_mean | 4.0 | 280 | 0.0 | 0.16785714285714284 | 0.0 | 0.0 | 79.96428571428571 | 0.30795635017698975 |
| unsafe_planning_best | unsafe_planning | 23 | assistant_answer_mean | -1.0 | 70 | 0.0 | 0.0 | 0.0 | 0.0 | 40.6 | 0.5157612605622877 |
| unsafe_planning_best | unsafe_planning | 23 | assistant_answer_mean | -2.0 | 70 | 0.0 | 0.0 | 0.0 | 0.0 | 40.628571428571426 | 0.516823531624559 |
| unsafe_planning_best | unsafe_planning | 23 | assistant_answer_mean | -4.0 | 70 | 0.0 | 0.0 | 0.0 | 0.0 | 40.9 | 0.5235633878316808 |
| unsafe_planning_best | unsafe_planning | 23 | assistant_answer_mean | 0.0 | 70 | 0.0 | 0.0 | 0.0 | 0.0 | 40.48571428571429 | 0.4999272571673088 |
| unsafe_planning_best | unsafe_planning | 23 | assistant_answer_mean | 1.0 | 70 | 0.0 | 0.0 | 0.0 | 0.0 | 40.48571428571429 | 0.5006241212788071 |
| unsafe_planning_best | unsafe_planning | 23 | assistant_answer_mean | 2.0 | 70 | 0.0 | 0.0 | 0.0 | 0.0 | 40.628571428571426 | 0.4945975442940418 |
| unsafe_planning_best | unsafe_planning | 23 | assistant_answer_mean | 4.0 | 70 | 0.0 | 0.0 | 0.0 | 0.0 | 40.8 | 0.45675497262849385 |
