# E001 Mean Direction

## Summary

- `experiment`: full_qwen35_2b_e001_mean_direction
- `backend`: qwen35_2b_base
- `rows`: 165
- `best_behavior`: hallucination
- `best_origin`: source_backed_contrasts
- `best_layer`: 0
- `best_activation_view`: assistant_answer_mean
- `best_direction_accuracy`: 1.0

## Metrics

| experiment | behavior | origin | layer | activation_view | n_train_pairs | n_eval_pairs | direction_accuracy | mean_projection_gap | median_projection_gap | mean_abs_margin | mean_positive_projection | mean_negative_projection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 0 | last_prompt_token | 140 | 60.0 | 0.0 | 0.0 | 0.0 | 0.0 | -0.011243322949437742 | -0.011243322949437742 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 0 | first_assistant_token | 140 | 60.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.009784064564301614 | 0.009784064564301614 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 0 | assistant_answer_mean | 140 | 60.0 | 1.0 | 0.2664360839654677 | 0.27075635605248005 | 0.2664360839654677 | 0.2479523985688444 | -0.0184836853966233 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 6 | last_prompt_token | 140 | 60.0 | 0.21666666666666667 | 0.00013482121168159795 | 0.0 | 0.001007499730741307 | 0.6588599994266767 | 0.6587251782149952 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 6 | first_assistant_token | 140 | 60.0 | 0.16666666666666666 | -0.00011297774281233963 | 0.0 | 0.0015410557291394094 | -1.1738370884757836 | -1.1737241107329714 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 6 | assistant_answer_mean | 140 | 60.0 | 0.9666666666666667 | 1.048814112260246 | 1.0625114476666928 | 1.0544483106978055 | 0.488820325860504 | -0.559993786399742 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 12 | last_prompt_token | 140 | 60.0 | 0.2 | 0.0002304540518371275 | 0.0 | 0.0018437248691301718 | -0.8715007055207068 | -0.871731159572544 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 12 | first_assistant_token | 140 | 60.0 | 0.23333333333333334 | 0.00033345086825876203 | 0.0 | 0.0017646596528833815 | -0.38605905026755 | -0.38639250113580875 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 12 | assistant_answer_mean | 140 | 60.0 | 0.9833333333333333 | 1.7917158008965588 | 1.8393955335043088 | 1.8017533029363675 | 0.3703467146478079 | -1.421369086248751 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 18 | last_prompt_token | 140 | 60.0 | 0.16666666666666666 | -0.00035259632697396775 | 0.0 | 0.014183506049116472 | 0.9067403898248912 | 0.9070929861518651 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 18 | first_assistant_token | 140 | 60.0 | 0.26666666666666666 | 0.004870192851075451 | 0.0 | 0.010551986267067456 | 1.6751057531951836 | 1.6702355603441077 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 18 | assistant_answer_mean | 140 | 60.0 | 0.9833333333333333 | 4.839105649406047 | 4.989256325561543 | 4.844737721103575 | 0.47639161880591707 | -4.362714030600128 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 23 | last_prompt_token | 140 | 60.0 | 0.18333333333333332 | 0.012600228890001211 | 0.0 | 0.05034250533718148 | -5.038587092130878 | -5.051187321020879 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 23 | first_assistant_token | 140 | 60.0 | 0.18333333333333332 | 0.017108385049269 | 0.0 | 0.05958845483246243 | -5.857093988857209 | -5.874202373906479 |
| full_qwen35_2b_e001_mean_direction | hallucination | source_backed_contrasts | 23 | assistant_answer_mean | 140 | 60.0 | 1.0 | 35.911587916909774 | 37.01082326495036 | 35.911587916909774 | 2.637601704535342 | -33.27398621237443 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 0 | last_prompt_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 0 | first_assistant_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 0 | assistant_answer_mean | 70 | 30.0 | 1.0 | 0.394490521284975 | 0.39498818719172446 | 0.394490521284975 | -0.26187870900048366 | -0.6563692302854586 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 6 | last_prompt_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 6 | first_assistant_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 6 | assistant_answer_mean | 70 | 30.0 | 1.0 | 1.4894987789016192 | 1.4916573587863795 | 1.4894987789016192 | 0.8439664812602384 | -0.6455322976413808 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 12 | last_prompt_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 12 | first_assistant_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 12 | assistant_answer_mean | 70 | 30.0 | 1.0 | 2.297584069493484 | 2.3005238743886682 | 2.297584069493484 | 1.4222419407135158 | -0.8753421287799678 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 18 | last_prompt_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 18 | first_assistant_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 18 | assistant_answer_mean | 70 | 30.0 | 1.0 | 7.63944254114466 | 7.617763842994851 | 7.63944254114466 | 6.762520596522749 | -0.8769219446219106 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 23 | last_prompt_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 23 | first_assistant_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | hallucination | synthetic_contrasts | 23 | assistant_answer_mean | 70 | 30.0 | 1.0 | 43.80989681281736 | 43.87973298787766 | 43.80989681281736 | 25.738806651900067 | -18.071090160917297 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 0 | last_prompt_token | 49 | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 0 | first_assistant_token | 49 | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 0 | assistant_answer_mean | 49 | 21.0 | 0.7142857142857143 | 0.005056956180325947 | 0.00037658685135427694 | 0.005977826950966406 | -0.13926824092439824 | -0.14432519710472413 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 6 | last_prompt_token | 49 | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 6 | first_assistant_token | 49 | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 6 | assistant_answer_mean | 49 | 21.0 | 0.42857142857142855 | 0.010095978476690894 | -0.003639465956255722 | 0.021491228791108343 | -0.05046412460941481 | -0.0605601030861057 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 12 | last_prompt_token | 49 | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 12 | first_assistant_token | 49 | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 12 | assistant_answer_mean | 49 | 21.0 | 0.2857142857142857 | -0.003415306657161463 | -0.018898223050704877 | 0.049494965612881255 | 0.15854599370736058 | 0.16196130036452203 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 18 | last_prompt_token | 49 | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 18 | first_assistant_token | 49 | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 18 | assistant_answer_mean | 49 | 21.0 | 0.9523809523809523 | 0.29871358134462667 | 0.23145419848767834 | 0.2995484884928137 | -0.012757597818245285 | -0.31147117916287187 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 23 | last_prompt_token | 49 | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 23 | first_assistant_token | 49 | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | source_backed_contrasts | 23 | assistant_answer_mean | 49 | 21.0 | 0.9047619047619048 | 1.7772847296115684 | 1.5155881590276294 | 1.8976617932780246 | -14.132140942596488 | -15.909425672208059 |
| full_qwen35_2b_e001_mean_direction | sycophancy | synthetic_contrasts | 0 | last_prompt_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | synthetic_contrasts | 0 | first_assistant_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | synthetic_contrasts | 0 | assistant_answer_mean | 70 | 30.0 | 1.0 | 0.3812200926030618 | 0.3804269719350556 | 0.3812200926030618 | 0.217187189524756 | -0.16403290307830568 |
| full_qwen35_2b_e001_mean_direction | sycophancy | synthetic_contrasts | 6 | last_prompt_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| full_qwen35_2b_e001_mean_direction | sycophancy | synthetic_contrasts | 6 | first_assistant_token | 70 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
