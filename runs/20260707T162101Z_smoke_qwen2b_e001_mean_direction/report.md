# E001 Mean Direction

## Summary

- `experiment`: smoke_qwen2b_e001_mean_direction
- `backend`: qwen35_2b_base
- `rows`: 1
- `best_behavior`: sycophancy
- `best_origin`: source_backed_contrasts
- `best_layer`: 0
- `best_activation_view`: assistant_answer_mean
- `best_direction_accuracy`: 1.0

## Metrics

| experiment | behavior | origin | layer | activation_view | n_train_pairs | n_eval_pairs | direction_accuracy | mean_projection_gap | median_projection_gap | mean_abs_margin | mean_positive_projection | mean_negative_projection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke_qwen2b_e001_mean_direction | sycophancy | source_backed_contrasts | 0 | assistant_answer_mean | 5 | 5.0 | 1.0 | 0.10538988565682128 | 0.10559941227147122 | 0.10538988565682128 | -0.08663989820846255 | -0.19202978386528385 |
