# E015 Layer-Fraction Transfer

## Summary

- `experiment`: phase2_qwen35_2b_e015_layer_transfer
- `backend`: qwen35_2b_base
- `behavior`: sycophancy
- `rows`: 25
- `best_source_layer`: 0
- `best_target_layer`: 0
- `best_direction_accuracy`: 1.0

## Metrics

| experiment | behavior | origin | source_layer | target_layer | activation_view | n_eval_pairs | direction_accuracy | mean_projection_gap | median_projection_gap | mean_abs_margin | mean_positive_projection | mean_negative_projection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 0 | 0 | assistant_answer_mean | 70.0 | 1.0 | 0.07621847439106719 | 0.10400036192708195 | 0.07621847439106719 | -0.0840944985552903 | -0.1603129729463575 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 0 | 6 | assistant_answer_mean | 70.0 | 0.9 | 0.0889051942345717 | 0.12236485311694177 | 0.09119243285491635 | -0.17643753682213587 | -0.2653427310567075 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 0 | 12 | assistant_answer_mean | 70.0 | 0.8 | 0.08737578921285659 | 0.12386721597029304 | 0.09396849364391668 | -0.2072084375533253 | -0.29458422676618196 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 0 | 18 | assistant_answer_mean | 70.0 | 0.8714285714285714 | 0.29045564575360877 | 0.39702134734422334 | 0.30240598534456886 | -0.9055676321041541 | -1.196023277857763 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 0 | 23 | assistant_answer_mean | 70.0 | 0.8571428571428571 | 0.3803994473162692 | 0.5225509177761252 | 0.5011556418842869 | -0.43638015162952487 | -0.8167795989457941 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 6 | 0 | assistant_answer_mean | 70.0 | 0.8 | 0.03791609359533812 | 0.05298449066820799 | 0.03817409199980518 | -0.01135669519712579 | -0.04927278879246391 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 6 | 6 | assistant_answer_mean | 70.0 | 1.0 | 0.17871614998950505 | 0.21193800608543328 | 0.17871614998950505 | -0.23216630436738325 | -0.41088245435688825 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 6 | 12 | assistant_answer_mean | 70.0 | 0.9714285714285714 | 0.12596936317376098 | 0.15054566579501902 | 0.126543502885317 | -0.3878323107111713 | -0.5138016738849325 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 6 | 18 | assistant_answer_mean | 70.0 | 0.9857142857142858 | 0.28576824365729453 | 0.36394671233472087 | 0.2871300508476715 | -0.7760746775101095 | -1.0618429211674036 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 6 | 23 | assistant_answer_mean | 70.0 | 0.9714285714285714 | 0.6241004134603253 | 0.6028569941283435 | 0.6339991267221254 | 0.7418310498958848 | 0.11773063643555938 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 12 | 0 | assistant_answer_mean | 70.0 | 0.8714285714285714 | 0.01991215697358699 | 0.02751199856053689 | 0.020095467095329722 | 0.055497072792205414 | 0.03558491581861843 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 12 | 6 | assistant_answer_mean | 70.0 | 0.9857142857142858 | 0.06731249340550634 | 0.0803369467660949 | 0.06781903398161172 | -0.044437826963813695 | -0.11175032036932003 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 12 | 12 | assistant_answer_mean | 70.0 | 1.0 | 0.3344514289111553 | 0.3006799982538306 | 0.3344514289111553 | -0.07274865751850937 | -0.4072000864296646 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 12 | 18 | assistant_answer_mean | 70.0 | 1.0 | 0.4500720606900394 | 0.4170491014451971 | 0.4500720606900394 | -0.06352918425782206 | -0.5136012449478615 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 12 | 23 | assistant_answer_mean | 70.0 | 1.0 | 1.4807741998061974 | 1.262224357806589 | 1.4807741998061974 | -2.2197263646543375 | -3.700500564460535 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 18 | 0 | assistant_answer_mean | 70.0 | 0.7571428571428571 | 0.019005301162477964 | 0.026732186894614415 | 0.019641213824867332 | -0.13452014353482436 | -0.15352544469730237 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 18 | 6 | assistant_answer_mean | 70.0 | 0.9857142857142858 | 0.043844230018122606 | 0.05227161787577678 | 0.043849416170113606 | -0.38635831734076576 | -0.43020254735888847 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 18 | 12 | assistant_answer_mean | 70.0 | 1.0 | 0.12922596724237245 | 0.11806744082611201 | 0.12922596724237245 | -0.47844844265361974 | -0.6076744098959921 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 18 | 18 | assistant_answer_mean | 70.0 | 1.0 | 1.1648374318487191 | 0.8411332549604209 | 1.1648374318487191 | -1.4468423014782417 | -2.6116797333269606 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 18 | 23 | assistant_answer_mean | 70.0 | 1.0 | 4.562226796855148 | 3.254912031245909 | 4.562226796855148 | -2.14285878325022 | -6.705085580105369 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 23 | 0 | assistant_answer_mean | 70.0 | 0.8142857142857143 | 0.0034311371476584513 | 0.005056969209779773 | 0.003980356817008937 | -0.0008373428871110378 | -0.004268480034769489 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 23 | 6 | assistant_answer_mean | 70.0 | 0.9142857142857143 | 0.013199461672759162 | 0.01611462055231152 | 0.014301287154225656 | -0.01860768634008743 | -0.0318071480128466 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 23 | 12 | assistant_answer_mean | 70.0 | 1.0 | 0.05860839706540352 | 0.050189968245377976 | 0.05860839706540352 | 0.01572216355397495 | -0.04288623351142857 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 23 | 18 | assistant_answer_mean | 70.0 | 1.0 | 0.6288978908004909 | 0.4686202085853507 | 0.6288978908004909 | 0.20439803480659696 | -0.4244998559938939 |
| phase2_qwen35_2b_e015_layer_transfer | sycophancy | source_backed_contrasts | 23 | 23 | assistant_answer_mean | 70.0 | 1.0 | 8.450103941032348 | 6.6078741484604535 | 8.450103941032348 | -3.526463473845794 | -11.976567414878142 |
