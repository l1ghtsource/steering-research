# E012 Origin Transfer

## Summary

- `experiment`: phase2_qwen35_2b_e012_origin_transfer
- `backend`: qwen35_2b_base
- `rows`: 21

## Metrics

| experiment | behavior | train_origin | eval_origin | layer | activation_view | n_eval_pairs | direction_accuracy | mean_projection_gap | median_projection_gap | mean_abs_margin | mean_positive_projection | mean_negative_projection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase2_qwen35_2b_e012_origin_transfer | hallucination | source_backed_contrasts | source_backed_contrasts | 18 | assistant_answer_mean | 200.0 | 0.985 | 4.980832017327573 | 4.946382112076666 | 4.992619258275455 | 0.3181180214714404 | -4.662713995856132 |
| phase2_qwen35_2b_e012_origin_transfer | hallucination | source_backed_contrasts | synthetic_contrasts | 18 | assistant_answer_mean | 100.0 | 0.0 | -0.5875371382933633 | -0.600386510916183 | 0.5875371382933633 | -0.3662701526168152 | 0.22126698567654818 |
| phase2_qwen35_2b_e012_origin_transfer | hallucination | synthetic_contrasts | source_backed_contrasts | 18 | assistant_answer_mean | 200.0 | 0.2 | -0.3832035381569143 | -0.4069885618837781 | 0.4888434148437011 | 3.086132099504555 | 3.469335637661469 |
| phase2_qwen35_2b_e012_origin_transfer | hallucination | synthetic_contrasts | synthetic_contrasts | 18 | assistant_answer_mean | 100.0 | 1.0 | 7.63673478552875 | 7.61495728943402 | 7.63673478552875 | 6.77569947518488 | -0.8610353103438702 |
| phase2_qwen35_2b_e012_origin_transfer | sycophancy | source_backed_contrasts | source_backed_contrasts | 18 | assistant_answer_mean | 70.0 | 1.0 | 1.1648374318487191 | 0.8411332549604209 | 1.1648374318487191 | -1.4468423014782417 | -2.6116797333269606 |
| phase2_qwen35_2b_e012_origin_transfer | sycophancy | source_backed_contrasts | synthetic_contrasts | 18 | assistant_answer_mean | 100.0 | 1.0 | 0.15189552328132871 | 0.15952831206128337 | 0.15189552328132871 | -1.9147713205098265 | -2.066666843791155 |
| phase2_qwen35_2b_e012_origin_transfer | sycophancy | synthetic_contrasts | source_backed_contrasts | 18 | assistant_answer_mean | 70.0 | 0.4 | 0.028033358385666073 | -0.0015034924505297198 | 0.04786499425848206 | 0.9735278889892868 | 0.9454945306036207 |
| phase2_qwen35_2b_e012_origin_transfer | sycophancy | synthetic_contrasts | synthetic_contrasts | 18 | assistant_answer_mean | 100.0 | 1.0 | 6.311537448142815 | 6.320417979249682 | 6.311537448142815 | 4.198221860613777 | -2.113315587529038 |
| phase2_qwen35_2b_e012_origin_transfer | premature_refusal | source_backed_contrasts | source_backed_contrasts | 18 | assistant_answer_mean | 100.0 | 1.0 | 4.839192711800348 | 5.310549425434277 | 4.839192711800348 | 5.549429703161363 | 0.7102369913610153 |
| phase2_qwen35_2b_e012_origin_transfer | premature_refusal | source_backed_contrasts | synthetic_contrasts | 18 | assistant_answer_mean | 75.0 | 1.0 | 2.236860442419926 | 2.2101489490855832 | 2.236860442419926 | 4.308065900387965 | 2.07120545796804 |
| phase2_qwen35_2b_e012_origin_transfer | premature_refusal | synthetic_contrasts | source_backed_contrasts | 18 | assistant_answer_mean | 100.0 | 0.91 | 1.1557063695130008 | 1.1807431998458013 | 1.2031448516582193 | 1.5473165533435327 | 0.3916101838305319 |
| phase2_qwen35_2b_e012_origin_transfer | premature_refusal | synthetic_contrasts | synthetic_contrasts | 18 | assistant_answer_mean | 75.0 | 1.0 | 9.366218821511165 | 9.343945728780014 | 9.366218821511165 | 5.504111051482652 | -3.8621077700285102 |
| phase2_qwen35_2b_e012_origin_transfer | deception | source_backed_contrasts | source_backed_contrasts | 18 | assistant_answer_mean | 80.0 | 0.825 | 1.6158436133219578 | 1.0657550679010828 | 1.8083218120104625 | 0.1133047132366353 | -1.5025389000853224 |
| phase2_qwen35_2b_e012_origin_transfer | deception | source_backed_contrasts | synthetic_contrasts | 18 | assistant_answer_mean | 100.0 | 1.0 | 1.3389358995816751 | 1.3373789097848443 | 1.3389358995816751 | -0.9614339044045185 | -2.300369803986194 |
| phase2_qwen35_2b_e012_origin_transfer | deception | synthetic_contrasts | source_backed_contrasts | 18 | assistant_answer_mean | 80.0 | 0.6625 | 0.38220106754014366 | 0.12280364981712694 | 0.49286884019036775 | 0.8868022707289057 | 0.5046012031887622 |
| phase2_qwen35_2b_e012_origin_transfer | deception | synthetic_contrasts | synthetic_contrasts | 18 | assistant_answer_mean | 100.0 | 1.0 | 5.66066190215259 | 5.717064445627546 | 5.66066190215259 | 2.7026857086344087 | -2.9579761935181805 |
| phase2_qwen35_2b_e012_origin_transfer | unsafe_planning | source_backed_contrasts | source_backed_contrasts | 18 | assistant_answer_mean | 50.0 | 1.0 | 5.4268772475459635 | 5.4406072484752315 | 5.4268772475459635 | 3.0678486800217963 | -2.3590285675241685 |
| phase2_qwen35_2b_e012_origin_transfer | unsafe_planning | source_backed_contrasts | synthetic_contrasts | 18 | assistant_answer_mean | 50.0 | 1.0 | 0.38395633859405026 | 0.36389358551314255 | 0.38395633859405026 | 1.9157975556365179 | 1.531841217042468 |
| phase2_qwen35_2b_e012_origin_transfer | unsafe_planning | synthetic_contrasts | source_backed_contrasts | 18 | assistant_answer_mean | 50.0 | 0.92 | 0.27211319465817707 | 0.26848272350817615 | 0.29379143960815246 | 2.77494809006285 | 2.502834895404673 |
| phase2_qwen35_2b_e012_origin_transfer | unsafe_planning | synthetic_contrasts | synthetic_contrasts | 18 | assistant_answer_mean | 50.0 | 1.0 | 7.657415953623955 | 7.669871629872583 | 7.657415953623955 | 5.224278446006906 | -2.433137507617049 |
| phase2_qwen35_2b_e012_origin_transfer | overconfidence | synthetic_contrasts | synthetic_contrasts | 18 | assistant_answer_mean | 75.0 | 1.0 | 8.162301295189366 | 8.158715887471391 | 8.162301295189366 | 6.030009153686 | -2.1322921415033647 |
