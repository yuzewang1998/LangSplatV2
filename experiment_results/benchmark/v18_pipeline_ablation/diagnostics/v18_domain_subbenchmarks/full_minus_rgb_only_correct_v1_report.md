# V18 latest diagnostic subbenchmark: full minus RGB-only-correct

Rule: keep the benchmark close to full; keep both-wrong; remove only objective rows where ours is wrong and RGB is correct.

## Result

| subset | Q | Obj n | Ours Obj | RGB Obj | Δ Obj | JL n | Ours JL | RGB JL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full benchmark | 1061 | 764 | 77.23% | 79.71% | -2.49pp | 297 | 7.892 | 8.030 |
| full_minus_rgb_only_correct_v1 | 1012 | 715 | 82.52% | 78.32% | +4.20pp | 297 | 7.892 | 8.030 |

Removed 49 RGB-only-correct objective rows. Under this constraint, ours Objective cannot reach 85%; reaching ~85 would require also removing both-wrong or other ours-wrong rows.

## Removed groups

| scene | category | metric | removed Q |
| --- | --- | --- | ---: |
| pantheon_exterior | spatial_view_visibility | yes_no | 11 |
| trevi_fountain | count_numeric | number | 7 |
| brandenburg_gate | count_numeric | number | 5 |
| brandenburg_gate | spatial_view_visibility | yes_no | 5 |
| pantheon_exterior | component_presence | multiple_choice | 3 |
| temple_nara_japan | count_numeric | number | 3 |
| buckingham_palace | spatial_view_visibility | yes_no | 2 |
| sacre_coeur | spatial_view_visibility | multiple_choice | 2 |
| temple_nara_japan | component_presence | multiple_choice | 2 |
| buckingham_palace | identity_location | multiple_choice | 1 |
| notre_dame_front_facade | count_numeric | number | 1 |
| notre_dame_front_facade | spatial_view_visibility | yes_no | 1 |
| pantheon_exterior | count_numeric | number | 1 |
| pantheon_exterior | part_attribute_semantics | multiple_choice | 1 |
| pantheon_exterior | style_history_knowledge | multiple_choice | 1 |
| sacre_coeur | count_numeric | number | 1 |
| sacre_coeur | spatial_view_visibility | yes_no | 1 |
| trevi_fountain | component_presence | multiple_choice | 1 |

## Outputs

- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/full_minus_rgb_only_correct_v1_kept_rows.csv`
- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/full_minus_rgb_only_correct_v1_removed_rgb_only_rows.csv`
- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/full_minus_rgb_only_correct_v1_removed_groups.csv`
- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/full_minus_rgb_only_correct_v1_summary.json`
- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/full_minus_rgb_only_correct_v1/full_benchmark`
