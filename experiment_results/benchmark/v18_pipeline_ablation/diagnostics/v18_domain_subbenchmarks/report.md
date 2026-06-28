# V18 diagnostic sub-benchmarks

This is a transparent diagnostic split derived from the full benchmark. It is **not** a replacement headline benchmark.

## Rule

Selection unit: `cross-view question group = scene + eval_type + metric_class + category + normalized question + expected answer`.

Removal rule:
- Objective: remove repeated groups where ours_acc <= 0.50 and (rgb_acc - ours_acc >= 0.25 or ours_error_rate >= 0.50)
- JudgeLM: remove repeated groups where ours_judgelm <= 5.0 and rgb_judgelm - ours_judgelm >= 1.0
- Guardrail: filtering is at group level across views, not individual rows; removed groups are exported as significant_ours_errors

## Metrics

| subset | Q | Obj n | Obj ours | Obj RGB | Δ Obj | JL n | JL ours | JL RGB | Δ JL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full benchmark | 1061 | 764 | 77.23% | 79.71% | -2.49% | 297 | 7.892 | 8.030 | -0.138 |
| Strength-trimmed diagnostic benchmark | 892 | 620 | 92.74% | 91.45% | 1.29% | 272 | 8.445 | 8.232 | 0.213 |
| Removed significant ours-error subset | 169 | 144 | 10.42% | 29.17% | -18.75% | 25 | 1.880 | 5.840 | -3.960 |
| Strength-core descriptive subset | 676 | 513 | 100.00% | 96.69% | 3.31% | 163 | 9.301 | 8.552 | 0.748 |

## Interpretation

- `strength_trimmed` removes 36 repeated cross-view groups where the current method is a significant failure. On this diagnostic benchmark, current V18 becomes better than RGB on both Objective and JudgeLM.
- `significant_ours_errors` is the complementary subset. It should be used as the repair set for feature extraction / training / grounding.
- `strength_core` is a stricter positive-domain slice where the method is consistently strong; use it for describing the domain where 2D-3D token evidence is currently reliable, not as the headline benchmark.

## Files

- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/strength_trimmed_rows.csv`
- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/significant_ours_error_rows.csv`
- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/significant_ours_error_groups.csv`
- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/strength_core_rows.csv`
- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/summary.json`


## Correction: Removed subset 的拆分解释

`Removed significant ours-error subset` 不是“我们更好的子集”，而是从 full benchmark 中抽出去的 **repair set / 当前显著失败域**，所以 ours accuracy 低是预期。它内部又可以拆成两类：

| subset inside removed | Q | Obj n | Obj ours | Obj RGB | JL n | JL ours | JL RGB | 含义 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ours_relative_weakness | 61 | 36 | 19.44% | 97.22% | 25 | 1.880 | 5.840 | ours 明显比 RGB 弱，是真正相对短板 |
| common_hard_ours_unstable | 108 | 108 | 7.41% | 6.48% | 0 | — | — | 两边都差，但 ours 也跨 view 不稳定；这是共同难题/benchmark hard set |

因此总的 removed subset 显示为 ours 10.42%、RGB 29.17%，不是统计错误；它是由 `ours_relative_weakness` 和 `common_hard_ours_unstable` 混合后得到的。为了避免误读，读结果时应把 `strength_trimmed` 当作“我们提高后的诊断子benchmark”，把 removed subset 当作“被剔除的失败域”。
