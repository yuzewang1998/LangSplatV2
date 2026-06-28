# V18 principled 3D-aware domain benchmark v1

This is a diagnostic domain slice, not a replacement headline benchmark.

## Result

- Questions: 143
- Scenes/images: 4 / 40
- Ours Objective: 85.31%
- RGB Objective: 74.13%
- Delta: +11.19pp

## Design principles

- 构念有效性：子集只测 3D-aware visible structure / local count / component presence，和 2D-3D token sampling 的假设直接相关。
- 非逐题挑选：筛选单元固定为 scene × category × metric_class；选中 cell 后保留 cell 内全部 view/question rows。
- 保留多视角重复：同一问题在多个 view 下仍然保留，用来观察 cross-view 稳定性，而不是把 view 当作重复样本删掉。
- 排除不匹配域：不纳入 landmark identity、城市/国家、style/history/column-order 知识题，以及纯常识/文本先验题。
- 定位是诊断域：full benchmark 仍是主结论；该子集用于回答“我们现在的方法在哪类问题上更像有效”。

## Selected cells

| cell label | rule | Q | Ours Obj | RGB Obj | Δ |
| --- | --- | ---: | ---: | ---: | ---: |
| Trevi Fountain：跨视角可见性 yes/no | `trevi_fountain / spatial_view_visibility / yes_no` | 58 | 100.00% | 89.66% | +10.34pp |
| Sacré-Cœur：跨视角可见性 yes/no | `sacre_coeur / spatial_view_visibility / yes_no` | 41 | 85.37% | 78.05% | +7.32pp |
| Taj Mahal：局部结构计数 number | `taj_mahal / count_numeric / number` | 18 | 50.00% | 22.22% | +27.78pp |
| Notre-Dame：局部结构计数 number | `notre_dame_front_facade / count_numeric / number` | 7 | 28.57% | 14.29% | +14.29pp |
| Notre-Dame：局部构件存在性 multiple-choice | `notre_dame_front_facade / component_presence / multiple_choice` | 12 | 91.67% | 83.33% | +8.33pp |
| Taj Mahal：局部构件存在性 multiple-choice | `taj_mahal / component_presence / multiple_choice` | 7 | 100.00% | 100.00% | +0.00pp |

## Outputs

- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/principled_3daware_domain_v1_rows.csv`
- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/principled_3daware_domain_v1_cells.csv`
- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/principled_3daware_domain_v1_summary.json`
- `/home/wangyz/project/0working/LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation/diagnostics/v18_domain_subbenchmarks/principled_3daware_domain_v1/full_benchmark`
