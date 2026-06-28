# V18 vs RGB 逐题误差画像 / Benchmark Sampling Diagnostic

生成时间：`2026-06-26T12:21:20+08:00`  
主分析方法：`mmtokscale_T576_R5` (`v18_tokpipe_best2d_best3d_m2tokcat_m3tokcat_mmtokscale_odirect_T576_R5_C1`)  
范围：8 scenes / 74 images / 1061 VQA；Objective 764 题、JudgeLM 297 题。

## 0. 读法

- **ours_only_correct**：V18 答对、RGB 答错，是我们方法相对 RGB 的正样本。
- **rgb_only_correct**：RGB 答对、V18 答错，是当前方法需要解释/修复的负样本。
- JudgeLM 同时给 exact 高低与 `±0.25` margin；`±0.25` 内先当近似持平，不强行过度解释。

## 1. 方法总览

| method | Obj n | Obj ours | Obj RGB | Δ Obj | ours-only | rgb-only | both-wrong | JL n | JL ours | JL RGB | Δ JL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmtokscale_T576_R5 | 764 | 77.23% | 79.71% | -2.49pp | 30 | 49 | 125 | 297 | 7.892 | 8.030 | -0.138 |
| mmtokscale_T512_R5 | 764 | 77.23% | 79.71% | -2.49pp | 29 | 48 | 126 | 297 | 7.875 | 8.030 | -0.155 |
| mmtok2d3d_T512_R30 | 764 | 74.87% | 79.71% | -4.84pp | 25 | 62 | 130 | 297 | 8.138 | 8.030 | 0.108 |

## 1.5 关键解释 / Bias hint

- Objective 总体仍是 RGB 更强：mmtokscale_T576_R5 为 77.23%，RGB 为 79.71%，差 -2.49pp；逐题上 V18-only 30 题，RGB-only 49 题。

- 按大类看，V18 没有出现稳定超过 RGB 的 Objective 大类；优势主要是少数 scene×question-type 切片，而不是全局能力提升。

- 当前最清楚的 V18 正切片是 taj_mahal / count_numeric / number：Δ +27.78pp，但要注意这是切片优势。

- 最需要修的是 count_numeric / number：V18 38.14% vs RGB 44.92%，RGB-only 18 题。

- 场景切片里最大负差是 pantheon_exterior / spatial_view_visibility / yes_no：Δ -35.48pp。

- JudgeLM 画像和 Objective 不完全一致：identity_location 上 V18 更高（Δ 0.319），但 component_presence 明显输给 RGB（Δ -1.741）。

- 初步方向：V18 的 3D token 更像提供 landmark/全局语义先验；短板在可见部件定位、可数实例、细粒度空间/部件 grounding。这更支持回到 feature extraction / training，让 3D feature 保留小部件和实例边界，而不是继续只调 post-training sampling。

## 2. Objective：按问题类别看 ours/RGB 各自优势

| category | metric | n | ours | RGB | Δ | ours-only | rgb-only | both-wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count_numeric | multiple_choice | 3 | 0.00% | 0.00% | +0.00pp | 0 | 0 | 3 |
| multiple_choice_general | multiple_choice | 42 | 100.00% | 100.00% | +0.00pp | 0 | 0 | 0 |
| part_attribute_semantics | multiple_choice | 25 | 96.00% | 96.00% | +0.00pp | 1 | 1 | 0 |
| part_attribute_semantics | yes_no | 2 | 100.00% | 100.00% | +0.00pp | 0 | 0 | 0 |
| spatial_view_visibility | multiple_choice | 97 | 75.26% | 75.26% | +0.00pp | 2 | 2 | 22 |
| yes_no_general | yes_no | 28 | 100.00% | 100.00% | +0.00pp | 0 | 0 | 0 |
| style_history_knowledge | multiple_choice | 135 | 81.48% | 82.22% | -0.74pp | 0 | 1 | 24 |
| spatial_view_visibility | yes_no | 229 | 84.28% | 86.46% | -2.18pp | 15 | 20 | 16 |
| component_presence | multiple_choice | 69 | 84.06% | 89.86% | -5.80pp | 2 | 6 | 5 |
| identity_location | multiple_choice | 16 | 93.75% | 100.00% | -6.25pp | 0 | 1 | 0 |
| count_numeric | number | 118 | 38.14% | 44.92% | -6.78pp | 10 | 18 | 55 |

## 3. Objective：按场景看偏差

| scene | metric | n | ours | RGB | Δ | ours-only | rgb-only | both-wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taj_mahal | number | 18 | 50.00% | 22.22% | +27.78pp | 5 | 0 | 9 |
| notre_dame_front_facade | number | 7 | 28.57% | 14.29% | +14.29pp | 2 | 1 | 4 |
| trevi_fountain | yes_no | 58 | 100.00% | 89.66% | +10.34pp | 6 | 0 | 0 |
| notre_dame_front_facade | yes_no | 24 | 66.67% | 58.33% | +8.33pp | 3 | 1 | 7 |
| sacre_coeur | yes_no | 41 | 85.37% | 78.05% | +7.32pp | 4 | 1 | 5 |
| notre_dame_front_facade | multiple_choice | 37 | 97.30% | 94.59% | +2.70pp | 1 | 0 | 1 |
| taj_mahal | multiple_choice | 65 | 86.15% | 84.62% | +1.54pp | 1 | 0 | 9 |
| brandenburg_gate | multiple_choice | 22 | 100.00% | 100.00% | +0.00pp | 0 | 0 | 0 |
| sacre_coeur | number | 21 | 85.71% | 85.71% | +0.00pp | 1 | 1 | 2 |
| taj_mahal | yes_no | 23 | 100.00% | 100.00% | +0.00pp | 0 | 0 | 0 |
| temple_nara_japan | yes_no | 21 | 90.48% | 90.48% | +0.00pp | 0 | 0 | 2 |
| trevi_fountain | multiple_choice | 65 | 76.92% | 76.92% | +0.00pp | 1 | 1 | 14 |
| buckingham_palace | multiple_choice | 51 | 90.20% | 92.16% | -1.96pp | 0 | 1 | 4 |
| sacre_coeur | multiple_choice | 41 | 60.98% | 63.41% | -2.44pp | 1 | 2 | 14 |
| buckingham_palace | yes_no | 31 | 90.32% | 93.55% | -3.23pp | 1 | 2 | 1 |
| temple_nara_japan | multiple_choice | 55 | 94.55% | 98.18% | -3.64pp | 0 | 2 | 1 |
| pantheon_exterior | multiple_choice | 51 | 68.63% | 76.47% | -7.84pp | 1 | 5 | 11 |
| brandenburg_gate | number | 26 | 23.08% | 34.62% | -11.54pp | 2 | 5 | 15 |
| pantheon_exterior | number | 7 | 71.43% | 85.71% | -14.29pp | 0 | 1 | 1 |
| brandenburg_gate | yes_no | 22 | 77.27% | 95.45% | -18.18pp | 1 | 5 | 0 |
| trevi_fountain | number | 28 | 0.00% | 25.00% | -25.00pp | 0 | 7 | 21 |
| temple_nara_japan | number | 11 | 45.45% | 72.73% | -27.27pp | 0 | 3 | 3 |
| pantheon_exterior | yes_no | 39 | 69.23% | 97.44% | -28.21pp | 0 | 11 | 1 |

## 4. JudgeLM：按问题类别看描述质量偏差

| category | n | ours | RGB | Δ | ours≥+0.25 | RGB≥+0.25 | near tie |
| --- | --- | --- | --- | --- | --- | --- | --- |
| identity_location | 182 | 8.890 | 8.571 | 0.319 | 65 | 30 | 87 |
| part_attribute_semantics | 14 | 6.500 | 6.857 | -0.357 | 2 | 5 | 7 |
| style_history_knowledge | 48 | 7.500 | 7.938 | -0.438 | 19 | 13 | 16 |
| open_description_general | 6 | 8.333 | 9.167 | -0.833 | 0 | 1 | 5 |
| spatial_view_visibility | 20 | 6.750 | 7.800 | -1.050 | 3 | 8 | 9 |
| component_presence | 27 | 3.333 | 5.074 | -1.741 | 6 | 12 | 9 |

## 5. 同题 side-by-side：坏例里两边分别答成什么样

### 5.1 错误模式计数（Objective）

| group | metric | category | V18 error | RGB error | count |
| --- | --- | --- | --- | --- | --- |
| both_wrong | multiple_choice | spatial_view_visibility | wrong_option(B!=A) | wrong_option(B!=A) | 21 |
| both_wrong | multiple_choice | style_history_knowledge | wrong_option(A!=C) | wrong_option(A!=C) | 18 |
| rgb_only_correct | yes_no | spatial_view_visibility | false_negative_yes->no | correct | 18 |
| ours_only_correct | yes_no | spatial_view_visibility | correct | false_negative_yes->no | 12 |
| both_wrong | yes_no | spatial_view_visibility | false_positive_no->yes | false_positive_no->yes | 11 |
| both_wrong | number | count_numeric | overcount(3>2) | overcount(3>2) | 10 |
| rgb_only_correct | number | count_numeric | overcount(3>2) | correct | 8 |
| both_wrong | number | count_numeric | overcount(10>5) | overcount(10>5) | 5 |
| both_wrong | yes_no | spatial_view_visibility | false_negative_yes->no | false_negative_yes->no | 5 |
| ours_only_correct | number | count_numeric | correct | overcount(4>3) | 5 |
| both_wrong | multiple_choice | style_history_knowledge | wrong_option(C!=A) | wrong_option(D!=A) | 4 |
| both_wrong | multiple_choice | count_numeric | wrong_option(B!=C) | wrong_option(B!=C) | 3 |
| both_wrong | number | count_numeric | overcount(4>0) | overcount(4>0) | 3 |
| both_wrong | number | count_numeric | overcount(12>1) | overcount(12>1) | 3 |
| ours_only_correct | yes_no | spatial_view_visibility | correct | false_positive_no->yes | 3 |
| rgb_only_correct | number | count_numeric | undercount(3<4) | correct | 3 |
| both_wrong | multiple_choice | component_presence | wrong_option(B!=A) | wrong_option(B!=A) | 2 |
| both_wrong | number | count_numeric | overcount(3>1) | overcount(2>1) | 2 |

### 5.2 RGB 对、V18 错：V18 坏例

| scene | metric | category | question | expected | expected option | V18 answer | V18 option | V18 error | RGB answer | RGB option | RGB error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pantheon_exterior | multiple_choice | component_presence | Which visible component stands in front of the portico columns in the image?<br>A. Obelisk<br>B. Portico columns<br>C. Triangular pediment<br>D. Rotunda wall | A | Obelisk | C | Triangular pediment | wrong_option(C!=A) | A | Obelisk | correct |
| temple_nara_japan | multiple_choice | component_presence | Which component is NOT visibly present in this image? A. eave B. roof C. golden Shibi D. column | C | golden Shibi | A | eave | wrong_option(A!=C) | C | golden Shibi | correct |
| trevi_fountain | multiple_choice | component_presence | Which visible component is most directly associated with central mythological figure? A. Oceanus B. papal coat of arms C. window D. water | A | Oceanus | D | water | wrong_option(D!=A) | A | Oceanus | correct |
| buckingham_palace | multiple_choice | identity_location | Which feature is centrally located on the palace façade in the image?<br>A. Royal balcony<br>B. Rose window<br>C. Onion dome<br>D. Minaret | A | Royal balcony | B | Rose window | wrong_option(B!=A) | A | Royal balcony | correct |
| pantheon_exterior | multiple_choice | part_attribute_semantics | What shape best describes the foreground monument in the image?<br>A. Tall tapering obelisk<br>B. Round dome<br>C. Curved arch<br>D. Flat pediment | A | Tall tapering obelisk | D | Flat pediment | wrong_option(D!=A) | A | Tall tapering obelisk | correct |
| sacre_coeur | multiple_choice | spatial_view_visibility | Which religious figure is represented by the central statue above the main entrance in the image?<br>A. Jesus Christ with a visible heart symbol<br>B. Saint Peter holding keys<br>C. The Virgin Mary with a child<br>D. Saint George slaying a dragon | A | Jesus Christ with a visible heart symbol | C | The Virgin Mary with a child | wrong_option(C!=A) | A | Jesus Christ with a visible heart symbol | correct |
| pantheon_exterior | multiple_choice | style_history_knowledge | Which visible component is most directly associated with piazza monument? A. front obelisk B. dome C. Corinthian column D. rose window | A | front obelisk | C | Corinthian column | wrong_option(C!=A) | A | front obelisk | correct |
| brandenburg_gate | number | count_numeric | How many horses are visible in the sculpture at the top? | 4 |  | 3 |  | undercount(3<4) | 4 |  | correct |
| notre_dame_front_facade | number | count_numeric | How many distinct rose window components are visible? | 1 |  | 2 |  | overcount(2>1) | 1 |  | correct |
| pantheon_exterior | number | count_numeric | How many separate triangular pediment instances are visible? | 1 |  | 2 |  | overcount(2>1) | 1 |  | correct |
| sacre_coeur | number | count_numeric | How many distinct bronze sculpture components are visible? | 2 |  | 3 |  | overcount(3>2) | 2 |  | correct |
| temple_nara_japan | number | count_numeric | How many distinct door components are visible? | 3 |  | 10 |  | overcount(10>3) | 3 |  | correct |
| trevi_fountain | number | count_numeric | How many large rectangular relief panels are visible above the side sections of the fountain façade? | 2 |  | 3 |  | overcount(3>2) | 2 |  | correct |
| brandenburg_gate | yes_no | spatial_view_visibility | Is the passage space behind the columns visible from this angle? | Yes |  | No |  | false_negative_yes->no | Yes |  | correct |
| buckingham_palace | yes_no | spatial_view_visibility | Is the fence clearly visible in this image? | No |  | Yes |  | false_positive_no->yes | no |  | correct |
| notre_dame_front_facade | yes_no | spatial_view_visibility | Is the Last Judgment scene visible in this image? | Yes |  | No. |  | false_negative_yes->no | Yes |  | correct |
| pantheon_exterior | yes_no | spatial_view_visibility | Is the triangular pediment visible in this image? | Yes |  | No |  | false_negative_yes->no | Yes |  | correct |
| sacre_coeur | yes_no | spatial_view_visibility | Is the rose window clearly visible in this image? | Yes |  | No |  | false_negative_yes->no | Yes |  | correct |

### 5.3 V18 对、RGB 错：RGB 坏例

| scene | metric | category | question | expected | expected option | V18 answer | V18 option | V18 error | RGB answer | RGB option | RGB error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| notre_dame_front_facade | multiple_choice | component_presence | Which of these architectural or landmark components is visibly present in this image? A. rose window B. tympanum C. door D. onion dome | A | rose window | A | rose window | correct | B | tympanum | wrong_option(B!=A) |
| trevi_fountain | multiple_choice | component_presence | Which component is NOT visibly present in this image? A. column B. Oceanus C. papal coat of arms D. winged horses | C | papal coat of arms | C | papal coat of arms | correct | D | winged horses | wrong_option(D!=C) |
| pantheon_exterior | multiple_choice | part_attribute_semantics | What shape best describes the foreground monument in the image?<br>A. Tall tapering obelisk<br>B. Round dome<br>C. Curved arch<br>D. Flat pediment | A | Tall tapering obelisk | A | Tall tapering obelisk | correct | D | Flat pediment | wrong_option(D!=A) |
| sacre_coeur | multiple_choice | spatial_view_visibility | Which feature is NOT part of the visible in the image?<br>A. Rose window<br>B. Bronze equestrian statue<br>C. Oceanus statue | C | Oceanus statue | C | Oceanus statue | correct | B | Bronze equestrian statue | wrong_option(B!=C) |
| taj_mahal | multiple_choice | spatial_view_visibility | Which dome form is most prominent at the center of the landmark in the image?<br>A. Bulbous onion dome<br>B. Hemispherical dome<br>C. Ribbed dome<br>D. Conical dome | A | Bulbous onion dome | A | Bulbous onion dome | correct | B | Hemispherical dome | wrong_option(B!=A) |
| brandenburg_gate | number | count_numeric | How many oval relief medallions are visible on the inner side walls in the image, including the partially visible one near the edge? | 3 |  | 3 |  | correct | 4 |  | overcount(4>3) |
| notre_dame_front_facade | number | count_numeric | How many distinct rose window components are visible? | 1 |  | 1 |  | correct | 12 |  | overcount(12>1) |
| sacre_coeur | number | count_numeric | How many distinct bronze sculpture components are visible? | 2 |  | 2 |  | correct | 3 |  | overcount(3>2) |
| taj_mahal | number | count_numeric | How many distinct onion dome components are visible? | 3 |  | 3 |  | correct | 4 |  | overcount(4>3) |
| brandenburg_gate | yes_no | spatial_view_visibility | Is the gate shown from an oblique angle rather than straight on? | Yes |  | Yes |  | correct | No |  | false_negative_yes->no |
| buckingham_palace | yes_no | spatial_view_visibility | Is the fence clearly visible in this image? | No |  | No |  | correct | Yes |  | false_positive_no->yes |
| notre_dame_front_facade | yes_no | spatial_view_visibility | Is the tympanum clearly visible in this image? | No |  | No. |  | correct | Yes |  | false_positive_no->yes |
| sacre_coeur | yes_no | spatial_view_visibility | Is the central religious statue visible in this image? | Yes |  | Yes |  | correct | No |  | false_negative_yes->no |
| trevi_fountain | yes_no | spatial_view_visibility | Are the winged horses visible in this image? | Yes |  | Yes. |  | correct | No |  | false_negative_yes->no |
| brandenburg_gate | number | count_numeric | How many fluted columns are clearly visible in this oblique view? | 2 |  | 2 |  | correct | 3 |  | overcount(3>2) |
| notre_dame_front_facade | number | count_numeric | How many distinct rose window components are visible? | 1 |  | 1 |  | correct | 12 |  | overcount(12>1) |
| taj_mahal | number | count_numeric | How many distinct onion dome components are visible? | 3 |  | 3 |  | correct | 4 |  | overcount(4>3) |
| notre_dame_front_facade | yes_no | spatial_view_visibility | Is the tympanum clearly visible in this image? | No |  | No |  | correct | Yes |  | false_positive_no->yes |

### 5.4 两边都错：共同盲点

| scene | metric | category | question | expected | expected option | V18 answer | V18 option | V18 error | RGB answer | RGB option | RGB error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| notre_dame_front_facade | multiple_choice | component_presence | Which of these architectural or landmark components is visibly present in this image? A. Last Judgment scene B. rose window C. tympanum D. door | A | Last Judgment scene | C | tympanum | wrong_option(C!=A) | C | tympanum | wrong_option(C!=A) |
| sacre_coeur | multiple_choice | component_presence | Which visible component is positioned directly above the main arched entrance in the image?<br>A. Central religious statue<br>B. Bronze equestrian statue<br>C. Front staircase<br>D. Side tower | A | Central religious statue | B | Bronze equestrian statue | wrong_option(B!=A) | B | Bronze equestrian statue | wrong_option(B!=A) |
| temple_nara_japan | multiple_choice | component_presence | Which visible component is most directly associated with deep protective eaves? A. eave B. roof C. golden Shibi D. door | A | eave | B | roof | wrong_option(B!=A) | B | roof | wrong_option(B!=A) |
| trevi_fountain | multiple_choice | component_presence | Which of these architectural or landmark components is visibly present in this image? A. column B. papal coat of arms C. window D. water | A | column | D | water | wrong_option(D!=A) | D | water | wrong_option(D!=A) |
| pantheon_exterior | multiple_choice | count_numeric | Approximately how many large portico columns are visible in the image?<br>A. 0–5<br>B. 6–10<br>C. 11–20<br>D. More than 20 | C | 11–20 | B | 6–10 | wrong_option(B!=C) | B | 6–10 | wrong_option(B!=C) |
| sacre_coeur | multiple_choice | spatial_view_visibility | Which religious figure is represented by the central statue above the main entrance in the image?<br>A. Jesus Christ with a visible heart symbol<br>B. Saint Peter holding keys<br>C. The Virgin Mary with a child<br>D. Saint George slaying a dragon | A | Jesus Christ with a visible heart symbol | B | Saint Peter holding keys | wrong_option(B!=A) | B | Saint Peter holding keys | wrong_option(B!=A) |
| taj_mahal | multiple_choice | spatial_view_visibility | Which dome form is most prominent at the center of the landmark in the image?<br>A. Bulbous onion dome<br>B. Hemispherical dome<br>C. Ribbed dome<br>D. Conical dome | A | Bulbous onion dome | B | Hemispherical dome | wrong_option(B!=A) | B | Hemispherical dome | wrong_option(B!=A) |
| buckingham_palace | multiple_choice | style_history_knowledge | Which of these architectural or landmark components is visibly present in this image? A. Ionic column B. fence C. Queen Victoria Memorial D. pediment | A | Ionic column | C | Queen Victoria Memorial | wrong_option(C!=A) | D | pediment | wrong_option(D!=A) |
| pantheon_exterior | multiple_choice | style_history_knowledge | Which column order do the visible portico columns most closely resemble?<br>A. Doric column<br>B. Ionic column<br>C. Corinthian column<br>D. Tuscan column | C | Corinthian column | A | Doric column | wrong_option(A!=C) | A | Doric column | wrong_option(A!=C) |
| trevi_fountain | multiple_choice | style_history_knowledge | Which column style do the visible columns or pilasters on the façade in the image most closely resemble?<br>A. Doric<br>B. Ionic<br>C. Corinthian<br>D. Gothic | C | Corinthian | A | Doric | wrong_option(A!=C) | A | Doric | wrong_option(A!=C) |
| brandenburg_gate | number | count_numeric | How many oval medallions are visible on the inner walls? | 3 |  | 2 |  | undercount(2<3) | 4 |  | overcount(4>3) |
| notre_dame_front_facade | number | count_numeric | How many distinct rose window components are visible? | 1 |  | 12 |  | overcount(12>1) | 12 |  | overcount(12>1) |
| pantheon_exterior | number | count_numeric | How many distinct front obelisk components are visible? | 1 |  | 3 |  | overcount(3>1) | 3 |  | overcount(3>1) |
| sacre_coeur | number | count_numeric | How many large rounded domes are visible above the main façade in the image? | 0 |  | 2 |  | overcount(2>0) | 1 |  | overcount(1>0) |
| taj_mahal | number | count_numeric | How many minarets are visible around the main mausoleum in this image? | 0 |  | 4 |  | overcount(4>0) | 4 |  | overcount(4>0) |
| temple_nara_japan | number | count_numeric | How many major roof tiers are visible on the main temple building? | 2 |  | 3 |  | overcount(3>2) | 3 |  | overcount(3>2) |
| trevi_fountain | number | count_numeric | How many sea horses are visible in the sculptural group of the landmark in the image? | 2 |  | 3 |  | overcount(3>2) | 3 |  | overcount(3>2) |
| buckingham_palace | yes_no | spatial_view_visibility | Is the fence clearly visible in this image? | No |  | Yes |  | false_positive_no->yes | Yes |  | false_positive_no->yes |

### 5.5 JudgeLM 大差异：两边回答文本对照

| group | scene | category | question | expected | V18 answer | V18 score | RGB answer | RGB score | ΔJL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v18_much_lower | trevi_fountain | component_presence | What visible component matches this description: the central mythological male figure? | Oceanus | The central mythological male figure is a statue. | 1.0 | The central mythological male figure is the one standing upright in the middle of the façade, holding what appears to be a spear or staff. He is surrounded by other sculptural elements and is part of a larger group of figures that are integral to the design of the building's central archway. This figure is likely to represent a deity or a significant historical or mythological character, given the grandeur and prominence of his position within the architectural composition. | 10.0 | -9.000 |
| v18_much_lower | trevi_fountain | component_presence | What visible component matches this description: the central mythological male figure? | Oceanus | The central mythological male figure is a statue. | 1.0 | The central mythological male figure is the one in the middle of the group of statues, who appears to be the focal point of the scene. He is standing upright and is holding what seems to be a spear or staff. This figure is part of the larger group of statues that are situated on the façade of the building, which is part of the Trevi Fountain in Rome, Italy. | 10.0 | -9.000 |
| v18_much_lower | trevi_fountain | component_presence | What visible component matches this description: the central mythological male figure? | Oceanus | The central mythological male figure is a statue. | 1.0 | The central mythological male figure is the one standing upright with a beard, located on the right side of the façade. He is holding what appears to be a scroll or a book in his left hand and has a muscular build. This figure is part of the sculptural group that adorns the central part of the building, above the cascading water. | 10.0 | -9.000 |
| v18_much_lower | trevi_fountain | component_presence | What visible component matches this description: the central mythological male figure? | Oceanus | The central mythological male figure is a statue. | 1.0 | The central figure is a male statue, likely representing a god or mythological character, who is standing with one foot on a rock and the other raised, holding what appears to be a cloth or garment in his left hand. He has a beard and is wearing a draped garment that falls over one shoulder, leaving the other bare. The statue is positioned above a fountain, with water flowing from its base. This figure is part of a larger group of statues that are part of the Trevi Fountain in Rome, Italy. | 10.0 | -9.000 |
| v18_much_lower | brandenburg_gate | spatial_view_visibility | What sculpture group is centered at the top of this image? | The Quadriga. | The sculpture group centered at the top of this image is the Brandenburg Gate. | 1.0 | The sculpture group centered at the top of the image is the Quadriga, which is a chariot drawn by four horses abreast. It is a famous landmark located on the Brandenburg Gate in Berlin, Germany. The Quadriga is topped with a figure of Victoria, the Roman goddess of victory, holding a laurel wreath and a spear with a circular emblem at its tip. | 9.0 | -8.000 |
| v18_much_lower | pantheon_exterior | style_history_knowledge | What does the Latin inscription above the columns of the landmark in the image mean? | It states that Marcus Agrippa, son of Lucius, built it when he was consul for the third time. | The Latin inscription above the columns of the landmark in the image reads "MVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVM | 1.0 | The Latin inscription above the columns of the landmark reads "MAGRIPPA L. F. COS. TERTIVM IMP. FECIT," which translates to "Magrippa, son of Lucius, made this third time by the third consul." This indicates that the structure was commissioned or built by a person named Magrippa, who was the son of Lucius and held the title of consul for the third time. The term "imperium" suggests that he may have been an emperor or held a high-ranking position in the Roman Empire at the time. | 8.0 | -7.000 |
| v18_much_lower | taj_mahal | component_presence | What visible component matches this description: large pointed arch openings? | arch-shaped doorways | The main dome of the Taj Mahal. | 1.0 | The large pointed arch openings are the domes on the main building. | 8.0 | -7.000 |
| v18_much_lower | trevi_fountain | component_presence | What visible component matches this description: the central mythological male figure? | Oceanus | The central mythological male figure is a statue. | 1.0 | The central mythological male figure is the statue of Oceanus. | 8.0 | -7.000 |
| v18_much_lower | brandenburg_gate | spatial_view_visibility | Which direction are the horses facing in the image? | Toward the front-right direction in the image. | The horses are facing left. | 2.0 | The horses in the image are facing forward, with their heads turned slightly to the left. This is evident from the position of their ears and the direction of their manes, which flow towards the left side of the image. | 8.0 | -6.000 |
| v18_much_lower | pantheon_exterior | style_history_knowledge | What does the Latin inscription above the columns of the landmark in the image mean? | It states that Marcus Agrippa, son of Lucius, built it when he was consul for the third time. | The Latin inscription above the columns of the landmark in the image reads "MAGISTERIUM VIVENS," which translates to "Living Masterpiece" in English. This phrase is often used to describe a work of art or architecture that is considered timeless and of enduring value. | 2.0 | The Latin inscription above the columns of the landmark reads "MAGRIFFAL L.F. COSTERTIVM FECIT," which translates to "Magrippa, son of Lucius, made this." This indicates that the structure was built or sponsored by a person named Magrippa, who was likely a son of someone named Lucius. The term "FECIT" is a common Latin verb meaning "made" or "built," suggesting that Magrippa was responsible for the construction or dedication of the building. | 8.0 | -6.000 |
| v18_much_lower | taj_mahal | component_presence | What visible component matches this description: large pointed arch openings? | arch-shaped doorways | The main entrance of the Taj Mahal. | 1.0 | The component that matches this description is the Main Gate. | 7.0 | -6.000 |
| v18_much_lower | taj_mahal | component_presence | What visible component matches this description: large pointed arch openings? | arch-shaped doorways | The main building of the Taj Mahal. | 1.0 | arch | 7.0 | -6.000 |
| v18_much_lower | buckingham_palace | open_description_general | Which memorial is commonly seen in front of the landmark shown in the image? | The Queen Victoria Memorial. | The memorial commonly seen in front of the landmark is Nelson's Column. | 5.0 | The memorial commonly seen in front of the landmark shown in the image is the Victoria Memorial. | 10.0 | -5.000 |
| v18_much_lower | brandenburg_gate | identity_location | What was the landmark in the image original main function? | The Brandenburg Gate originally served as a ceremonial city gate and entrance to Berlin | The landmark in the image was a monument. | 5.0 | The Brandenburg Gate was originally a triumphal arch, intended to celebrate the victory of King Frederick William I of Prussia over the French in 1760. It was designed by architect Georg Wenzeslaus von Knobelsdorff and sculpted by Johann Gottfried Schadow. The gate was part of the city wall that surrounded Berlin and was meant to commemorate the Prussian victory and the unity of the German states under Prussian leadership. Over time, it became a symbol of Berlin and Germany, and after World War II, it was used as a border crossing between East and West Berlin. Today, it is one of Berlin's most famous landmarks and a UNESCO World Heritage Site. | 9.0 | -4.000 |
| v18_much_lower | brandenburg_gate | style_history_knowledge | Which ancient gateway is often cited as the design inspiration for the landmark shown in the image? | The Propylaea of the Acropolis in Athens. | The Brandenburg Gate is often cited as the design inspiration for the landmark shown in the image. | 1.0 | The Brandenburg Gate is often cited as the design inspiration for the landmark shown in the image. It was originally built in 1799 to commemorate the Prussian victory over Napoleon and was a symbol of Berlin's status as the capital of the Kingdom of Prussia. The gate has been rebuilt several times, with its current structure dating back to 1876. | 5.0 | -4.000 |
| v18_much_lower | brandenburg_gate | style_history_knowledge | Which ancient gateway is often cited as the design inspiration for the landmark shown in the image? | The Propylaea of the Acropolis in Athens. | The Brandenburg Gate is often cited as the design inspiration for the landmark shown in the image. | 1.0 | The Brandenburg Gate is often cited as the design inspiration for the landmark shown in the image. It was originally inspired by the Arc de Triomphe in Paris, France, and its neoclassical design features columns and a central archway, similar to what is seen in the image. | 5.0 | -4.000 |
| v18_much_lower | brandenburg_gate | style_history_knowledge | Which ancient gateway is often cited as the design inspiration for the landmark shown in the image? | The Propylaea of the Acropolis in Athens. | The Brandenburg Gate is often cited as the design inspiration for the landmark shown in the image. | 1.0 | The Brandenburg Gate is often cited as the design inspiration for the landmark shown in the image. It was originally built in 1799 to commemorate the Prussian victory over Napoleon and was modeled after the ancient Roman triumphal arches, such as the Arch of Titus in Rome. | 5.0 | -4.000 |
| v18_much_lower | brandenburg_gate | style_history_knowledge | Which ancient gateway is often cited as the design inspiration for the landmark shown in the image? | The Propylaea of the Acropolis in Athens. | The Brandenburg Gate is often cited as the design inspiration for the landmark shown in the image. | 1.0 | The Brandenburg Gate in Berlin is often cited as the design inspiration for the landmark shown in the image. This is because both structures share a neoclassical style, with a central archway flanked by columns and topped with sculptures. The Brandenburg Gate was originally built in 1763 and has been a symbol of Berlin ever since, with its design influencing many other monumental gateways around the world. | 5.0 | -4.000 |
| rgb_much_lower | taj_mahal | identity_location | What type of landmark scene is visible in this image? | a mausoleum | The image depicts a grand architectural landmark, likely a mausoleum or monument. | 9.0 | The image depicts a famous historical monument, which is the Taj Mahal, located in Agra, India. It is an iconic example of Mughal architecture and is known for its large, white marble dome and four minarets at the corners. The scene captures the grandeur of the monument with visitors gathered around, indicating its status as a significant tourist attraction. | 1.0 | 8.000 |
| rgb_much_lower | taj_mahal | identity_location | What type of landmark scene is visible in this image? | a mausoleum | The image depicts a grand architectural landmark, likely a mosque or a temple. | 8.0 | The image depicts a famous historical monument, which is the Taj Mahal, located in Agra, India. It is an iconic example of Mughal architecture and is recognized as a UNESCO World Heritage Site. The scene captures the main building with its large central dome and four minarets at the corners, set against a hazy sky, with visitors visible at the base, indicating it's a popular tourist destination. | 1.0 | 7.000 |
| rgb_much_lower | taj_mahal | identity_location | What type of landmark scene is visible in this image? | a mausoleum | The image shows a grand architectural landmark, likely a mosque or a mausoleum. | 8.0 | The image depicts a famous historical monument, which is the Taj Mahal. It is an iconic white marble mausoleum located in Agra, India. The Taj Mahal is known for its stunning architecture, intricate carvings, and beautiful gardens. It was built by Mughal Emperor Shah Jahan in memory of his wife Mumtaz Mahal. The structure features a large central dome flanked by four minarets at the corners, with smaller domes on the corners of the main building. The image captures the grandeur and serene beauty of this UNESCO World Heritage Site. | 1.0 | 7.000 |
| rgb_much_lower | taj_mahal | component_presence | What visible component matches this description: large pointed arch openings? | arch-shaped doorways | The large pointed arch openings are visible. | 8.0 | mausoleum | 1.0 | 7.000 |
| rgb_much_lower | taj_mahal | identity_location | What type of landmark scene is visible in this image? | a mausoleum | The image shows a grand architectural landmark, likely a mosque or a mausoleum. | 8.0 | The image depicts a famous historical monument, which is the Taj Mahal. It is located in Agra, India, and is known for its stunning white marble architecture, large central dome, and intricate carvings. The scene captures the grandeur of the monument against a clear blue sky, with visitors visible at the base, indicating it's a popular tourist attraction. | 1.0 | 7.000 |
| rgb_much_lower | taj_mahal | component_presence | What visible component matches this description: large pointed arch openings? | arch-shaped doorways | The main building has large pointed arch openings. | 8.0 | The component that matches this description is the main entrance of the Taj Mahal. | 2.0 | 6.000 |

## 6. 原始样例抽样

### 6.1 V18 对、RGB 错（ours-only objective）

| scene | metric | category | question | expected | ours | RGB | ours ok | RGB ok | ΔJL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taj_mahal | number | count_numeric | Answer with a number only. How many distinct onion dome components are visible? | 3 | 3 | 4 | True | False | None |
| buckingham_palace | yes_no | spatial_view_visibility | Answer with yes or no only. Is the fence clearly visible in this image? | No | No | Yes | True | False | None |
| sacre_coeur | yes_no | spatial_view_visibility | Answer with yes or no only. Is the central religious statue visible in this image? | Yes | Yes | No | True | False | None |
| trevi_fountain | yes_no | spatial_view_visibility | Answer with yes or no only. Are the winged horses visible in this image? | Yes | Yes. | No | True | False | None |
| taj_mahal | number | count_numeric | Answer with a number only. How many distinct onion dome components are visible? | 3 | 3 | 4 | True | False | None |
| taj_mahal | number | count_numeric | Answer with a number only. How many distinct onion dome components are visible? | 3 | 3 | 4 | True | False | None |
| trevi_fountain | multiple_choice | component_presence | Answer with the option letter only. Which component is NOT visibly present in this image? A. column B. Oceanus C. papal coat of arms D. winged horses | C | C | D | True | False | None |
| brandenburg_gate | number | count_numeric | Answer with a number only. How many oval relief medallions are visible on the inner side walls in the image, including the partially visible one near the edge? | 3 | 3 | 4 | True | False | None |
| notre_dame_front_facade | number | count_numeric | Answer with a number only. How many distinct rose window components are visible? | 1 | 1 | 12 | True | False | None |
| notre_dame_front_facade | number | count_numeric | Answer with a number only. How many distinct rose window components are visible? | 1 | 1 | 12 | True | False | None |
| trevi_fountain | yes_no | spatial_view_visibility | Answer with yes or no only. Are the winged horses visible in this image? | Yes | Yes. | No | True | False | None |
| notre_dame_front_facade | yes_no | spatial_view_visibility | Answer with yes or no only. Is the tympanum clearly visible in this image? | No | No. | Yes | True | False | None |

### 6.2 RGB 对、V18 错（rgb-only objective）

| scene | metric | category | question | expected | ours | RGB | ours ok | RGB ok | ΔJL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brandenburg_gate | number | count_numeric | Answer with a number only. How many horses are visible in the sculpture at the top? | 4 | 3 | 4 | False | True | None |
| sacre_coeur | yes_no | spatial_view_visibility | Answer with yes or no only. Is the rose window clearly visible in this image? | Yes | No | Yes | False | True | None |
| brandenburg_gate | number | count_numeric | Answer with a number only. How many horses are visible in the sculpture at the top? | 4 | 2 | 4 | False | True | None |
| trevi_fountain | number | count_numeric | Answer with a number only. How many large rectangular relief panels are visible above the side sections of the fountain façade? | 2 | 3 | 2 | False | True | None |
| pantheon_exterior | yes_no | spatial_view_visibility | Answer with yes or no only. Is the triangular pediment visible in this image? | Yes | No | Yes | False | True | None |
| pantheon_exterior | multiple_choice | component_presence | Answer with the option letter only. Which visible component stands in front of the portico columns in the image?<br>A. Obelisk<br>B. Portico columns<br>C. Triangular pediment<br>D. Rotunda wall | A | C | A | False | True | None |
| sacre_coeur | multiple_choice | spatial_view_visibility | Answer with the option letter only. Which religious figure is represented by the central statue above the main entrance in the image?<br>A. Jesus Christ with a visible heart symbol<br>B. Saint Peter holding keys<br>C. The Virgin Mary with a child<br>D. Saint George slaying a dragon | A | C | A | False | True | None |
| pantheon_exterior | yes_no | spatial_view_visibility | Answer with yes or no only. Is the front obelisk visible in this image? | Yes | No | Yes | False | True | None |
| pantheon_exterior | yes_no | spatial_view_visibility | Answer with yes or no only. Is the triangular pediment visible in this image? | Yes | No | Yes | False | True | None |
| pantheon_exterior | yes_no | spatial_view_visibility | Answer with yes or no only. Is the front obelisk visible in this image? | Yes | No | yes | False | True | None |
| sacre_coeur | multiple_choice | spatial_view_visibility | Answer with the option letter only. Which religious figure is represented by the central statue above the main entrance in the image?<br>A. Jesus Christ with a visible heart symbol<br>B. Saint Peter holding keys<br>C. The Virgin Mary with a child<br>D. Saint George slaying a dragon | A | B | A | False | True | None |
| trevi_fountain | multiple_choice | component_presence | Answer with the option letter only. Which visible component is most directly associated with central mythological figure? A. Oceanus B. papal coat of arms C. window D. water | A | D | A | False | True | None |

### 6.3 两者都错（objective both-wrong）

| scene | metric | category | question | expected | ours | RGB | ours ok | RGB ok | ΔJL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pantheon_exterior | multiple_choice | style_history_knowledge | Answer with the option letter only. Which column order do the visible portico columns most closely resemble?<br>A. Doric column<br>B. Ionic column<br>C. Corinthian column<br>D. Tuscan column | C | A | A | False | False | None |
| trevi_fountain | multiple_choice | style_history_knowledge | Answer with the option letter only. Which column style do the visible columns or pilasters on the façade in the image most closely resemble?<br>A. Doric<br>B. Ionic<br>C. Corinthian<br>D. Gothic | C | A | A | False | False | None |
| taj_mahal | multiple_choice | spatial_view_visibility | Answer with the option letter only. Which dome form is most prominent at the center of the landmark in the image?<br>A. Bulbous onion dome<br>B. Hemispherical dome<br>C. Ribbed dome<br>D. Conical dome | A | B | B | False | False | None |
| buckingham_palace | multiple_choice | style_history_knowledge | Answer with the option letter only. Which of these architectural or landmark components is visibly present in this image? A. Ionic column B. fence C. Queen Victoria Memorial D. pediment | A | C | D | False | False | None |
| buckingham_palace | yes_no | spatial_view_visibility | Answer with yes or no only. Is the fence clearly visible in this image? | No | Yes | Yes | False | False | None |
| sacre_coeur | multiple_choice | spatial_view_visibility | Answer with the option letter only. Which religious figure is represented by the central statue above the main entrance in the image?<br>A. Jesus Christ with a visible heart symbol<br>B. Saint Peter holding keys<br>C. The Virgin Mary with a child<br>D. Saint George slaying a dragon | A | B | B | False | False | None |
| trevi_fountain | number | count_numeric | Answer with a number only. How many sea horses are visible in the sculptural group of the landmark in the image? | 2 | 3 | 3 | False | False | None |
| brandenburg_gate | number | count_numeric | Answer with a number only. How many oval medallions are visible on the inner walls? | 3 | 2 | 4 | False | False | None |
| notre_dame_front_facade | yes_no | spatial_view_visibility | Answer with yes or no only. Is the Last Judgment scene visible in this image? | Yes | No. | No | False | False | None |
| notre_dame_front_facade | yes_no | spatial_view_visibility | Answer with yes or no only. Is the tympanum clearly visible in this image? | No | Yes. | Yes | False | False | None |
| trevi_fountain | multiple_choice | style_history_knowledge | Answer with the option letter only. Which column style do the visible columns or pilasters on the façade in the image most closely resemble?<br>A. Doric<br>B. Ionic<br>C. Corinthian<br>D. Gothic | C | A | A | False | False | None |
| trevi_fountain | number | count_numeric | Answer with a number only. How many sea horses are visible in the sculptural group of the landmark in the image? | 2 | Three | 3 | False | False | None |

### 6.4 JudgeLM：V18 明显更高

| scene | metric | category | question | expected | ours | RGB | ours ok | RGB ok | ΔJL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taj_mahal | judgelm | identity_location | What type of landmark scene is visible in this image? | a mausoleum | The image depicts a grand architectural landmark, likely a mausoleum or monument. | The image depicts a famous historical monument, which is the Taj Mahal, located in Agra, India. It is an iconic example of Mughal architecture and is known for its large, white marble dome and four minarets at the corners. The scene captures the grandeur of the monument with visitors gathered around, indicating its status as a significant tourist attraction. |  |  | 8.000 |
| taj_mahal | judgelm | identity_location | What type of landmark scene is visible in this image? | a mausoleum | The image depicts a grand architectural landmark, likely a mosque or a temple. | The image depicts a famous historical monument, which is the Taj Mahal, located in Agra, India. It is an iconic example of Mughal architecture and is recognized as a UNESCO World Heritage Site. The scene captures the main building with its large central dome and four minarets at the corners, set against a hazy sky, with visitors visible at the base, indicating it's a popular tourist destination. |  |  | 7.000 |
| taj_mahal | judgelm | identity_location | What type of landmark scene is visible in this image? | a mausoleum | The image shows a grand architectural landmark, likely a mosque or a mausoleum. | The image depicts a famous historical monument, which is the Taj Mahal. It is an iconic white marble mausoleum located in Agra, India. The Taj Mahal is known for its stunning architecture, intricate carvings, and beautiful gardens. It was built by Mughal Emperor Shah Jahan in memory of his wife Mumtaz Mahal. The structure features a large central dome flanked by four minarets at the corners, with smaller domes on the corners of the main building. The image captures the grandeur and serene beauty of this UNESCO World Heritage Site. |  |  | 7.000 |
| taj_mahal | judgelm | component_presence | What visible component matches this description: large pointed arch openings? | arch-shaped doorways | The large pointed arch openings are visible. | mausoleum |  |  | 7.000 |
| taj_mahal | judgelm | identity_location | What type of landmark scene is visible in this image? | a mausoleum | The image shows a grand architectural landmark, likely a mosque or a mausoleum. | The image depicts a famous historical monument, which is the Taj Mahal. It is located in Agra, India, and is known for its stunning white marble architecture, large central dome, and intricate carvings. The scene captures the grandeur of the monument against a clear blue sky, with visitors visible at the base, indicating it's a popular tourist attraction. |  |  | 7.000 |
| taj_mahal | judgelm | component_presence | What visible component matches this description: large pointed arch openings? | arch-shaped doorways | The main building has large pointed arch openings. | The component that matches this description is the main entrance of the Taj Mahal. |  |  | 6.000 |
| sacre_coeur | judgelm | part_attribute_semantics | What is the name of the bronze sculpture in the image? | King Saint Louis IX | The name of the bronze sculpture is "Equestrian Statue". | The bronze sculpture in the image is Napoleon I on horseback. |  |  | 4.000 |
| sacre_coeur | judgelm | identity_location | What famous landmark or landmark scene is shown in this image? | Sacré-Cœur Basilica. | The image shows the famous landmark, Sacré Coeur. | The image shows the Arc de Triomphe with its famous horse-mounted statues, known as the "Tomb of the Unknown Soldier" or "Tombe des Morts." These statues are part of the monument's design and represent the soldiers who have fallen in battle. The perspective is from a low angle looking up towards the monument, emphasizing its grandeur and historical significance. |  |  | 4.000 |
| brandenburg_gate | judgelm | spatial_view_visibility | What architectural feature dominates this image? | The Quadriga and the upper entablature dominate this image. | The Brandenburg Gate dominates the image. | The Brandenburg Gate |  |  | 3.000 |
| brandenburg_gate | judgelm | part_attribute_semantics | Describe the overall facade layout in one sentence. | The facade is symmetrical, with a colonnaded front, five passage openings, and a broad entablature supporting the Quadriga. | The facade features a series of tall, ornate columns with intricate carvings and statues atop them. | The facade has a large archway with four columns on each side, and there is a sign that reads "Pariser Platz" in front of it. |  |  | 3.000 |
| buckingham_palace | judgelm | style_history_knowledge | What is the landmark's general function or cultural role? | It is the official London residence of the British monarch. | The landmark is a palace, which suggests it serves as a residence for royalty or an important government building. | Based solely on the image, the landmark appears to be a monument or memorial, likely commemorating an important historical figure or event. The presence of the seated figure with a scepter and the surrounding angels suggests themes of authority, peace, and possibly victory or divine blessing. The grandeur of the building and the prominence of the statue indicate that it is a place of significance, possibly related to governance or national history. |  |  | 3.000 |
| buckingham_palace | judgelm | style_history_knowledge | What is the landmark's general function or cultural role? | It is the official London residence of the British monarch. | The landmark is a grand building with intricate sculptures and statues, suggesting it may be a significant cultural or historical site. | Based on the image, the landmark appears to be a monument or memorial, likely commemorating an important historical event or figure. The presence of the flag suggests it may also serve as a symbol of national pride or identity. Its grandeur and central location in what seems to be a significant square indicate that it is meant to be a focal point for public viewing and possibly a place for ceremonial events or gatherings. |  |  | 3.000 |

### 6.5 JudgeLM：RGB 明显更高

| scene | metric | category | question | expected | ours | RGB | ours ok | RGB ok | ΔJL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trevi_fountain | judgelm | component_presence | What visible component matches this description: the central mythological male figure? | Oceanus | The central mythological male figure is a statue. | The central mythological male figure is the one standing upright in the middle of the façade, holding what appears to be a spear or staff. He is surrounded by other sculptural elements and is part of a larger group of figures that are integral to the design of the building's central archway. This figure is likely to represent a deity or a significant historical or mythological character, given the grandeur and prominence of his position within the architectural composition. |  |  | -9.000 |
| trevi_fountain | judgelm | component_presence | What visible component matches this description: the central mythological male figure? | Oceanus | The central mythological male figure is a statue. | The central mythological male figure is the one in the middle of the group of statues, who appears to be the focal point of the scene. He is standing upright and is holding what seems to be a spear or staff. This figure is part of the larger group of statues that are situated on the façade of the building, which is part of the Trevi Fountain in Rome, Italy. |  |  | -9.000 |
| trevi_fountain | judgelm | component_presence | What visible component matches this description: the central mythological male figure? | Oceanus | The central mythological male figure is a statue. | The central mythological male figure is the one standing upright with a beard, located on the right side of the façade. He is holding what appears to be a scroll or a book in his left hand and has a muscular build. This figure is part of the sculptural group that adorns the central part of the building, above the cascading water. |  |  | -9.000 |
| trevi_fountain | judgelm | component_presence | What visible component matches this description: the central mythological male figure? | Oceanus | The central mythological male figure is a statue. | The central figure is a male statue, likely representing a god or mythological character, who is standing with one foot on a rock and the other raised, holding what appears to be a cloth or garment in his left hand. He has a beard and is wearing a draped garment that falls over one shoulder, leaving the other bare. The statue is positioned above a fountain, with water flowing from its base. This figure is part of a larger group of statues that are part of the Trevi Fountain in Rome, Italy. |  |  | -9.000 |
| brandenburg_gate | judgelm | spatial_view_visibility | What sculpture group is centered at the top of this image? | The Quadriga. | The sculpture group centered at the top of this image is the Brandenburg Gate. | The sculpture group centered at the top of the image is the Quadriga, which is a chariot drawn by four horses abreast. It is a famous landmark located on the Brandenburg Gate in Berlin, Germany. The Quadriga is topped with a figure of Victoria, the Roman goddess of victory, holding a laurel wreath and a spear with a circular emblem at its tip. |  |  | -8.000 |
| pantheon_exterior | judgelm | style_history_knowledge | What does the Latin inscription above the columns of the landmark in the image mean? | It states that Marcus Agrippa, son of Lucius, built it when he was consul for the third time. | The Latin inscription above the columns of the landmark in the image reads "MVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVMVM | The Latin inscription above the columns of the landmark reads "MAGRIPPA L. F. COS. TERTIVM IMP. FECIT," which translates to "Magrippa, son of Lucius, made this third time by the third consul." This indicates that the structure was commissioned or built by a person named Magrippa, who was the son of Lucius and held the title of consul for the third time. The term "imperium" suggests that he may have been an emperor or held a high-ranking position in the Roman Empire at the time. |  |  | -7.000 |
| taj_mahal | judgelm | component_presence | What visible component matches this description: large pointed arch openings? | arch-shaped doorways | The main dome of the Taj Mahal. | The large pointed arch openings are the domes on the main building. |  |  | -7.000 |
| trevi_fountain | judgelm | component_presence | What visible component matches this description: the central mythological male figure? | Oceanus | The central mythological male figure is a statue. | The central mythological male figure is the statue of Oceanus. |  |  | -7.000 |
| brandenburg_gate | judgelm | spatial_view_visibility | Which direction are the horses facing in the image? | Toward the front-right direction in the image. | The horses are facing left. | The horses in the image are facing forward, with their heads turned slightly to the left. This is evident from the position of their ears and the direction of their manes, which flow towards the left side of the image. |  |  | -6.000 |
| pantheon_exterior | judgelm | style_history_knowledge | What does the Latin inscription above the columns of the landmark in the image mean? | It states that Marcus Agrippa, son of Lucius, built it when he was consul for the third time. | The Latin inscription above the columns of the landmark in the image reads "MAGISTERIUM VIVENS," which translates to "Living Masterpiece" in English. This phrase is often used to describe a work of art or architecture that is considered timeless and of enduring value. | The Latin inscription above the columns of the landmark reads "MAGRIFFAL L.F. COSTERTIVM FECIT," which translates to "Magrippa, son of Lucius, made this." This indicates that the structure was built or sponsored by a person named Magrippa, who was likely a son of someone named Lucius. The term "FECIT" is a common Latin verb meaning "made" or "built," suggesting that Magrippa was responsible for the construction or dedication of the building. |  |  | -6.000 |
| taj_mahal | judgelm | component_presence | What visible component matches this description: large pointed arch openings? | arch-shaped doorways | The main entrance of the Taj Mahal. | The component that matches this description is the Main Gate. |  |  | -6.000 |
| taj_mahal | judgelm | component_presence | What visible component matches this description: large pointed arch openings? | arch-shaped doorways | The main building of the Taj Mahal. | arch |  |  | -6.000 |

## 7. 文件

- `summary.json`：机器可读摘要。
- `objective_by_category.csv`, `objective_by_scene.csv`, `objective_by_scene_category.csv`：Objective 分解。
- `judgelm_by_category.csv`, `judgelm_by_scene.csv`, `judgelm_by_scene_category.csv`：JudgeLM 分解。
- `bad_case_side_by_side_all.csv`, `bad_case_side_by_side_selected.csv`：同题 Expected / V18 / RGB 坏例对照。
- `bad_case_pair_patterns.csv`, `bad_case_method_error_patterns.csv`：错误模式计数。
- `objective_ours_only_correct.csv`, `objective_rgb_only_correct.csv`, `objective_both_wrong.csv`, `judgelm_ours_higher.csv`, `judgelm_rgb_higher.csv`：逐题抽样。
