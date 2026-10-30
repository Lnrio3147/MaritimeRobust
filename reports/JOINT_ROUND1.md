# C 线第一轮：联合训练消融（C1/C2/C3）

## 0. 准入规则与指标

- 主指标为 Macro-F1；家族值为该家族条件 Macro-F1 的未加权平均（条件数见 `reports/joint_round1.json`）。
- 配对比较：同一骨干、同一种子、同一评测条件下，变体减组合增强基线，乘 100 得 pp；跨种子取均值与样本 sd。
- 准入：matched 组合 Δ ≥ +1.0 pp，干净 Δ ≥ −1.0 pp，至少 2/3 种子 matched 组合为正，没有种子干净 Δ 低于 −2.0 pp，且 matched 单退化、shifted 单退化、shifted 组合、最差条件四项均值均不低 2.0 pp 以上。
- 三个种子是判定单位；单种子正负不构成结论。
- C2 的验证熵为验证集逐样本预测分布的平均熵（8 类，均匀=2.079）；验证 max prob 为平均首位概率；两者与类别直方图用于发现一致性塌缩。

## 0.1 结论

- `swin_t_c1`：未通过；matched 组合各种子 Δ = +0.4, +4.0, -1.1 pp，均值 +1.1 pp；干净均值 -1.0 pp。
  - 该组种子基线组合为 78.7, 72.6, 74.8；最大增益出现在基线最低的 seed 29，与回归均值一致，不能当作处理效应。
- `swin_t_c2`：未通过；matched 组合各种子 Δ = -6.3, -0.5, +2.2 pp，均值 -1.5 pp；干净均值 -1.1 pp。
- `swin_t_c3`：未通过；matched 组合各种子 Δ = -1.4, +7.6, -3.0 pp，均值 +1.1 pp；干净均值 -0.4 pp。
  - 该组种子基线组合为 78.7, 72.6, 74.8；最大增益出现在基线最低的 seed 29，与回归均值一致，不能当作处理效应。
- `convnext_tiny_c3`：未通过；matched 组合各种子 Δ = +0.9, -1.1, -2.3 pp，均值 -0.8 pp；干净均值 -1.0 pp。

三种子方向不一致时，均值不能作为增益证据；本轮没有任何变体在 matched 组合上同时满足“均值 ≥ +1 pp、至少 2/3 种子为正、没有种子干净图低于 −2 pp”。

## Swin-Tiny / localized（seeds [17, 29, 43]）

**逐种子家族值**

| 模型 | 变体 | seed | 基线 matched 组合 | 变体 matched 组合 | 干净 | matched 单 | shifted 单 | shifted 组合 | 最差 |
|---|---|---|---|---|---|---|---|---|---|
| Swin-Tiny | localized | 17 | 78.7 | 79.1 | 89.9 | 87.4 | 87.8 | 81.2 | 60.7 |
| Swin-Tiny | localized | 29 | 72.6 | 76.6 | 84.6 | 84.7 | 85.2 | 77.6 | 61.7 |
| Swin-Tiny | localized | 43 | 74.8 | 73.6 | 86.9 | 84.4 | 84.4 | 77.1 | 57.2 |

**相对同种子组合基线的配对 Δ**

| 模型 | 变体 | 家族 | Δ pp 均值 | 种子 sd |
|---|---|---|---|---|
| Swin-Tiny | localized | clean | -1.0 | 3.0 |
| Swin-Tiny | localized | matched single | -0.2 | 1.1 |
| Swin-Tiny | localized | shifted single | +0.1 | 0.8 |
| Swin-Tiny | localized | matched combination | +1.1 | 2.6 |
| Swin-Tiny | localized | shifted combination | +0.8 | 1.8 |
| Swin-Tiny | localized | worst | +2.9 | 4.2 |

**训练诊断（C2/C3）**

| 模型 | 变体 | seed | 一致性/特征损失 | 验证熵 | 验证 max prob | 验证预测直方图 |
|---|---|---|---|---|---|---|
| Swin-Tiny | localized | 17 | 0.0000 | — | — | — |
| Swin-Tiny | localized | 29 | 0.0000 | — | — | — |
| Swin-Tiny | localized | 43 | 0.0000 | — | — | — |

**高置信错误（干净 + matched 组合，236×5 样本）**

HCE 定义为错误且首位概率 ≥ 0.9 的样本数 / 该组样本总数。

| 模型 | 变体 | seed | 基线 HCE(%) | 变体 HCE(%) | Δ pp | 基线错误平均置信 | 变体错误平均置信 |
|---|---|---|---|---|---|---|---|
| Swin-Tiny | localized | 17 | 6.3 | 5.9 | -0.3 | 0.737 | 0.743 |
| Swin-Tiny | localized | 29 | 5.2 | 6.9 | +1.7 | 0.708 | 0.723 |
| Swin-Tiny | localized | 43 | 6.9 | 5.9 | -1.0 | 0.735 | 0.695 |

单条件 Δ 最大/最小五项：matched_glare_3_center +3.6；matched_haze+glare_5 +2.9；shifted_haze+glare_5 +2.5；matched_haze+glare_3 +2.0；matched_glare_4 +1.8 / shifted_cloud_3 -1.3；shifted_cloud_2 -1.4；matched_haze+cloud_3 -1.5；matched_glare_1 -1.5；matched_haze_4 -2.3


## Swin-Tiny / consistency（seeds [17, 29, 43]）

**逐种子家族值**

| 模型 | 变体 | seed | 基线 matched 组合 | 变体 matched 组合 | 干净 | matched 单 | shifted 单 | shifted 组合 | 最差 |
|---|---|---|---|---|---|---|---|---|---|
| Swin-Tiny | consistency | 17 | 78.7 | 72.4 | 85.4 | 84.0 | 84.5 | 78.6 | 49.2 |
| Swin-Tiny | consistency | 29 | 72.6 | 72.1 | 85.7 | 83.5 | 83.0 | 72.7 | 53.4 |
| Swin-Tiny | consistency | 43 | 74.8 | 77.0 | 90.0 | 87.1 | 87.8 | 78.5 | 60.9 |

**相对同种子组合基线的配对 Δ**

| 模型 | 变体 | 家族 | Δ pp 均值 | 种子 sd |
|---|---|---|---|---|
| Swin-Tiny | consistency | clean | -1.1 | 2.0 |
| Swin-Tiny | consistency | matched single | -0.8 | 2.2 |
| Swin-Tiny | consistency | shifted single | -0.6 | 2.8 |
| Swin-Tiny | consistency | matched combination | -1.5 | 4.3 |
| Swin-Tiny | consistency | shifted combination | -1.2 | 2.8 |
| Swin-Tiny | consistency | worst | -2.5 | 9.5 |

**训练诊断（C2/C3）**

| 模型 | 变体 | seed | 一致性/特征损失 | 验证熵 | 验证 max prob | 验证预测直方图 |
|---|---|---|---|---|---|---|
| Swin-Tiny | consistency | 17 | 0.2777 | 0.218 | 0.920 | [16, 12, 56, 20, 20, 15, 8, 10] |
| Swin-Tiny | consistency | 29 | 0.2663 | 0.208 | 0.926 | [17, 12, 54, 20, 17, 18, 9, 10] |
| Swin-Tiny | consistency | 43 | 0.2500 | 0.219 | 0.920 | [17, 12, 58, 18, 17, 17, 8, 10] |

**高置信错误（干净 + matched 组合，236×5 样本）**

HCE 定义为错误且首位概率 ≥ 0.9 的样本数 / 该组样本总数。

| 模型 | 变体 | seed | 基线 HCE(%) | 变体 HCE(%) | Δ pp | 基线错误平均置信 | 变体错误平均置信 |
|---|---|---|---|---|---|---|---|
| Swin-Tiny | consistency | 17 | 6.3 | 3.8 | -2.5 | 0.737 | 0.648 |
| Swin-Tiny | consistency | 29 | 5.2 | 6.1 | +0.9 | 0.708 | 0.720 |
| Swin-Tiny | consistency | 43 | 6.9 | 5.5 | -1.4 | 0.735 | 0.751 |

单条件 Δ 最大/最小五项：matched_glare_3_center +2.0；shifted_haze_2 +1.4；shifted_haze+glare_5 +1.1；shifted_haze_5 +1.0；shifted_haze_3 +0.8 / shifted_cloud_5 -2.8；shifted_cloud_3 -2.9；matched_glare_5 -3.1；matched_haze+cloud_5 -3.4；shifted_haze+cloud_5 -3.5


## Swin-Tiny / feature（seeds [17, 29, 43]）

**逐种子家族值**

| 模型 | 变体 | seed | 基线 matched 组合 | 变体 matched 组合 | 干净 | matched 单 | shifted 单 | shifted 组合 | 最差 |
|---|---|---|---|---|---|---|---|---|---|
| Swin-Tiny | feature | 17 | 78.7 | 77.3 | 86.9 | 86.2 | 86.3 | 80.7 | 58.2 |
| Swin-Tiny | feature | 29 | 72.6 | 80.2 | 88.7 | 86.3 | 86.7 | 80.5 | 64.3 |
| Swin-Tiny | feature | 43 | 74.8 | 71.8 | 87.7 | 85.4 | 85.8 | 77.4 | 48.4 |

**相对同种子组合基线的配对 Δ**

| 模型 | 变体 | 家族 | Δ pp 均值 | 种子 sd |
|---|---|---|---|---|
| Swin-Tiny | feature | clean | -0.4 | 0.9 |
| Swin-Tiny | feature | matched single | +0.2 | 0.6 |
| Swin-Tiny | feature | shifted single | +0.5 | 1.1 |
| Swin-Tiny | feature | matched combination | +1.1 | 5.7 |
| Swin-Tiny | feature | shifted combination | +1.7 | 1.9 |
| Swin-Tiny | feature | worst | -0.0 | 8.7 |

**训练诊断（C2/C3）**

| 模型 | 变体 | seed | 一致性/特征损失 | 验证熵 | 验证 max prob | 验证预测直方图 |
|---|---|---|---|---|---|---|
| Swin-Tiny | feature | 17 | 0.0000 | 0.102 | 0.958 | [17, 13, 50, 22, 19, 17, 9, 10] |
| Swin-Tiny | feature | 29 | 0.0000 | 0.128 | 0.949 | [17, 12, 49, 25, 21, 15, 8, 10] |
| Swin-Tiny | feature | 43 | 0.0000 | 0.113 | 0.955 | [17, 12, 53, 23, 16, 17, 9, 10] |

**高置信错误（干净 + matched 组合，236×5 样本）**

HCE 定义为错误且首位概率 ≥ 0.9 的样本数 / 该组样本总数。

| 模型 | 变体 | seed | 基线 HCE(%) | 变体 HCE(%) | Δ pp | 基线错误平均置信 | 变体错误平均置信 |
|---|---|---|---|---|---|---|---|
| Swin-Tiny | feature | 17 | 6.3 | 7.0 | +0.8 | 0.737 | 0.730 |
| Swin-Tiny | feature | 29 | 5.2 | 6.4 | +1.3 | 0.708 | 0.758 |
| Swin-Tiny | feature | 43 | 6.9 | 5.3 | -1.7 | 0.735 | 0.681 |

单条件 Δ 最大/最小五项：shifted_haze+glare_5 +4.3；matched_glare_3_center +4.0；shifted_haze+cloud_5 +2.9；shifted_cloud_5 +2.7；shifted_haze_4 +2.3 / shifted_cloud_3 -0.7；matched_haze_5 -0.8；matched_haze_3 -0.8；shifted_cloud_2 -0.8；shifted_glare_5 -0.9


## ConvNeXt-Tiny / feature（seeds [17, 29, 43]）

**逐种子家族值**

| 模型 | 变体 | seed | 基线 matched 组合 | 变体 matched 组合 | 干净 | matched 单 | shifted 单 | shifted 组合 | 最差 |
|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | feature | 17 | 78.9 | 79.8 | 88.2 | 86.6 | 87.9 | 82.6 | 66.1 |
| ConvNeXt-Tiny | feature | 29 | 78.7 | 77.6 | 85.4 | 84.8 | 85.3 | 78.7 | 64.2 |
| ConvNeXt-Tiny | feature | 43 | 75.2 | 72.9 | 88.4 | 84.9 | 86.3 | 77.7 | 50.9 |

**相对同种子组合基线的配对 Δ**

| 模型 | 变体 | 家族 | Δ pp 均值 | 种子 sd |
|---|---|---|---|---|
| ConvNeXt-Tiny | feature | clean | -1.0 | 2.2 |
| ConvNeXt-Tiny | feature | matched single | -1.0 | 0.6 |
| ConvNeXt-Tiny | feature | shifted single | -0.6 | 1.3 |
| ConvNeXt-Tiny | feature | matched combination | -0.8 | 1.6 |
| ConvNeXt-Tiny | feature | shifted combination | +0.4 | 1.9 |
| ConvNeXt-Tiny | feature | worst | +2.4 | 2.9 |

**训练诊断（C2/C3）**

| 模型 | 变体 | seed | 一致性/特征损失 | 验证熵 | 验证 max prob | 验证预测直方图 |
|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | feature | 17 | 0.0000 | 0.166 | 0.939 | [16, 14, 51, 22, 15, 21, 8, 10] |
| ConvNeXt-Tiny | feature | 29 | 0.0000 | 0.182 | 0.934 | [17, 13, 55, 21, 15, 17, 9, 10] |
| ConvNeXt-Tiny | feature | 43 | 0.0000 | 0.180 | 0.929 | [17, 13, 53, 23, 18, 16, 8, 9] |

**高置信错误（干净 + matched 组合，236×5 样本）**

HCE 定义为错误且首位概率 ≥ 0.9 的样本数 / 该组样本总数。

| 模型 | 变体 | seed | 基线 HCE(%) | 变体 HCE(%) | Δ pp | 基线错误平均置信 | 变体错误平均置信 |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | feature | 17 | 6.1 | 5.8 | -0.3 | 0.741 | 0.755 |
| ConvNeXt-Tiny | feature | 29 | 4.6 | 5.3 | +0.7 | 0.727 | 0.743 |
| ConvNeXt-Tiny | feature | 43 | 5.5 | 4.6 | -0.9 | 0.687 | 0.669 |

单条件 Δ 最大/最小五项：matched_haze+glare_5 +2.4；shifted_haze+glare_5 +1.8；shifted_haze+cloud_5 +1.6；shifted_cloud_4 +1.4；matched_cloud_4 +0.3 / matched_haze+cloud_5 -1.9；matched_haze+cloud_3 -2.0；matched_glare_3_center -2.4；shifted_cloud_5 -2.6；matched_glare_5 -3.5

