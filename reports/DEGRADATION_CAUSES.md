# 退化性能下降的机制诊断

数据只来自组合增强基线（民船 8 类、seed 17、256 条件中的关键项），所有指标由保存的逐样本预测重算。目标是回答“下降来自哪里、哪部分还可恢复”，而不是再列一遍准确率。

## 1. 翻转：干净答对、退化答错

| 条件 | 模型 | 准确率 | 干净答对样本 | 翻转数 | 翻转率 | 错误均值置信度 | 高置信错误占比 |
|---|---|---:|---:|---:|---:|---:|---:|
| clean | ResNet-50 | 89.8 | 212 | 0 | 0.0 | 76.2 | 33.3 |
| clean | ConvNeXt-Tiny | 87.7 | 207 | 0 | 0.0 | 83.0 | 48.3 |
| clean | Swin-Tiny | 87.7 | 207 | 0 | 0.0 | 81.5 | 44.8 |
| matched_haze_5 | ResNet-50 | 82.6 | 212 | 23 | 10.8 | 67.6 | 19.5 |
| matched_haze_5 | ConvNeXt-Tiny | 85.6 | 207 | 9 | 4.3 | 79.2 | 41.2 |
| matched_haze_5 | Swin-Tiny | 86.4 | 207 | 6 | 2.9 | 82.5 | 40.6 |
| matched_cloud_5 | ResNet-50 | 83.1 | 212 | 22 | 10.4 | 68.2 | 22.5 |
| matched_cloud_5 | ConvNeXt-Tiny | 86.9 | 207 | 6 | 2.9 | 81.2 | 48.4 |
| matched_cloud_5 | Swin-Tiny | 85.2 | 207 | 10 | 4.8 | 82.1 | 45.7 |
| matched_glare_5 | ResNet-50 | 74.6 | 212 | 40 | 18.9 | 65.8 | 13.3 |
| matched_glare_5 | ConvNeXt-Tiny | 82.6 | 207 | 21 | 10.1 | 76.1 | 43.9 |
| matched_glare_5 | Swin-Tiny | 79.7 | 207 | 24 | 11.6 | 79.7 | 43.8 |
| matched_haze+glare_5 | ResNet-50 | 47.0 | 212 | 106 | 50.0 | 45.2 | 1.6 |
| matched_haze+glare_5 | ConvNeXt-Tiny | 65.7 | 207 | 61 | 29.5 | 64.8 | 18.5 |
| matched_haze+glare_5 | Swin-Tiny | 63.1 | 207 | 65 | 31.4 | 63.1 | 19.5 |
| matched_haze+cloud_5 | ResNet-50 | 71.2 | 212 | 51 | 24.1 | 60.8 | 8.8 |
| matched_haze+cloud_5 | ConvNeXt-Tiny | 82.2 | 207 | 21 | 10.1 | 74.0 | 26.2 |
| matched_haze+cloud_5 | Swin-Tiny | 80.1 | 207 | 26 | 12.6 | 78.9 | 38.3 |

解读：翻转率的分母是该模型在干净集答对的样本，衡量“已学会的样本在退化下丢了多少”；错误均值置信度越高，说明模型不是不知道，而是自信地错。

## 2. 跨模型共识：信息还在不在

三个架构（ResNet-50 / ConvNeXt-Tiny / Swin-Tiny，同种子、同组合增强）对同一测试样本的判定。前两列以全部样本为分母；后三列以该模型自身的错误为分母，衡量它的错误里有多少是“三个模型都不会”。

| 条件 | 三模型全错样本占比 | 三模型全对占比 | Swin 错误中全错 | ConvNeXt 错误中全错 | ResNet 错误中全错 | 样本数分布 (0/1/2/3 模型对) |
|---|---:|---:|---:|---:|---:|---|
| clean | 5.5% | 82.2% | 44.8% | 44.8% | 54.2% | 13/14/15/194 |
| matched_haze_5 | 7.6% | 75.4% | 56.2% | 52.9% | 43.9% | 18/13/27/178 |
| matched_cloud_5 | 7.6% | 75.8% | 51.4% | 58.1% | 45.0% | 18/13/26/179 |
| matched_glare_5 | 9.7% | 66.5% | 47.9% | 56.1% | 38.3% | 23/24/32/157 |
| matched_haze+glare_5 | 21.2% | 35.6% | 57.5% | 61.7% | 40.0% | 50/41/61/84 |
| matched_haze+cloud_5 | 9.3% | 61.9% | 46.8% | 52.4% | 32.4% | 22/23/45/146 |

全错的样本更可能是信息已不可辨认；至少一个模型答对的样本，说明判别信息仍在，下降来自模型利用信息的方式，这是新模块的可争取空间。

## 3. 物理参数与错误的关系

透射率越低表示雾越厚；云面积、受影响面积、饱和面积越大表示遮挡/过曝越重。ρ 为参数与逐样本正确性（1=对）的 Spearman 相关，负值表示参数越大越容易错。

### haze

| 参数 | 模型 | 合并 ρ | 严重度内平均 ρ | 五级低三分位准确率 | 五级高三分位准确率 |
|---|---|---:|---:|---:|---:|
| transmission_mean | ResNet-50 | +0.05 | -0.07 | 89.7 | 78.2 |
| transmission_mean | ConvNeXt-Tiny | +0.01 | -0.03 | 89.7 | 85.9 |
| transmission_mean | Swin-Tiny | +0.01 | -0.04 | 92.3 | 89.7 |
| center_proxy_affected_fraction | ResNet-50 | — | — | — | — |
| center_proxy_affected_fraction | ConvNeXt-Tiny | — | — | — | — |
| center_proxy_affected_fraction | Swin-Tiny | — | — | — | — |

### cloud

| 参数 | 模型 | 合并 ρ | 严重度内平均 ρ | 五级低三分位准确率 | 五级高三分位准确率 |
|---|---|---:|---:|---:|---:|
| cloud_area_fraction | ResNet-50 | -0.04 | +0.02 | 82.1 | 83.3 |
| cloud_area_fraction | ConvNeXt-Tiny | -0.01 | +0.01 | 88.5 | 83.3 |
| cloud_area_fraction | Swin-Tiny | -0.04 | -0.01 | 89.7 | 79.5 |
| affected_fraction | ResNet-50 | -0.03 | +0.08 | 80.8 | 88.5 |
| affected_fraction | ConvNeXt-Tiny | -0.00 | +0.06 | 85.9 | 91.0 |
| affected_fraction | Swin-Tiny | -0.03 | +0.07 | 79.5 | 89.7 |
| center_proxy_affected_fraction | ResNet-50 | -0.07 | -0.05 | 85.9 | 75.6 |
| center_proxy_affected_fraction | ConvNeXt-Tiny | -0.01 | -0.01 | 84.6 | 84.6 |
| center_proxy_affected_fraction | Swin-Tiny | -0.01 | +0.03 | 80.8 | 84.6 |

### glare

| 参数 | 模型 | 合并 ρ | 严重度内平均 ρ | 五级低三分位准确率 | 五级高三分位准确率 |
|---|---|---:|---:|---:|---:|
| saturated_fraction | ResNet-50 | -0.15 | -0.03 | 85.9 | 67.9 |
| saturated_fraction | ConvNeXt-Tiny | -0.05 | +0.00 | 94.9 | 71.8 |
| saturated_fraction | Swin-Tiny | -0.06 | +0.03 | 92.3 | 73.1 |
| affected_fraction | ResNet-50 | -0.17 | -0.09 | 91.0 | 65.4 |
| affected_fraction | ConvNeXt-Tiny | -0.08 | -0.06 | 91.0 | 69.2 |
| affected_fraction | Swin-Tiny | -0.10 | -0.06 | 88.5 | 67.9 |
| center_proxy_affected_fraction | ResNet-50 | -0.17 | -0.07 | 88.5 | 61.5 |
| center_proxy_affected_fraction | ConvNeXt-Tiny | -0.06 | -0.01 | 89.7 | 67.9 |
| center_proxy_affected_fraction | Swin-Tiny | -0.09 | -0.03 | 88.5 | 66.7 |

### haze+glare

| 参数 | 模型 | 合并 ρ | 严重度内平均 ρ | 五级低三分位准确率 | 五级高三分位准确率 |
|---|---|---:|---:|---:|---:|
| transmission_mean | ResNet-50 | +0.32 | +0.03 | 41.0 | 51.3 |
| transmission_mean | ConvNeXt-Tiny | +0.22 | +0.07 | 59.0 | 66.7 |
| transmission_mean | Swin-Tiny | +0.26 | +0.04 | 59.0 | 70.5 |
| saturated_fraction | ResNet-50 | -0.34 | -0.07 | 47.4 | 44.9 |
| saturated_fraction | ConvNeXt-Tiny | -0.27 | -0.16 | 73.1 | 55.1 |
| saturated_fraction | Swin-Tiny | -0.29 | -0.10 | 70.5 | 55.1 |
| center_proxy_affected_fraction | ResNet-50 | — | — | — | — |
| center_proxy_affected_fraction | ConvNeXt-Tiny | — | — | — | — |
| center_proxy_affected_fraction | Swin-Tiny | — | — | — | — |

### haze+cloud

| 参数 | 模型 | 合并 ρ | 严重度内平均 ρ | 五级低三分位准确率 | 五级高三分位准确率 |
|---|---|---:|---:|---:|---:|
| transmission_mean | ResNet-50 | +0.13 | +0.00 | 71.8 | 70.5 |
| transmission_mean | ConvNeXt-Tiny | +0.04 | -0.03 | 84.6 | 80.8 |
| transmission_mean | Swin-Tiny | +0.05 | -0.08 | 88.5 | 74.4 |
| cloud_area_fraction | ResNet-50 | -0.10 | +0.07 | 67.9 | 71.8 |
| cloud_area_fraction | ConvNeXt-Tiny | -0.02 | +0.07 | 78.2 | 82.1 |
| cloud_area_fraction | Swin-Tiny | -0.10 | -0.00 | 84.6 | 74.4 |
| center_proxy_affected_fraction | ResNet-50 | — | — | — | — |
| center_proxy_affected_fraction | ConvNeXt-Tiny | — | — | — | — |
| center_proxy_affected_fraction | Swin-Tiny | — | — | — | — |

## 4. 类别级损失（Swin-Tiny，组合增强）

| 类别 | n | 干净 | 眩光5 | 雾+眩光5 | 雾+云5 | 雾+眩光5 主要去向 |
|---|---:|---:|---:|---:|---:|---|
| container ship | 18 | 77.8 | 72.2 | 61.1 | 77.8 | bulk carrier×6、car carrier×1 |
| car carrier | 14 | 100.0 | 85.7 | 92.9 | 85.7 | fishing boat×1 |
| bulk carrier | 69 | 89.9 | 87.0 | 73.9 | 76.8 | oil tanker×8、fishing boat×4 |
| oil tanker | 33 | 75.8 | 60.6 | 45.5 | 75.8 | bulk carrier×13、liquefied gas ship×2 |
| fishing boat | 20 | 65.0 | 45.0 | 55.0 | 70.0 | bulk carrier×5、passenger ship×3 |
| passenger ship | 18 | 88.9 | 83.3 | 55.6 | 61.1 | bulk carrier×6、fishing boat×2 |
| liquefied gas ship | 51 | 100.0 | 94.1 | 62.7 | 94.1 | barge×13、fishing boat×3 |
| barge | 13 | 92.3 | 84.6 | 46.2 | 92.3 | liquefied gas ship×4、bulk carrier×2 |

## 5. 机制判断

- 五级雾+眩光下 21.2% 的样本三个模型全错；Swin-Tiny 的 87 个错误中 50 个（57.5%）属于这种全错样本，其余 42.5% 至少有一个模型能答对，是可恢复空间。
- 五级眩光下三模型全错只占全部样本的 9.7%，单退化下判别信息基本还在，下降更多来自模型没有利用好，而不是信息消失。（Swin 错误中全错占比 47.9%）
- Swin-Tiny 在五级雾+眩光下错误样本的平均最大置信度 63.1%，其中 19.5% 的错误置信度 ≥0.9：错误不是低置信犹豫，而是高置信误判，置信度本身不能直接当拒识依据。
- haze：退化参数与正确性的合并 |ρ| 均值只有 0.02，同一严重度内 0.05。样本级的失败几乎不能由这些可观测参数解释，说明起作用的不是“强度连续变大”，而是具体图像内容与破坏位置的组合。
- cloud：退化参数与正确性的合并 |ρ| 均值只有 0.03，同一严重度内 0.04。样本级的失败几乎不能由这些可观测参数解释，说明起作用的不是“强度连续变大”，而是具体图像内容与破坏位置的组合。
- glare：退化参数与正确性的合并 |ρ| 均值只有 0.10，同一严重度内 0.04。样本级的失败几乎不能由这些可观测参数解释，说明起作用的不是“强度连续变大”，而是具体图像内容与破坏位置的组合。
- haze+glare：合并 |ρ| 均值 0.28，同一严重度内降到 0.08。合并相关主要来自严重度之间的差异；给定同一强度后，参数对失败样本的区分力有限。
- haze+cloud：退化参数与正确性的合并 |ρ| 均值只有 0.07，同一严重度内 0.04。样本级的失败几乎不能由这些可观测参数解释，说明起作用的不是“强度连续变大”，而是具体图像内容与破坏位置的组合。
- 所有参数中，同一严重度内区分力最强的也只是 haze+glare 的 saturated_fraction（平均 |ρ|=0.11），仍然很弱。失败与否更多由图像内容决定，而不是由某一个可测量的退化强度决定。

## 6. 对新模块的含义

- 可争取空间：五级组合下仍有 78.8% 的样本至少有一个模型答对；Swin-Tiny 的 57.5% 错误属于“三模型全错”，剩下的 42.5% 才是模块的现实目标。
- 单退化下三模型全错只占 7.6%–9.7%，说明增强训练已经把大部分单退化缺口补上；新模块的主战场是组合退化，且不能以牺牲干净图和单退化为代价。
- 失败与全局退化参数在同一严重度内几乎无关，“先估计雾浓度/饱和比例再复原”不是关键；更合理的是让网络判断哪些局部证据已不可信，并在聚合时降低其权重。
- 类别级损失集中在结构相近类别（油轮↔散货船、LPG↔驳船、客船↔散货船），与“判别性部件被遮挡/饱和”一致，而不是全局颜色或亮度偏移。

第一版模块（Reliability-Gated Part Aggregation，RGPA）：在骨干最后一阶段之后，用两路信号计算每个 token 的可靠性——输入侧的局部饱和比例与局部对比度（固定算子，无参数），以及一个 1×1 卷积学到的可靠性头；用可靠性加权池化替代全局平均池化，并保留一条原始 GAP 的残差路径。参数量目标 <0.1M，不改数据配方与训练步数。

验证顺序：先做同骨干、同增强、同种子的“无门控”对照，确认能复现组合增强基线；再加门控，并同时报告干净、单退化、组合退化与成本。

## 复现

```bash
.venv/bin/python scripts/degradation_causes.py
```
