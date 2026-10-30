# CDLNet 修复版退化诊断（research split，seed 17）

数据：CDLNet 修复版（单阶段 CE、256 输入、40 epoch）与同一 research split、同一种子 17 的干净训练基线
（ResNet-50 / ConvNeXt-Tiny / Swin-Tiny，224 输入、40 epoch）的 43 条件逐样本预测，全部指标由保存预测重算。
CDLNet 是修复版复现，不是作者原成绩（详见 `reports/CDLNET_REPRO.md`）。

主实验为民船 8 类；全 23 类作为外部模型参照单列。CDLNet 的干净集与基线接近或略低，因此同时给出
**保持率**（干净正确样本在退化后仍正确的比例），避免把“起点低”误读成“退化更敏感”。

## 民船主实验（8 类）

### 总体落差（Macro-F1）

| 条件族（Macro-F1，括号内为相对干净的变化） | CDLNet(修复版) | ResNet-50 | ConvNeXt-Tiny | Swin-Tiny |
|---|---|---|---|---|
| 干净 | 84.3 | 80.0 | 84.3 | 87.4 |
| 单退化 matched | 68.8 (-15.5) | 73.1 (-6.9) | 80.2 (-4.1) | 79.4 (-8.0) |
| 单退化 shifted | 70.1 (-14.2) | 72.6 (-7.4) | 79.2 (-5.1) | 81.6 (-5.7) |
| 组合 matched | 33.6 (-50.7) | 42.2 (-37.7) | 60.6 (-23.7) | 56.7 (-30.7) |
| 组合 shifted | 37.9 (-46.4) | 45.4 (-34.6) | 57.8 (-26.5) | 60.7 (-26.7) |

“组合”= 雾+眩光 / 雾+云，各 3、5 级，共 4 条。matched 组合均值：CDLNet(修复版) 33.6、ResNet-50 42.2、ConvNeXt-Tiny 60.6、Swin-Tiny 56.7；
CDLNet 的干净集比同骨干 ResNet-50 基线高 4.3 个百分点。

### 干净正确样本的保持率

保持率 = 在干净图上预测正确的样本中，退化后仍然正确的比例（15 或 4 条条件合并计算）。

| 条件族 | CDLNet(修复版) | ResNet-50 | ConvNeXt-Tiny | Swin-Tiny |
|---|---|---|---|---|
| 单退化 matched | 81.8 | 84.4 | 93.1 | 90.4 |
| 单退化 shifted | 81.5 | 84.0 | 91.8 | 92.5 |
| 组合 matched | 44.5 | 48.0 | 72.0 | 67.4 |
| 组合 shifted | 47.6 | 50.5 | 68.7 | 72.0 |

这个口径把“干净起点”扣掉了。matched 单退化保持率：CDLNet(修复版) 81.8、ResNet-50 84.4、ConvNeXt-Tiny 93.1、Swin-Tiny 90.4；matched 组合保持率：CDLNet(修复版) 44.5、ResNet-50 48.0、ConvNeXt-Tiny 72.0、Swin-Tiny 67.4。

### 逐级曲线

CDLNet：

| 退化 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| haze（matched） | 80.0 | 77.4 | 71.8 | 58.8 | 33.6 |
| haze（shifted） | 81.7 | 77.1 | 69.8 | 59.8 | 30.6 |
| cloud（matched） | 83.4 | 79.8 | 75.6 | 60.3 | 42.9 |
| cloud（shifted） | 84.8 | 77.0 | 72.8 | 52.2 | 37.3 |
| glare（matched） | 83.9 | 82.2 | 79.0 | 68.6 | 55.0 |
| glare（shifted） | 84.3 | 82.2 | 84.4 | 82.3 | 74.9 |

基线对照（matched）：

| 模型 / 退化（matched） | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| ResNet-50 haze | 78.2 | 77.3 | 76.0 | 68.6 | 57.8 |
| ResNet-50 cloud | 79.7 | 81.3 | 78.3 | 71.4 | 52.7 |
| ResNet-50 glare | 81.1 | 83.3 | 79.8 | 71.3 | 60.2 |
| ConvNeXt-Tiny haze | 84.4 | 83.5 | 81.3 | 76.7 | 72.3 |
| ConvNeXt-Tiny cloud | 83.3 | 84.3 | 83.9 | 80.9 | 72.8 |
| ConvNeXt-Tiny glare | 83.9 | 84.8 | 80.9 | 79.1 | 71.2 |
| Swin-Tiny haze | 85.1 | 84.7 | 81.2 | 78.6 | 76.1 |
| Swin-Tiny cloud | 87.8 | 84.6 | 81.3 | 76.3 | 63.2 |
| Swin-Tiny glare | 87.0 | 83.7 | 78.5 | 76.8 | 66.0 |

### 类别级变化（CDLNet，按单退化跌幅排序）

| 类别 | 测试数 | 干净召回 | 单退化均值 | 组合均值 | 单退化跌幅(pp) | 组合跌幅(pp) |
|---|---|---|---|---|---|---|
| passenger ship | 18 | 94.4 | 66.3 | 25.0 | 28.1 | 69.4 |
| barge | 13 | 92.3 | 69.7 | 15.4 | 22.6 | 76.9 |
| container ship | 18 | 77.8 | 63.0 | 27.8 | 14.8 | 50.0 |
| bulk carrier | 69 | 88.4 | 75.1 | 51.8 | 13.3 | 36.6 |
| fishing boat | 20 | 50.0 | 39.0 | 20.0 | 11.0 | 30.0 |
| liquefied gas ship | 51 | 100.0 | 90.6 | 45.1 | 9.4 | 54.9 |
| oil tanker | 33 | 75.8 | 68.3 | 44.7 | 7.5 | 31.1 |
| car carrier | 14 | 71.4 | 76.7 | 50.0 | -5.2 | 21.4 |

小样本类别（如测试数低于 25 的类别）单类召回波动大，需结合“测试数”列解读。
各模型单退化跌幅最大的 5 类对比：

| 模型 | 单退化跌幅前 5 类 | 与 CDLNet 前 5 的重合数 |
|---|---|---|
| ResNet-50 | barge、bulk carrier、passenger ship、car carrier、oil tanker | 3 |
| ConvNeXt-Tiny | bulk carrier、car carrier、barge、oil tanker、container ship | 3 |
| Swin-Tiny | fishing boat、bulk carrier、car carrier、oil tanker、liquefied gas ship | 2 |

### 混淆结构

干净图（CDLNet）前 8 个错误对：

| 真实 | 预测 | 次数 |
|---|---|---|
| fishing boat | bulk carrier | 9 |
| oil tanker | bulk carrier | 8 |
| bulk carrier | oil tanker | 4 |
| container ship | bulk carrier | 3 |
| car carrier | bulk carrier | 2 |
| bulk carrier | fishing boat | 2 |
| bulk carrier | container ship | 2 |
| container ship | oil tanker | 1 |

matched 五级雾+眩光（CDLNet）：

| 真实 | 预测 | 次数 |
|---|---|---|
| liquefied gas ship | bulk carrier | 41 |
| oil tanker | bulk carrier | 20 |
| bulk carrier | oil tanker | 19 |
| passenger ship | bulk carrier | 13 |
| container ship | bulk carrier | 11 |
| fishing boat | bulk carrier | 11 |
| barge | bulk carrier | 9 |
| bulk carrier | car carrier | 7 |

同一条件下 ConvNeXt-Tiny 的前 8 个错误对：

| 真实 | 预测 | 次数 |
|---|---|---|
| bulk carrier | oil tanker | 13 |
| bulk carrier | liquefied gas ship | 11 |
| liquefied gas ship | barge | 10 |
| container ship | bulk carrier | 8 |
| liquefied gas ship | fishing boat | 8 |
| barge | liquefied gas ship | 8 |
| fishing boat | bulk carrier | 7 |
| oil tanker | liquefied gas ship | 6 |

shifted 五级雾+云（CDLNet）：

| 真实 | 预测 | 次数 |
|---|---|---|
| liquefied gas ship | fishing boat | 23 |
| bulk carrier | car carrier | 22 |
| liquefied gas ship | bulk carrier | 18 |
| oil tanker | bulk carrier | 17 |
| fishing boat | bulk carrier | 10 |
| passenger ship | bulk carrier | 10 |
| container ship | car carrier | 7 |
| bulk carrier | fishing boat | 7 |

### 预测塌缩

各模型首位预测类别及占比：

| 模型 | clean 首位预测 | matched 5 级雾+眩光首位预测 | shifted 5 级雾+云首位预测 |
|---|---|---|---|
| CDLNet(修复版) | bulk carrier (35.2%) | bulk carrier (64.0%) | bulk carrier (41.5%) |
| ResNet-50 | bulk carrier (29.7%) | oil tanker (63.1%) | fishing boat (33.5%) |
| ConvNeXt-Tiny | bulk carrier (34.7%) | liquefied gas ship (27.5%) | liquefied gas ship (30.9%) |
| Swin-Tiny | bulk carrier (31.4%) | liquefied gas ship (40.3%) | liquefied gas ship (34.7%) |

matched 五级雾+眩光下，CDLNet 的首位预测为 bulk carrier（64.0%），与基线中最高占比相差 +0.8 个百分点。所有模型在该条件下的首位类别占比都在 20% 以上（27.5%–64.0%），单一类别塌缩是共性问题。

### 高置信错误与翻转

高置信错误 = 预测错误且最大 softmax ≥ 0.9；“干净对→错”= 干净图正确、该条件下错误的样本数。

| 条件 | 错误数 | 高置信错误(≥0.9) | 占错误比例(%) | 干净对→错（CDLNet） | 干净对→错（ConvNeXt） |
|---|---|---|---|---|---|
| clean | 36 | 10 | 27.8 | 0 | 0 |
| matched_haze_5 | 128 | 17 | 13.3 | 102 | 23 |
| matched_cloud_5 | 122 | 11 | 9.0 | 95 | 38 |
| matched_glare_5 | 93 | 11 | 11.8 | 64 | 32 |
| matched_haze+glare_5 | 179 | 5 | 2.8 | 148 | 100 |
| matched_haze+cloud_5 | 190 | 8 | 4.2 | 158 | 86 |
| shifted_haze+cloud_5 | 187 | 4 | 2.1 | 161 | 102 |

干净图上 CDLNet 的 36 个错误中有 10 个（27.8%）置信度 ≥0.9；
matched 五级组合下该比例降到 2.8%–4.2%，即组合退化下的错误大多不再伴随高置信，
更接近“结构已丢失”而非“细节混淆”。

### 位置对照与分组 bootstrap 区间

三级位置对照（Macro-F1）：

| 模型 | 云-中心 | 云-边缘 | 眩光-中心 | 眩光-边缘 |
|---|---|---|---|---|
| CDLNet(修复版) | 73.2 | 78.6 | 62.1 | 83.9 |
| ResNet-50 | 79.4 | 79.0 | 64.0 | 80.4 |
| ConvNeXt-Tiny | 82.2 | 82.1 | 78.6 | 83.1 |
| Swin-Tiny | 78.8 | 84.0 | 74.8 | 86.9 |

CDLNet 分组 bootstrap（按 source_group_id 重采样）：

| 条件 | Macro-F1 | 2.5% | 97.5% |
|---|---|---|---|
| clean | 84.3 | 78.9 | 88.7 |
| matched_haze_5 | 33.6 | 26.2 | 40.1 |
| matched_cloud_5 | 42.9 | 35.9 | 49.9 |
| matched_glare_5 | 55.0 | 48.0 | 61.8 |
| matched_haze+glare_5 | 11.3 | 7.5 | 14.8 |
| matched_haze+cloud_5 | 12.9 | 9.3 | 16.4 |
| shifted_haze+cloud_5 | 14.1 | 9.9 | 18.9 |

### 该口径下的判断

- 4 条 matched 组合（3、5 级）的 Macro-F1 在 11.3–55.7 之间，全部 43 条件最差为
  matched_haze+glare_5（11.3）；matched 单退化均值仍为 68.8。组合退化不是单退化曲线的线性外推。
- 即使扣掉干净集起点，CDLNet 的 matched 组合保持率仍比保持率最高的基线 ConvNeXt-Tiny（72.0%）低 27.5 个百分点。这与“CDLNet 特有分支（区域选择/响应融合/门控注意力）在低对比度下失效”的假设一致，
  但本报告没有区域响应可视化，不能直接断言。
- 单退化跌幅前 5 类与各基线的重合数：ResNet-50 3/5、ConvNeXt-Tiny 3/5、Swin-Tiny 2/5。重合的类别更可能是数据集共享的细粒度瓶颈；
  只在 CDLNet 上出现的跌幅类别，才值得优先检查区域选择/响应融合分支。
- 高置信错误比例用于判断错误是否“结构已丢失”：比例越高，越不应把错误当作可学习的细节差异。


## 全 23 类（外部模型参照）

### 总体落差（Macro-F1）

| 条件族（Macro-F1，括号内为相对干净的变化） | CDLNet(修复版) | ResNet-50 | ConvNeXt-Tiny | Swin-Tiny |
|---|---|---|---|---|
| 干净 | 86.8 | 89.0 | 90.7 | 89.0 |
| 单退化 matched | 73.3 (-13.6) | 76.6 (-12.5) | 82.9 (-7.8) | 80.1 (-9.0) |
| 单退化 shifted | 72.8 (-14.0) | 77.3 (-11.8) | 82.3 (-8.4) | 81.2 (-7.8) |
| 组合 matched | 40.5 (-46.3) | 42.6 (-46.4) | 54.7 (-36.0) | 54.9 (-34.1) |
| 组合 shifted | 39.2 (-47.6) | 44.9 (-44.1) | 57.4 (-33.3) | 56.9 (-32.2) |

“组合”= 雾+眩光 / 雾+云，各 3、5 级，共 4 条。matched 组合均值：CDLNet(修复版) 40.5、ResNet-50 42.6、ConvNeXt-Tiny 54.7、Swin-Tiny 54.9；
CDLNet 的干净集比同骨干 ResNet-50 基线低 2.2 个百分点。

### 干净正确样本的保持率

保持率 = 在干净图上预测正确的样本中，退化后仍然正确的比例（15 或 4 条条件合并计算）。

| 条件族 | CDLNet(修复版) | ResNet-50 | ConvNeXt-Tiny | Swin-Tiny |
|---|---|---|---|---|
| 单退化 matched | 84.4 | 85.0 | 90.4 | 88.7 |
| 单退化 shifted | 83.4 | 85.0 | 89.6 | 89.2 |
| 组合 matched | 50.2 | 47.0 | 60.5 | 61.0 |
| 组合 shifted | 49.8 | 48.0 | 63.4 | 62.1 |

这个口径把“干净起点”扣掉了。matched 单退化保持率：CDLNet(修复版) 84.4、ResNet-50 85.0、ConvNeXt-Tiny 90.4、Swin-Tiny 88.7；matched 组合保持率：CDLNet(修复版) 50.2、ResNet-50 47.0、ConvNeXt-Tiny 60.5、Swin-Tiny 61.0。

### 逐级曲线

CDLNet：

| 退化 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| haze（matched） | 84.5 | 82.0 | 72.7 | 59.8 | 46.1 |
| haze（shifted） | 84.9 | 80.9 | 72.5 | 57.1 | 40.9 |
| cloud（matched） | 86.1 | 83.5 | 76.4 | 65.2 | 50.0 |
| cloud（shifted） | 87.6 | 82.5 | 74.2 | 56.6 | 40.5 |
| glare（matched） | 86.9 | 85.6 | 82.3 | 77.4 | 60.3 |
| glare（shifted） | 86.8 | 86.6 | 83.7 | 82.4 | 74.3 |

基线对照（matched）：

| 模型 / 退化（matched） | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| ResNet-50 haze | 87.3 | 84.9 | 82.3 | 73.8 | 59.4 |
| ResNet-50 cloud | 88.8 | 86.0 | 79.9 | 65.5 | 47.2 |
| ResNet-50 glare | 88.6 | 87.3 | 83.0 | 73.8 | 60.7 |
| ConvNeXt-Tiny haze | 88.7 | 87.9 | 85.1 | 82.2 | 76.7 |
| ConvNeXt-Tiny cloud | 89.9 | 88.0 | 83.2 | 78.0 | 62.8 |
| ConvNeXt-Tiny glare | 89.9 | 89.7 | 87.2 | 82.1 | 71.4 |
| Swin-Tiny haze | 87.6 | 86.6 | 83.3 | 79.7 | 72.4 |
| Swin-Tiny cloud | 88.8 | 85.0 | 81.8 | 73.4 | 63.0 |
| Swin-Tiny glare | 88.6 | 87.1 | 83.3 | 76.1 | 64.3 |

### 类别级变化（CDLNet，按单退化跌幅排序）

| 类别 | 测试数 | 干净召回 | 单退化均值 | 组合均值 | 单退化跌幅(pp) | 组合跌幅(pp) |
|---|---|---|---|---|---|---|
| combat boat | 29 | 89.7 | 62.8 | 18.1 | 26.9 | 71.6 |
| barge | 13 | 92.3 | 67.2 | 17.3 | 25.1 | 75.0 |
| fishing boat | 20 | 55.0 | 33.0 | 5.0 | 22.0 | 50.0 |
| passenger ship | 18 | 77.8 | 57.8 | 12.5 | 20.0 | 65.3 |
| landing craft | 22 | 81.8 | 62.4 | 29.5 | 19.4 | 52.3 |
| submarine | 48 | 93.8 | 76.0 | 47.9 | 17.8 | 45.8 |
| hovercraft | 24 | 100.0 | 82.2 | 36.5 | 17.8 | 63.5 |
| bulk carrier | 69 | 75.4 | 57.7 | 22.5 | 17.7 | 52.9 |

小样本类别（如测试数低于 25 的类别）单类召回波动大，需结合“测试数”列解读。
各模型单退化跌幅最大的 5 类对比：

| 模型 | 单退化跌幅前 5 类 | 与 CDLNet 前 5 的重合数 |
|---|---|---|
| ResNet-50 | combat boat、Tarawa-class amphibious assault ship、command ship、landing craft、passenger ship | 3 |
| ConvNeXt-Tiny | fishing boat、non-ship、auxiliary ship、command ship、Tarawa-class amphibious assault ship | 1 |
| Swin-Tiny | non-ship、bulk carrier、landing craft、command ship、fishing boat | 2 |

### 混淆结构

干净图（CDLNet）前 8 个错误对：

| 真实 | 预测 | 次数 |
|---|---|---|
| non-ship | bulk carrier | 6 |
| bulk carrier | non-ship | 6 |
| non-ship | auxiliary ship | 4 |
| fishing boat | bulk carrier | 4 |
| destroyer | cruiser | 4 |
| non-ship | passenger ship | 3 |
| container ship | bulk carrier | 3 |
| bulk carrier | oil tanker | 3 |

matched 五级雾+眩光（CDLNet）：

| 真实 | 预测 | 次数 |
|---|---|---|
| frigate | destroyer | 18 |
| destroyer | cruiser | 15 |
| liquefied gas ship | cruiser | 15 |
| non-ship | cruiser | 14 |
| frigate | cruiser | 14 |
| cruiser | destroyer | 14 |
| auxiliary ship | non-ship | 12 |
| hovercraft | non-ship | 12 |

同一条件下 ConvNeXt-Tiny 的前 8 个错误对：

| 真实 | 预测 | 次数 |
|---|---|---|
| frigate | destroyer | 35 |
| non-ship | destroyer | 32 |
| bulk carrier | destroyer | 23 |
| cruiser | destroyer | 23 |
| non-ship | submarine | 20 |
| non-ship | frigate | 19 |
| auxiliary ship | destroyer | 19 |
| liquefied gas ship | frigate | 17 |

shifted 五级雾+云（CDLNet）：

| 真实 | 预测 | 次数 |
|---|---|---|
| non-ship | liquefied gas ship | 23 |
| non-ship | destroyer | 18 |
| cruiser | destroyer | 18 |
| non-ship | frigate | 16 |
| bulk carrier | destroyer | 14 |
| frigate | destroyer | 14 |
| destroyer | cruiser | 13 |
| auxiliary ship | destroyer | 12 |

### 预测塌缩

各模型首位预测类别及占比：

| 模型 | clean 首位预测 | matched 5 级雾+眩光首位预测 | shifted 5 级雾+云首位预测 |
|---|---|---|---|
| CDLNet(修复版) | destroyer (13.1%) | non-ship (20.3%) | destroyer (19.5%) |
| ResNet-50 | destroyer (12.7%) | fishing boat (25.6%) | fishing boat (48.1%) |
| ConvNeXt-Tiny | destroyer (13.3%) | destroyer (45.4%) | destroyer (35.0%) |
| Swin-Tiny | destroyer (12.7%) | liquefied gas ship (19.5%) | frigate (15.5%) |

matched 五级雾+眩光下，CDLNet 的首位预测为 non-ship（20.3%），与基线中最高占比相差 -25.1 个百分点。该条件下各模型首位类别占比为 19.5%–45.4%。

### 高置信错误与翻转

高置信错误 = 预测错误且最大 softmax ≥ 0.9；“干净对→错”= 干净图正确、该条件下错误的样本数。

| 条件 | 错误数 | 高置信错误(≥0.9) | 占错误比例(%) | 干净对→错（CDLNet） | 干净对→错（ConvNeXt） |
|---|---|---|---|---|---|
| clean | 112 | 48 | 42.9 | 0 | 0 |
| matched_haze_5 | 391 | 68 | 17.4 | 295 | 137 |
| matched_cloud_5 | 405 | 82 | 20.2 | 310 | 267 |
| matched_glare_5 | 297 | 99 | 33.3 | 206 | 154 |
| matched_haze+glare_5 | 641 | 67 | 10.5 | 546 | 498 |
| matched_haze+cloud_5 | 640 | 77 | 12.0 | 537 | 516 |
| shifted_haze+cloud_5 | 678 | 63 | 9.3 | 576 | 543 |

干净图上 CDLNet 的 112 个错误中有 48 个（42.9%）置信度 ≥0.9；
matched 五级组合下该比例降到 10.5%–12.0%，即组合退化下的错误大多不再伴随高置信，
更接近“结构已丢失”而非“细节混淆”。

### 位置对照与分组 bootstrap 区间

三级位置对照（Macro-F1）：

| 模型 | 云-中心 | 云-边缘 | 眩光-中心 | 眩光-边缘 |
|---|---|---|---|---|
| CDLNet(修复版) | 73.2 | 75.8 | 77.0 | 86.7 |
| ResNet-50 | 75.7 | 79.0 | 75.3 | 88.2 |
| ConvNeXt-Tiny | 82.6 | 82.4 | 83.3 | 89.9 |
| Swin-Tiny | 76.8 | 81.9 | 74.5 | 88.5 |

CDLNet 分组 bootstrap（按 source_group_id 重采样）：

| 条件 | Macro-F1 | 2.5% | 97.5% |
|---|---|---|---|
| clean | 86.8 | 84.0 | 88.9 |
| matched_haze_5 | 46.1 | 42.1 | 49.5 |
| matched_cloud_5 | 50.0 | 45.7 | 53.2 |
| matched_glare_5 | 60.3 | 56.2 | 63.3 |
| matched_haze+glare_5 | 18.7 | 15.7 | 21.4 |
| matched_haze+cloud_5 | 20.7 | 17.3 | 23.7 |
| shifted_haze+cloud_5 | 12.8 | 10.7 | 14.8 |

### 该口径下的判断

- 4 条 matched 组合（3、5 级）的 Macro-F1 在 18.7–62.9 之间，全部 43 条件最差为
  shifted_haze+cloud_5（12.8）；matched 单退化均值仍为 73.3。组合退化不是单退化曲线的线性外推。
- 即使扣掉干净集起点，CDLNet 的 matched 组合保持率仍比保持率最高的基线 Swin-Tiny（61.0%）低 10.8 个百分点。这与“CDLNet 特有分支（区域选择/响应融合/门控注意力）在低对比度下失效”的假设一致，
  但本报告没有区域响应可视化，不能直接断言。
- 单退化跌幅前 5 类与各基线的重合数：ResNet-50 3/5、ConvNeXt-Tiny 1/5、Swin-Tiny 2/5。重合的类别更可能是数据集共享的细粒度瓶颈；
  只在 CDLNet 上出现的跌幅类别，才值得优先检查区域选择/响应融合分支。
- 高置信错误比例用于判断错误是否“结构已丢失”：比例越高，越不应把错误当作可学习的细节差异。


## 方法说明与限制

- 保持率、翻转、高置信错误的统计都要求逐样本配对，因此只使用同一条条件内全部测试样本；区间按
  `source_group_id` 做整组重采样。
- 混淆表按错误次数排序，不按比例；类别支持度差异很大，跨条件比较时需同时看“测试数”。
- 类别跌幅取 15 条 matched 单退化、4 条 matched 组合的平均召回，异常值可能来自单条条件或小样本类别。
- CDLNet 的 256 输入与基线的 224 输入不同，这是修复版复现的既定设置；分辨率差会同时影响干净与退化结果，
  不能把差值全部归给架构。
- 本页只做已知框分类的诊断，不涉及检测或跟踪，也不支持船只身份结论。

## 复现

```bash
.venv/bin/python scripts/cdlnet_report.py
.venv/bin/python scripts/cdlnet_diagnostics.py
```
