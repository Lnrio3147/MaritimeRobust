# 深度条件雾验证（val 157，纯推理）

2026-10-29。目的：确认"给雾加真实深度结构"是否改变冻结分类器在验证集上的表现，
从而判断它应作为**难度手段**还是仅作为**真实性手段**进入新课题的评测。

数据：`data/manifests/fgsc23_research_civilian.jsonl` 的 val 157 张；
分类器：7 个冻结组合增强模型（ConvNeXt-T/Swin-T 各 3 种子、ResNet-50 s17）；
深度：Depth Anything V2 Small，缓存于 `artifacts/depth_cache/depth-anything`（1,011 张，约 1.2 图/s）；
脚本：`scripts/depth_fog_eval.py`；数字：`reports/depth_fog_check.json`（matched）、
`reports/depth_fog_check_shifted.json`（shifted）。未训练任何模型。

三个条件（同一批样本）：
`flat_haze` = 现有配方；`depth_haze` = 深度散射，远场透射率对齐同一五级网格；
`equal_flat` = 平坦雾，平均透射率设成等于同图 `depth_haze` 的均值。
**`depth_haze − equal_flat` 才是纯空间结构效应**；`depth_haze − flat_haze` 混入了深度雾整体更淡。

## 结果（7 个分类器 accuracy 均值）

| 域 | 级 | flat_haze (t̄) | depth_haze (t̄) | equal_flat | depth−flat | **depth−equal** |
|---|---|---|---|---|---|---|
| matched | 1 | 84.6 (0.85) | 85.3 (0.90) | 85.3 | +0.6 | **−0.0** |
| matched | 3 | 84.7 (0.55) | 84.6 (0.70) | 84.5 | −0.1 | **+0.1** |
| matched | 5 | 81.6 (0.25) | 82.6 (0.45) | 83.5 | +1.0 | **−0.9** |
| shifted | 1 | 84.6 (0.85) | 85.3 (0.90) | 85.4 | +0.6 | **−0.1** |
| shifted | 3 | 81.7 (0.55) | 84.5 (0.70) | 84.7 | +2.8 | **−0.2** |
| shifted | 5 | 81.1 (0.25) | 83.3 (0.45) | 82.9 | +2.3 | **+0.5** |

## 结论

1. **空间深度结构本身没有可测效应**：`depth − equal` 在 −0.9 至 +0.5 pp 之间，六格均值 −0.1 pp，
   量级小于种子波动。深度条件雾应被定位为**真实性升级**，不是难度调节旋钮。
2. **所有"深度雾更友好"的读数都来自平均透射率更高**（s5：0.45 vs 0.25）。任何把
   `depth−flat = +0.6…+2.8 pp` 解释成"模型对真实雾更鲁棒"的说法都是口径错误。
3. **更值得注意的次要结果**：在 1–5 级雾下，这批模型 val accuracy 只掉约 3.5 pp
   （matched 84.6→81.6，shifted 84.6→81.1）。它们正是在 matched/shifted 这套合成雾上训练的，
   所以"用现有 5 级雾证明退化显著"在 val 上站不住；G1（≥5 pp 且 CI 不跨零）需要更强设定：
   更高 β 档、跨域组合，或换成未见过该雾族的模型（外部 SOTA）作为被测对象。

## 边界

- 仅 val 157 张、8 类；未做种子间区间，量级判断以 pp 为主。
- 单一深度模型（Depth Anything V2 Small），俯视航拍图上单体深度的语义有效性有限；
  `turbulence=0` 以隔离结构效应，与正式条件的默认湍流不同。
- 未测检测任务与侧视场景；深度先验在侧视/水平线场景的价值不能由本表否定。
