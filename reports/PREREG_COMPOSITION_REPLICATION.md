# 预注册：复合退化结构的跨折复现

写于 2026-10-31 01:05，**在看到 fold1 / fold2 任何结果之前**。发现单元是 fold0 / seed17（`reports/composition/interaction_fold0_s17.json`）。以下判据用于决定该结构是"一个单元的探索性观察"还是"可复现的规律"。

## 数据

6 个单元：fold ∈ {0,1,2} × seed ∈ {17,29}，冻结基线 `convnext_tiny_none_e40`，无训练。
每个单元：6 个族在**匹配边际下降 10 / 20 pp** 处标定水平；15 对；逐图预测。

两个量：

- `E(a,b) = drop(a+b) − drop(a) − drop(b)`（相对 clean 的下降空间）
- `delta(a,b) = e(a+b) − |E_a ∪ E_b|`（错误率空间，扣除两族错误集重叠）

族的主效应由加性模型 `delta(a,b) ≈ m_a + m_b` 最小二乘拟合。

## 判据（任一不成立 → 只按单单元探索性观察报告，不得写成规律）

| 编号 | 判据 | 门槛 |
|---|---|---|
| C1 | 发现单元中 \|delta\| ≥ 5 pp 的对，在其余 5 个单元中 **符号一致** | ≥ 4/5 单元 |
| C2 | 低通×低通对（defocus/motion/pixelate 两两）平均 delta < 0 | ≥ 5/6 单元 |
| C3 | 含 noise 的对平均 delta > 0 | ≥ 5/6 单元 |
| C4 | 加性主效应模型对 delta 的 R² | ≥ 0.40，且 ≥ 5/6 单元 |
| C5 | 四个单退化预测器（平均位移余弦、逐图位移对齐、高频能量积、逐类错误相关）|ρ| | 均 < 0.60，或至多一个 ≥ 0.60 —— 即"没有可用几何预测器" |

C5 是**阴性预期**：发现单元上四个预测器的 |ρ| 为 0.29–0.43，全部不显著。若复现中某个预测器在每个单元都稳定 ≥ 0.6，则需改写结论，不能沿用"无预测器"的说法。

## 边界

- 只覆盖 6 个人工附加族 + 1 个骨干 + 1 个数据集；不复现论文原始的 haze/glare/cloud 复合。
- 匹配边际为 10 / 20 pp；30 pp 档因准确率地板被排除出主分析。
- delta 依赖逐图错误集，样本量 206，单对 delta 的抽样误差约 ±5 pp；只对跨单元一致性作判断，不对单对数值作强解释。

## 产物

`scripts/composition_interaction.py`、`scripts/composition_predictors.py`、`scripts/composition_aggregate.py` → `reports/composition/{interaction,predictors}_fold{f}_s{s}.json`。
