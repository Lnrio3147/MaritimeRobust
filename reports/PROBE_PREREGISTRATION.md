# 冻结特征双探针：预注册（写于提取任何特征之前）

2026-10-30。本文件在**运行前**冻结设计、阈值与决策阶梯。任何事后改动必须在结果文件中单列为一节「偏离」。

## 1. 目的

判定组合退化下 8 类民用船识别失败的层级，并在三种可能结论中机械地选一个 G2 方法假设，或终止新模块开发。**不训练任何模型，不产生 checkpoint，不触碰 test 集。**

## 2. 数据

- 训练/拟合集：`data/manifests/fgsc23_research_civilian.jsonl` 中 `split=="train"` 的 618 张（clean）。
- 测试集：同清单 `split=="val"` 的 157 张（152 个 `source_group_id`）。
- 条件（5 个，取自 `G2_P0_FAILURE_MECHANISM.md` §3）：`clean`、`matched_glare_5`、`matched_haze+glare_5`、`matched_haze+cloud_5`、`shifted_haze+glare_5`。
- 组合条件集 **C = {matched_haze+glare_5, matched_haze+cloud_5, shifted_haze+glare_5}**，共 3 个。
- 图像管线：`ImageOps.contain(224, BICUBIC)` → `corrupt(kind, severity, domain, placement, corruption_seed(sample_id, "efmnet", ...))` → `ImageOps.pad(224, color=(124,116,104))` → ImageNet 归一化。与 `scripts/baseline_val_conditions.py` 逐字节一致。
- 退化强度、种子、domain **不改动**。

## 3. 四个骨干家族（每家族 1 个种子）

| 家族 | 运行 | 特征位置 | 维度 |
|---|---|---|---|
| convnext_tiny | `runs/fgsc23_research_civilian/convnext_tiny_combo_s17` | `model.avgpool` 输出 | 768 |
| swin_t | `runs/fgsc23_research_civilian/swin_t_combo_s17` | `model.avgpool` 输出 | 768 |
| resnet50 | `runs/fgsc23_research_civilian/resnet50_combo_s17` | `model.avgpool` 输出 | 2048 |
| efmnet_convnext_b | `runs_external/fgsc23_civ8/fgsc23_civ8_10-29-17-34-01/checkpoint-best.pth` | `stages[3]` 输出的全局平均池化 | 1024 |

EFM-Net 用 ConvNeXt-B 主干池化特征，**位于其 MHA/BPP 头之前**；即对该模型是保守位置（其真实头还能再加工），若探针仍胜出则结论更强。

## 4. 双探针定义

对每个家族 f、每个条件 c：

- **P1（跨域探针）**：在 `(F_train_clean, y_train)` 上拟合多项逻辑回归，在 val 的 c 条件特征上评估。回答：「干净数据上拟合的线性边界能否迁移到退化特征」。
- **P2（同域探针）**：在 `(F_train_c, y_train)` 上拟合，在 val 的 c 条件特征上评估。`F_train_c` 为 618 张 train 图经**同一退化参数**（同 kind/severity/domain，按 sample_id 生成种子）处理后提的特征。回答：「退化特征中线性可用的类别信息上限」。
- **对照 1（先验地板）**：忽略特征，恒预测 train 中样本最多的类。
- **对照 2（标签打乱）**：在 `(F_train_clean, permute(y_train))` 上拟合，在 val 的 c 上评估。应接近先验地板；显著高于则说明管线泄漏，实验判废。
- **参照（模型自身）**：该家族原始分类头的准确率（EFM-Net 先做 efmnet 索引→项目标签映射）。

`clean` 条件下 P2 与 P1 相同，只报一次。

### 关于 P2 使用退化 train 图的声明

P2 需要 618 张 train 图按固定参数退化一次。这些图**仅用于拟合诊断探针**，不进入任何部署模型、不产生 checkpoint、不进入任何数据集统计。这是 `G2_P0` §3 判据得以「区分表征层与判别层」的必要条件。若认为这仍违反「生成图不得以继承标签入账」，则 P2 应整体剔除，此时只剩 P1，无法区分 R1/R2/R3——请直接否决而非事后删除。

## 5. 超参数（固定）

- 预处理：`StandardScaler` 在拟合集上 fit，应用到测试集。
- 分类器：`LogisticRegression(multi_class="multinomial", max_iter=3000, solver="lbfgs")`。
- 正则强度 C：在拟合集上做 5 折分层交叉验证，网格 `{0.01, 0.03, 0.1, 0.3}`，按准确率选最优；同一家族的 P1/P2/打乱对照**复用同一个 C**。
- 不做特征选择、不做 PCA、不做类别重加权。

## 6. 统计

- Δ 的置信区间：按 `source_group_id`（152 组）分组 bootstrap，2000 次，2.5/97.5 分位。
- 多重比较：对 4 家族 × 3 组合条件 = 12 个检验分别对 Δ_P1 与 Δ_P2 做 Benjamini–Hochberg（α=0.05）。单边 p 值取 bootstrap 分布中 ≥0 的比例。
- 汇总量（对 C 内 12 个 (家族, 条件) 取均值）：
  - `H = mean(P2 − model)`：域匹配线性头的净空。
  - `S = mean(P2 − P1)`：干净边界迁移到退化域付出的代价（分布偏移量）。
  - `T = mean(P1 − model)`：干净拟合线性边界相对模型头的增益。

## 7. 决策阶梯（自上而下，首个命中即采用）

1. **R1 终止**：`H < 3.0 pp`。退化后线性可用的类别信息已无净空 —— 表征层损失，低成本头部/校准类模块没有空间。**终止新模块开发，记录负结果。**
2. **R3 决策层假设**：`H ≥ 3.0 pp` 且 `S < 5.0 pp`。信息存活，干净拟合的边界几乎追平域匹配边界 —— **模型自身分类头是短板**。G2 假设 = 决策层校准 + margin/原型约束（新增参数 ≈ 0）。
3. **R2 特征分布偏移假设**：`H ≥ 3.0 pp` 且 `S ≥ 5.0 pp`。信息存活，但必须用域匹配边界才能取出 —— **特征分布发生偏移**。G2 假设 = 轻量特征对齐（BN 统计重估 / 测试时适应），不新建表征模块。

**生效护栏**：所采用分支的判定量（H 或 S）须满足 (a) 分组 bootstrap 95% CI 不含 0，且 (b) 其 12 个组成检验中至少 7 个经 BH 显著。任一不满足 → 判为 **R0 不确定**，按「终止新模块开发」处理。

阈值理由：单检验分组 bootstrap SE ≈ 3.2 pp；3.0 pp 与 5.0 pp 分别约为均值的 2 SE 与 3 SE（12 个检验聚合后有效 SE 更低）。阈值不因结果调整。

## 8. 预算与停止

- GPU：4 家族 × 3,875 张前向 + 约 1,200 次逻辑回归拟合，预计 **< 20 min**，计入 72 GPU-h 总预算。
- 磁盘：特征缓存 `reports/probe_features/*.npz`，合计 < 150 MB。
- 输出：`reports/frozen_probe.json`、`reports/FROZEN_PROBE.md`、`reports/G2_METHOD_DECISION.md`。
- 完成后**停止机制探索**，不再追加探针变体。

## 9. 执行期修正记录（2026-10-30，在向用户报告结果之前）

以下三处是**实现与本文档不符**，不是判据变更。第 1、2 点是在看到第一版脚本输出之后发现的，两版数字全部保留在 `frozen_probe.json` 与 `FROZEN_PROBE.md` 中；两版得出的分支相同（R0，终止）。

1. **p 值方向。** 本文档 §6 原文写作「单边 p 值取 bootstrap 分布中 ≥0 的比例」，但 §7 的假设是「探针优于模型」，对应 `H1: Δ > 0`，p 值应为 `P(Δ ≤ 0)`。首版按原文取了 `P(Δ ≥ 0)`，把检验方向整个翻转（会把「探针显著更差」误判为「更优」）。已改为 `P(Δ ≤ 0)`。原文措辞本身有歧义，故在此改写该句为：**单边 p = bootstrap 重采样中 Δ ≤ 0 的比例（H1：探针优于模型）。**
2. **BH 检验族。** §6 规定 12 个检验（4 家族 × 3 组合条件）。首版脚本实际把 `matched_glare_5` 也算进去，变成 16 个。已改回 12 个；16 个版本保留为敏感性分析一并报告。
3. **泄漏检查方向。** §2 写的是「打乱对照**显著高于**先验则判废」。首版渲染脚本用绝对偏差 `|shuffle − prior|` 判定，在对照**低于**先验时会产生假警报。已改为有符号偏差。

修正前后 `H / S / T` 三个汇总量完全不变（它们不依赖上述任何一项）。
