# CDLNet 修复版复现：干净训练 → 43 条件退化评测

> **这不是作者的原始成绩。** 官方仓库 `MIPCLab/CDLNetcode` 缺训练入口、缺权重、含未注册层，
> 只能按其公开结构修复后单阶段训练。结果只用于本项目内部的退化敏感性分析，
> 不得与论文报告的 FGSC-23 指标并列比较。

## 训练设置

| 项 | 值 |
|---|---|
| 源码 | `external/cdlnet_repro/`（本次运行哈希已写入 `reports/cdlnet_repro.json`） |
| 数据 | `fgsc23_research_all.jsonl`（research split，23 类，train 2565 / val 643） |
| 输入 | 256×256（4×4 网格 + 3×3 窗口需要 32 的整数倍） |
| 骨干 | ResNet-50 ImageNet-1K V2（`resnet50-11ad3fa6.pth`） |
| 训练 | 单阶段交叉熵，SGD lr 0.01，momentum 0.9，wd 0.0001，cosine，40 epoch，batch 16，AMP |
| 选择 | 干净 val Macro-F1（最佳 0.8890，第 30/40 轮） |
| 资源 | 训练 6.5 min，峰值显存 2.36 GB，参数 40.87 M |

## 与官方实现的差异（必须随结果一起报告）

1. 官方 `project/main.py` 导入不存在的 `train` 模块，且 `CKPT_*` 未定义：无法复原三阶段训练，
   本复现只做单阶段分类训练，去掉二分类头、代理样本与对比损失。
2. `FeatureExtractor` 在 `forward` 内新建 `response_head`，参数不注册、每次前向随机：
   本复现将其注册为模块（否则该分支不可能被训练）。
3. `GatedSelfAttention` 的布尔索引在反向时语义不明：改为 `torch.where`；推理时使用全 1 mask，保证可复现。
4. 官方在 2048 通道全分辨率特征图上做上采样响应，256 输入下 batch 16 会触发 CUDA
   `invalid configuration argument`：本复现将 1×1 响应头放在上采样之前（线性算子可交换，
   内部像素等价），显存从 >19 GB 降到 2.4 GB。
5. 官方硬编码 42 类 / `FGSCR-test` / 双卡：本复现使用 research split 的 23 类与单卡。
6. 评测退化与本项目基线完全同源（同一 `corruption_seed`、同一 `corrupt`、同 letterbox），
   仅分辨率 256 与基线的 224 不同。
7. 两个运行完成后，`run.py` 又加入了官方划分 sanity check 用的 `--allow-official-leakage`
   开关（只跳过原图组重叠检查，不改变训练/评测路径）；research 运行 `complete.json` 记录的
   `run.py` 哈希对应加入开关前的版本，`cdlnet_model.py` 与 `cdlnet_data.py` 哈希仍与当前文件一致。

## 为什么干净集不是论文的 93.23%

仓库 README 声称 FGSC-23 上 93.23%（RGCRL-Net 92.89%）。本复现的官方划分 OA 为
86.17%，差约 7.1 个百分点。差距有明确来源，
不是评测代码写错：

1. **作者的方法训练代码没有公开。** `project/main.py` 的 Step 2（分类训练）与 Step 3（二分类 gate 训练）
   只剩注释和模型构建，训练调用被删除；`res_bin.py` 只有 `warmup_train`，没有分类/二分类训练函数。
   公开代码无法产出作者的 checkpoint，93.23% 不能由它复现。
2. **官方评测是双模型 + 门控 + KNN。** `project/evaluate.py` 同时使用 `ckpt_cls`（分类）与
   `ckpt_final`（二分类 gate），测试时每样本提供弱/强/原图三视图，先用二分类头决定是否启用 CIMAM，
   再对特征做 k=5 的 KNN 修正后才分类。本复现是单模型、单视图、单次前向。
3. **预处理与调度不同。** 官方 `dataloader.py` 用 `Resize([256,256])` 拉伸，本项目用 letterbox；
   官方 SGD lr 0.01 + StepLR(10, 0.1)、batch 12，本项目 cosine、batch 16。
4. **训练数据量不同。** 本项目留出 654 张 val 做选型，实际只用 2,602 张训练；文献常用约 1:4 的
   训练/测试划分（约 3,240 张训练），且不一定留 val。

因此 86% 应读作“CDLNet 架构在单阶段 CE 下的水平”，而不是“CDLNet 方法的水平”。
按 FGSC-23 数据集原文的基线（ResNet-50 81.33%，Xception 87.76%），普通骨干在 81–88%；本复现落在该区间，
与“方法组件缺失”一致。同一官方划分上，项目的 ResNet-50 基线干净 test Macro-F1 为 87.05%，
说明评测 harness 本身没有系统性偏低。

### 协议对齐对照（诊断）

为排除“预处理/调度/数据量导致 86%”的解释，另跑 `runs_external/fgsc23_official_all/cdlnet_protocol_check_s17`：
train+val 训练（3256 张）、拉伸到 256×256、SGD lr 0.01 +
StepLR(10, 0.1)、batch 12、40 epoch、不留 val（取最后一轮），
训练 8.7 min。

| 变体 | OA | Macro-F1 | balanced acc |
|---|---|---|---|
| 主复现（train only，letterbox，cosine） | 86.17% | 85.79% | 85.06% |
| 协议对齐（train+val，stretch，StepLR） | 86.17% | 86.98% | 86.52% |

两个变体的总正确数都是 710/824，但逐类召回与混淆矩阵不同，是巧合而非同一模型。
协议对齐只带来约 +1.2 个 Macro-F1 点、OA 不变，仍比 93.23% 低约 7.1 个百分点。
因此差距不能归因于预处理、调度或训练数据量，剩余差距只能来自未公开的方法组件
（分类/二分类训练、双模型门控、KNN 修正）。



## 官方划分 sanity check

| 项 | 值 |
|---|---|
| 官方划分（原图组按官方 train/val/test） | train 2602 / val 654 |
| 最佳干净 val Macro-F1 | 0.8694（第 40/40 轮） |
| 干净 test accuracy | 0.8617 |
| 干净 test Macro-F1 | 0.8579 |

该划分下 `source_group_id` 仍有 train/test 18 组、val/test 2 组重叠
（FGSC-23 官方划分本身如此），因此只能作为“干净集精度量级”检查：修复版 OA 86.17%
仍低于论文报告的干净集成绩，与 research split 上“未复现出干净集增益”的结论一致。
官方划分不做退化评测，退化结论以 research split 为准。

## 结果（test，43 条件）

| 条件族 | CDLNet 修复版 | ResNet-50（基线，s17） | 差（CDLNet − 基线） |
|---|---|---|---|
| 干净 | 0.8682 | 0.8903 | -0.0221 |
| matched 单退化（15 条均值） | 0.7325 | 0.7657 | -0.0331 |
| shifted 单退化（15 条均值） | 0.7277 | — | — |
| matched 组合（4 条均值） | 0.4048 | 0.4261 | -0.0213 |
| shifted 组合（4 条均值） | 0.3923 | 0.4490 | -0.0567 |
| 最差单条件 | 0.1283 | 0.1207 | +0.0076 |

干净准确率 0.8702。同表 ConvNeXt-Tiny（s17）为干净 0.9065、
matched 单退化 0.8285、matched 组合 0.5470、
最差 0.2505，可作为“更强干净基线在退化下掉得更慢”的参照。

### 逐级退化曲线（Macro-F1，severity 1→5）

| 退化 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| haze（matched） | 0.845 | 0.82 | 0.727 | 0.598 | 0.461 |
| cloud（matched） | 0.861 | 0.835 | 0.764 | 0.652 | 0.5 |
| glare（matched） | 0.869 | 0.856 | 0.823 | 0.774 | 0.603 |
| haze（shifted） | 0.849 | 0.809 | 0.725 | 0.571 | 0.409 |
| cloud（shifted） | 0.876 | 0.825 | 0.742 | 0.566 | 0.405 |
| glare（shifted） | 0.868 | 0.866 | 0.837 | 0.824 | 0.743 |

### 组合退化

| 条件 | Macro-F1 |
|---|---|
| matched_haze+glare_3 | 0.629 |
| matched_haze+glare_5 | 0.187 |
| matched_haze+cloud_3 | 0.596 |
| matched_haze+cloud_5 | 0.207 |
| shifted_haze+glare_3 | 0.67 |
| shifted_haze+glare_5 | 0.21 |
| shifted_haze+cloud_3 | 0.561 |
| shifted_haze+cloud_5 | 0.128 |

### 最差 8 个条件

- `shifted_haze+cloud_5` 0.128
- `matched_haze+glare_5` 0.187
- `matched_haze+cloud_5` 0.207
- `shifted_haze+glare_5` 0.210
- `shifted_cloud_5` 0.405
- `shifted_haze_5` 0.409
- `matched_haze_5` 0.461
- `matched_cloud_5` 0.500

## 民船 8 类范围（主实验）

与主实验同数据、同划分、同 43 条件：train 618 / val 157，
最佳干净 val Macro-F1 0.8562（第 17/40 轮），
训练 2.6 min。

| 条件族 | CDLNet 修复版 | ResNet-50（civilian，s17） | 差（CDLNet − 基线） |
|---|---|---|---|
| 干净 | 0.8429（acc 0.8475） | 0.8000 | +0.0429 |
| matched 单退化（15 条均值） | 0.6882 | 0.7314 | -0.0432 |
| shifted 单退化（15 条均值） | 0.7007 | — | — |
| matched 组合（4 条均值） | 0.3356 | 0.4225 | -0.0869 |
| shifted 组合（4 条均值） | 0.3786 | 0.4544 | -0.0758 |
| 最差单条件（`matched_haze+glare_5`） | 0.1133 | 0.1520 | -0.0387 |

民船范围内 CDLNet 干净集比同骨干基线高 4.3 个百分点，
但单退化低 4.3、组合退化低
8.7 / 7.6 个百分点，
即“干净更高、退化更差”的不对称在 8 类上更明显。

## 观察

- all 23 类上 CDLNet 修复版干净集比同骨干 ResNet-50 基线低 2.2 个百分点（民船 8 类相反，见上节），
  说明单阶段 CE 修复版没有复现出论文的干净集增益；这不排除是训练流程差异（无二分类预热/代理样本）造成。
- 单退化中 haze/cloud 比 glare 更伤、shifted glare 最温和，方向与基线一致；但五级差异变大：
  haze 0.461（基线 0.594）、cloud 0.500（基线 0.472）、glare 0.603（基线 0.607）。
- 组合退化比同骨干 ResNet-50 基线掉得更深（shifted 组合低 5.7 个百分点）：matched 组合均值 0.405，最差 0.128。
- 组合退化在 severity 3 时已跌到 0.56–0.67，severity 5 跌到 0.13–0.21。
  一种可能的解释是“区域选择 + 响应融合”在低对比度、遮挡叠加时选错区域并把误差放大；
  这仍是假设，需在下一轮用区域响应可视化与干净/退化区域命中率来验证，不能从当前指标直接断言。
- 局限：单种子、单阶段训练、256 输入、未复现二分类/对比分支；结论只适用于该修复版。

## 复现命令

```bash
.venv/bin/python external/cdlnet_repro/run.py train --run-dir runs_external/fgsc23_research_all/cdlnet_clean_s17
.venv/bin/python external/cdlnet_repro/run.py eval  --run-dir runs_external/fgsc23_research_all/cdlnet_clean_s17
.venv/bin/python scripts/cdlnet_report.py
```
