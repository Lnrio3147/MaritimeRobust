# 三个领域模型的复现准入审计（2026-10-28）

结论先行：**CDLNet、MA-Net、RGCRL-Net 目前都不能按“官方模型 + 官方权重”直接跑退化实验。**
直接训练只能得到本项目的复现版本，不能把结果标成原论文成绩。下面分别记录证据与缺口。

## 状态总表

| 模型 | 论文/来源 | 官方代码 | 官方权重 | 当前可否直接评测 | 主要缺口 |
|---|---|---|---|---|---|
| CDLNet | ISPRS JPRS 2026, DOI 10.1016/j.isprsjprs.2026.04.051 | 有，但为不完整代码包 | 未发现 | 否；已做修复版复现 | 缺训练入口、若干常量/模块缺失、无权重、代码含硬编码路径与未注册层 |
| MA-Net | Remote Sensing 2026, 18(3):462, DOI 10.3390/rs18030462 | 未发现 | 未发现 | 否 | 无公开实现；仅摘要/模块级描述，缺精确层结构与训练细节 |
| RGCRL-Net | IEEE TAES 2025, 61(2):2631–2643, DOI 10.1109/TAES.2024.3480052 | 未发现 | 未发现 | 否 | 无公开实现；论文付费墙，仅有摘要级结构描述 |

## CDLNet

- 官方仓库：https://github.com/MIPCLab/CDLNetcode
- 已克隆到 `external/CDLNetcode`，仅用于审计，未运行。
- 代码包包含 `dataloader.py`、`loss.py`、`res_bin.py`、`project/{main,evaluate,models,losses}.py`。
- 阻断项：
  - `project/main.py` 导入 `train` 模块，但仓库没有 `train.py`；
  - `main.py` 的 Step 2（分类训练）与 Step 3（二分类 gate 训练）只剩注释和模型构建，训练调用被删除；
    `res_bin.py` 只有 `warmup_train`，没有分类/二分类训练函数；
  - `CKPT_WARMUP`、`CKPT_CLS`、`CKPT_FINAL` 未定义；
  - 官方评测 `project/evaluate.py` 需要两个模型（`ckpt_cls` 分类 + `ckpt_final` 二分类门控），
    测试时每样本提供弱/强/原图三视图，并对特征做 k=5 KNN gate 修正后才分类；
  - 多处硬编码 `FGSCR-test`、42 类、`CUDA_VISIBLE_DEVICES=2,3`；
  - `project/models.py` 使用 `random` 但未导入；
  - `FeatureExtractor.forward` 内部临时创建 `response_head`，参数未注册、每次前向重新随机初始化，不能随优化器更新；
  - `GatedSelfAttention` 的空 mask 分支行为不明确；
  - 仓库没有预训练权重或 release。
- 因此：公开代码无法产出作者 checkpoint；即使修到能跑，也只能称为“基于官方代码修复的 CDLNet 架构复现”，
  不能声称是作者的方法或成绩。README 声称 FGSC-23 93.23%（RGCRL-Net 92.89%），该数字依赖未公开的训练与门控推理。

### 已执行的修复版复现（2026-10-28）

实现位于 `external/cdlnet_repro/`（官方仓库保持原样，只读）。修复内容、训练设置与全部结果见
`reports/CDLNET_REPRO.md`；机读汇总见 `reports/cdlnet_repro.json`；与三种基线的退化诊断对照
（保持率、逐级曲线、混淆对、预测塌缩、高置信错误、位置对照、分组区间）见
`reports/CDLNET_DIAGNOSTICS.md`；两个附加分支（门控自注意力、响应图/裁剪重跑）的消融见
`reports/CDLNET_ABLATION.md`。

| 项目 | 结果 |
|---|---|
| research split、23 类、256×256、40 epoch、单阶段 CE | 最佳干净 val Macro-F1 0.8890（第 30 轮） |
| research 干净 test | Macro-F1 0.8682 / accuracy 0.8702 |
| matched 单退化 / shifted 单退化均值 | 0.7325 / 0.7277 |
| matched 组合 / shifted 组合均值 | 0.4048 / 0.3923 |
| 最差条件 | shifted 五级雾+云 0.1283 |
| 民船 8 类（干净 test） | Macro-F1 0.8429 / accuracy 0.8475；matched 单退化 0.6882、matched 组合 0.3356、shifted 组合 0.3786、最差 0.1133 |
| 官方划分 sanity check（干净 test） | OA 86.17%，val Macro-F1 0.8694（第 40 轮） |

同骨干 ResNet-50 干净基线（seed 17）：干净 0.8903、matched 单退化 0.7657、matched 组合 0.4261、
shifted 组合 0.4490、最差 0.1207。修复版在干净集与各类退化上都没有超过该基线，
且官方划分干净 OA 比论文报告的 93.23% 低 7.1 个百分点。为排除协议因素，另做协议对齐诊断
（`runs_external/fgsc23_official_all/cdlnet_protocol_check_s17`：train+val、拉伸 256、StepLR(10,0.1)、
batch 12、40 epoch）：OA 仍为 86.17%，Macro-F1 86.98%（+1.2 个百分点），8.7 min。
因此差距不能归因于预处理、调度或训练数据量，剩余约 7 个百分点只能来自未公开的方法组件：
被删除的分类/二分类训练代码、双模型门控与 KNN 修正。官方划分上本项目 ResNet-50 基线为 87.05%，
说明 harness 本身没有系统性偏低，86% 只能读作“CDLNet 架构在单阶段 CE 下的水平”。

评测协议：退化与基线共用 `corruption_seed` / `corrupt` / letterbox，仅分辨率 256 与基线 224 不同；
官方划分存在 train/test 18 组、val/test 2 组原图组重叠（FGSC-23 官方划分本身如此），故只做干净集检查。
研究运行的 `run.py` 哈希对应加入 `--allow-official-leakage` 开关之前的版本，`cdlnet_model.py`、
`cdlnet_data.py` 哈希与当前文件一致。

## MA-Net

- 论文：https://www.mdpi.com/2072-4292/18/3/462
- 检索未发现 GitHub/代码/权重入口；Crossref 与索引页只有元数据和摘要。
- 已见模块级信息：ResNet-50 金字塔特征；ALFA（自适应局部特征注意力，重叠块 + 局部自注意力）；DMGFF（动态多粒度融合 + 通道注意力）；FBDA（基于注意力的掩码训练分支，θ_mask=0.5，λ=0.75）；AdamW、lr=1e-3、batch 16、100 epoch。
- 缺口：ALFA 的具体分块/窗口参数、DMGFF 的融合公式、FBDA 的掩码生成与损失权重、数据增强与划分细节未在可访问材料中完整给出。按现有信息实现只能算“思路复现”，不能标成 MA-Net。

## RGCRL-Net

- 论文：https://ieeexplore.ieee.org/document/10716440
- 检索未发现官方代码/权重；论文在付费墙后，当前只能读到摘要级描述：区域特征学习、判别性部件发现、全局-局部对比学习。
- 缺口：部件发现网络结构、区域采样与对比损失的具体形式、训练超参数均不足以做忠实复现。

## 建议的准入顺序

1. **先拿到作者代码/权重或完整论文**：这是唯一能称为“跑 CDLNet/MA-Net/RGCRL-Net”的路径。
2. **CDLNet 修复版复现已完成**（见上节）。按原定准入条件，官方划分干净 OA 未接近论文报告值，
   因此其退化结果只作为本项目内部的敏感性参照，不得当作 CDLNet 的论文水平参与 SOTA 对比。
   若后续拿到作者权重/完整训练代码，应重跑干净集检查后再替换该参照。
3. MA-Net 需要完整 PDF 后再评估；RGCRL-Net 在拿到论文/代码前不做猜测实现。
4. 退化实验仍使用本项目统一的 43 条件协议；外部模型需在同一 research split 上重训，或同时报告“官方划分复现”和“本项目划分对照”两套结果，不能混用。

## 对当前基线的影响

现有 45 个基线运行不受影响，继续保持冻结。外部模型接入应使用新的模型名、新的运行目录和新的源码哈希，不改动已绑定运行。
