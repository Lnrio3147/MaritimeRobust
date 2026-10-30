# EFM-Net 架构复现记录（8 类民用船型，非官方成绩）

2026-10-30。外部 SOTA 复现，用于新课题的退化观测与 G0/G1 判定。所有数字由 `scripts/efmnet_eval.py`、`scripts/baseline_val_conditions.py` 与 `scripts/cross_model_degradation.py` 生成，工件见 `reports/efmnet_val_conditions*.json`、`reports/cross_model_degradation.json`。

## 1. 复现范围

**架构级重实现，无官方权重。** 参数随机初始化后从头训练，只使用 ImageNet-22K 的 ConvNeXt-Base 主干权重作为初始化；MHA（多层级注意力）与 BPP 细化头全部随机初始化。因此本文结果不是论文成绩的复现，只能称为"同架构、本项目协议"的一次独立训练。

代码：`external/audit/EFM-Net-Pytorch`（commit `7d04323`，Apache-2.0），工作副本 `external/efmnet_repro/`。

## 2. 使其可运行的补丁

| 文件 | 问题 | 处理 |
|---|---|---|
| `util/utils.py` | `from torch._six import inf`（torch≥2.0 移除） | 改为 `from torch import inf` |
| `util/utils.py` | `save_input_image` 依赖 cv2 | 改用 PIL 保存 |
| `main.py` | `dist.get_rank()` 未导入 `dist` | 改为 `utils.get_rank()` |
| `main.py` | 非分布式分支 `logger = None` 后调用 logger | 改为 `create_logger(...)` |
| `main.py` | `logger('...')` 把 logger 当函数调用 | 改为 `logger.info(...)` |
| `models/miner.py` | `torch.autograd.set_detect_anomaly(True)` 拖慢训练 | 移除 |
| `processor/optim_factory.py` | timm 0.9.16 已移除 `novograd` 等模块，导入即失败 | 改为可选导入（默认优化器 adamw 不受影响） |
| `datasets/FGSC-23` | 软链路径少一级 | 重建软链指向 `data/raw/fgsc23/FGSC-23` |

## 3. 预训练权重来源

- 文件：`external/efmnet_repro/pretrained/convnext_base_22k_224.pth`（420 MB，344 个张量）
- 来源：ImageNet-22K 监督预训练 ConvNeXt-Base，官方 `convnext_base_22k_224` 权重
- 载入日志明确列出未初始化模块：`mha.*`、`head.*`（共 26 项），其余主干完全由预训练权重填充

## 4. 数据划分

由 `scripts/efmnet_dataset.py` 生成 ImageFolder 布局（软链，不复制像素）：

| EFM-Net 目录 | 项目划分 | 数量 | 用途 |
|---|---|---|---|
| `train/` | manifest `split=train` | 618 | 训练 |
| `test/` | manifest `split=val` | 157 | 模型选择（EFM-Net 代码里叫 test） |
| 未链接 | manifest `split=test` | 236 | 官方测试集，保持冻结 |

类别为 8 类民用船型（container ship、car carrier、bulk carrier、oil tanker、fishing boat、passenger ship、liquefied gas ship、barge）。

## 5. 训练配置

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | EFM-Net = ConvNeXt-B + MHA(M=16) + BPP | 87,750,292 参数 |
| 优化器 | AdamW，lr 1e-3，wd 1e-8，layer_decay 0.8 | 代码默认 |
| batch × 累积 | 16 × 30 = 480 | 与论文有效批大小一致 |
| 每 epoch 更新次数 | 1 | `len(dataset)//480 = 1` |
| 计划 epoch | 1800 | 使总更新次数 = 1800，与论文 300 epoch × 6 次一致 |
| 实际 epoch | 693 | 见 §6 |
| 增强 | timm 默认训练变换 | 无 mixup / cutmix / EMA |
| 种子 | 17 | |
| 设备 | 单卡 RTX 4090 24 GB | |

## 6. 为何在 693 epoch 停止

| 观测 | 值 |
|---|---|
| 最好 val 准确率 | 85.99%（record 中 epoch 索引 71，即第 72 epoch） |
| 之后连续未刷新的 epoch 数 | 621 |
| 停止时 LR | 6.779e-04（余弦尚未走完） |
| 停止时训练损失 | 0.355（首 epoch 2.08） |

理由：最优权重在第 72 epoch 出现后 621 个 epoch 内未被刷新，继续训练只增加过拟合风险；按 `checkpoint-best.pth` 停止，并把"余弦退火尾段未执行"作为已知偏离记录在 §10。

保留的 checkpoint：`checkpoint-best.pth`（最优）、`checkpoint-499.pth`、`checkpoint-599.pth`。

## 7. 两套预处理口径

| 口径 | 定义 | clean val acc |
|---|---|---|
| EFM-Net 原生 | timm eval transform：缩放到 256×256（squash，非等比）→ 中心裁剪 224 | 85.99% |
| 项目管线 | `contain(224)` → 退化 → `pad(224, 灰边 124,116,104)` → ImageNet 归一化 | 78.34% |

两套口径相差 7.64 pp。**85.99% 是在 EFM-Net 自己的预处理下得到的，不能作为相同评估协议下的 SOTA 对比数，也不能与原论文 23 类官方划分的 OA 直接比较。** 跨模型比较一律使用项目管线口径（`reports/CROSS_MODEL_DEGRADATION.md`），并且只比较同一口径内的差值。

## 8. 退化结果（项目管线，val 157）

配对 bootstrap（按 `source_group_id` 分组重采样 2000 次）相对 clean 的差值：

| 条件 | acc | Δ (pp) | 95% CI |
|---|---|---|---|
| matched_haze_1 | 76.43 | -1.9 | [-5.0, 0.6] |
| matched_haze_2 | 71.34 | -7.0 | [-11.5, -3.1] |
| matched_haze_3 | 73.89 | -4.5 | [-9.1, 0.0] |
| matched_haze_4 | 71.97 | -6.4 | [-11.7, -1.3] |
| matched_haze_5 | 72.61 | -5.7 | [-11.6, 0.0] |
| matched_cloud_1 | 77.71 | -0.6 | [-5.0, 3.2] |
| matched_cloud_2 | 75.80 | -2.5 | [-6.5, 1.3] |
| matched_cloud_3 | 75.16 | -3.2 | [-7.8, 1.3] |
| matched_cloud_4 | 73.25 | -5.1 | [-11.7, 1.2] |
| matched_cloud_5 | 68.15 | -10.2 | [-17.5, -2.6] |
| matched_glare_1 | 79.62 | +1.3 | [-1.3, 3.8] |
| matched_glare_2 | 77.07 | -1.3 | [-5.7, 2.6] |
| matched_glare_3 | 73.89 | -4.5 | [-9.2, 0.0] |
| matched_glare_4 | 73.89 | -4.5 | [-10.5, 1.3] |
| matched_glare_5 | 63.69 | -14.6 | [-22.6, -6.9] |
| matched_haze+glare_3 | 70.70 | -7.6 | [-13.8, -1.3] |
| matched_haze+glare_5 | 47.77 | -30.6 | [-39.1, -22.2] |
| matched_haze+cloud_3 | 71.97 | -6.4 | [-13.4, 0.0] |
| matched_haze+cloud_5 | 49.04 | -29.3 | [-37.7, -20.7] |
| shifted_haze_1 | 75.16 | -3.2 | [-7.2, 0.0] |
| shifted_haze_2 | 72.61 | -5.7 | [-10.6, -1.9] |
| shifted_haze_3 | 75.16 | -3.2 | [-7.5, 0.7] |
| shifted_haze_4 | 71.34 | -7.0 | [-12.1, -2.6] |
| shifted_haze_5 | 65.61 | -12.7 | [-19.1, -7.1] |
| shifted_cloud_1 | 78.34 | +0.0 | [-3.8, 3.3] |
| shifted_cloud_2 | 78.34 | +0.0 | [-5.1, 5.1] |
| shifted_cloud_3 | 75.16 | -3.2 | [-7.7, 0.7] |
| shifted_cloud_4 | 74.52 | -3.8 | [-9.6, 1.9] |
| shifted_cloud_5 | 66.88 | -11.5 | [-18.5, -4.5] |
| shifted_glare_1 | 78.34 | +0.0 | [-1.9, 1.8] |
| shifted_glare_2 | 80.25 | +1.9 | [-0.6, 4.6] |
| shifted_glare_3 | 77.07 | -1.3 | [-5.9, 3.1] |
| shifted_glare_4 | 74.52 | -3.8 | [-8.3, 0.6] |
| shifted_glare_5 | 74.52 | -3.8 | [-10.3, 2.5] |
| shifted_haze+glare_3 | 71.97 | -6.4 | [-11.0, -1.9] |
| shifted_haze+glare_5 | 48.41 | -29.9 | [-38.9, -20.8] |
| shifted_haze+cloud_3 | 71.34 | -7.0 | [-13.5, -1.3] |
| shifted_haze+cloud_5 | 49.68 | -28.7 | [-37.0, -20.0] |
| matched_cloud_3_center | 73.89 | -4.5 | [-9.6, 0.6] |
| matched_glare_3_center | 71.97 | -6.4 | [-12.3, -0.6] |
| matched_cloud_3_border | 73.25 | -5.1 | [-10.5, 0.0] |
| matched_glare_3_border | 75.80 | -2.5 | [-6.3, 0.7] |

原生预处理下的对照（仅 clean 标定 + 与项目管线的差值比较）见 `reports/cross_model_degradation.json` 的 `preprocess_ablation` 字段，汇总见 `reports/CROSS_MODEL_DEGRADATION.md` §4。

## 9. 与论文数字的关系（明确不可比）

| 差异 | 论文 | 本次 |
|---|---|---|
| 类别数 | 23 | 8 |
| 划分 | 官方 FGSC-23 train/test | 项目 8 类划分 618/157 |
| 权重 | 官方训练结果 | 随机初始化后从头训练 |
| 评估预处理 | 论文自己的管线 | 项目 contain+灰边 pad |
| 优化步数 | 300 epoch × 6 次/epoch = 1800 | 693 次 |
| 数字 | OA 约 89.70（23 类） | 85.99%（原生口径）/ 78.34%（项目口径） |

结论：本运行只能作为"同架构在本项目协议下的独立训练"，不构成对论文成绩的复现，也不能声称高于或低于论文。G0 判定因此改为"架构复现 + 协议声明"，其可比性仅限本项目内部（与 7 个冻结基线同管线、同图、同种子）。

## 10. 协议偏离清单

1. 类别数与划分：8 类、项目划分，非官方 23 类官方划分。
2. 预处理：训练用 timm 增强与原生 eval 变换；跨模型评估改用项目管线，导致干净条件净降 7.64 pp。
3. 优化步数：693 次，低于计划的 1800 次；余弦退火未完成（停止时 LR 6.779e-04）。
4. 无 EMA、无 mixup/cutmix、无分布式训练（代码默认值）。
5. 训练集 618 张，远小于论文 23 类训练集。
6. 未使用官方权重，故任何绝对准确率都不应作为 SOTA 行进入对比表，除非同时标注口径。

## 11. 工件

| 工件 | 内容 |
|---|---|
| `runs_external/fgsc23_civ8/fgsc23_civ8_10-29-17-34-01checkpoint-best.pth` | 最优权重（1.05 GB） |
| `runs_external/fgsc23_civ8/fgsc23_civ8_10-29-17-34-01record.txt` | 逐 epoch 训练/验证记录 |
| `reports/efmnet_val_conditions.json` | 项目管线 43 条件精度 |
| `reports/efmnet_val_conditions_samples.json` | 项目管线逐样本预测与概率 |
| `reports/efmnet_val_conditions_native.json` | 原生预处理 43 条件精度 |
| `reports/cross_model_degradation.json` | 跨模型统计（Δ、CI、闸门、转移矩阵） |
