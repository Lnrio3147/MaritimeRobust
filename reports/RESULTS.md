# 实验结果（首轮 + 组合增强基线）

所有数值均来自本地保存的预测；以下不是论文原始成绩。训练列区分干净、单退化增强与组合增强；组合增强是第二轮基线，不参与首轮架构排名。

| 协议 | 类别 | 模型 | 训练 | 种子数 | 干净 Macro-F1 (%) | 单退化平均 Macro-F1 (%) | matched 组合平均 Macro-F1 (%) | 最差条件 Macro-F1 (%) |
|---|---|---|---|---:|---:|---:|---:|---:|
| official | all | convnext_tiny | 干净 | 1 | 90.23（单种子） | 82.81（单种子） | 53.08（单种子） | 24.35（单种子） |
| official | all | resnet50 | 干净 | 1 | 87.05（单种子） | 73.39（单种子） | 37.10（单种子） | 8.65（单种子） |
| official | all | swin_t | 干净 | 1 | 87.20（单种子） | 79.43（单种子） | 52.73（单种子） | 26.80（单种子） |
| research | all | convnext_tiny | 单退化增强 | 3 | 89.75 ± 0.75 | 88.43 ± 0.75 | 77.23 ± 0.72 | 57.42 ± 1.89 |
| research | all | convnext_tiny | 干净 | 3 | 89.90 ± 0.72 | 82.55 ± 0.69 | 55.22 ± 0.48 | 27.08 ± 1.82 |
| research | all | convnext_tiny | DCP + 干净 | 3 | 89.39 ± 0.75 | 84.60 ± 0.37 | 65.35 ± 1.52 | 42.94 ± 1.74 |
| research | all | convnext_tiny | 组合增强 | 3 | 89.83 ± 0.67 | 88.11 ± 0.16 | 80.31 ± 0.48 | 63.42 ± 1.61 |
| research | all | resnet50 | 单退化增强 | 1 | 87.90（单种子） | 85.41（单种子） | 68.32（单种子） | 47.97（单种子） |
| research | all | resnet50 | 干净 | 1 | 89.03（单种子） | 76.57（单种子） | 42.61（单种子） | 12.07（单种子） |
| research | all | resnet50 | 组合增强 | 1 | 89.39（单种子） | 86.29（单种子） | 77.24（单种子） | 59.33（单种子） |
| research | all | swin_t | 单退化增强 | 3 | 88.90 ± 0.99 | 86.93 ± 0.51 | 74.97 ± 0.66 | 54.14 ± 1.69 |
| research | all | swin_t | 干净 | 3 | 87.98 ± 0.94 | 79.37 ± 0.70 | 53.28 ± 1.74 | 27.62 ± 1.49 |
| research | all | swin_t | 组合增强 | 3 | 88.77 ± 0.37 | 86.75 ± 0.52 | 79.10 ± 0.97 | 63.67 ± 2.52 |
| research | civilian | convnext_tiny | 单退化增强 | 3 | 88.35 ± 1.11 | 86.52 ± 1.19 | 74.65 ± 2.32 | 54.19 ± 3.19 |
| research | civilian | convnext_tiny | 干净 | 3 | 87.50 ± 3.18 | 80.96 ± 2.01 | 60.43 ± 2.08 | 40.12 ± 2.56 |
| research | civilian | convnext_tiny | 组合增强 | 3 | 88.36 ± 0.55 | 86.39 ± 0.39 | 77.61 ± 2.08 | 58.06 ± 5.40 |
| research | civilian | resnet50 | 单退化增强 | 1 | 81.13（单种子） | 83.44（单种子） | 63.60（单种子） | 38.53（单种子） |
| research | civilian | resnet50 | 干净 | 1 | 80.00（单种子） | 73.14（单种子） | 42.25（单种子） | 15.20（单种子） |
| research | civilian | resnet50 | 组合增强 | 1 | 88.60（单种子） | 83.68（单种子） | 69.61（单种子） | 44.84（单种子） |
| research | civilian | swin_t | 单退化增强 | 3 | 89.02 ± 0.64 | 86.05 ± 1.19 | 69.96 ± 4.41 | 48.17 ± 3.77 |
| research | civilian | swin_t | 干净 | 3 | 88.15 ± 0.70 | 79.03 ± 0.86 | 53.91 ± 2.91 | 30.83 ± 4.57 |
| research | civilian | swin_t | DCP + 干净 | 3 | 86.73 ± 1.03 | 80.27 ± 0.41 | 61.33 ± 1.84 | 37.40 ± 4.44 |
| research | civilian | swin_t | 组合增强 | 3 | 88.13 ± 0.59 | 85.72 ± 0.56 | 75.34 ± 3.09 | 56.98 ± 4.18 |

不确定性：重复图分组不等于按港口/船只划分；模拟眩光与雾均为近似模型。中心区域覆盖率不是船体遮挡率。组合增强只在 matched 域、3/5 级训练；shifted 组合仍是分布偏移测试，不能据此声称真实组合泛化。

逐样本概率、参数与混淆矩阵位于各 runs 子目录；diagnostics.json 包含按原图组 bootstrap 的置信区间和高置信错误。
真实退化数据是否可独立验证，见 reports/fgsrcs_audit.json 和 docs/DATA_PROTOCOL.md。

## 相同架构与种子的增强对照

下列变化是增强版本减干净训练版本，单位为 Macro-F1 百分点，不代表新模块收益。

| 类别 | 模型 | 训练 | 种子 | 干净图变化 | 单退化平均变化 | matched 组合平均变化 |
|---|---|---|---:|---:|---:|---:|
| all | convnext_tiny | 单退化增强 | 17 | -1.23 | +4.77 | +22.81 |
| all | convnext_tiny | 组合增强 | 17 | -1.03 | +5.31 | +26.14 |
| all | convnext_tiny | 单退化增强 | 29 | +0.79 | +6.04 | +21.07 |
| all | convnext_tiny | 组合增强 | 29 | +0.76 | +5.20 | +24.84 |
| all | convnext_tiny | 单退化增强 | 43 | -0.00 | +6.84 | +22.15 |
| all | convnext_tiny | 组合增强 | 43 | +0.06 | +6.16 | +24.27 |
| all | resnet50 | 单退化增强 | 17 | -1.12 | +8.84 | +25.71 |
| all | resnet50 | 组合增强 | 17 | +0.36 | +9.72 | +34.63 |
| all | swin_t | 单退化增强 | 17 | -1.26 | +6.32 | +20.66 |
| all | swin_t | 组合增强 | 17 | -0.23 | +6.87 | +25.17 |
| all | swin_t | 单退化增强 | 29 | +2.33 | +8.71 | +22.82 |
| all | swin_t | 组合增强 | 29 | +1.19 | +7.50 | +26.70 |
| all | swin_t | 单退化增强 | 43 | +1.68 | +7.66 | +21.58 |
| all | swin_t | 组合增强 | 43 | +1.39 | +7.79 | +25.57 |
| civilian | convnext_tiny | 单退化增强 | 17 | +2.85 | +4.95 | +11.38 |
| civilian | convnext_tiny | 组合增强 | 17 | +3.71 | +6.62 | +18.28 |
| civilian | convnext_tiny | 单退化增强 | 29 | -1.30 | +4.16 | +14.02 |
| civilian | convnext_tiny | 组合增强 | 29 | -1.69 | +2.91 | +16.29 |
| civilian | convnext_tiny | 单退化增强 | 43 | +1.02 | +7.58 | +17.26 |
| civilian | convnext_tiny | 组合增强 | 43 | +0.58 | +6.78 | +16.97 |
| civilian | resnet50 | 单退化增强 | 17 | +1.14 | +10.30 | +21.35 |
| civilian | resnet50 | 组合增强 | 17 | +8.60 | +10.55 | +27.36 |
| civilian | swin_t | 单退化增强 | 17 | +1.13 | +6.09 | +16.21 |
| civilian | swin_t | 组合增强 | 17 | +0.21 | +6.97 | +21.99 |
| civilian | swin_t | 单退化增强 | 29 | +0.44 | +7.78 | +17.94 |
| civilian | swin_t | 组合增强 | 29 | -0.36 | +5.69 | +18.42 |
| civilian | swin_t | 单退化增强 | 43 | +1.06 | +7.20 | +14.01 |
| civilian | swin_t | 组合增强 | 43 | +0.08 | +7.42 | +23.87 |

## DCP 预处理对照

| 运行 | 干净图 F1 变化/百分点 | 五级 matched 雾平均 F1 变化/百分点 |
|---|---:|---:|
| runs/fgsc23_research_all/convnext_tiny_clean_s17 | -0.58 | +4.30 |
| runs/fgsc23_research_all/convnext_tiny_clean_s29 | -0.31 | +4.17 |
| runs/fgsc23_research_all/convnext_tiny_clean_s43 | -0.63 | +4.90 |
| runs/fgsc23_research_civilian/swin_t_clean_s17 | -1.74 | +2.23 |
| runs/fgsc23_research_civilian/swin_t_clean_s29 | -1.50 | +2.96 |
| runs/fgsc23_research_civilian/swin_t_clean_s43 | -1.01 | +8.85 |

## 同设备成本（种子 17）

端到端延迟包括读图、缩放、可选 DCP、归一化、传输及推理。FLOPs 为 PyTorch 支持算子的部分计数。

| 协议/类别 | 模型/训练 | 参数量 M | 端到端中位 ms | P95 ms | 训练墙钟 GPU h |
|---|---|---:|---:|---:|---:|
| research/all | convnext_tiny/single | 27.84 | 8.24 | 12.81 | 0.055 |
| research/all | convnext_tiny/clean | 27.84 | 7.15 | 10.48 | 0.051 |
| research/all | convnext_tiny/DCP | 27.84 | 13.01 | 18.72 | 0.051 |
| research/all | convnext_tiny/combined | 27.84 | 6.93 | 10.40 | 0.064 |
| research/all | resnet50/single | 23.56 | 8.12 | 14.00 | 0.054 |
| research/all | resnet50/clean | 23.56 | 8.10 | 14.04 | 0.072 |
| research/all | resnet50/combined | 23.56 | 8.04 | 13.95 | 0.063 |
| research/all | swin_t/single | 27.54 | 12.43 | 18.97 | 0.117 |
| research/all | swin_t/clean | 27.54 | 12.39 | 18.04 | 0.065 |
| research/all | swin_t/combined | 27.54 | 12.44 | 18.33 | 0.066 |
| research/civilian | convnext_tiny/single | 27.83 | 8.07 | 11.04 | 0.025 |
| research/civilian | convnext_tiny/clean | 27.83 | 7.97 | 11.27 | 0.059 |
| research/civilian | convnext_tiny/combined | 27.83 | 8.01 | 10.95 | 0.022 |
| research/civilian | resnet50/single | 23.52 | 8.35 | 11.16 | 0.021 |
| research/civilian | resnet50/clean | 23.52 | 8.43 | 11.42 | 0.019 |
| research/civilian | resnet50/combined | 23.52 | 8.30 | 11.07 | 0.022 |
| research/civilian | swin_t/single | 27.53 | 12.67 | 15.47 | 0.033 |
| research/civilian | swin_t/clean | 27.53 | 12.60 | 15.32 | 0.037 |
| research/civilian | swin_t/DCP | 27.53 | 21.48 | 26.99 | 0.037 |
| research/civilian | swin_t/combined | 27.53 | 12.39 | 15.16 | 0.023 |
