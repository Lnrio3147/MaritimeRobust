# 技术地形：模型、真实数据、退化合成

本文件是 `docs/THESIS_SCOPE.md` 的配套清单，服务于启动指标 R1/R2/R6。
网页检索于 2026-10-29 完成；标注 UNVERIFIED 的条目**必须自己核验后才能写进论文或计划**。
本地已有审计（`docs/MODEL_REPRODUCTION_AUDIT.md`、`docs/LITERATURE.md`）优先于网页信息。

## 0. 结论摘要

1. **模型**：FGSC-23 的纯 RGB SOTA 集群在 OA 92.9–93.2%，公开代码的只有 EFM-Net、HSMFNet、CDLNet
   三家；**没有任何一家公开训练权重**，全部只能是"架构复现"。
2. **真实数据**：不存在同时具备"民用船型标签"与"逐图真实退化标注"的公开数据集。
   本项目手工标注的 341 张是当前唯一可用的分层依据，扩标是唯一出路。
3. **合成**：物理参数化（深度 + 大气散射）是唯一按构造保留标签且 0 GPU 成本的路线；
   扩散类生成的标签有效性未被任何海事数据验证。

## 1. A 线：SOTA 民用船型识别模型

FGSC-23（23 类，约 4,081 张，官方固定划分 3,256/825）是事实标准；FGSCR-42 已饱和（98–99.6%），
只作副证；FGSRCS（17 类，复杂场景）刚起步，尚未成为标准基准。

| 方法 | 年份/出处 | 骨干 | FGSC-23 报告值 | 公开代码 | 额外模态 |
|---|---|---|---|---|---|
| AMEFRN | 2020 IEEE JSTARS 13:1271 | 属性引导多层 CNN | OA 93.58 (VGG16) / 93.09 (ResNet50) | 无官方；SOAP-KD 有重实现 | 是（属性标注） |
| EFM-Net | 2023 IEEE TGRS | ConvNeXt-B | OA 89.70（另有论文引作 87.62） | 有 `JACYI/EFM-Net-Pytorch` | 否 |
| Cog-Net | 2024 IEEE TGRS 62 | 因果多粒度 + 视觉概念 | OA 93.02 | 未找到 | 否 |
| RGCRL-Net | 2024/2025 IEEE TAES 61 | 区域全局-局部对比 | OA 92.89 | 未找到 | 否 |
| MSCL-Net | 2024 IEEE JSTARS 17:9989 | ResNet50+FPN 对比 | ~91.5（UNVERIFIED，仅由消融推得） | 未找到 | 否 |
| MA-Net | 2026 Remote Sensing 18(3):462 | ResNet50 + ALFA/DMGFF/FBDA | OA 93.12 / AA 92.38 | 未找到 | 否 |
| CDLNet | 2026 ISPRS JPRS 238 | 类差异学习（IVI/SWRA/CIMAM） | OA 93.23 | 有 `MIPCLab/CDLNetcode`，**本地审计为不可运行** | 否 |
| HSMFNet | 2026 Int. J. Remote Sens. | 层级语义多尺度融合 | OA 93.21 / macro-F1 92.99 | 论文所列仓库 **404**（2026-10-29 审计） | 否 |
| APS | 2025 IEEE conf. | ViT_Base + 部件发现 | OA 91.68 | 未找到 | 否 |

注意：

- AMEFRN 的 93.58 依赖属性监督，和纯 RGB 不是同一口径；纯 RGB 的真实天花板约 93.2。
- 各论文的指标几乎都是 **OA**，macro-F1/AA 少数才给；跨论文比较只在官方固定划分上成立。
- **无权重发布**：G0 只能是"架构复现 + 自己训练"，不能声称作者成绩。这与
  `docs/MODEL_REPRODUCTION_AUDIT.md` 对 CDLNet 的结论一致（代码包缺训练入口、缺权重）。

### 可复现排序（72 GPU h / 单 4090 内）

1. **EFM-Net**：4.15 GFLOPs，29.93 M 参数；仓库有完整训练入口（`main.py:113/168/224`），
   修 6 处（`torch._six`、单卡 `dist.get_rank`、Logger 调用、eval logger、timm<1.0、
   `set_detect_anomaly`）后约 2–4 h 可首次训练。无 EFM-Net 权重，只有 ConvNeXt-B 22K 初始化。
2. **CDLNet**：代码已有本地修复版复现（`runs_external/`），但那是修复版，不是原方法。
3. **HSMFNet**：论文所列仓库 `xndy002605/HSMFNet` 实测 **404**（2026-10-29），
   所有者主页无相关仓库，GitHub/grep.app 搜索无结果 → 不可复现；除非自行按论文重写（>100 h）。

AMEFRN / DKSCN 需要属性或知识标注，不纳入。

### 与本地资产的衔接

- 本地已有 23 类基线：`runs/fgsc23_research_all/`（clean / aug / combo × 3 骨干 × 3 种子），
  以及官方划分 `runs/fgsc23_official_all/`（3 个运行）。
- **G0 判定必须用官方划分**，否则和论文数字不可比；本项目研究划分折叠了重复图，数字会不同。

## 2. B 线：真实退化数据

| 数据集 | 任务/规模 | 逐图退化标注 | 船型标签 | 许可/获取 |
|---|---|---|---|---|
| WUTDet (2026) | 检测，100,576 图 | **有**：fog / glare / low-light / rain / normal | 只有 "ship" 一类 | 公开仓库 `MAPGroup/WUTDet` |
| ShipRSImageNet (2021) | 检测，3,435 图 | 无 | 50 个细类 | 公开，无明确许可 |
| McShips (2020) | 检测+细类，14,709 图 | 无（真实"in the wild"退化存在但未标） | 13 类，其中民用 7 类 | 仓库可用，要求引用 |
| MVDD13 (2023/24) | 检测，35,474 图 | 无（fog/rain/昼夜仅作多样性描述） | 13 类含民船 | 公开仓库 |
| SeaShips (2018) | 检测，31,455 图 | 无 | 6 类民船 | 公开 |
| SMD (2017) | 检测视频，81 段 | 场景级（haze/rain/轻雾） | 10 类粗分 | 公开，研究用 |
| WSODD (2021) | 检测，7,467 图 | 场景级 sunny/cloudy/foggy（fog 约 589 图） | 只有 ship/boat | 百度网盘，无许可 |
| ShipSG (2022) | 分割，3,505 图 | 无 | 7 类民用 | CC BY-NC-ND，需申请 |
| SeaDronesSee (2022) | 检测，~14.2k | 无 | 只有 boat | CC0 |
| WaterScenes (2024) | 检测/分割，54,120 帧 | 有天气场景标签（无雾） | 无船型 | 公开 |
| "Ship Weather" (2026) | 检测，2,054 图 | **有**：阴/雾/夜/眩光 | 粗类：货/油/未知/吊驳 | 下载 UNVERIFIED |
| AIMO/RMO (2025) | 分类/无监督域适应 | AIMO 合成、RMO 真实长尾天气 | 类别未明确 | 仓库可用性 UNVERIFIED |

**直接结论：不存在支持"民用船型 × 逐图真实退化"的公开数据。** 每个候选都缺一条轴。

对课题的处置：

1. **扩标 McShips**（14,709 图，7 个民用类，真实野外场景）。用本项目已建好的审阅页 →
   `merge_degradation_labels.py` 流程加标退化层，成本最低、可直接产出 R6 的消融数据。
2. **WUTDet 只做副作用检验**：真实雾/眩光/低光下，增强后的分类或检测器是否在"单类 ship"上
   也保持收益——不是主证据，是外部合理性检查。
3. **FGSRCS 收紧为跨库域差**，只有 2 个严格映射类，不出退化结论。
4. 现有 341 张（157 FGSC-23 + 184 FGSRCS）保留为第一层证据，并披露单标注者、无一致性。

## 3. C 线：退化合成

| 方法 | 范式 | 可控强度 | 代码/权重 | 海事验证 | 主要风险 |
|---|---|---|---|---|---|
| 物理参数化（散射 + 眩光饱和 + 模糊） | 物理，无学习 | 是（β、A、饱和度） | 自写，0 GPU | 间接（本项目 43 条件） | 真实性上限低 |
| FoggySea (2026 Ocean Eng.) | 物理：MiDaS 深度 + 散射 | 是（β=0.005–0.020 四档） | 代码未找到（UNVERIFIED） | 是（SeaShips） | 论文未开源，只能仿写 |
| GenDeg (CVPR 2025) | 扩散 + 结构校正 | 是（μ,σ） | 代码公开；权重 UNVERIFIED | 无 | 生成结构改写船体，标签失效 |
| GenHaze (ICCV 2025) | 单步 SD-Turbo 生成雾 | 间接（参考图风格） | 代码未找到（UNVERIFIED） | 无 | 同上 |
| Neptune-X (NeurIPS 2025) | SD1.5 布局条件扩散 | 无连续强度 | 代码+权重公开 | 是（检测 mAP +7–10） | 从零训练约 100 GPU h，超预算 |
| UniRestore (CVPR 2025) | 复原（非生成） | — | — | — | 不能当合成器用 |

决策：**A（物理，深度条件）+ B（GenDeg 仅作真实性探针）**；Neptune-X 超预算，不纳入。

### 与本地管线的差距（这是最便宜的一步升级）

`src/maritime_robust/degradations.py:10` 现在用固定 transmission 网格 `[0.85, 0.70, 0.55, 0.40, 0.25]`
配平滑场，雾是**整体发灰**，光学上不对（真实雾远端浓、近端淡）。补深度条件后
`t = exp(-β·d)` 才是文献口径（RESIDE / FoggySea）。这一步不训练、可复现、和已有 43 条件兼容。

## 4. 阶段 0 行动清单（未开始）

1. ~~审计 HSMFNet 与 EFM-Net 仓库~~ 已完成：HSMFNet 仓库 404；EFM-Net Apache-2.0、有训练入口、
   需修 6 处，约 2–4 h 到首次训练（详见 `reports/SOTA_MODEL_SURVEY.md` §2.3）。
2. 在官方 3,256/825 划分上跑一个本地强基线（ConvNeXt-B 或 ViT-B/16，224 输入），
   作为 G0 的"我们自己的下限"，避免把复现失败误判成方法弱。
3. 给退化管线加深度条件雾（MiDaS 单目深度，离线缓存深度图，约 4,080 张）。
4. 向导师确认 §1.1 的六个验收口径与预算（72 h / 50 GB 是否沿用）。

## 未验证项（写论文前必须自己核）

- HSMFNet 代码是否完整可训练；EFM-Net 代码与论文表格是否对应。
- MSCL-Net / RGCRL 的 FGSC-23 绝对值（只拿到二手引用）。
- "Ship Weather" 数据集能否下载；AIMO/RMO 仓库是否开放。
- GenDeg 权重是否真的公开、海事迁移是否成立；FoggySea 是否开源。
