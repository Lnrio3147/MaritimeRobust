# 文献与复现清单（核查日期：2026-10-27）

本表区分领域方法、通用架构和退化/恢复方法。论文报告领先并不等于在本项目划分下领先。未复现结果不进入本地性能表。

| 方法/数据 | 原始来源 | 用途与复现状态 | 数据与成本注意事项 |
|---|---|---|---|
| FGSC-23 / 属性引导多层特征网络（2020） | [论文 DOI](https://doi.org/10.1109/JSTARS.2020.2981686)、[作者数据仓库](https://github.com/xiong577/ship-datasets) | 数据已获取；作者数据仓库仅含 README，未发现可核验的原模型训练实现；本轮不猜测复现 | 23 类含军民类别与非船；本地包 4,080 图；论文协议、包版本须单列 |
| FGSRCS | [作者仓库](https://github.com/dwddw/FGSRCS) | 已完成准入：4,500 张原图、11,187 个已知框裁剪；开发/封存按原图组 30%/70%，仅 container/oil-tanker 严格映射到 FGSC-23；未做真实天气分层识别评测 | 包约 6.6GB；17 类以军舰为主，民船裁剪 862 个；不是逐样本天气标签集，缺少港口/时相身份 |
| CARP（2025） | [原论文](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1662024/full) | 最直接的云遮挡细粒度分类相关工作；论文页面未给可确认的代码/权重入口，数据声明为作者提供；保留文献参照 | CLIP/CoOp、少样本任务，SeaCloud-Ship 为合成云基准；不能与本轮全监督训练直接排名 |
| Efficient Prompt Tuning（2024） | [原论文](https://arxiv.org/abs/2403.08271) | 遥感船舶视觉语言模型路线的后续候选；本轮未复现 | FGSCM-52、新的预训练与少样本口径应核对 |
| HSMFNet（2026） | [期刊原文](https://www.tandfonline.com/doi/full/10.1080/01431161.2026.2724159)、[论文所列代码](https://github.com/xndy002605/HSMFNet) | 近期领域比较候选，论文报告 FGSC-23 OA 93.21%；本轮未复现，代码链接不等于本地已认证可运行 | 论文报告 224 输入约 2.72 GFLOPs；划分、宏指标与数据版本需复核，不能拿论文数字替代本地对照 |
| MA-Net（2026） | [期刊原文](https://www.mdpi.com/2072-4292/18/3/462) | 多粒度、局部注意力及特征增强方向；本轮未复现 | 与未来局部区域方法的创新边界有关，不凭摘要认定代码可运行 |
| ResNet-50 | [torchvision 官方说明](https://pytorch.org/vision/0.20/models/generated/torchvision.models.resnet50.html) | 本地通用基线，ImageNet-1K V2 | 约 25M 原始参数；本轮替换分类头后实测；权重哈希绑定运行 |
| ConvNeXt-Tiny | [torchvision 官方说明](https://pytorch.org/vision/0.20/models/generated/torchvision.models.convnext_tiny.html) | 本地通用基线，ImageNet-1K V1 | 官方权重；不称为船舶专用 SOTA |
| Swin-T | [torchvision 官方说明](https://pytorch.org/vision/0.20/models/generated/torchvision.models.swin_t.html) | 本地通用基线，ImageNet-1K V1 | 不使用 ImageNet-22K 权重混入对照 |
| GenHaze（ICCV 2025） | [CVF 原论文](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_GenHaze_Pioneering_Controllable_One-Step_Realistic_Haze_Generation_for_Real-World_Dehazing_ICCV_2025_paper.html) | 单步、参考可控雾合成；已存论文；未确认官方代码/权重，不用同名猜测仓库替代 | 原论文验证去雾，未证明遥感细粒度船体结构保持；本地未运行时不声称通过质量筛查 |
| GenDeg（CVPR 2025） | [论文](https://arxiv.org/abs/2411.17687)、[官方代码](https://github.com/sudraj2002/GenDeg)、[官方项目](https://sudraj2002.github.io/gendegpage/) | 已完成官方权重分块校验、EMA 转换和 8 类×2 强度的验证集小试；结论见 generative_feasibility.json 与 reports/gendeg_pilot_sheet.png | InstructPix2Pix/SD 路线，六种退化不包含船舶眩光和碎云；高 mu 下 3/16 样本结构失效，未用于训练，任何使用须逐图标签有效性审查 |
| UniRestore（CVPR 2025） | [CVF 原论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.pdf)、[官方代码](https://github.com/unirestore/UniRestore) | 感知与任务导向图像恢复，作为后续恢复路线参照；本轮不训练 | 不能当作退化生成器；视觉恢复指标改善不自动等于识别改善 |
| UMDATrack（ICCV 2025） | [CVF 原论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Yao_UMDATrack_Unified_Multi-Domain_Adaptive_Tracking_Under_Adverse_Weather_Conditions_ICCV_2025_paper.pdf)、[官方代码](https://github.com/Z-Z188/UMDATrack) | 跟踪任务的天气适配与合成思路；本轮不作为分类对照 | DCA/TCA 的跟踪定位收益不能直接迁移为分类有效性证明 |
| DCP（CVPR 2009 / TPAMI 2011） | [作者页面](https://people.csail.mit.edu/kaiming/cvpr09/index.html) | 实现暗通道+guided filter 去雾对照；与合成雾成像模型分开 | 白色船体和强高亮可能破坏先验；必须报告去雾后的识别与全部预处理延迟 |

## 领域复现席位结论

按既定优先级检查了 FGSC-23 原属性引导方法和 CARP。作者数据仓库没有模型实现，CARP 页面没有可确认的训练代码与权重链接；因此当前席位保留为文献参照，不以自拟实现冒充原方法。这里表达的是本轮检索证据，不断言整个互联网不存在代码；找到作者可运行实现后可单独追加对照，保留预训练与训练预算差异。

## 对后续选题的约束

“加云后分类掉点”“使用提示/适配器处理云遮挡”“增加合成数据”本身已有直接相关研究，不能作为充分创新点。下一阶段应先从本地结果确定具体失败机制，再验证一个针对性的改进，并与同数据增强的强基线比较。

若严重遮挡已消除可辨认信息，应研究不确定性/拒识或利用后续多时相证据，而不是以图像生成补出的细节当作型号识别证据。若主要问题是背景或分布漂移，则优先评估有效区域聚合、特征适配等假设。以上是待实验支持的候选，不代表本轮已得结论。
