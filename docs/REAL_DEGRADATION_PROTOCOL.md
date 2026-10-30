# 真实退化分层评测协议

本协议补充 `docs/DATA_PROTOCOL.md`，只约束真实退化标签的建立、分层与评测口径。
合成退化实验仍按原有预注册规则执行。

## 1. 数据范围

| 数据集 | 角色 | 数量 | 用途 |
|---|---|---|---|
| FGSC-23 验证集 | 开发/验证 | 157 张，8 类 | 主分层评测，标签确认后可用 |
| FGSRCS development（严格映射） | 开发/验证 | 184 个裁剪，container-ship / oil-tanker | 跨库参考分层，只评 2 类 |
| FGSC-23 测试集 | 封存 | 236 张 | 本轮不读、不评、不调参 |
| FGSRCS sealed | 封存 | 317 个裁剪 | 本轮不读、不评、不调参 |

标签只从开发数据建立。封存数据不用于确定标签细则、阈值或分层定义。

## 2. 标签类型

每张图可多选类型，另给一个总体强度和一个场景杂乱标记。

| 字段 | 取值 | 操作定义 |
|---|---|---|
| `degradation_types` | `cloud` 厚云 | 连续、成片的云层遮挡船体或海天线，局部对比度因遮挡而下降；不是薄雾或单纯阴天 |
| | `haze` 雾/霾 | 大气散射造成的远距离发灰、地平线褪色、远景对比度下降；船体轮廓仍基本连续 |
| | `shadow` 阴影 | 船体或关键结构被强投影/背光压暗，细节落在暗部；不等于整体曝光不足 |
| | `glare` 强反光/过曝 | 镜面高光、太阳耀斑或相机饱和，像素接近 250 并丢失结构 |
| | `blur` 模糊/低分辨率 | 失焦、运动模糊或原始分辨率不足，边缘不可分辨；不等于压缩块效应 |
| `severity` | `none` | 无上述类型 |
| | `mild` | 存在退化但船体类别判别信息基本完整 |
| | `medium` | 判别信息明显受损，仍可人工判类 |
| | `severe` | 判别信息大面积丢失，人工判类困难 |
| | `unable` | 无法判断（图像本身不可读或证据不足） |
| `clutter` | 布尔 | 场景杂乱（港口设施、其他船、岸线遮挡）；是场景因素，不是退化类型 |
| `uncertain` | 布尔 | 标注者对类型或强度不确定；不替代 `unable` |

约束：

- 有类型时强度必须是 `mild`/`medium`/`severe`/`unable`；无类型时强度必须是 `none`/`unable`。
- 同一类型不重复；类型顺序归一化为字典序。
- 不允许根据亮度、饱和度、对比度等图像统计自动推断雾/云标签（`docs/DATA_PROTOCOL.md:37`）。
  AI 初标只作为人工审阅的预填，必须逐图确认。
- 单张图可同时有多个类型；`multi_type` 分层指类型数 ≥ 2。

## 3. 标注流程

1. `scripts/degradation_review_sheets.py` 生成带稳定 `review_index` 的联系表。
2. `scripts/build_degradation_review.py` 生成离线 HTML 审阅页，预填 AI 初标。
3. 人工逐图确认或修改；页面把状态存在浏览器 localStorage，导出 `labels_<dataset>.json`。
4. `scripts/merge_degradation_labels.py` 校验：全部 `confirmed=true`、索引/sample_id 一一对应、
   类型与强度合法；任一不满足即拒绝合并。脚本不会补标签或改标签。
5. 合并结果写 `reviews/real_degradation/<dataset>/labels.jsonl` 与 `labels_summary.json`，
   记录导出文件与标签文件的 SHA-256。

AI 初标只使用图像外观与类别名，不接触分类器预测，避免用模型结果反向定义分层。

## 4. 分层定义

在标签表上预先定义，评测前不按结果增删：

| 分层 | 定义 |
|---|---|
| `clear` | 无类型且强度 `none` |
| `any_degraded` | 至少一个类型 |
| `type:cloud` / `haze` / `shadow` / `glare` / `blur` | 含该类型 |
| `severity:mild` / `medium` / `severe` | 总体强度等于该值 |
| `multi_type` | 类型数 ≥ 2 |
| `clutter` | 场景杂乱为真 |
| `clear_no_clutter` | 清晰且不杂乱，作为干净对照的严格子集 |

`clear` 包含杂乱图；因此 `clear_no_clutter` 与 `clutter` 的对比用于区分“退化”与“场景杂乱”。
没有类型但强度为 `unable` 的图不进入 `clear` 或 `any_degraded`，单独计数。

## 5. 指标定义

对每个分层、每个分类器分别计算；分母均为该分层内的样本数。

- **accuracy** = 分层内预测正确数 / 分层样本数。
- **balanced accuracy** = 分层内出现的每个类别的 recall 的算术平均；
  recall_c = 分层内类别 c 正确数 / 分层内类别 c 样本数。分层内未出现的类别不参与平均。
- **macro-F1** = 分层内出现的每个类别的 F1 的算术平均；未出现类别不参与平均。
  报告必须列出参与平均的类别；类别缺失时不得与全测试集 macro-F1 直接比较。
- **delta vs clear** = 100 × (分层指标 − 同一分类器在 `clear` 上的指标)，单位 pp。
  只做同分类器配对，不跨骨干比较。
- **置信区间** = 95% 百分位组自助法：在分层内按 `source_group_id` 有放回重采样
  2,000 次；同一原图组的样本一起进出。区间反映抽样不确定性，不消除分层选择偏差。

`N < 15` 或分层内原图组数 `< 8` 标为 exploratory：只描述，不做方向性结论，
不计算 delta 区间。

## 6. 评测口径

- 分类器为已冻结的组合增强基线：ConvNeXt-Tiny s17/s29/s43、Swin-Tiny s17/s29/s43、
  ResNet-50 s17（单种子，仅诊断）。不因分层结果重新训练或选检查点。
- FGSC-23 验证集：8 类标签，按原始 224 管线（contain → 可选退化 → pad → ImageNet 归一化）；
  真实退化图不施加合成退化，只做 contain → pad → 归一化。
- FGSRCS development：严格映射到 FGSC-23 民用 8 类中的 container ship(0) 与 oil tanker(3)，
  只报告这两类的 accuracy/balanced accuracy；不外推 8 类 macro-F1。
- 推理脚本 `scripts/real_degradation_predict.py` 缓存逐样本概率；标签确认后
  `scripts/real_degradation_eval.py` 只做连接与统计，不重跑模型。
- 报告必须同时给出样本数、原图组数、类别支持度与 exploratory 标记。

## 7. 解释边界

- 分层是观察性的，不是随机干预；退化类型与类别、港口、拍摄条件可能混杂。
- 单轮人工标注，无第二标注者，未计算标注者间一致性；标签噪声未建模。
- 现有两库中厚云与重度退化样本极少（本轮初标为 0 张 `severe`），无法支持
  “厚云/重度退化下性能如何”的统计结论；只能描述 `any_degraded`、`haze`、`mild` 等层。
- FGSRCS 只有 2 个映射类别，且裁剪尺度与 FGSC-23 不同；跨库差异不能归因于退化。
- 封存测试保持冻结；本协议不授权在测试集上做分层探索。

## 8. 变更记录

- 2026-10-29：建立协议；FGSC-23 val 与 FGSRCS development 的审阅页与 AI 初标完成，
  用户逐图确认并合并。确认后 FGSC-23 val 为清晰 64 / 退化 93（厚云 0、重度 2），
  FGSRCS 为清晰 151 / 退化 33（厚云 4、重度 2）；分层评测见
  `reports/REAL_DEGRADATION_STRATIFIED.md`。厚云与重度样本仍不足，报告只做描述性比较。
