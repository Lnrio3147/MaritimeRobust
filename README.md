# MaritimeRobust

复杂退化条件下光学遥感民用船舶细粒度识别。第一轮研究以基线、退化诊断和计算成本为交付，不预设创新模块；第二轮增量建立组合退化增强基线。**2026-10-29 模块线停止**：现有候选均未通过准入，真实退化分层也未发现稳定的退化准确率缺口；详见 `reports/ROUND2_DECISION.md`。

项目位于 `/home/uestc/mount_2T/uestc/lnrio/MaritimeRobust`。旧 Ship_air 的数据、代码和暂停状态不受本项目修改。

## 阅读顺序

1. `reports/STATUS.md`：交付状态和已知限制。
2. `reports/RESULTS.md`、`reports/results.json`：完成评估后的实验结果。
3. `reports/ROUND2_DECISION.md`：第二轮路线决策与 2026-10-29 停止决定（复原/局部反事实/联合训练/真实退化的结论、H1/H2 判断）。
4. `reports/COMBINED_BASELINE.md`：组合退化增强基线的配方、配对比较与边界。
5. `reports/CDLNET_REPRO.md`、`reports/cdlnet_repro.json`：CDLNet 修复版复现（非官方成绩）及其 43 条件退化结果。
6. `reports/CDLNET_DIAGNOSTICS.md`、`reports/cdlnet_diagnostics.json`：CDLNet 与三种基线的退化对照（保持率、逐级曲线、混淆对、预测塌缩、高置信错误、位置对照、分组区间）。
7. `reports/CDLNET_ABLATION.md`、`reports/cdlnet_ablation.json`：分支消融（去门控 / 去裁剪分支 / 去两者 / 同管线 ResNet-50(256)），23 类与 8 类两套口径。
8. `reports/RGPA_ROUND1.md`、`reports/rgpa_round1.json`：第一版新模块（可靠性门控部件聚合）的构建、管线复现检验与第一轮结果（未通过准入）。
9. `reports/PART_OCCLUSION.md`、`reports/part_occlusion.json`：中心/角落遮挡干预与退化错误重叠（图像区域遮挡，不是船体部件 mask）。
10. `reports/ROUTE_AB_ROUND1.md`、`reports/restoration_probe.json`、`reports/local_recovery.json`：零训练复原探测（CLAHE/饱和填充/概率融合）与 3×3 局部反事实恢复。
11. `reports/JOINT_ROUND1.md`、`reports/joint_round1.json`：联合训练消融 C1/C2/C3（局部放置增强、预测一致性、饱和加权特征一致性），三种子配对比较。
12. `reports/PAIRED_RESTORER_ROUND1.md`、`reports/paired_restorer_round1.json`：同配方配对数据训练的轻量复原器，验证集闸门结果与 `reports/figures/restorer_contact_sheet.png` 定性对照。
13. `reports/REAL_DEGRADATION_STRATIFIED.md`、`reports/real_degradation_stratified_*.json`：真实退化分层评测（FGSC-23 val 157 张、FGSRCS development 184 裁剪，7 个冻结分类器）。`docs/REAL_DEGRADATION_PROTOCOL.md` 与 `reviews/real_degradation/` 为分类、审阅页、分层与指标定义。
14. `docs/MODEL_REPRODUCTION_AUDIT.md`：CDLNet / MA-Net / RGCRL-Net 的代码与权重准入结论。
14b. `reports/DEPTH_FOG_CHECK.md`：深度条件雾在 val 上的三条件对照（结论：结构本身无效应，只影响平均不透明度）。
15. `reports/fgsc23_audit.json`：实际数据量、重复图、标签映射和划分统计。
16. `docs/DATA_PROTOCOL.md`、`docs/EXPERIMENT_PROTOCOL.md`：数据与实验定义。
17. `docs/LITERATURE.md`：论文、代码及复现取舍。
18. `docs/THESIS_SCOPE.md`、`docs/SOTA_LANDSCAPE.md`：2026-10-29 新课题的范围、启动指标映射、四道闸门，以及模型/真实数据/退化合成三条线的可用性清单。
19. `reports/EFMNET_REPRO.md`：EFM-Net 架构复现（补丁、预训练来源、训练配置、693 epoch 停止原因、85.99% / 78.34% 两套预处理口径、43 条件结果与协议偏离）。**不得作为相同协议下的 SOTA 对比数，也不能与论文 23 类 OA 直接比较。**
20. `reports/CROSS_MODEL_DEGRADATION.md`：8 个模型 × 43 条件的配对退化评估（4 个骨干家族、按来源组 bootstrap、G1 判定、预处理消融、错误转移、逐类召回、置信度）。G1 在 9 个组合/眩光条件上达到 4/4 家族；含预定义组统计、BH 多重比较、margin 与四家族共同错误。
21. `reports/G2_P0_FAILURE_MECHANISM.md`：失效机制假设（组合退化的 margin 压缩 + 类别原型吸引）、竞争解释、最低成本证伪实验（冻结特征线性探针，<30 min、无训练）与普通增强基线要求。不启动模块训练。

## 环境与命令

已创建项目内 `.venv`，通过 system-site-packages 只读复用 `copt_cuda` 中的 PyTorch；新增的小型依赖安装在项目虚拟环境内。完整版本见 `artifacts/environment.freeze.txt`。CUDA 为 12.1，torch 2.5.1，torchvision 0.20.1。无需 OpenCV。

```bash
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust
.venv/bin/python -m pytest -q
.venv/bin/python -m maritime_robust.cli environment
.venv/bin/python -m maritime_robust.cli prepare --root data/raw/fgsc23
.venv/bin/python scripts/run_first_round.py --stage all
.venv/bin/python scripts/run_combined_baseline.py --stage all
# RGPA 第一轮（结果写入 runs_modules/，不改动 45 个基线运行）
.venv/bin/python scripts/train_rgpa.py --manifest data/manifests/fgsc23_research_civilian.jsonl --backbone swin_t --mode full
.venv/bin/python scripts/rgpa_report.py
# 遮挡干预
.venv/bin/python scripts/part_occlusion.py
.venv/bin/python scripts/part_occlusion_report.py
# 零训练复原与局部反事实探测
.venv/bin/python scripts/restoration_probe.py
.venv/bin/python scripts/local_recovery.py
.venv/bin/python scripts/route_ab_report.py
# 联合训练消融 C1/C2/C3（结果写入 runs_joint/）
.venv/bin/python scripts/train_joint.py --manifest data/manifests/fgsc23_research_civilian.jsonl \
  --backbone swin_t --mode c1 --seed 17
.venv/bin/python scripts/train_joint.py --manifest data/manifests/fgsc23_research_civilian.jsonl \
  --backbone swin_t --mode c2 --seed 17
.venv/bin/python scripts/train_joint.py --manifest data/manifests/fgsc23_research_civilian.jsonl \
  --backbone swin_t --mode c3 --seed 17
.venv/bin/python scripts/joint_report.py
# 配对监督复原器（restorers/，只做验证集；闸门未过不跑测试集）
.venv/bin/python scripts/train_restorer.py --seed 17
.venv/bin/python scripts/train_restorer.py --seed 29
.venv/bin/python scripts/train_restorer.py --seed 43
.venv/bin/python scripts/restorer_eval.py --split val
.venv/bin/python scripts/restorer_report.py
.venv/bin/python scripts/restorer_contact_sheet.py
# 深度条件雾（新增条件，不改动 degradations.py 与既有 43 条件）
.venv/bin/python scripts/depth_cache.py --limit 0          # 缓存 Depth Anything V2 深度图
.venv/bin/python scripts/depth_fog_preview.py              # 联系表，仅用于目视检查
.venv/bin/python scripts/depth_fog_eval.py                 # val 对照：平坦雾 / 深度雾 / 等均值平坦雾

# 真实退化标注与分层评测（标签确认后才生成正式报告）
.venv/bin/python scripts/degradation_review_sheets.py --dataset fgsc23 --split val
.venv/bin/python scripts/degradation_review_sheets.py --dataset fgsrcs
.venv/bin/python scripts/build_degradation_review.py --dataset fgsc23_val
.venv/bin/python scripts/build_degradation_review.py --dataset fgsrcs_dev
# 审阅页导出 labels_<dataset>.json 后：
.venv/bin/python scripts/merge_degradation_labels.py --dataset fgsc23_val
.venv/bin/python scripts/merge_degradation_labels.py --dataset fgsrcs_dev
.venv/bin/python scripts/real_degradation_predict.py --dataset fgsc23_val
.venv/bin/python scripts/real_degradation_predict.py --dataset fgsrcs_dev
.venv/bin/python scripts/real_degradation_eval.py --dataset fgsc23_val
.venv/bin/python scripts/real_degradation_eval.py --dataset fgsrcs_dev
.venv/bin/python scripts/real_degradation_report.py
```

队列依次执行初始模型、验证集选择、补充种子、测试、成本测量和报告。组合增强基线单独使用 `reports/combined_queue_status.json` 和 `reports/combined_baseline.log`，不覆盖首轮队列状态。已完成且配置相同的运行会跳过；失败记录保留在预算账本中。需要恢复时先检查对应队列状态和日志，再使用同一命令；不要同时启动两个队列。

单项运行示例：

```bash
.venv/bin/python -m maritime_robust.cli train \
  --manifest data/manifests/fgsc23_research_civilian.jsonl \
  --model resnet50 --seed 17 --augmented
.venv/bin/python -m maritime_robust.cli train \
  --manifest data/manifests/fgsc23_research_civilian.jsonl \
  --model resnet50 --seed 17 --augmentation combined
.venv/bin/python -m maritime_robust.cli evaluate \
  --run runs/fgsc23_research_civilian/resnet50_combo_s17
.venv/bin/python -m maritime_robust.cli report
```

外部模型修复版复现（独立于 `runs/`，不参与基线统计）：

```bash
.venv/bin/python external/cdlnet_repro/run.py train \
  --run-dir runs_external/fgsc23_research_all/cdlnet_clean_s17
.venv/bin/python external/cdlnet_repro/run.py eval \
  --run-dir runs_external/fgsc23_research_all/cdlnet_clean_s17
.venv/bin/python external/cdlnet_repro/protocol_check.py   # 官方划分协议对齐诊断（干净集）
.venv/bin/python scripts/cdlnet_report.py
.venv/bin/python scripts/cdlnet_diagnostics.py   # 与基线的退化诊断对照
# EFM-Net 复现与跨模型退化评估（不参与 runs/ 统计；测试集未链接）
.venv/bin/python scripts/efmnet_dataset.py                 # train=618 / test(val)=157
(cd external/efmnet_repro && ../../.venv/bin/python main.py --data_path ./datasets/ \
  --dataset fgsc23_civ8 --output_dir ../../runs_external/fgsc23_civ8 \
  --batch_size 16 --epochs 1800 --update_freq 30 --seed 17 \
  --save_ckpt True --save_ckpt_freq 100 --save_ckpt_num 2)
.venv/bin/python scripts/efmnet_curve.py --run-dir runs_external/fgsc23_civ8/<run> --step 100
.venv/bin/python scripts/efmnet_eval.py --run-dir runs_external/fgsc23_civ8/<run> --split val --conditions all
.venv/bin/python scripts/efmnet_eval.py --run-dir runs_external/fgsc23_civ8/<run> --split val --conditions all \
  --preprocess efmnet --output reports/efmnet_val_conditions_native.json
.venv/bin/python scripts/baseline_val_conditions.py         # 7 个冻结基线 × 43 条件（val）
.venv/bin/python scripts/cross_model_degradation.py         # Δ、组 bootstrap、G1 闸门计数
.venv/bin/python scripts/failure_analysis.py                # 组统计、BH-FDR、margin、四家族共同错误
.venv/bin/python scripts/efmnet_reports.py                  # 生成上面三份报告
.venv/bin/python scripts/frozen_probe.py                    # 冻结特征双探针（P1/P2 + 对照），无训练
.venv/bin/python scripts/frozen_probe_report.py             # 生成 FROZEN_PROBE.md 与 G2_METHOD_DECISION.md
bash scripts/run_cdlnet_ablation.sh              # 分支消融队列（8 次训练 + 评测）
.venv/bin/python scripts/cdlnet_ablation_report.py
```

主入口之外：`scripts/prepare_fgsrcs.py` 处理真实标注裁剪；`scripts/audit_delivery.py` 验收工件。退化按需生成，无需落盘所有副本。不要修改已训练运行绑定的 manifest、配置或配方；新协议使用新项目版本/运行命名。

## 结果的含义

民船主实验是八种用途/船型类别识别；它不支持制造型号或某一艘船的身份结论。完整 23 类实验另行训练。模拟雾、云和眩光用于可控诊断，不能替代真实复杂场景验证。已知目标位置的裁剪分类不代表检测或跟踪性能。
