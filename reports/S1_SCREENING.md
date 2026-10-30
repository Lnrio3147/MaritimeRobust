# S1 筛选结果：T1 / I1 / R3

2026-10-30。第一批三个候选 + 参照臂，同一训练/评估入口，同折同种子配对。

- 数据：`data/manifests/folds/fgsc23_civ8_fold{0,1,2}.jsonl`（train 412 / val 206，组不相交）
- 训练：ConvNeXt-T，combined 增强，24 epoch，batch 64，AdamW（`configs/protocol.json`）
- 种子：17、29；每个候选 3 折 × 2 种子 = **6 个配对测量**
- 评估：`scripts/explore_eval.py`，clean + 8 个 `combo_both` 条件，`ROLE="efmnet"`，逐图配对
- 基线：`convnext_tiny_none_s{seed}`（同脚本、同调度、同增强；仅去掉干预）

## 1. 主结果（8 个 combo 条件的平均 Δ，pp）

| 方法 | 均值 Δ | 配对聚类 bootstrap 95% CI | 为正的测量 | clean Δ | 参数量 | 单次训练 |
|---|---:|---|---:|---:|---:|---:|
| **T1** 组合一致性 | **+1.66** | **[+0.63, +2.68]** | **5/6** | −0.24 | +0.00% | 55 s |
| I1 退化条件化归一化 | +0.94 | [−0.09, +1.98] | 4/6 | +0.08 | +4.11% | 45 s |
| R3 特征去噪 | −0.69 | [−1.76, +0.35] | 2/6 | −0.97 | +1.40% | 42 s |
| 参照复核：engine combo 臂 | +0.40 | [−0.70, +1.54] | — | — | — | — |

CI 用按 `source_group_id`（609 个簇，每簇含两个种子）的 4000 次 bootstrap；「为正的测量」按折×种子计。

## 2. 逐折逐种子

| 方法 | fold0/s17 | fold0/s29 | fold1/s17 | fold1/s29 | fold2/s17 | fold2/s29 |
|---|---:|---:|---:|---:|---:|---:|
| T1 | +3.4 | +2.4 | +3.6 | −2.2 | +1.1 | +1.7 |
| I1 | −1.1 | +5.6 | +1.3 | −0.7 | +0.5 | +0.6 |
| R3 | +1.5 | +2.7 | −2.2 | −3.2 | −2.5 | −0.1 |

单测量噪声 sd ≈ 2.1–2.4 pp，与实测种子噪声地板（sd 1.9 pp）一致；因此逐格数字不单独解释，只看 6 个测量的合并量。

T1 的 λ 在 fold0/s17 上于 {0.1, 0.5, 1.0} 中一次性选定（combo 均值 71.4 / 74.2 / 74.0），冻结为 **λ=0.5**；因此 fold0/s17 那一格是 λ 选择污染的，其余 5 格不是。

## 3. 判定（按 `NEXT_PHASE_PLAN.md` §3 预注册规则）

- **R3 淘汰**：三折平均 Δ 为负（−0.69）。方向排除，不进入 S2。
- **I1 保留但不晋级**：均值为正但 CI 含零且低于 +1.5 判据。若 S2 有富余名额可作为次选；单次 +4.11% 参数的成本效益不成立。
- **T1 晋级 S2**：均值 +1.66 ≥ +1.5，CI 下界 +0.63 > 0，5/6 测量为正，clean −0.24 ≥ −1.0，**零新增参数、零新增推理 FLOPs**（代价仅在训练期多一次前向）。

## 4. 结论

第一批三个方向里，唯一通过预注册判据的是 **T1（组合一致性）**，且它是唯一直接对准本数据实测现象（组合超加性）的候选。R3 作为负结果排除。I1 不足以支撑额外参数成本。

T1 的效应量（+1.66 pp）接近但未大幅超过其自身的噪声水平（CI 宽 2.05 pp），因此下一阶段的 S2 必须按计划做全量训练 + 3 种子确认，不能以 S1 数字作为结论。

## 5. 复现

```bash
for f in 0 1 2; do for s in 17 29; do for m in none r3 i1 t1; do
  .venv/bin/python scripts/explore_train.py --fold $f --seed $s --method $m --epochs 24 --lambda-consistency 0.5
  .venv/bin/python scripts/explore_eval.py --run runs/fgsc23_civ8_fold$f/convnext_tiny_${m}_s$s \
    --manifest data/manifests/folds/fgsc23_civ8_fold$f.jsonl
done; done; done
.venv/bin/python scripts/explore_summary.py --out reports/s1_summary.json
```

原始数据：`reports/s1/*.json`、`reports/s1_summary.json`。
