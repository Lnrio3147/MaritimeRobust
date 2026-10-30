# 项目收束记录

2026-10-30。本文件是这一阶段工作的**正式锁定**。锁定后不再新增训练、不改方法、不改配方、不开第三批。

## 1. 四项锁定

### 1.1 归档

- 全部训练与评估产物保留原位，不删除、不覆盖。
- 代码、配置、划分、报告、逐样本预测进入 git（`c9dbff2` 起）。
- 权重、裁剪图、原始图像不进 git（体积原因），改由哈希清单覆盖：`reports/snapshot.json`。
- 不再为「拿到一个正结果」而训练任何新模型。

### 1.2 数据披露

236 张 `split == "test"` 的图像在 2026-10-29 已被三项诊断脚本使用（`part_occlusion.py`、`local_recovery.py`、`restoration_probe.py`），**不再是未接触的盲测集**。

- 正式记录：`data/manifests/TEST_SET_USAGE.md`
- 不重命名、不改写划分文件——使用历史不能用改名抹去
- 允许的表述：「探索阶段已用于诊断性分析，非盲测」
- 禁止的表述：「一次性盲测」「hold-out」「未接触的测试集」

### 1.3 贡献边界

只声称**实证鲁棒性分析**。不声称新算法、不声称性能上限、不声称首个基准。

| 可以声称 | 依据 |
|---|---|
| 组合退化造成大幅、非均匀、超加性的下降 | E1、E2；43 条件 × 4 骨干，逐图配对 |
| 冻结线性读出的净空很小 | E3；H=+3.3 [1.0,5.6]，BH 0/12 |
| 受限训练预算会制造假增益 | E4；T1 +1.66 → −0.18 |
| 七个候选与一个初始化对照均无稳定增益 | E5；含 R1 的 CI 跨零表述修正 |
| clean 与组合退化性能不同向 | E6 |
| 改动破坏多于修复（仅描述方向） | E7；9888 个预测非独立观测 |

| 不可以声称 | 原因 |
|---|---|
| 组合退化存在性能上限 | R2 未执行，多因子、更深骨干、更大数据未测 |
| 预训练质量对组合退化不重要 | 只比较了两个具体权重 |
| 某单一机制导致了失败 | 全部为待检验假设（`FINAL_CONCLUSIONS.md` §2） |
| 真实退化上也如此 | 真实分层为观察性，重度样本 n≤2 |
| 本文为盲测结果 | 见 1.2 |

### 1.4 不再开第三批

- 不开第三批候选搜索；不实现 R2；不为挽救失败候选调参；不自动拼接模块。
- 保留方向（供将来的独立研究，不属于本项目）：R2 频域、多因子组合、非线性读出上限、真实重度退化数据。

## 2. 已投入与剩余

| 项 | 数值 |
|---|---|
| 训练 run（`runs/`） | 126 |
| 外部基线 run（`runs_external/`） | 12 |
| 训练 GPU 时间 | 约 3.3 h（预算 72 h） |
| 新增磁盘 | < 50 GB 上限内（runs 15G + data 15G + runs_external 4.9G + artifacts 5.1G） |

## 3. 统一版本快照

`reports/snapshot.json` 由 `scripts/snapshot.py` 生成，覆盖 git 无法记录的部分：

| 字段 | 内容 |
|---|---|
| `git` | commit、branch、dirty 标志 |
| `code` | `src/` `scripts/` `configs/` 每个文件的 sha256 |
| `data` | 11 个划分/清单文件 sha256（含 `folds/*.jsonl`） |
| `reports` | 143 个报告与评估 JSON 的 sha256 |
| `runs[]` | 每个 run 的 method、init、seed、epochs、manifest/config/recipe/source 哈希、`best_val_macro_f1`、参数量、耗时、产物清单 |
| `runs[].checkpoint_sha256_matches` | 重新计算的权重哈希是否等于训练时记录值 |

最近一次校验（`--verify-weights`，覆盖 `runs/` 与 `runs_external/` 全部权重）：

```
138 runs, 0 sha mismatches, 1 missing checkpoint
runs_external/fgsc23_official_all/cdlnet_protocol_check_s17
```

唯一缺失的是协议核查 run（未保存权重），既有报告中未使用其权重数值。

复现命令：

```bash
python scripts/snapshot.py                # 快速：存在性 + 体积
python scripts/snapshot.py --verify-weights  # 全量重算哈希（约 20 GB 读取）
```

## 4. 收束后的状态

- 结论以 `FINAL_CONCLUSIONS.md` 为准（实证 vs 假设分开陈述）。
- 可发表性判断以 `PUBLICATION_REVIEW.md` 为准（算法论文：不成立；评测研究：可能，有前置条件）。
- 本阶段产出定位为**研究阶段的交付物**，不是论文投稿物。
- 下一研究问题（路线 B）另行选题，不在本记录范围内；选题应在实现之前先确认数据与文献可行性。
