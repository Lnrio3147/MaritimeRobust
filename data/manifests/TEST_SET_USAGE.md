# 数据使用披露：236 张 test 划分

**状态：已用于探索性诊断，不是未接触的盲测集。**

记录日期 2026-10-30。本文件是使用历史的正式记录；不重命名、不改写任何划分文件。

## 事实

`fgsc23_research_civilian.jsonl` 中的 `split == "test"` 共 **236 张**。它们在 2026-10-29 被以下三项诊断脚本读取并评估：

| 脚本 | 输出 | 输出时间 | n |
|---|---|---|---|
| `scripts/part_occlusion.py:117` | `reports/part_occlusion.json` | 2026-10-29 11:18 | 236 |
| `scripts/local_recovery.py` | `reports/local_recovery.json` | 2026-10-29 11:41 | 236 |
| `scripts/restoration_probe.py` | `reports/restoration_probe.json` | 2026-10-29 11:40 | 236 |

## 性质界定

- 这些评估**没有**把测试图放入训练。它们是模型行为诊断（区域遮挡、局部恢复、复原探针），不参与任何训练或超参选择。
- 因此这**不是训练集/测试集泄漏**。
- 但它们的结论参与了研究者对模型行为的理解，并影响了后续的研究判断。因此这 236 张**不再构成「未接触的最终盲测集」**。

## 允许的表述

- 「236 张测试图在探索阶段被用于诊断性分析，本文所有基于该划分的数字均非盲测。」
- 「该划分在方法筛选阶段未被使用」（可核实：`scripts/explore_*.py` 全部只读 `data/manifests/folds/*`，其 fiducial 行来自 `split == "train"`）。

## 禁止的表述

- 「一次性盲测」「未接触的测试集」「hold-out 集合」。
- 用重命名（如改称 validation-2）来抹去使用历史。

## 核实方法

```bash
grep -n 'split.*test' scripts/part_occlusion.py scripts/local_recovery.py scripts/restoration_probe.py
python -c "import json;print({p:json.load(open(p))['n'] for p in ['reports/part_occlusion.json','reports/local_recovery.json','reports/restoration_probe.json']})"
grep -rn 'split' scripts/explore_folds.py   # 只取 split=="train"
```

2026-10-30 实测：

| 检查 | 结果 |
|---|---|
| `scripts/explore_folds.py:36` | `rows = [r for r in read_jsonl(...) if r["split"] == "train"]`，只取 train |
| `scripts/explore_train.py` 中的 `"test"` | 仅出现在 `L114` 的**泄漏断言**（检查 source_group 是否跨划分重叠），不读取 test 行用于训练或评估 |
| `scripts/explore_eval.py` | 读 fold 清单的 `split == "val"` 行 |
