#!/usr/bin/env python
"""Write a focused summary of the combined-degradation augmentation baseline."""
import json
from statistics import mean

from maritime_robust.common import ROOT

MATCHED_COMBOS = ('matched_haze+glare_3', 'matched_haze+glare_5',
                  'matched_haze+cloud_3', 'matched_haze+cloud_5')
SHIFTED_COMBOS = tuple(name.replace('matched_', 'shifted_') for name in MATCHED_COMBOS)
SEVERE_COMBOS = MATCHED_COMBOS + SHIFTED_COMBOS


def pct(value, digits=2):
    return '—' if value is None else f'{100 * value:.{digits}f}'


def main():
    results = json.loads((ROOT / 'reports/results.json').read_text())
    diagnostics_path = ROOT / 'reports/diagnostics.json'
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    runs = results['runs']
    combined = [run for run in runs if run.get('protocol') == 'research'
                and run.get('augmentation') == 'combined' and not run.get('restoration')]
    single = [run for run in runs if run.get('protocol') == 'research'
              and run.get('augmentation') == 'single' and not run.get('restoration')]

    lines = ['# 组合退化增强基线', '',
             '本页只汇总第二轮增量：在相同数据、架构、种子和训练步数下，把单退化增强替换为“单退化 + 组合退化”混合增强。'
             '它不是新模块，也不是对真实复杂场景的验证。', '',
             '## 配方（`configs/combined_augmentation.json`）', '',
             '- 每个样本每个 epoch 以 50% 概率保持干净；其余 50% 为退化。',
             '- 退化抽样中 50% 为单退化（雾/云/眩光均匀，强度 1–5），50% 为组合退化（雾+眩光/雾+云均匀，强度 3 或 5）。',
             '- 组合使用 matched 域、随机位置；shifted 组合仍只出现在测试中，属于分布偏移。',
             '- 除增强配方外，训练超参数、几何增强、验证选型与 43 个测试条件均与首轮一致。', '',
             '## 测试集汇总', '',
             '| 范围 | 模型 | 种子数 | 干净 | 单退化平均 | matched 组合平均 | shifted 组合平均 | 最差条件 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in sorted(results['aggregates'], key=lambda r: (r['scope'], r['model'], r['augmentation'])):
        if row['augmentation'] != 'combined':
            continue
        lines.append(f"| {row['scope']} | {row['model']} | {len(row['seeds'])} | "
                     f"{pct(row['clean_macro_f1_mean'])} | {pct(row['matched_single_macro_f1_mean'])} | "
                     f"{pct(row['matched_combination_macro_f1_mean'])} | "
                     f"{pct(row['shifted_combination_macro_f1_mean'])} | {pct(row['worst_macro_f1_mean'])} |")

    lines.extend(['', '## 与单退化增强的同模型、同种子配对', '',
                  '变化 = 组合增强 − 单退化增强；正值表示组合增强更好。干净图变化用于检查基线的干净性能代价。', '',
                  '| 范围 | 模型 | 种子 | matched 组合−单退化/百分点 | shifted 组合−单退化/百分点 | 干净图−单退化/百分点 |',
                  '|---|---|---:|---:|---:|---:|'])
    differences = []
    shifted_differences = []
    clean_differences = []
    for combined_run in sorted(combined, key=lambda r: (r['scope'], r['model'], r['seed'])):
        match = [run for run in single if all(run[key] == combined_run[key] for key in ('scope', 'model', 'seed'))]
        if not match:
            continue
        single_run = match[0]
        combination_difference = combined_run['matched_combination_macro_f1'] - single_run['matched_combination_macro_f1']
        shifted_difference = combined_run['shifted_combination_macro_f1'] - single_run['shifted_combination_macro_f1']
        clean_difference = combined_run['clean_macro_f1'] - single_run['clean_macro_f1']
        differences.append(combination_difference)
        shifted_differences.append(shifted_difference)
        clean_differences.append(clean_difference)
        lines.append(f"| {combined_run['scope']} | {combined_run['model']} | {combined_run['seed']} | "
                     f"{100 * combination_difference:+.2f} | {100 * shifted_difference:+.2f} | "
                     f"{100 * clean_difference:+.2f} |")
    if differences:
        lines.extend(['', f'配对平均：matched 组合 {100 * mean(differences):+.2f} 个百分点，'
                          f'shifted 组合 {100 * mean(shifted_differences):+.2f} 个百分点，'
                          f'干净图 {100 * mean(clean_differences):+.2f} 个百分点。'])

    if diagnostics:
        lines.extend(['', '## 组合条件（按运行平均）', '',
                      '| 条件 | 单退化增强 | 组合增强 | 组合−单退化/百分点 |',
                      '|---|---:|---:|---:|'])
        for name in SEVERE_COMBOS:
            base = diagnostics.get('condition_means', {}).get(name)
            combo = diagnostics.get('condition_means_combined', {}).get(name)
            if base is None or combo is None:
                continue
            lines.append(f'| {name} | {pct(base, 1)} | {pct(combo, 1)} | {100 * (combo - base):+.2f} |')

    lines.extend(['', '## 解读边界', '',
                  '- 组合增强只训练 matched 域的 3/5 级组合；shifted 组合、1/2 级组合和真实天气都不在训练分布内。',
                  '- checkpoint 仍按干净验证 Macro-F1 选择，组合性能不是选型目标；这避免测试条件调参，但可能低估组合增强潜力。',
                  '- 干净图与单退化性能必须与组合收益一起报告；不能只用最差条件挑结果。',
                  '- 该基线用于第二轮新模块的同数据、同种子对照，不构成新方法收益或真实场景泛化证据。', ''])
    (ROOT / 'reports/COMBINED_BASELINE.md').write_text('\n'.join(lines))
    return {'combined_runs': len(combined), 'paired_runs': len(differences)}


if __name__ == '__main__':
    print(main())
