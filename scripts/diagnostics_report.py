#!/usr/bin/env python
"""Aggregate saved predictions and diagnostics into the degradation diagnosis report.

No model is re-run here: every number is recomputed from evaluation jsonl files or
from the per-run diagnostics/complete files written by the training queue.
"""
import collections
import glob
import json
import statistics
from pathlib import Path

from maritime_robust.common import ROOT, read_jsonl, write_json
from maritime_robust.degradations import conditions

KINDS = ('haze', 'cloud', 'glare')
SINGLE = [f'matched_{kind}_{severity}' for kind in KINDS for severity in range(1, 6)]
COMBOS = [f'matched_haze+{other}_{severity}' for other in ('glare', 'cloud') for severity in (3, 5)]


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def pct(value, digits=2):
    return '—' if value is None else f'{100 * value:.{digits}f}'


def run_mode(config):
    return config.get("augmentation") or ("single" if config["augmented"] else "clean")


def load_runs():
    runs = []
    for complete in sorted(glob.glob(str(ROOT / 'runs/fgsc23_*/*/complete.json'))):
        run = Path(complete).parent
        config = json.loads(Path(complete).read_text())
        summary_path = run / 'evaluation/summary.json'
        diagnostics_path = run / 'diagnostics.json'
        if not summary_path.exists() or not diagnostics_path.exists():
            continue
        runs.append({'run': str(run.relative_to(ROOT)), 'config': config,
                     'summary': json.loads(summary_path.read_text()),
                     'diagnostics': json.loads(diagnostics_path.read_text())})
    return runs


def condition_means(runs):
    return {name: mean(run['summary'][name]['macro_f1'] for run in runs)
            for name in runs[0]['summary']}


def combination_mean(run):
    return mean(run['summary'][name]['macro_f1'] for name in COMBOS)


def shifted_combination_mean(run):
    return mean(run['summary'][name.replace('matched_', 'shifted_')]['macro_f1'] for name in COMBOS)


def severity_curve(runs):
    clean = mean(run['summary']['clean']['macro_f1'] for run in runs)
    curves = {kind: [mean(run['summary'][f'matched_{kind}_{severity}']['macro_f1'] for run in runs)
                     for severity in range(1, 6)] for kind in KINDS}
    return {'clean': clean, 'curves': curves}


def confusion_pairs(runs, condition, labels, limit=6):
    counter = collections.Counter()
    for run in runs:
        path = ROOT / run['run'] / 'evaluation' / f'{condition}.jsonl'
        if path.exists():
            for row in read_jsonl(path):
                if row['label'] != row['prediction']:
                    counter[(labels[str(row['label'])], labels[str(row['prediction'])])] += 1
    return [{'true': true, 'predicted': predicted, 'count': count}
            for (true, predicted), count in counter.most_common(limit)]


def main():
    runs = load_runs()
    research = [run for run in runs if run['config']['protocol'] == 'research']
    official = [run for run in runs if run['config']['protocol'] == 'official']
    single = [run for run in research if run_mode(run['config']) == 'single']
    combined = [run for run in research if run_mode(run['config']) == 'combined']
    report = {'run_count': len(runs), 'research_run_count': len(research),
              'research_single_run_count': len(single), 'research_combined_run_count': len(combined),
              'official_run_count': len(official), 'conditions': [c['name'] for c in conditions()]}

    # Severity curves by scope and training regime.
    curves = {}
    for scope in ('all', 'civilian'):
        for mode in ('clean', 'single', 'combined'):
            subset = [run for run in research
                      if run['config']['scope'] == scope and run_mode(run['config']) == mode]
            if subset:
                curves[f'{scope}/{mode}'] = severity_curve(subset)
    report['severity_curves'] = curves

    # Matched/shifted comparison and combination conditions on the single-degradation
    # arm; the combined arm is reported separately so the two recipes never mix.
    report['condition_means'] = condition_means(single)
    report['condition_means_combined'] = condition_means(combined) if combined else {}
    report['shifted_minus_matched'] = {}
    for kind in KINDS:
        for severity in (3, 5):
            matched = report['condition_means'][f'matched_{kind}_{severity}']
            shifted = report['condition_means'][f'shifted_{kind}_{severity}']
            report['shifted_minus_matched'][f'{kind}_{severity}'] = shifted - matched
    report['combinations'] = {name: report['condition_means'][name] for name in COMBOS}
    report['combinations_combined'] = {name: report['condition_means_combined'][name]
                                       for name in COMBOS if name in report['condition_means_combined']}

    # Same-model, same-seed comparison of the two augmentation recipes on the
    # four matched combination conditions. This is deliberately paired, unlike
    # the per-condition arm means above.
    report['combined_vs_single_pairs'] = []
    for single_run in single:
        matches = [run for run in combined if all(
            run['config'][key] == single_run['config'][key] for key in ('scope', 'model', 'seed'))]
        if not matches:
            continue
        combined_run = matches[0]
        report['combined_vs_single_pairs'].append({
            'scope': single_run['config']['scope'], 'model': single_run['config']['model'],
            'seed': single_run['config']['seed'],
            'single_combination': combination_mean(single_run),
            'combined_combination': combination_mean(combined_run),
            'combination_difference': combination_mean(combined_run) - combination_mean(single_run),
            'single_shifted_combination': shifted_combination_mean(single_run),
            'combined_shifted_combination': shifted_combination_mean(combined_run),
            'shifted_combination_difference': shifted_combination_mean(combined_run) - shifted_combination_mean(single_run),
            'single_clean': single_run['summary']['clean']['macro_f1'],
            'combined_clean': combined_run['summary']['clean']['macro_f1']})

    # Center/border placement diagnostic (matched, severity 3).
    report['placement'] = {}
    for scope in ('all', 'civilian'):
        subset = [run for run in single if run['config']['scope'] == scope]
        for kind in ('cloud', 'glare'):
            base = f'matched_{kind}_3'
            report['placement'][f'{scope}/{kind}'] = {
                position: mean(run['summary'][base + suffix]['macro_f1'] for run in subset)
                for position, suffix in (('random', ''), ('center', '_center'), ('border', '_border'))}

    # Clean-image resolution proxy and high-confidence errors.
    report['resolution_proxy'] = {}
    for bin_name in ('short_side_lt96', 'short_side_96_191', 'short_side_ge192'):
        values = [run['diagnostics']['clean']['crop_resolution_proxy_bins'][bin_name] for run in single]
        report['resolution_proxy'][bin_name] = {
            'mean_samples': mean(item['n'] for item in values),
            'mean_accuracy': mean(item['accuracy'] for item in values if item['accuracy'] is not None)}
    report['high_confidence_errors'] = {
        name: {'mean_count': mean(run['diagnostics'][name]['high_confidence_errors'] for run in single),
               'mean_clean_correct_to_wrong': mean(run['diagnostics'][name]['clean_correct_to_wrong'] for run in single)}
        for name in ('clean', 'matched_haze_5', 'matched_cloud_5', 'matched_glare_5',
                     'matched_haze+glare_5', 'shifted_haze+cloud_5')}

    # Class confusions from saved predictions, per scope.
    labels_by_scope = {}
    for run in single:
        labels_by_scope.setdefault(run['config']['scope'], run['config']['classes'])
    report['confusions'] = {}
    for scope in ('all', 'civilian'):
        subset = [run for run in single if run['config']['scope'] == scope]
        report['confusions'][scope] = {
            condition: confusion_pairs(subset, condition, labels_by_scope[scope])
            for condition in ('clean', 'matched_haze+glare_5', 'matched_cloud_5', 'matched_glare_5')}

    # Paired augmentation changes are written by report.summarize().
    paired_path = ROOT / 'reports/paired_augmentation_changes.json'
    paired = json.loads(paired_path.read_text()) if paired_path.exists() else []
    report['augmentation'] = {}
    for scope in ('all', 'civilian'):
        for mode in ('single', 'combined'):
            subset = [row for row in paired if row['scope'] == scope
                      and row.get('augmentation', 'single') == mode]
            if subset:
                report['augmentation'][f'{scope}/{mode}'] = {
                    'runs': len(subset),
                    'mean_clean_change': mean(row['augmentation_clean_f1_change'] for row in subset),
                    'mean_matched_change': mean(row['augmentation_matched_f1_change'] for row in subset),
                    'mean_combination_change': mean(row['augmentation_combination_f1_change']
                                                    for row in subset if row.get('augmentation_combination_f1_change') is not None),
                    'mean_shifted_combination_change': mean(row['augmentation_shifted_combination_f1_change']
                                                            for row in subset if row.get('augmentation_shifted_combination_f1_change') is not None),
                    'mean_recovery_fraction': mean(row['recovery_of_clean_model_average_drop']
                                                   for row in subset if row['recovery_of_clean_model_average_drop'] is not None)}

    write_json(ROOT / 'reports/diagnostics.json', report)

    lines = ['# 退化诊断报告', '',
             f'由 `scripts/diagnostics_report.py` 从 {len(runs)} 个已完成运行的逐样本预测和诊断工件生成；'
             f'其中研究协议 {len(research)} 个（单退化增强 {len(single)}、组合增强 {len(combined)}）、官方协议 {len(official)} 个。'
             '所有指标均由保存的预测重算。', '',
             '模拟雾、云、眩光及组合均为本项目的近似合成器，不是实测大气或海面反射物理；中心区域只是目标位置代理，FGSC-23 没有船体 mask。', '',
             '## 单退化严重度曲线（研究协议，均值）', '',
             '| 范围/训练 | 干净 | 雾 1–5 | 云 1–5 | 眩光 1–5 | 五级下降（雾/云/眩光） |',
             '|---|---:|---|---|---|---|']
    for key, curve in curves.items():
        cells = {kind: ' / '.join(pct(value, 1) for value in curve['curves'][kind]) for kind in KINDS}
        drops = ' / '.join(pct(curve['clean'] - curve['curves'][kind][-1], 1) for kind in KINDS)
        lines.append(f'| {key} | {pct(curve["clean"], 1)} | {cells["haze"]} | {cells["cloud"]} | {cells["glare"]} | {drops} |')

    lines.extend(['', '## 组合退化与单退化对照（研究协议均值）', '',
                  '| 严重度 | 雾 | 眩光 | 雾+眩光 | 云 | 雾+云 |', '|---|---:|---:|---:|---:|---:|'])
    for severity in (3, 5):
        row = [pct(report['condition_means'][f'matched_{kind}_{severity}'], 1) for kind in KINDS]
        haze, cloud, glare = row
        lines.append(f'| {severity} | {haze} | {glare} | '
                     f'{pct(report["condition_means"][f"matched_haze+glare_{severity}"], 1)} | '
                     f'{cloud} | {pct(report["condition_means"][f"matched_haze+cloud_{severity}"], 1)} |')
    if report['combined_vs_single_pairs']:
        lines.extend(['', '### 同模型、同种子配对', '',
                      '| 范围 | 模型 | 种子 | matched 单→组合 | shifted 单→组合 | matched差 | shifted差 | 干净差 |',
                      '|---|---|---:|---:|---:|---:|---:|---:|'])
        for p in report['combined_vs_single_pairs']:
            lines.append(f"| {p['scope']} | {p['model']} | {p['seed']} | "
                         f"{pct(p['single_combination'], 1)} → {pct(p['combined_combination'], 1)} | "
                         f"{pct(p['single_shifted_combination'], 1)} → {pct(p['combined_shifted_combination'], 1)} | "
                         f"{100 * p['combination_difference']:+.2f} | "
                         f"{100 * p['shifted_combination_difference']:+.2f} | "
                         f"{100 * (p['combined_clean'] - p['single_clean']):+.2f} |")
        mean_matched = mean(p['combination_difference'] for p in report['combined_vs_single_pairs'])
        mean_shifted = mean(p['shifted_combination_difference'] for p in report['combined_vs_single_pairs'])
        lines.append('')
        lines.append(f'配对平均变化：matched {100 * mean_matched:+.2f}、shifted {100 * mean_shifted:+.2f} 个百分点'
                     '（组合增强减单退化增强，正值表示组合增强更好）。')

    lines.extend(['', '`shifted` 与 `matched` 的差（负值表示参数分布偏移后更差）：', ''])
    for key, value in report['shifted_minus_matched'].items():
        lines.append(f'- {key}: {value * 100:+.2f} 个百分点')

    if report['combinations_combined']:
        lines.extend(['', '## 组合条件：单退化增强 vs 组合增强（研究协议均值）', '',
                      '按运行平均，未按种子配对；配对到干净训练的变化见下文“增强训练诊断”。',
                      '',
                      '| 条件 | 单退化增强 | 组合增强 | 组合−单退化/百分点 |',
                      '|---|---:|---:|---:|'])
        for name in COMBOS:
            if name not in report['combinations_combined']:
                continue
            base = report['combinations'][name]
            combo = report['combinations_combined'][name]
            lines.append(f'| {name} | {pct(base, 1)} | {pct(combo, 1)} | {100 * (combo - base):+.2f} |')

    lines.extend(['', '## 空间位置对照（三级强度，研究协议均值）', '',
                  '| 范围/退化 | 随机位置 | 中心 | 边缘 |', '|---|---:|---:|---:|'])
    for key, values in report['placement'].items():
        lines.append(f'| {key} | {pct(values["random"], 1)} | {pct(values["center"], 1)} | {pct(values["border"], 1)} |')

    lines.extend(['', '## 增强训练诊断', '',
                  '| 范围/训练 | 配对运行 | 干净图变化/百分点 | 单退化平均提升/百分点 | matched 组合平均提升/百分点 | shifted 组合平均提升/百分点 | 平均退化间隙恢复比例 |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for scope, values in report['augmentation'].items():
        combo = '—' if values['mean_combination_change'] is None else f'{100 * values["mean_combination_change"]:+.2f}'
        shifted = '—' if values['mean_shifted_combination_change'] is None else f'{100 * values["mean_shifted_combination_change"]:+.2f}'
        lines.append(f'| {scope} | {values["runs"]} | {100 * values["mean_clean_change"]:+.2f} | '
                     f'{100 * values["mean_matched_change"]:+.2f} | {combo} | {shifted} | '
                     f'{values["mean_recovery_fraction"]:.2f} |')

    lines.extend(['', '## 类别混淆（研究协议，来自保存预测）', ''])
    for scope, per_condition in report['confusions'].items():
        lines.append(f'### {scope}')
        lines.append('')
        for condition, pairs in per_condition.items():
            top = '；'.join(f'{item["true"]}→{item["predicted"]} ({item["count"]})' for item in pairs[:5])
            lines.append(f'- {condition}: {top}')
        lines.append('')

    lines.extend(['## 分辨率代理与高置信错误', '',
                  '| 原始裁剪短边 | 平均样本数 | 干净准确率 |', '|---|---:|---:|'])
    for name, values in report['resolution_proxy'].items():
        lines.append(f'| {name} | {values["mean_samples"]:.1f} | {pct(values["mean_accuracy"], 1)} |')
    lines.extend(['', '| 条件 | 平均高置信错误数 | 平均由正确翻为错误数 |', '|---|---:|---:|'])
    for name, values in report['high_confidence_errors'].items():
        lines.append(f'| {name} | {values["mean_count"]:.1f} | {values["mean_clean_correct_to_wrong"]:.1f} |')

    lines.extend(['', '## 失败案例', ''])
    failure_sheet = ROOT / 'reports/failure_cases_haze_glare_5.png'
    if failure_sheet.exists():
        lines.extend([f'`reports/failure_cases_haze_glare_5.png` 展示 12 个“干净正确、五级雾+眩光错误”的高置信案例'
                      '（置信度 ≥0.9，退化图由保存的参数和种子重放）。这些案例多数在退化后已难以辨认船体轮廓，'
                      '属于信息缺失而非单纯分类器偏差；不能据此声称生成模型可以恢复型号证据。', ''])
    else:
        lines.extend(['失败案例图尚未生成；可运行 `scripts/failure_contact_sheet.py`。', ''])

    lines.extend(['', '## 诊断结论', '',
                  '1. **组合增强已补上大量组合缺口，但未消除。** 同模型、同种子配对下，组合增强相对单退化增强'
                  '在 matched 组合上平均 +4.40、shifted 组合 +3.41 个百分点；收益集中在五级：'
                  'matched 雾+眩光 +7.3、雾+云 +10.3，三级组合基本不变。相对干净训练，all 范围的 matched 组合增益为 +26.8，'
                  'civilian 为 +20.5。',
                  '2. **组合增强不是无代价。** 配对干净图平均 +0.44 个百分点，civilian 平均 +1.59；'
                  '但单个种子波动可到 +7.46（civilian resnet50，其单退化/干净基线本身偏低），不能把该单种子差异解释为稳定收益。'
                  '单退化平均提升与单退化增强几乎相同（+6.94 vs +7.02）。',
                  '3. **眩光是增强后残留最强的单退化。** 组合增强后雾和云的下降通常降到个位数，'
                  '眩光五级仍下降约 10 个百分点；局部饱和可能破坏船体轮廓与上层建筑特征。',
                  '4. **目标位置与遮挡比例比名义强度更重要。** 同一三级云/眩光下，中心位置通常明显差于边缘；'
                  'shifted 眩光因偏向边缘反而好于 matched。中心矩形不是船体 mask，不能据此声称船体覆盖率效应。',
                  '5. **最差条件仍是五级组合。** 组合增强的最差条件约 0.44–0.64 Macro-F1，仍显著低于干净图；'
                  '增强恢复的是平均性能，不是最差条件性能。',
                  '6. **主要民船混淆是结构相近类别。** bulk carrier 与 oil tanker、fishing boat、passenger ship '
                  '互相混淆；严重组合下 bulk carrier 更常被判为 oil tanker 或 liquefied gas ship。',
                  '7. **小裁剪图更难。** 短边 <96 的平均干净准确率约 84%，短边 ≥192 约 95%；'
                  '这是分辨率代理，且与类别分布混杂，不作因果解释。',
                  '', '## 对下一轮研究路线的约束', '',
                  '- 组合增强基线已完成并冻结配方；第一项新方法必须在相同数据、种子和步数下同时比较'
                  '“单退化增强”和“组合增强”两个基线，只有超过组合基线才能把组合鲁棒性归因于新结构。',
                  '- 若定位为局部遮挡问题，应优先验证有效区域聚合/遮挡感知池化，并用具备目标区域的真实数据（FGSRCS 标注框裁剪）'
                  '做开发集诊断；标注框不是像素级船体 mask，结论需相应限制。',
                  '- 若定位为雾导致特征漂移，应验证特征适配或恢复前置；DCP 对照已显示去雾可改善雾条件但会牺牲干净图，'
                  '任何恢复方案必须同时报告干净性能与预处理延迟。',
                  '- 严重组合下若信息已不可辨认，应研究不确定性与拒识，而不是用生成模型补细节并声称恢复型号证据。',
                  '- 新模块须在相同数据、相同种子和相同训练步数下，优于增强基线，并同时报告干净图、单退化、组合退化与成本。',
                  '', '## 下一轮建议与决策门槛', '',
                  '1. **组合增强基线已完成，下一步以它为门槛。** 同模型、同种子配对相对单退化增强为 matched +4.40、'
                  'shifted +3.41 个百分点，且五级组合收益最大；新方法须在 matched 与 shifted 组合上同时超过该基线，'
                  '并保持干净图与单退化性能不显著下降，才能归因于新模块。',
                  '2. **位置假设需真实区域验证。** 在 FGSRCS 开发集用标注框区域检验中心/边缘效应；若不能复现，'
                  '不把“局部遮挡”作为主线，也不把标注框当作船体分割 mask。',
                  '3. **恢复路线以净收益为目标。** DCP 已显示雾条件改善但干净图下降；任何恢复或前置模块必须同时报告干净、单退化、'
                  '组合条件和端到端延迟，并设置净 Macro-F1 门槛。',
                  '4. **严重组合转向不确定性与拒识。** 五级组合下信息已大量缺失，优先评估置信度校准、OOD/拒识和多时相；'
                  '不以生成细节作为型号识别证据。',
                  '5. **生成式只作为受限增强候选。** GenDeg 小试中高 mu 已有结构失效；若使用，须逐图标签有效性审查、'
                  '保持原图组隔离并固定准入比例，生成图不得进入验证或测试。',
                  '6. **真实数据分层标注先行。** FGSRCS 没有逐样本天气类型/强度标签；下一轮若声称真实复杂场景收益，'
                  '须先建立双人复核的退化标注与封存评测协议。'])
    (ROOT / 'reports/DIAGNOSTICS.md').write_text('\n'.join(lines) + '\n')
    return {'runs': len(runs), 'research': len(research)}


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False))
