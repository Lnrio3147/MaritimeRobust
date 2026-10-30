#!/usr/bin/env python
"""Render degradation-induced high-confidence failures for a completed run.

The degraded image is regenerated deterministically from the saved seed and
degradation parameters; the saved prediction files remain the source of truth.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from maritime_robust.common import ROOT, read_jsonl, write_json
from maritime_robust.degradations import corrupt


def load_cases(run, condition, count):
    config = json.loads((run / 'complete.json').read_text())
    manifest_path = (ROOT / config['manifest'])
    rows = {row['sample_id']: row for row in read_jsonl(manifest_path)}
    clean = {row['sample_id']: row for row in read_jsonl(run / 'evaluation/clean.jsonl')}
    degraded = read_jsonl(run / 'evaluation' / f'{condition}.jsonl')
    cases = []
    for row in degraded:
        previous = clean[row['sample_id']]
        if previous['prediction'] == row['label'] and row['prediction'] != row['label']:
            probability = max(row['probabilities'])
            if probability >= .9:
                cases.append((probability, row))
    cases.sort(reverse=True, key=lambda item: item[0])
    selected = []
    for probability, row in cases[:count]:
        source = ROOT / rows[row['sample_id']]['path']
        with Image.open(source) as image:
            rgb = image.convert('RGB')
            content = ImageOps.contain(rgb, (224, 224), Image.Resampling.BICUBIC)
        params = row['degradation']
        degraded_image, _ = corrupt(content, kind=params['kind'], severity=params['severity'],
                                    seed=params['seed'], domain=params['domain'],
                                    placement=params.get('placement', 'random'))
        image = ImageOps.pad(degraded_image, (224, 224), method=Image.Resampling.BICUBIC, color=(124, 116, 104))
        selected.append({'sample_id': row['sample_id'], 'path': rows[row['sample_id']]['path'],
                         'label': config['classes'][str(row['label'])],
                         'prediction': config['classes'][str(row['prediction'])],
                         'confidence': probability, 'image': image,
                         'degradation': params})
    return selected, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--condition', default='matched_haze+glare_5')
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--output', default='reports/failure_cases.png')
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    selected, config = load_cases(run, args.condition, args.count)
    if not selected:
        raise SystemExit('No matching high-confidence failures')
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * 240, rows * 270), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, case in enumerate(selected):
        x, y = (index % columns) * 240 + 8, (index // columns) * 270 + 8
        sheet.paste(case['image'], (x, y))
        caption = [f"{case['label']} -> {case['prediction']}",
                   f"conf {case['confidence']:.3f} | {args.condition}"]
        for line, text in enumerate(caption):
            draw.text((x, y + 228 + line * 12), text, fill=(20, 20, 20), font=font)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    write_json(output.with_suffix('.json'), {
        'run': str(run.relative_to(ROOT)), 'condition': args.condition, 'count': len(selected),
        'checkpoint_sha256': config['checkpoint_sha256'], 'manifest_sha256': config['manifest_sha256'],
        'selection': 'clean-correct and degraded wrong, sorted by confidence, confidence >= 0.9',
        'degraded_images_regenerated_from_saved_seed': True,
        'cases': [{key: value for key, value in case.items() if key != 'image'} for case in selected]})
    print(json.dumps({'output': str(output.relative_to(ROOT)), 'cases': len(selected)}))


if __name__ == '__main__':
    main()
