#!/usr/bin/env python
"""Screen FGSC-23 against FGSRCS for exact and conservative near-duplicate overlap.

This is a source-level screen, not a claim of scene, port or time independence.
Images are only read for integrity, pixel hashes and thumbnail-based duplicate checks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from maritime_robust.common import ROOT, digest, read_jsonl, write_json

HAMMING_MAX = 4
MAD_MAX = 0.035


def fingerprint(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        pixels = np.asarray(rgb)
        pixel_hash = hashlib.sha256(str(rgb.size).encode() + pixels.tobytes()).hexdigest()
        small = np.asarray(rgb.resize((32, 32)), dtype=np.float32) / 255
        coeff = dctn(np.asarray(rgb.convert('L').resize((32, 32))), norm='ortho')[:8, :8]
        bits = coeff.ravel() > np.median(coeff.ravel()[1:])
        bits[0] = False
    return pixel_hash, bits, small.ravel()


def load_group(rows, name, limit=None):
    records = []
    for row in rows[:limit]:
        path = ROOT / row['path']
        pixel_hash, bits, small = fingerprint(path)
        records.append({'id': row.get('sample_id') or row['source_image_id'], 'path': row['path'], 'pixel_hash': pixel_hash,
                        'bits': bits, 'small': small, 'group': row.get('source_group_id') or name})
    return records


def screen(queries, references):
    reference_bits = np.stack([row['bits'] for row in references])
    reference_small = np.stack([row['small'] for row in references])
    by_pixel = {}
    for index, row in enumerate(references):
        by_pixel.setdefault(row['pixel_hash'], []).append(index)
    exact = []
    near = []
    for query in queries:
        for index in by_pixel.get(query['pixel_hash'], []):
            exact.append({'query': query['id'], 'reference': references[index]['id'],
                          'query_path': query['path'], 'reference_path': references[index]['path']})
        distances = np.count_nonzero(reference_bits != query['bits'], axis=1)
        candidates = np.flatnonzero(distances <= HAMMING_MAX)
        if not len(candidates):
            continue
        differences = np.abs(reference_small[candidates] - query['small']).mean(axis=1)
        for position in np.flatnonzero(differences <= MAD_MAX):
            index = int(candidates[position])
            near.append({'query': query['id'], 'reference': references[index]['id'],
                         'query_path': query['path'], 'reference_path': references[index]['path'],
                         'hamming': int(distances[index]), 'mean_absolute_difference': float(differences[position])})
    return exact, near


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fgsc23', default='data/manifests/fgsc23_research_all.jsonl')
    parser.add_argument('--fgsrcs-sources', default='data/manifests/fgsrcs_sources.jsonl')
    parser.add_argument('--fgsrcs-crops', default='data/manifests/fgsrcs_known_box_crops.jsonl')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    fgsc23_rows = read_jsonl(ROOT / args.fgsc23)
    source_rows = read_jsonl(ROOT / args.fgsrcs_sources)
    crop_rows = read_jsonl(ROOT / args.fgsrcs_crops)
    reference = load_group(fgsc23_rows, 'fgsc23', args.limit)
    sources = load_group(source_rows, 'fgsrcs_source', args.limit)
    crops = load_group(crop_rows, 'fgsrcs_crop', args.limit)
    source_exact, source_near = screen(sources, reference)
    crop_exact, crop_near = screen(crops, reference)
    result = {
        'fgsc23_images': len(reference), 'fgsrcs_source_images': len(sources), 'fgsrcs_crop_images': len(crops),
        'method': {'pixel_sha256': 'sha256(size bytes + decoded RGB bytes)',
                   'near_duplicate': f'63-bit DCT thumbnail Hamming <= {HAMMING_MAX} AND '
                                     f'32x32 RGB mean absolute difference <= {MAD_MAX}',
                   'crop_scope': 'FGSRCS known-box crops use 10% context, so crop-level matches approximate '
                                 'same-ship/same-scene overlap but are not proof of identical source scenes'},
        'exact_matches': {'sources': source_exact, 'crops': crop_exact},
        'near_duplicate_candidates': {'sources': source_near, 'crops': crop_near},
        'counts': {'exact_source_pairs': len(source_exact), 'exact_crop_pairs': len(crop_exact),
                   'near_source_pairs': len(source_near), 'near_crop_pairs': len(crop_near)},
        'limitations': ['Global thumbnails cannot detect crops, rotations, re-scaling or heavy reprocessing.',
                        'No port, capture-time or vessel identity is available for either release; a negative screen '
                        'does not establish scene or port independence.',
                        'Crop-level near-duplicates are expected when the same ship appears in both datasets and do '
                        'not by themselves prove shared source imagery.']}
    write_json(ROOT / 'reports/cross_dataset_overlap.json', result)
    print(json.dumps(result['counts'], indent=2))
    return result


if __name__ == '__main__':
    main()
