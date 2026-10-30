#!/usr/bin/env python
"""Prepare the author's FGSRCS level-2 annotations as known-box classification crops.

No weather labels are inferred, and no recognition scores are produced here.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from maritime_robust.common import ROOT, digest, directory_bytes, stable_seed, write_json, write_jsonl
from maritime_robust.prepare import safe_extract

CLASSES = ['aircraft-carrier','amphibious-assault-ship','amphibious-transport-dock',
           'corvette','cruiser','destroyer','frigate','landing-ship','littoral-combat-ship',
           'mine-warfare-vessel','patrol-vessel','replenishment-oiler','submarine',
           'container-ship','cruise-ship','oil-tanker','sand-carrier']
CIVILIAN = {'container-ship','cruise-ship','oil-tanker','sand-carrier'}
STRICT_MAP = {'container-ship':{'fgsc23_original_label':14,'label_name':'container ship'},
              'oil-tanker':{'fgsc23_original_label':18,'label_name':'oil tanker'}}


def parse_annotation(path,allowed_labels=None):
    allowed_labels=set(CLASSES if allowed_labels is None else allowed_labels)
    records=[]
    for line_no,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(),1):
        tokens=line.split()
        if not tokens or line.startswith(('imagesource:','gsd:')):
            continue
        if len(tokens)!=10:
            raise ValueError(f'{path}:{line_no}: expected 8 coordinates, class and difficulty')
        polygon=np.asarray([float(v) for v in tokens[:8]]).reshape(4,2)
        if not np.isfinite(polygon).all() or tokens[8] not in allowed_labels:
            raise ValueError(f'{path}:{line_no}: invalid polygon or unknown class {tokens[8]}')
        records.append({'polygon':polygon.tolist(),'label_name':tokens[8],'difficulty':int(tokens[9]),'line':line_no})
    return records


def crop_box(polygon,size,context=.1):
    polygon=np.asarray(polygon,dtype=float)
    lo=polygon.min(0);hi=polygon.max(0)
    extent=hi-lo
    if (extent<=0).any():
        raise ValueError('Degenerate object polygon')
    lo=np.floor(lo-context*extent).astype(int)
    hi=np.ceil(hi+context*extent).astype(int)
    x0,y0=np.maximum(lo,0)
    x1,y1=np.minimum(hi,np.asarray(size))
    if x1<=x0 or y1<=y0:
        raise ValueError('Object outside source image')
    return [int(x0),int(y0),int(x1),int(y1)]


def prepare(archive,root):
    archive=Path(archive).resolve();root=Path(root).resolve()
    if not archive.is_file():
        raise FileNotFoundError('Completed author archive is required; partial sparse file is not admitted')
    marker=root/'EXTRACTED.json'
    archive_hash=digest(archive)
    if marker.exists():
        if json.loads(marker.read_text())['archive_sha256']!=archive_hash:
            raise ValueError('Archive changed after extraction')
    else:
        if directory_bytes(ROOT)+8*10**9>49*10**9:
            raise RuntimeError('Insufficient project disk allowance for source extraction')
        # ZIP extraction verifies per-file CRC on every read.
        safe_extract(archive,root,max_bytes=9*10**9)
        write_json(marker,{'archive_sha256':archive_hash,'crc_verified_by_extraction':True})
    originals=[]
    for publisher_split in ('train','val'):
        annotations=sorted((root/publisher_split/'annfiles_lv2').glob('*.txt'))
        images={p.stem:p for p in (root/publisher_split/'images').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}}
        if not annotations or not images:
            raise ValueError('Expected author train/val, images and annfiles_lv2 hierarchy')
        for annotation in annotations:
            if annotation.stem not in images:
                raise ValueError('Missing source image: '+str(annotation))
            image_path=images[annotation.stem]
            objects=parse_annotation(annotation)
            coarse_path=root/publisher_split/'annfiles_lv1'/annotation.name
            coarse=parse_annotation(coarse_path,allowed_labels=['military','civilian'])
            if len(coarse)!=len(objects):
                raise ValueError('Annotation level object counts disagree: '+str(annotation))
            for obj,parent in zip(objects,coarse):
                if not np.allclose(obj['polygon'],parent['polygon'],atol=.01):
                    raise ValueError('Annotation level object geometry disagrees: '+str(annotation))
                obj['civilian']=parent['label_name']=='civilian'
                if obj['civilian']!=(obj['label_name'] in CIVILIAN):
                    raise ValueError('Civilian mapping disagrees with author level-1 annotation')
            with Image.open(image_path) as f:
                rgb=f.convert('RGB')
                pixel_hash=hashlib.sha256(str(rgb.size).encode()+np.asarray(rgb).tobytes()).hexdigest()
                size=list(rgb.size)
            originals.append({'source_image_id':f'{publisher_split}/{image_path.stem}',
                              'source_group_id':'fgsrcs_source_'+pixel_hash[:24],
                              'path':str(image_path.relative_to(ROOT)),
                              'sha256':digest(image_path),'pixel_sha256':pixel_hash,
                              'annotation_path':str(annotation.relative_to(ROOT)),
                              'annotation_sha256':digest(annotation),'publisher_split':publisher_split,
                              'size':size,'objects':objects,'port_id':None,'capture_time':None})
    all_groups={r['source_group_id'] for r in originals}
    val_groups={r['source_group_id'] for r in originals if r['publisher_split']=='val'}
    eligible=sorted(all_groups-val_groups,key=lambda g:stable_seed(20261027,g))
    development=set(eligible[:min(len(eligible),round(.3*len(all_groups)))])
    crops=[];bad=[]
    destination=ROOT/'data/fgsrcs_crops';destination.mkdir(parents=True,exist_ok=True)
    for source in originals:
        with Image.open(ROOT/source['path']) as f:
            rgb=f.convert('RGB')
            for obj in source['objects']:
                sid='fgsrcs_'+source['source_image_id'].replace('/','_')+f"_o{obj['line']:03d}"
                try:
                    box=crop_box(obj['polygon'],rgb.size)
                except ValueError as e:
                    bad.append({'sample_id':sid,'reason':str(e)})
                    continue
                image_path=destination/(sid+'.png');mask_path=destination/(sid+'_region.png')
                if not image_path.exists():rgb.crop(box).save(image_path)
                if not mask_path.exists():
                    mask=Image.new('L',(box[2]-box[0],box[3]-box[1]),0)
                    points=[(x-box[0],y-box[1]) for x,y in obj['polygon']]
                    ImageDraw.Draw(mask).polygon(points,fill=255);mask.save(mask_path)
                strict=STRICT_MAP.get(obj['label_name'])
                crops.append({'sample_id':sid,'path':str(image_path.relative_to(ROOT)),
                              'source_dataset':'FGSRCS','source_url':'https://github.com/dwddw/FGSRCS',
                              'source_image_id':source['source_image_id'],'source_scene_id':None,
                              'source_group_id':source['source_group_id'],'source_group_basis':'exact decoded source image pixels',
                              'publisher_split':source['publisher_split'],'port_id':None,'vessel_id':None,
                              'split':'development' if source['source_group_id'] in development else 'sealed_test',
                              'label_name':obj['label_name'],'original_label':CLASSES.index(obj['label_name']),
                              'civilian':obj['civilian'],'difficulty':obj['difficulty'],
                              'crop_box_xyxy':box,'source_polygon':obj['polygon'],'context_fraction':.1,
                              'target_mask':str(mask_path.relative_to(ROOT)),'target_mask_kind':'annotated oriented box region, not ship segmentation',
                              'width':box[2]-box[0],'height':box[3]-box[1],
                              'sha256':digest(image_path),'degradation_type':None,'degradation_severity':None,
                              'strict_fgsc23_mapping':strict,'degradation_recipe':None})
        if len(crops)%1000<30 and directory_bytes(ROOT)>49*10**9:
            raise RuntimeError('Project disk allowance exhausted during crop preparation')
    write_jsonl(ROOT/'data/manifests/fgsrcs_sources.jsonl',originals)
    write_jsonl(ROOT/'data/manifests/fgsrcs_known_box_crops.jsonl',crops)
    dev={r['source_group_id'] for r in crops if r['split']=='development'}
    sealed={r['source_group_id'] for r in crops if r['split']=='sealed_test'}
    assert not dev&sealed
    report={'dataset':'FGSRCS','archive_sha256':archive_hash,'source_image_count':len(originals),
            'unique_source_pixel_groups':len(all_groups),'object_crops':len(crops),'invalid_objects':bad,
            'class_counts':dict(collections.Counter(r['label_name'] for r in crops)),
            'publisher_source_counts':dict(collections.Counter(r['publisher_split'] for r in originals)),
            'publisher_cross_split_exact_groups':len({r['source_group_id'] for r in originals if r['publisher_split']=='train'}&val_groups),
            'split_crop_counts':dict(collections.Counter(r['split'] for r in crops)),
            'split_group_counts':{'development':len(dev),'sealed_test':len(sealed)},
            'group_intersection':len(dev&sealed),'civilian_crops':sum(r['civilian'] for r in crops),
            'strict_mapping':STRICT_MAP,
            'strict_mapping_counts':{split:dict(collections.Counter(r['label_name'] for r in crops if r['split']==split and r['strict_fgsc23_mapping'])) for split in ['development','sealed_test']},
            'weather_annotations_available':False,'recognition_evaluation_performed':False,
            'readiness':'Crops and source-image grouping ready; strict independent weather validation not yet certified',
            'limitations':['Port, capture-time and vessel identities absent in released hierarchy; no port-independent claim.',
                           'Oriented box region is not a pixel-level ship segmentation mask.',
                           'Per-sample weather type and severity need annotation; not inferred from dataset-level descriptions.',
                           'Cross-dataset near-duplicate/source overlap with FGSC-23 remains to be audited before independent cross-dataset claims.',
                           'Only container ship and oil tanker have direct strict label mappings; cruise/sand carrier are not force-mapped.']}
    write_json(ROOT/'reports/fgsrcs_audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',default=str(ROOT/'data/raw/fgsrcs.archive'))
    parser.add_argument('--root',default=str(ROOT/'data/raw/fgsrcs'))
    args=parser.parse_args();prepare(args.archive,args.root)
