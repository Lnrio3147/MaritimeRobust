"""Replay the fixed validation-only calibration grid; never reads test samples."""
from PIL import Image, ImageDraw

from maritime_robust.common import ROOT, digest, read_jsonl, write_json
from maritime_robust.engine import ShipDataset

rows=read_jsonl(ROOT/'data/manifests/fgsc23_research_civilian.jsonl')
chosen=[next(r for r in rows if r['split']=='val' and r['label']==label) for label in range(3)]
canvas=Image.new('RGB',(6*180,3*190),'white');draw=ImageDraw.Draw(canvas)
for k,kind in enumerate(['haze','cloud','glare']):
    for severity in range(6):
        condition={'kind':kind if severity else 'clean','severity':severity,'domain':'matched','placement':'random'}
        image,_=ShipDataset([chosen[k]],condition=condition).image(0)
        image.thumbnail((175,165));canvas.paste(image,(severity*180,k*190))
        draw.text((severity*180,k*190+167),f'{kind} {severity}',fill='black')
canvas.save(ROOT/'reports/validation_calibration.png')
write_json(ROOT/'reports/validation_calibration.json',{
    'sample_ids':[r['sample_id'] for r in chosen], 'split':'val',
    'recipe_sha256':digest(ROOT/'configs/degradations.json'),
    'review':'Approximate fog/soft cloud/overexposure levels span mild to severe; no model test scores used to choose parameters.',
    'limitations':'Not calibrated against measured maritime weather or physical visibility; no real vessel segmentation.'})
