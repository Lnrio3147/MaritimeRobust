#!/usr/bin/env python
from PIL import Image,ImageDraw

from maritime_robust.common import ROOT,read_jsonl,write_json,digest
from maritime_robust.engine import ShipDataset

rows=read_jsonl(ROOT/'data/manifests/fgsc23_research_civilian.jsonl')
chosen=[next(r for r in rows if r['split']=='val' and r['label']==label) for label in range(3)]
canvas=Image.new('RGB',(6*180,3*190),'white');draw=ImageDraw.Draw(canvas);records=[]
for k,kind in enumerate(['haze','cloud','glare']):
    for severity in range(6):
        ds=ShipDataset([chosen[k]],condition={'kind':kind if severity else 'clean','severity':severity,'domain':'matched','placement':'random'})
        im,meta=ds.image(0);im.thumbnail((175,165));canvas.paste(im,(severity*180,k*190))
        draw.text((severity*180,k*190+167),f'{kind} {severity}',fill='black')
        records.append({'sample_id':chosen[k]['sample_id'],'degradation':meta})
canvas.save(ROOT/'reports/validation_calibration.png')
write_json(ROOT/'reports/validation_calibration.json',{'split':'val','recipe_sha256':digest(ROOT/'configs/degradations.json'),'samples':records})
