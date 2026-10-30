#!/usr/bin/env python
"""CPU synthesis cost measured on validation images, separate from classifier inference."""
import time
import numpy as np
from PIL import Image,ImageOps
from maritime_robust.common import ROOT,read_jsonl,write_json,environment
from maritime_robust.degradations import corrupt,dcp,corruption_seed

rows=[r for r in read_jsonl(ROOT/'data/manifests/fgsc23_research_civilian.jsonl') if r['split']=='val'][:64]
images=[]
for r in rows:
    with Image.open(ROOT/r['path']) as f:images.append(ImageOps.contain(f.convert('RGB'),(224,224),Image.Resampling.BICUBIC))
results=[]
for kind in ('haze','cloud','glare','haze+glare','haze+cloud','DCP'):
    times=[]
    for image,row in zip(images,rows):
        start=time.perf_counter()
        if kind=='DCP':dcp(image)
        else:corrupt(image,kind,3,corruption_seed(row['sample_id'],'val',kind,3))
        times.append((time.perf_counter()-start)*1000)
    results.append({'method':kind,'severity':3 if kind!='DCP' else None,'n':len(times),
                    'cpu_ms_median':float(np.median(times)),'cpu_ms_p95':float(np.percentile(times,95)),
                    'total_cpu_wall_seconds':sum(times)/1000})
write_json(ROOT/'reports/degradation_cost.json',{'split':'val','includes_image_decode':False,'environment':environment(),'results':results})
