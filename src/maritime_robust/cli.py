import argparse
import json
from pathlib import Path

from .common import ROOT, environment, write_json


def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("environment")
    p=commands.add_parser("prepare")
    p.add_argument("--root",required=True)
    p=commands.add_parser("extract")
    p.add_argument("archive");p.add_argument("destination")
    p=commands.add_parser('degrade')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--kind',required=True,choices=['haze','cloud','glare','haze+glare','haze+cloud'])
    p.add_argument('--severity',type=int,required=True,choices=range(1,6));p.add_argument('--seed',type=int,default=17)
    p.add_argument('--domain',choices=['matched','shifted'],default='matched')
    p.add_argument('--placement',choices=['random','center','border'],default='random')
    p.add_argument('--mask');p.add_argument('--native-resolution',action='store_true')
    p=commands.add_parser("train")
    p.add_argument("--manifest",required=True);p.add_argument("--model",required=True,choices=["resnet50","convnext_tiny","swin_t"])
    p.add_argument("--augmented",action="store_true");p.add_argument("--augmentation",choices=["clean","single","combined"])
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--epochs",type=int);p.add_argument("--batch-size",type=int);p.add_argument("--workers",type=int)
    p.add_argument("--limit",type=int)
    for name in ("evaluate","profile"):
        p=commands.add_parser(name);p.add_argument("--run",required=True);p.add_argument("--restore",action="store_true")
        if name=="evaluate":
            p.add_argument("--split",default="test",choices=["val","test"]);p.add_argument("--validation-only",action="store_true")
    commands.add_parser("report")
    args=parser.parse_args()
    kwargs=vars(args);command=kwargs.pop("command")
    if command=="environment":
        result=environment();write_json(ROOT/"artifacts/environment.json",result)
    elif command=="prepare":
        from .prepare import audit_fgsc23
        result=audit_fgsc23(**kwargs)
    elif command=="extract":
        from .prepare import safe_extract
        result=safe_extract(**kwargs)
    elif command=="report":
        from .report import summarize
        result=summarize()
    elif command=='degrade':
        import numpy as np
        from PIL import Image,ImageOps
        from .common import digest
        from .degradations import corrupt
        source=Path(kwargs.pop('input'));destination=Path(kwargs.pop('output'))
        mask_path=kwargs.pop('mask');native=kwargs.pop('native_resolution')
        if source.resolve()==destination.resolve():raise ValueError('Output must differ from original image')
        with Image.open(source) as f:image=f.convert('RGB')
        original_size=image.size
        if not native:image=ImageOps.contain(image,(224,224),Image.Resampling.BICUBIC)
        mask=None
        if mask_path:
            with Image.open(mask_path) as f:mask=np.asarray(f.convert('L').resize(image.size,Image.Resampling.NEAREST))>0
        output,result=corrupt(image,target_mask=mask,**kwargs)
        if not native:output=ImageOps.pad(output,(224,224),method=Image.Resampling.BICUBIC,color=(124,116,104))
        destination.parent.mkdir(parents=True,exist_ok=True);output.save(destination)
        result.update(input_sha256=digest(source),original_size=list(original_size),content_size=list(image.size),output_size=list(output.size),
                      recipe_sha256=digest(ROOT/'configs/degradations.json'))
        write_json(destination.with_suffix(destination.suffix+'.json'),result)
    else:
        from . import engine
        if command=="train":
            kwargs["model_name"]=kwargs.pop("model")
        result=getattr(engine,{"train":"train","evaluate":"evaluate","profile":"profile"}[command])(**kwargs)
    if command == "evaluate":
        print(json.dumps({"run":kwargs["run"],"split":kwargs["split"],"conditions":len(result),
                          "summary":"saved in run directory"},ensure_ascii=False))
    else:
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
