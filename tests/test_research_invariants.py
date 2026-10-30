import hashlib
import json
import zipfile

import numpy as np
import pytest
from PIL import Image
import importlib.util
from pathlib import Path

from maritime_robust.degradations import conditions, corrupt, corruption_seed, dcp
from maritime_robust.engine import metrics, ShipDataset, resolve_augmentation
from maritime_robust.prepare import safe_extract
from maritime_robust.report import grouped_f1_interval


def sample():
    y,x=np.mgrid[:64,:96]
    return Image.fromarray(np.uint8(np.stack([x/96,y/64,(x+y)/160],-1)*220))


def test_combined_augmentation_mode_and_sampling():
    assert resolve_augmentation(False,None)=="clean"
    assert resolve_augmentation(True,None)=="single"
    assert resolve_augmentation(False,"combined")=="combined"
    with pytest.raises(ValueError):
        resolve_augmentation(True,"combined")
    ds=ShipDataset([],train=True,augmentation="combined")
    rng=np.random.default_rng(17)
    draws=[ds.augmentation_condition(rng) for _ in range(4000)]
    kinds={d["kind"] for d in draws}
    assert {"haze","cloud","glare","haze+glare","haze+cloud"} <= kinds
    assert all(1<=d["severity"]<=5 for d in draws)
    combined=[d for d in draws if "+" in d["kind"]]
    assert {d["severity"] for d in combined} == {3,5}
    # Among degraded draws the recipe is half single, half combined.
    assert abs(len(combined)/len(draws)-0.5)<0.03
    assert len({d["severity"] for d in draws if "+" not in d["kind"]})==5


def test_replay_and_train_test_seed_isolation():
    image=sample()
    train=corruption_seed("same-source","train","cloud",3)
    test=corruption_seed("same-source","test","cloud",3)
    assert train!=test
    a,meta=corrupt(image,"cloud",3,test)
    b,other=corrupt(image,"cloud",3,test)
    c,_=corrupt(image,"cloud",3,train)
    assert np.array_equal(a,b) and meta==other
    assert not np.array_equal(a,c)
    assert meta["target_affected_fraction"] is None


def test_all_conditions_are_finite_and_preserve_geometry():
    for condition in conditions():
        out,meta=corrupt(sample(),seed=17,**{k:v for k,v in condition.items() if k!="name"})
        assert out.size==sample().size
        assert np.isfinite(np.asarray(out)).all()
        json.dumps(meta,allow_nan=False)


def test_severity_and_spatial_diagnostic():
    image=sample()
    low,lm=corrupt(image,"haze",1,seed=17)
    high,hm=corrupt(image,"haze",5,seed=17)
    assert hm["transmission_mean"]<lm["transmission_mean"]
    _,center=corrupt(image,"glare",3,seed=17,placement="center")
    _,border=corrupt(image,"glare",3,seed=17,placement="border")
    assert center["center_proxy_affected_fraction"]>border["center_proxy_affected_fraction"]


def test_dcp_handles_uniform_and_extreme_images():
    for level in (0,128,255):
        out=dcp(Image.new("RGB",(48,32),(level,level,level)))
        assert out.size==(48,32) and np.isfinite(np.asarray(out)).all()


def test_fixed_label_metrics_and_cluster_unit():
    result=metrics([0,0,1,1],[0,1,1,1],3)
    assert result["accuracy"]==.75
    assert result["macro_f1"]==pytest.approx((2/3+.8)/3)
    rows=[{"label":0,"prediction":0,"source_group_id":"one"}]*10
    interval=grouped_f1_interval(rows,1,repeats=20)
    assert interval["group_count"]==1 and interval["low"]==1


def test_archive_traversal_rejected(tmp_path):
    archive=tmp_path/"bad.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("../escaped.txt","bad")
    with pytest.raises(ValueError):
        safe_extract(archive,tmp_path/"output")
    assert not (tmp_path/"escaped.txt").exists()


def test_real_data_crop_clips_context_without_distorting_coordinates(tmp_path):
    path=Path(__file__).resolve().parents[1]/"scripts/prepare_fgsrcs.py"
    spec=importlib.util.spec_from_file_location("prepare_fgsrcs",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    polygon=[[2,4],[42,4],[42,24],[2,24]]
    assert module.crop_box(polygon,(50,30))==[0,2,46,26]
    annotation=tmp_path/"scene.txt"
    annotation.write_text("imagesource:GoogleEarth\ngsd:0.5\n2 4 42 4 42 24 2 24 container-ship 0\n")
    objects=module.parse_annotation(annotation)
    assert len(objects)==1 and objects[0]["label_name"]=="container-ship"
    annotation.write_text("2 4 42 4 42 24 2 24 made-up-model 0\n")
    with pytest.raises(ValueError):module.parse_annotation(annotation)


def test_download_lock_prevents_duplicate_network_job(tmp_path):
    import fcntl
    path=Path(__file__).resolve().parents[1]/"scripts/download_public.py"
    spec=importlib.util.spec_from_file_location("download_public",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    destination=tmp_path/"weights.bin"
    with destination.with_suffix('.bin.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            module.download('https://example.invalid/never-accessed',destination)


def test_ema_conversion_preserves_tensors_and_rejects_unknown_pickle_globals(tmp_path):
    import io
    import pickle
    import torch
    path=Path(__file__).resolve().parents[1]/"scripts/gendeg_ema_checkpoint.py"
    spec=importlib.util.spec_from_file_location("gendeg_ema_checkpoint",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=tmp_path
    original={'model.layer.weight':torch.zeros(2,3),'model_ema.layerweight':torch.arange(6,dtype=torch.float32).reshape(2,3),
              'cond_stage_model.position_ids':torch.arange(5).reshape(1,5),'model_ema.decay':torch.tensor(.99)}
    buffer=io.BytesIO();torch.save({'state_dict':original},buffer)
    source=tmp_path/'original.pt';source.write_bytes(buffer.getvalue())
    output=tmp_path/'inference.pt'
    module.convert(source,output)
    loaded=torch.load(output,weights_only=True)['state_dict']
    assert torch.equal(loaded['model.layer.weight'],original['model_ema.layerweight'])
    assert torch.equal(loaded['cond_stage_model.position_ids'],original['cond_stage_model.position_ids'])
    assert not any(k.startswith('model_ema.') for k in loaded)
    with pytest.raises(ValueError):module.MetadataReader(io.BytesIO(pickle.dumps(Path))).load()
