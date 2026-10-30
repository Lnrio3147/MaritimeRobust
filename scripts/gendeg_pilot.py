#!/usr/bin/env python
"""Small, validation-only GenDeg compatibility pilot; never admits images to training."""
import contextlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from maritime_robust.common import Budget,digest,read_jsonl,seed_all,write_json


def main():
    # Offline mode must be set before transformers/huggingface_hub import.
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    import numpy as np
    import torch
    from PIL import Image,ImageOps
    from omegaconf import OmegaConf
    from transformers import CLIPTextConfig,CLIPTextModel,CLIPTokenizer
    import k_diffusion as K
    clip_dir=ROOT/'artifacts/generation_assets/clip_text'
    upstream=ROOT/'artifacts/third_party/GenDeg-main/GenDeg'
    for path in (upstream,upstream/'stable_diffusion',upstream/'NAFNet'):
        sys.path.insert(0,str(path))
    os.chdir(upstream)
    os.environ['WANDB_MODE']='disabled'
    original_pretrained=CLIPTextModel.from_pretrained
    original_tokenizer=CLIPTokenizer.from_pretrained
    # All learned text parameters must subsequently come from the author's checkpoint.
    # Use the pinned public config/tokenizer files; do not fetch the 1.7GB CLIP model.
    def initialize_config_only(version,*args,**kwargs):
        return CLIPTextModel(CLIPTextConfig.from_pretrained(str(clip_dir)))
    def local_tokenizer(version,*args,**kwargs):
        return original_tokenizer(str(clip_dir),*args,**kwargs)
    CLIPTextModel.from_pretrained=initialize_config_only
    CLIPTokenizer.from_pretrained=local_tokenizer
    from run_single import CFGDenoiser,NAFNet_Combine
    from ldm.util import instantiate_from_config
    config=OmegaConf.load(upstream/'configs/generate.yaml')
    # The downloaded subset is converted to a compact inference checkpoint with the
    # author's EMA mapping applied, so ema_scope() must not be used afterwards.
    converted=ROOT/'artifacts/generation_weights/gendeg_inference_ema.pt'
    checkpoint=converted if converted.exists() else ROOT/'artifacts/generation_weights/gendeg.ckpt'
    use_ema_scope=not converted.exists()
    scm_checkpoint=ROOT/'artifacts/generation_weights/nafnet_sd.ckpt'
    if not checkpoint.exists() or not scm_checkpoint.exists():
        raise FileNotFoundError('Both official generator and structure-correction weights are required')
    out=ROOT/'artifacts/gendeg_pilot';out.mkdir(exist_ok=True)
    with Budget('preflight') as budget:
        seed_all(17)
        torch.set_num_threads(4)
        state=torch.load(checkpoint,map_location='cpu',weights_only=not use_ema_scope)
        model=instantiate_from_config(config.model)
        missing,unexpected=model.load_state_dict(state['state_dict'],strict=False)
        # Converted weights already contain the EMA values under the regular names;
        # the unused model_ema copies are intentionally absent.
        if not use_ema_scope:
            missing=[key for key in missing if not key.startswith('model_ema.')]
        learned=set(dict(model.named_parameters()))
        if learned&set(missing):
            raise RuntimeError('Missing learned generator parameters: '+str(sorted(learned&set(missing))[:30]))
        del state
        model=model.cuda().eval().requires_grad_(False)
        scm=NAFNet_Combine(img_channel=6,width=64,enc_blk_nums=[2,2,4,8],middle_blk_num=12,dec_blk_nums=[2,2,2,2])
        scm.load_state_dict(torch.load(scm_checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        scm=scm.cuda().eval().requires_grad_(False)
        wrapped=K.external.CompVisDenoiser(model);guided=CFGDenoiser(wrapped)
        rows=read_jsonl(ROOT/'data/manifests/fgsc23_research_civilian.jsonl')
        chosen=[next(r for r in rows if r['split']=='val' and r['label']==i) for i in range(8)]
        records=[]
        scope=model.ema_scope() if use_ema_scope else contextlib.nullcontext()
        with torch.inference_mode(),torch.autocast('cuda'),scope:
            null=model.get_learned_conditioning([''])
            for row in chosen:
                with Image.open(ROOT/row['path']) as f:
                    image=ImageOps.pad(f.convert('RGB'),(512,512),method=Image.Resampling.LANCZOS,color=(124,116,104))
                image.save(out/(row['sample_id']+'_clean.png'))
                x=torch.from_numpy(np.array(image).astype(np.float32)/127.5-1).permute(2,0,1)[None].cuda()
                for mu,sigma in ((.10,.05),(.25,.10)):
                    seed_all(17)
                    stats=torch.zeros((1,2,129),device='cuda')
                    stats[0,0,min(127,int(mu/.7*128))]=1
                    stats[0,1,min(127,int(sigma/.4*128))]=1
                    null_stats=torch.zeros_like(stats);null_stats[:,:,-1]=1
                    text='An optical satellite image of a civilian ship at sea, in hazy conditions.'
                    cond={'c_crossattn':[model.cc_projection(torch.cat([model.get_learned_conditioning([text]),model.stats_proj(stats).transpose(1,2)],dim=-1))],
                          'c_concat':[model.encode_first_stage(x).mode()]}
                    uncond={'c_crossattn':[model.cc_projection(torch.cat([null,model.stats_proj(null_stats).transpose(1,2)],dim=-1))],
                            'c_concat':[torch.zeros_like(cond['c_concat'][0])]}
                    sigmas=wrapped.get_sigmas(30)
                    start=time.perf_counter()
                    z=torch.randn_like(cond['c_concat'][0])*sigmas[0]
                    z=K.sampling.sample_euler_ancestral(guided,z,sigmas,extra_args={'cond':cond,'uncond':uncond,'text_cfg_scale':7.5,'image_cfg_scale':1.5},disable=True)
                    generated=model.decode_first_stage(z).clamp(-1,1)
                    generated=scm(torch.cat([x,generated],dim=1)).clamp(-1,1)
                    array=((generated[0].permute(1,2,0).float().cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
                    target=out/f"{row['sample_id']}_mu{mu}_sigma{sigma}.png"
                    Image.fromarray(array).save(target)
                    records.append({'sample_id':row['sample_id'],'split':'val','mu':mu,'sigma':sigma,'seed':17,'steps':30,
                                    'path':str(target.relative_to(ROOT)),'sha256':digest(target),'seconds':time.perf_counter()-start,
                                    'quality_review':'pending','accepted_for_training':False})
                    write_json(out/'samples.json',records);budget.check(force=True)
        write_json(out/'provenance.json',{'generator_sha256':digest(checkpoint),'structure_correction_sha256':digest(scm_checkpoint),
                   'source_zip_sha256':digest(ROOT/'artifacts/sources/GenDeg_source.zip'),'missing_keys':missing,'unexpected_keys':unexpected,
                   'inference_checkpoint_kind':'converted EMA weights' if not use_ema_scope else 'author checkpoint with ema_scope',
                   'ema_scope_used':use_ema_scope,
                   'compatibility_changes':['new isolated Python/torch environment','config-only CLIP initialization followed by complete learned-parameter checkpoint audit','aspect-preserving input padding','explicit deterministic seed'],
                   'not_paper_reproduction':True,'all_generated_images_excluded_from_classifier_training':True})
    CLIPTextModel.from_pretrained=original_pretrained
    CLIPTokenizer.from_pretrained=original_tokenizer


if __name__=='__main__':
    main()
