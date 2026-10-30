"""Extract inference-equivalent EMA weights from the author's public Torch ZIP.

The author uses model.ema_scope(): UNet weights are replaced with model_ema.
Auxiliary modules keep their regular weights. We apply that exact key mapping,
preserving dtype and values, then require CRC verification of every selected ZIP
entry. Sparse source files must have all selected chunks downloaded first.
"""
import argparse
import collections
import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import torch

from maritime_robust.common import ROOT, digest, write_json


class MetadataReader(pickle.Unpickler):
    """Accept only the five metadata globals used by this official checkpoint."""
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
        if module=='torch' and name in ('FloatStorage','LongStorage','IntStorage'):return name
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):
            return lambda storage,offset,size,stride,*args:{'storage':storage,'offset':offset,'size':size,'stride':stride}
        raise ValueError('Unrecognized checkpoint metadata global: '+module+'.'+name)

    def persistent_load(self,pid):
        if len(pid)!=5 or pid[0]!='storage':raise ValueError('Unrecognized storage reference')
        return {'dtype':pid[1],'key':pid[2],'length':pid[4]}


def selected_metadata(archive):
    metadata_bytes=archive.read('archive/data.pkl')
    full=MetadataReader(io.BytesIO(metadata_bytes)).load()['state_dict']
    selected={};mapping={}
    for key,value in full.items():
        if key.startswith('model_ema.'):continue
        source_key='model_ema.'+key[len('model.'):].replace('.','') if key.startswith('model.') else key
        if source_key not in full:raise ValueError('Missing EMA counterpart: '+key)
        chosen=full[source_key]
        if value['size']!=chosen['size'] or value['storage']['dtype']!=chosen['storage']['dtype']:
            raise ValueError('EMA shape/dtype mismatch: '+key)
        selected[key]=chosen;mapping[key]=source_key
    return selected,mapping,metadata_bytes


def plan(source):
    chunk=32*2**20
    with zipfile.ZipFile(source) as archive:
        selected,mapping,data=selected_metadata(archive)
        keys={v['storage']['key'] for v in selected.values()}
        starts={0}
        total=Path(source).stat().st_size
        # Include full local record headers and a conservative 512-byte alignment allowance.
        for key in keys:
            info=archive.getinfo('archive/data/'+key)
            if info.compress_type!=zipfile.ZIP_STORED:raise ValueError('Expected uncompressed Torch storages')
            lo=info.header_offset;hi=min(total,lo+len(info.filename.encode())+512+info.file_size)
            starts.update(range(lo//chunk*chunk,(hi-1)//chunk*chunk+chunk,chunk))
        # ZIP64 directory may span two tail chunks. Preserve both and metadata.
        starts.update(range(max(0,total//chunk-1)*chunk,total,chunk))
        write_json(ROOT/'artifacts/generation_weights/gendeg_required_ranges.json',sorted(starts))
        write_json(ROOT/'reports/gendeg_ema_plan.json',{
            'selected_tensors':len(selected),'selected_storage_bytes':sum(archive.getinfo('archive/data/'+k).file_size for k in keys),
            'required_chunks':len(starts),'total_chunks':(total+chunk-1)//chunk,
            'metadata_sha256':hashlib.sha256(data).hexdigest(),
            'mapping':mapping,'equivalence':'Exact author EMA-to-UNet name mapping; no quantization; use_ema=False after conversion',
            'source_whole_file_sha256':None,'source_whole_file_complete':False})
    return len(starts)


def convert(source,output):
    dtype={'FloatStorage':torch.float32,'LongStorage':torch.int64,'IntStorage':torch.int32}
    tensors={};checksums={}
    with zipfile.ZipFile(source) as archive:
        selected,mapping,data=selected_metadata(archive)
        for name,spec in selected.items():
            member='archive/data/'+spec['storage']['key']
            # ZipFile.read verifies the authoritative per-entry CRC, including sparse holes.
            raw=archive.read(member)
            tensor=torch.frombuffer(bytearray(raw),dtype=dtype[spec['storage']['dtype']])
            if tensor.numel()!=spec['storage']['length']:raise ValueError('Storage size mismatch')
            tensors[name]=torch.as_strided(tensor,spec['size'],spec['stride'],spec['offset'])
            checksums[member]=hashlib.sha256(raw).hexdigest()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix('.tmp.pt')
    torch.save({'state_dict':tensors,'inference_ema_applied':True},tmp);tmp.replace(output)
    result={'path':str(output.relative_to(ROOT)),'sha256':digest(output),'tensor_count':len(tensors),
            'bytes':output.stat().st_size,'storage_sha256':checksums,'source_mapping':mapping,
            'all_selected_zip_crcs_verified':True,'source_whole_file_complete':False,
            'equivalence':'EMA UNet + regular VAE/CLIP/statistics projections, same float32 tensors as author ema_scope inference'}
    write_json(ROOT/'reports/gendeg_ema_conversion.json',result)
    return {'tensor_count':len(tensors),'bytes':output.stat().st_size,'sha256':result['sha256']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['plan','convert'])
    parser.add_argument('--source',default=str(ROOT/'artifacts/generation_weights/gendeg.ckpt.part'))
    parser.add_argument('--output',default=str(ROOT/'artifacts/generation_weights/gendeg_inference_ema.pt'))
    args=parser.parse_args()
    print(plan(args.source) if args.action=='plan' else convert(args.source,args.output))
