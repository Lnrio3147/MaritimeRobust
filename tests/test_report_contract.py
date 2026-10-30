"""Saved-prediction/report consistency on a tiny, explicitly synthetic fixture."""
import json

import pytest

from maritime_robust.common import write_json,write_jsonl
from maritime_robust.engine import metrics
from maritime_robust import report


def test_report_keeps_paired_gain_and_restoration_separate(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    (tmp_path/'reports').mkdir()
    def save(run,folder,clean_pred,haze_pred):
        summaries={}
        for name,pred in [('clean',clean_pred),('matched_haze_1',haze_pred)]:
            condition={'name':name,'kind':'clean' if name=='clean' else 'haze',
                       'domain':'matched','placement':'random','severity':0 if name=='clean' else 1}
            records=[]
            for i,(label,guess) in enumerate(zip([0,0,1,1],pred)):
                records.append({'sample_id':str(i),'source_group_id':'g'+str(label),'label':label,
                                'prediction':guess,'probabilities':[.95,.05] if guess==0 else [.05,.95],
                                'original_width':64,'original_height':128,
                                'degradation':{'saturated_fraction':0.,'center_proxy_affected_fraction':0.}})
            score=metrics([0,0,1,1],pred,2);score['condition']=condition
            summaries[name]=score
            write_jsonl(run/f'{folder}/{name}.jsonl',records)
        write_json(run/f'{folder}/summary.json',summaries)
    for augmented in [False,True]:
        run=tmp_path/'runs/fgsc23_research_civilian'/('resnet50_aug_s17' if augmented else 'resnet50_clean_s17')
        write_json(run/'complete.json',{'model':'resnet50','scope':'civilian','protocol':'research',
                   'augmented':augmented,'seed':17,'num_classes':2,'classes':{'0':'cargo','1':'tanker'},'training_wall_seconds':10.})
        save(run,'evaluation',[0,1,1,1] if augmented else [0,0,1,1],
             [0,0,1,1] if augmented else [0,0,0,0])
        if not augmented:save(run,'evaluation_dcp',[0,0,1,1],[0,0,1,0])
    result=report.summarize()
    saved=json.loads((tmp_path/'reports/results.json').read_text())
    assert result['runs']==3  # two classifiers and one restoration pipeline
    assert len(saved['augmentation_pairs'])==1
    assert saved['augmentation_pairs'][0]['augmentation_matched_f1_change']==pytest.approx(2/3)
    assert len(saved['dcp_comparisons'])==1
    assert (tmp_path/'reports/results.csv').exists()
    text=(tmp_path/'reports/RESULTS.md').read_text()
    assert 'DCP + 干净' in text and '百分点' in text
