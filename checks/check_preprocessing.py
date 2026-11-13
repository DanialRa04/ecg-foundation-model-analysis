"""Synthetic tests for signal recipes, dataset boundaries, and training scalers."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import wfdb
from common.artifacts import file_hash
from common.signals import bandpass, center_crop, fit_scalar, resample, zscore
from integrations.datasets import ptbxl
from integrations.models import REGISTRY, get_integration
from scripts.preprocess import run

def fixture_dataset(root):
    data = root / 'dataset'
    (data/'records100').mkdir(parents=True)
    rows=[]
    for i,(fold,offset) in enumerate([(1,1.),(2,3.),(9,100.),(10,200.)],1):
        t=np.arange(1000)/100
        signal=np.stack([offset+np.sin(2*np.pi*(1+j/10)*t) for j in range(12)],axis=1)
        wfdb.wrsamp(f'record{i}',fs=100,units=['mV']*12,sig_name=ptbxl.CANONICAL_LEADS,p_signal=signal,fmt=['16']*12,write_dir=str(data/'records100'))
        rows.append({'ecg_id':str(i),'patient_id':str(i),'strat_fold':str(fold),'filename_lr':f'records100/record{i}','filename_hr':f'records500/record{i}'})
    with (data/'ptbxl_database.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return data

def run_checks():
    root=Path(__file__).resolve().parents[1]
    passed=[]
    x=np.stack([np.sin(2*np.pi*(2+i)*np.arange(5000)/500) for i in range(12)]).astype(np.float32)
    for name in REGISTRY:
        module=get_integration(name);r=module.RECIPE
        values=np.stack([x[i % 12] for i in range(len(r['leads']))])
        if r['source_hz']==100:values=resample(values,500,100)
        out=module.preprocess(values,r['source_hz'],{'mean':.1,'scale':2.})
        assert list(out.shape)==r['shape'] and out.dtype==np.float32 and np.isfinite(out).all(),name
    passed.append('all model input recipes')
    moments=fit_scalar([np.array([[1.,2.]]),np.array([[3.,4.]])])
    assert np.isclose(moments['mean'],2.5) and np.isclose(moments['scale'],np.std([1.,2.,3.,4.]))
    assert np.all(zscore(np.ones((2,50)),per_lead=True)==0)
    passed.append('streaming population moments and constant leads')
    assert np.array_equal(center_crop(np.arange(10)[None,:],4),[[3,4,5,6]])
    assert np.array_equal(center_crop(np.array([[1,2]]),5),[[0,1,2,0,0]])
    assert resample(x,500,240,'resampy').shape==(12,2400)
    assert abs(bandpass(np.ones((2,5000)),500,.67,40).max())<1e-8
    passed.append('crop, padding, anti-aliased resampling and zero-phase filtering')
    (root/'outputs').mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='preprocessing_checks_',dir=root/'outputs') as directory:
        base=Path(directory).resolve()
        assert (root/'outputs').resolve() in base.parents
        dataset=fixture_dataset(base)
        paths=base/'paths.json';paths.write_text(json.dumps({'paths':{'dataset_root':str(dataset)}}))
        config=root/'configs/experiment.json'
        args=argparse.Namespace(config=config,paths=paths,models=['ECGFM-KED'],output=base/'run',limit=4,ids=None,scalers=None)
        run(args)
        saved=json.loads((base/'run/scalers.json').read_text())['ECGFM-KED']
        rows=ptbxl.metadata(dataset,{'train':[1,2,3,4,5,6,7,8],'validation':[9],'test':[10]})
        train=[ptbxl.read_record(dataset,row,100,ptbxl.CANONICAL_LEADS) for row in rows if row['split']=='train']
        assert saved['fit_records']==2 and np.isclose(saved['mean'],np.concatenate(train,axis=1).mean(),atol=1e-5)
        assert saved['mean']<10
        passed.append('training-only fitting with extreme held-out values')
        original_hash=file_hash(dataset/'ptbxl_database.csv')
        try:run(args)
        except FileExistsError:pass
        else:raise AssertionError('Existing output was overwritten')
        assert original_hash==file_hash(dataset/'ptbxl_database.csv')
        passed.append('input preservation and output collision rejection')
        malicious=dict(rows[0],filename_lr='../outside')
        try:ptbxl.record_path(dataset,malicious,100)
        except ValueError:pass
        else:raise AssertionError('Dataset path escaped its root')
        with (dataset/'ptbxl_database.csv').open(newline='') as stream:raw=list(csv.DictReader(stream))
        raw[-1]['patient_id']=raw[0]['patient_id']
        with (dataset/'ptbxl_database.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
        try:ptbxl.metadata(dataset,{'train':[1,2,3,4,5,6,7,8],'validation':[9],'test':[10]})
        except ValueError:pass
        else:raise AssertionError('Patient crossed partitions')
        passed.append('path confinement and patient separation')
    return {'status':'PASS','checks':passed}

if __name__=='__main__':print(json.dumps(run_checks(),indent=2))
