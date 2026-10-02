from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from .runtime import save,sha
from .shared.structure_features import STRUCTURE_FEATURES,structure_feature_values

FLOORS={'band_gap':.02,'formation_energy':.01}

def fit_manifest(manifest,destination):
    manifest=Path(manifest).resolve();dest=Path(destination).resolve()
    if dest.exists():raise FileExistsError(dest)
    pools={p:defaultdict(list) for p in FLOORS};ids=set();structure_splits={}
    for line in manifest.read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line);prop=row['property'];split=row['split']
        if prop not in pools or split not in ['fit','dre_source','calibration','diagnostic']:raise ValueError('Invalid property/split')
        for name in ['id','group']:
            if not isinstance(row.get(name),str) or not row[name].strip():
                raise ValueError('Every row requires a nonempty string '+name)
        key=(prop,row['id'])
        if key in ids:raise ValueError('Duplicate property/id')
        ids.add(key)
        path=None
        if row.get('cif_path'):
            path=Path(row['cif_path']);path=path if path.is_absolute() else manifest.parent/path
            digest=sha(path)
            if row.get('cif_sha256') and digest!=row['cif_sha256']:raise ValueError('CIF hash mismatch')
            structure_key=(prop,digest)
            if structure_key in structure_splits and structure_splits[structure_key]!=split:
                raise ValueError('Repeated CIF across splits: '+prop)
            structure_splits[structure_key]=split
        if 'features' not in row:
            if path is None:raise ValueError('Every row requires features or a CIF path')
            row['features']=structure_feature_values(path)
        if not isinstance(row['features'],dict) or set(STRUCTURE_FEATURES)-set(row['features']):
            raise ValueError('Missing structure features')
        for name in STRUCTURE_FEATURES:
            value=row['features'][name]
            if value is None:continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or math.isinf(float(value)):
                raise ValueError('Structure features must be numeric or missing: '+name)
        if split!='dre_source':
            if not all(math.isfinite(float(row[k])) for k in ['prediction','truth']):raise ValueError('Nonfinite labels/predictions')
        pools[prop][split].append(row)
    for prop,pool in pools.items():
        if not pool['fit'] or not pool['dre_source'] or len(pool['calibration'])<19:raise ValueError('Each property needs fit/DRE and at least19 calibration records')
        for a,b in itertools.combinations(['fit','dre_source','calibration','diagnostic'],2):
            if {r['group'] for r in pool[a]}&{r['group'] for r in pool[b]}:raise ValueError('Composition leakage: '+prop+' '+a+'/'+b)
    dest.mkdir(parents=True);qs={};summary={};diagnostics={}
    def vector(row):return [row['features'][k] for k in STRUCTURE_FEATURES]+[float(row['prediction'])]
    def safe(row):return {k:float(v) if v is not None and math.isfinite(float(v)) else None for k,v in row['features'].items()}
    for prop,pool in pools.items():
        rows=pool['fit'];x=np.asarray([vector(r) for r in rows],dtype=float)
        y=np.log(np.maximum([abs(float(r['prediction'])-float(r['truth'])) for r in rows],FLOORS[prop]))
        model=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,min_samples_leaf=30,learning_rate=.05,l2_regularization=1,early_stopping=False).fit(x,y)
        joblib.dump(model,dest/f'{prop}_scale.joblib')
        c=pool['calibration'];sc=np.maximum(np.exp(model.predict(np.asarray([vector(r) for r in c],dtype=float))),FLOORS[prop])
        residual=np.abs([float(r['prediction'])-float(r['truth']) for r in c])/sc
        k=math.ceil((len(c)+1)*.95);q=float(np.sort(residual)[k-1]);qs[prop]=dict(q=q,n=len(c),k=k,alpha=.05)
        save(dest/f'{prop}_residuals.json',dict(rows=[dict(id=r['id'],features=safe(r),score=float(v),scale=float(s)) for r,v,s in zip(c,residual,sc)],
            dre_source=[dict(id=r['id'],features=safe(r)) for r in pool['dre_source']]))
        summary[prop]={role:len(rows) for role,rows in pool.items()}
        diagnostic=pool['diagnostic']
        if diagnostic:
            ds=np.maximum(np.exp(model.predict(np.asarray([vector(r) for r in diagnostic],dtype=float))),FLOORS[prop])
            errors=np.abs([float(r['prediction'])-float(r['truth']) for r in diagnostic])
            diagnostics[prop]=dict(n=len(diagnostic),unweighted_source_coverage=float(np.mean(errors<=q*ds)),
                interval_width_quantiles=list(map(float,np.quantile(2*q*ds,[0,.1,.5,.9,1]))))
    for task in ['wbg','sse','pv']:
        props={'band_gap':{'lower':qs['band_gap']}}
        if task=='pv':props['band_gap']['upper']=qs['band_gap']
        props['formation_energy']={'upper':qs['formation_energy']}
        save(dest/f'{task}.json',dict(properties=props,alpha_joint=.1,method='normalized_absolute_residual_scale_assets'))
    save(dest/'training_manifest.json',dict(source_sha256=sha(manifest),counts=summary,diagnostics=diagnostics,
        estimator='Fixed log-absolute-residual HistGradientBoostingRegressor; no QR offset trained by this command'))
