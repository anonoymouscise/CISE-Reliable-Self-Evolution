from pathlib import Path
import shutil
from .runtime import CONFIG,EXPERIMENT as R,save,load,sha

PROPS=['band_gap','formation_energy']
CALIBRATION_METADATA=['training_manifest.json','fixed_basis.json']+[f'{prop}_fixed_basis.json' for prop in PROPS]

def import_calibration(source,destination):
    source=Path(source).resolve();dest=Path(destination).resolve()
    if dest.exists():raise FileExistsError(dest)
    files=[source/'interval'/f'{p}{ext}' for p in PROPS for ext in ['_scale.joblib','_residuals.json']]
    files += [source/'interval'/f'{t}.json' for t in ['wbg','sse','pv']]
    for p in files:
        if not p.is_file():raise FileNotFoundError(p)

    for prop in PROPS:
        metadata=load(source/'interval'/f'{prop}_residuals.json')
        declared=[]
        if metadata.get('quantile_model'):
            declared.append((metadata['quantile_model'],metadata.get('quantile_model_sha256')))
        for side,name in metadata.get('quantile_models',{}).items():
            declared.append((name,metadata.get('quantile_models_sha256',{}).get(side)))
        for name,expected in declared:
            if not isinstance(name,str) or Path(name).name!=name or name in ['.','..']:
                raise ValueError('Quantile model must be a local filename')
            path=source/'interval'/name
            if not path.is_file():raise FileNotFoundError(path)
            if expected is not None and sha(path)!=expected:raise ValueError('Quantile model hash mismatch')
            if path not in files:files.append(path)
    files += [source/'interval'/name for name in CALIBRATION_METADATA if (source/'interval'/name).is_file()]
    dest.mkdir(parents=True)
    for p in files:shutil.copy2(p,dest/p.name)
    save(dest/'import_manifest.json',dict(source=str(source),files={p.name:sha(p) for p in files},
        note='Frozen calibration import; source data, model weights and API credentials are not bundled.'))

def demo_calibration():
    import numpy as np
    import joblib
    from sklearn.dummy import DummyRegressor
    from .shared.structure_features import structure_feature_values
    seeds=load(R/'prototypes/manifest.json');feature=structure_feature_values(seeds[0]['path'])
    safe={k:float(v) if np.isfinite(v) else None for k,v in feature.items()}
    folder=R/'interval';folder.mkdir()
    for prop in PROPS:
        model=DummyRegressor(strategy='constant',constant=float(np.log(.02)))
        model.fit(np.zeros((2,28)),np.zeros(2));joblib.dump(model,folder/f'{prop}_scale.joblib')
        rows=[dict(id=str(i),features=safe,score=1.,scale=.02) for i in range(40)]
        metadata=dict(rows=rows,dre_source=rows,demo=True)
        if CONFIG['cci'].get('quantile_offset'):
            offset=DummyRegressor(strategy='constant',constant=0.)
            offset.fit(np.zeros((2,28)),np.zeros(2));joblib.dump(offset,folder/f'{prop}_quantile.joblib')
            metadata.update(score_definition='normalized_absolute_residual_minus_frozen_quantile',
                quantile_model=f'{prop}_quantile.joblib',quantile_model_sha256=sha(folder/f'{prop}_quantile.joblib'))
        save(folder/f'{prop}_residuals.json',metadata)
    for task in ['wbg','sse','pv']:
        props={'band_gap':{'lower':{'q':1.}}}
        if task=='pv':props['band_gap']['upper']={'q':1.}
        props['formation_energy']={'upper':{'q':1.}}
        save(folder/f'{task}.json',dict(properties=props,demo=True))

def initialize():
    from .shared.structure_edits import prepare_library
    from .provenance import seal,verify_run
    if (R/'run_seal.json').exists():verify_run();return
    if R.exists() and any(R.iterdir()):raise RuntimeError('Output directory is nonempty without a valid run seal')
    if not CONFIG['demo']:
        s=CONFIG['surrogates']
        required=[(s['mgt_bundle'],'MGT bundle'),(s['alignn_root'],'ALIGNN root')]
        if CONFIG['method']=='cci':
            required += [(CONFIG['cci']['calibration_dir'],'CCI calibration directory')]
        if CONFIG['qe']['enabled']:required += [(CONFIG['qe'][k],k) for k in ['pseudo_dir','reference_file','smoke_evidence']]
        for p,label in required:
            if not p or not Path(p).exists():raise FileNotFoundError('Missing '+label)
        for rel in ['mgt_normalizer.json','mgt_validation.json','repos/MGT/config/finetune.yml',
                    'repos/MGT/ckpt/finetuned/gap pbe/gap pbe_checkpoint_best.pt',
                    'upstream_preprocessing/cif2dataset_finetune_megnet.py']:
            if not (Path(s['mgt_bundle'])/rel).is_file():raise FileNotFoundError('MGT bundle: '+rel)
        for name in ['jv_formation_energy_peratom_alignn']:
            if not (Path(s['alignn_root'])/'alignn'/(name+'.zip')).is_file():raise FileNotFoundError('ALIGNN: '+name)
    R.mkdir(parents=True,exist_ok=True)
    for name in ['audit','results','runs','qe']:(R/name).mkdir()
    prepare_library(R)
    if CONFIG['method']=='cci':
        if CONFIG['demo']:demo_calibration()
        else:
            folder=R/'interval';folder.mkdir()
            source=Path(CONFIG['cci']['calibration_dir'])
            for name in [f'{p}{s}' for p in PROPS for s in ['_scale.joblib','_residuals.json']]+[f'{t}.json' for t in CONFIG['tasks']]:
                shutil.copy2(source/name,folder/name)
            if CONFIG['cci'].get('quantile_offset'):
                for prop in PROPS:
                    meta=load(source/f'{prop}_residuals.json')
                    names=[]
                    if meta.get('quantile_model'):names.append(meta['quantile_model'])
                    names.extend(meta.get('quantile_models',{}).values())
                    if not names:raise ValueError('Adjusted score metadata lacks quantile model: '+prop)
                    for name in names:
                        if Path(name).name != name:raise ValueError('Quantile model must be a local filename')
                        shutil.copy2(source/name,folder/name)
            for name in CALIBRATION_METADATA:
                if (source/name).exists():shutil.copy2(source/name,folder/name)
    save(R/'config_locked.json',CONFIG)
    seal()
