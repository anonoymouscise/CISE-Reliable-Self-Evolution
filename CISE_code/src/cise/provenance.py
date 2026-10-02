from pathlib import Path
from .runtime import CONFIG,EXPERIMENT as R,CODE,load,save,sha,now

def asset_files():
    paths=list((R/'interval').glob('*'))+list((R/'prototypes').glob('*'))
    if not CONFIG['demo']:
        s=CONFIG['surrogates'];m=Path(s['mgt_bundle']);a=Path(s['alignn_root'])
        paths += [m/'mgt_normalizer.json',m/'mgt_validation.json',m/'repos/MGT/config/finetune.yml',
                  m/'repos/MGT/ckpt/finetuned/gap pbe/gap pbe_checkpoint_best.pt',
                  m/'upstream_preprocessing/cif2dataset_finetune_megnet.py']
        paths += list((m/'repos/MGT/models').rglob('*.py'))+list((a/'alignn').rglob('*.py'))
        paths += [a/'alignn'/f'{n}.zip' for n in ['jv_formation_energy_peratom_alignn']]
    if CONFIG['qe']['enabled']:
        q=CONFIG['qe'];paths += [Path(q['reference_file']),Path(q['smoke_evidence'])]
        paths += list(Path(q['pseudo_dir']).glob('*.UPF'))
    return [p for p in paths if p.is_file()]

def seal():
    files=asset_files()
    save(R/'run_seal.json',dict(at=now(),configuration=CONFIG,
        code={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*.py')},
        assets={str(p.resolve()):sha(p) for p in files}))

def verify_run():
    record=load(R/'run_seal.json')
    if record['configuration']!=CONFIG:raise RuntimeError('CONFIG_CHANGED: resume requires the original configuration')
    for rel,h in record['code'].items():
        if sha(CODE/rel)!=h:raise RuntimeError('CODE_CHANGED: '+rel)
    for p,h in record['assets'].items():
        if not Path(p).is_file() or sha(p)!=h:raise RuntimeError('ASSET_CHANGED: '+p)
