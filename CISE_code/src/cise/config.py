import copy
import json
import math
from pathlib import Path
import sys

TASKS = {'wbg':'Stable Wide-Bandgap Semiconductors','sse':'Solid-State Electrolytes',
         'pv':'Photovoltaic Absorbers'}
DEFAULT = dict(method='llema', output_dir='runs/example', tasks=list(TASKS), iterations=100,
    demo=False, parallel_tasks=1, archive_reset_seconds=43200,
    llema=dict(max_attempts=3), cci=dict(probe_min_valid=8, admission_min_valid=2, max_attempts=3,
        engine='gibbs', alpha_total=.10, quantile_offset=False, calibration_dir=None),
    llm=dict(model='openai/gpt-5-mini', api_key_env='OPENROUTER_API_KEY', max_output_tokens=4096),
    surrogates=dict(mgt_python=sys.executable, alignn_python=sys.executable,
        mgt_bundle=None, alignn_root=None, device='cuda:0'),
    qe=dict(enabled=False, python=sys.executable, protocol='qetb_pbesol',
        pseudo_dir=None, reference_file=None, smoke_evidence=None, pw='pw.x',
        slots=2, threads=4, timeout=3600, min_disk_gib=80, min_memory_gib=64))


def load_config(path):
    path=Path(path).resolve();given=json.loads(path.read_text());cfg=copy.deepcopy(DEFAULT)
    if not isinstance(given,dict):raise ValueError('Configuration must be an object')
    unknown=set(given)-set(cfg)
    if unknown:raise ValueError('Unknown configuration fields: '+','.join(sorted(unknown)))
    for key,value in given.items():
        if isinstance(cfg[key],dict):
            if not isinstance(value,dict) or set(value)-set(cfg[key]):raise ValueError('Invalid configuration section: '+key)
            cfg[key].update(value)
        else:cfg[key]=value
    if cfg['method'] not in ['llema','cci']:raise ValueError('method must be llema or cci')
    if type(cfg['demo']) is not bool or type(cfg['qe']['enabled']) is not bool:raise ValueError('demo and qe.enabled must be booleans')
    if type(cfg['llm']['max_output_tokens']) is not int or cfg['llm']['max_output_tokens']<1:raise ValueError('Invalid max_output_tokens')
    for key in ['model','api_key_env']:
        if not isinstance(cfg['llm'][key],str) or not cfg['llm'][key].strip():raise ValueError('Invalid llm.'+key)
    if type(cfg['archive_reset_seconds']) not in [int,float] or not math.isfinite(cfg['archive_reset_seconds']) or cfg['archive_reset_seconds']<=0:
        raise ValueError('archive_reset_seconds must be finite and positive')
    if not isinstance(cfg['tasks'],list) or not cfg['tasks'] or any(not isinstance(t,str) for t in cfg['tasks']):raise ValueError('Invalid tasks')
    if len(set(cfg['tasks']))!=len(cfg['tasks']) or set(cfg['tasks'])-set(TASKS):raise ValueError('Invalid tasks')
    for key in ['iterations','parallel_tasks']:
        if type(cfg[key]) is not int or cfg[key]<1:raise ValueError(key+' must be positive')
    for section,keys in [('cci',['probe_min_valid','admission_min_valid','max_attempts']),('llema',['max_attempts'])]:
        for key in keys:
            if type(cfg[section][key]) is not int or cfg[section][key]<1:raise ValueError(key+' must be positive')
    if cfg['cci']['engine']!='gibbs':raise ValueError('Supported CISE engine: gibbs')
    if type(cfg['cci']['quantile_offset']) is not bool:raise ValueError('cci.quantile_offset must be boolean')
    if type(cfg['cci']['alpha_total']) not in [int,float] or not 0<cfg['cci']['alpha_total']<1:raise ValueError('Invalid joint alpha')
    if cfg['method']=='cci' and cfg['cci']['admission_min_valid']!=2:raise ValueError('CISE uses two final candidate slots')
    if cfg['qe']['protocol']!='qetb_pbesol':raise ValueError('Supported portable QE protocol: qetb_pbesol')
    for key in ['slots','threads','timeout']:
        if type(cfg['qe'][key]) is not int or cfg['qe'][key]<1:raise ValueError('Invalid QE '+key)
    if cfg['demo'] and cfg['qe']['enabled']:raise ValueError('Demo predictions must never be sent to real QE')
    def resolve(value):
        if value is None:return None
        if not isinstance(value,str) or not value.strip():raise ValueError('Asset paths must be nonempty strings')
        p=Path(value).expanduser()
        return str((path.parent/p).resolve()) if not p.is_absolute() else str(p.resolve())
    cfg['output_dir']=resolve(cfg['output_dir'])
    if cfg['output_dir'] is None:raise ValueError('output_dir is required')
    for section,keys in [('cci',['calibration_dir']),('surrogates',['mgt_bundle','alignn_root']),
                          ('qe',['pseudo_dir','reference_file','smoke_evidence'])]:
        for key in keys:cfg[section][key]=resolve(cfg[section][key])
    return cfg
