import importlib.util
import shutil
import subprocess
from pathlib import Path

def inspect(config):
    checks={}
    for name in ['numpy','scipy','sklearn','pymatgen','joblib','jsonschema','requests']:
        checks['python:'+name]=importlib.util.find_spec(name) is not None
    if not config['demo']:
        for kind in ['mgt','alignn']:
            py=config['surrogates'][kind+'_python']
            checks[kind+':python']=bool(shutil.which(py))
            if checks[kind+':python']:
                modules=['torch','torch_geometric','jarvis','pymatgen','yaml'] if kind=='mgt' else ['torch','dgl','jarvis','pymatgen']
                code='import importlib.util; assert all(importlib.util.find_spec(x) for x in '+repr(modules)+')'
                p=subprocess.run([py,'-c',code],capture_output=True,timeout=30)
                checks[kind+':dependencies']=p.returncode==0
        for key in ['mgt_bundle','alignn_root']:
            value=config['surrogates'][key];checks['asset:'+key]=bool(value and Path(value).is_dir())
        if config['method']=='cci':
            for key in ['calibration_dir']:
                value=config['cci'][key];checks['asset:'+key]=bool(value and Path(value).exists())
    if config['qe']['enabled']:
        for key in ['pw']:checks['qe:'+key]=bool(shutil.which(config['qe'][key]))
        for key in ['pseudo_dir','reference_file','smoke_evidence']:
            value=config['qe'][key];checks['qe:'+key]=bool(value and Path(value).exists())
    return dict(passed=all(checks.values()),checks=checks,demo=config['demo'],method=config['method'],
                note='Checks local dependencies and assets only; no API request is sent.')
