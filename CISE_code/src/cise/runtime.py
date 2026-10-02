import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from .config import DEFAULT, TASKS, load_config

CONFIG=load_config(os.environ['CISE_CONFIG']) if os.environ.get('CISE_CONFIG') else copy.deepcopy(DEFAULT)
EXPERIMENT=Path(CONFIG['output_dir']).resolve()
ROOT=EXPERIMENT/'runs'/('B' if CONFIG['method']=='cci' else 'A')
PROTOTYPES=EXPERIMENT/'prototypes'
CODE=Path(__file__).resolve().parent
PYTHON=CONFIG['surrogates']['alignn_python']
MGT_ROOT=Path(CONFIG['surrogates']['mgt_bundle'] or '.')

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)
