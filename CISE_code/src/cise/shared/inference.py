import hashlib, json, os, pickle, select, subprocess
from pathlib import Path
import numpy as np
from pymatgen.core import Structure
from cise.runtime import CONFIG,ROOT,CODE,TASKS,save,load,sha,now
from cise.shared.scoring import simple_score, _violations_for

DUPLICATE_POLICY = 'current_task_initial_seeds_and_adopted_candidates_v1'


def initial_seen(seeds):

    return {row['fingerprint'] for row in seeds}


class Server:
    def __init__(self,kind,run):
        self.kind=kind;self.demo=CONFIG['demo']
        if self.demo:return
        python=CONFIG['surrogates'][kind+'_python']
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE.parent)+os.pathsep+os.environ.get('PYTHONPATH',''))
        self.log=(run/f'{kind}_worker.log').open('a')
        self.proc=subprocess.Popen([python,'-m','cise.workers.models',kind],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.log,text=True,bufsize=1,env=env,cwd=run)
        try:
            if self.read().get('ready')!=kind:raise RuntimeError('SURROGATE_INIT_FAILED')
        except BaseException:
            self.proc.terminate();self.proc.wait(timeout=10);self.log.close();raise
    def read(self):
        ready,_,_=select.select([self.proc.stdout],[],[],240)
        if not ready:raise RuntimeError('SURROGATE_WORKER_TIMEOUT')
        line=self.proc.stdout.readline()
        if not line:raise RuntimeError('SURROGATE_WORKER_EXITED')
        return json.loads(line)
    def infer(self,cif,properties):
        if self.demo:
            return {p:{'band_gap':3.5,'formation_energy':-2.}[p] for p in properties}
        self.proc.stdin.write(json.dumps(dict(cif=str(cif),properties=properties))+'\n');self.proc.stdin.flush()
        result=self.read()
        if result.get('status')!='completed':raise RuntimeError('SURROGATE_INFERENCE_FAILED_'+result.get('error_type','unknown'))
        return result['values']
    def close(self):
        if self.demo:return
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=10)
        self.log.close()

def fingerprint(path):
    s = Structure.from_file(path)
    data = {'lattice': np.round(s.lattice.matrix, 6).tolist(),
            'sites': sorted((str(site.specie), *np.round(site.frac_coords % 1, 6).tolist()) for site in s)}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def score_values(task, values, species):
    predictions = {k: {'value': float(v)} for k, v in values.items()}
    candidate = {'species': species}
    violations = _violations_for(TASKS[task], values, candidate)
    return float(simple_score(predictions, None, TASKS[task], candidate)), violations


def evaluate(task, desc, servers, run, apply_interval=True):
    fp = fingerprint(desc['path'])
    cached = run / 'prediction_cache' / f'{fp}.json'
    if cached.exists():
        values = load(cached)['property_values']
    else:
        values = servers[0].infer(desc['path'], ['band_gap'])
        values.update(servers[1].infer(desc['path'], ['formation_energy']))
        save(cached, dict(cif_path=desc['path'], cif_sha256=sha(desc['path']), fingerprint=fp,
                         property_values=values, models={'band_gap': 'MGT', 'other': 'ALIGNN'}, at=now()))
    species = [site['element'] for site in desc['sites']]
    score, violations = score_values(task, values, species)
    assert not any(isinstance(v[-1], set) for v in violations), 'categorical validation must precede scoring'
    row = dict(candidate_id=desc['parent_id'], task=task, status='evaluated',
                cif_path=desc['path'], cif_sha256=sha(desc['path']), formula=desc['formula'],
                species=species, property_values=values, score=score, proxy_pass=not violations,
                failed_constraints=violations, fingerprint=fp, structure_metadata=desc)
    row.update(method_id=ROOT.name, raw_property_values=dict(values), raw_score=score, proxy_hit=not violations)
    if ROOT.name == 'B' and apply_interval:
        from cise.cci.interval_bridge import correct
        row = correct(row, score_values)
    row['output_selected'] = bool(row['proxy_pass'])
    return row


def feedback_record(row):
    return {k: row[k] for k in ['candidate_id', 'formula', 'property_values', 'score', 'proxy_pass', 'failed_constraints', 'conformal_bounds'] if k in row}


def persist_state(path, state):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as handle:
        pickle.dump(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)
