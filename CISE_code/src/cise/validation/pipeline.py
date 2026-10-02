import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone

from cise.runtime import CONFIG, EXPERIMENT, CODE
ROOT = EXPERIMENT/'qe'
PY = CONFIG['qe']['python']
RY = 13.605693122994

def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n'); tmp.replace(p)
def event(kind, **kw):
    with (ROOT/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=now(), event=kind, **kw))+'\n')
def summary(j, phase):


    if phase == 'final' and (ROOT/'streaming_manifest.json').exists():
        for prior in load(ROOT/'streaming_manifest.json'):
            if prior['cif_sha256'] != j['cif_sha256']: continue
            path = summary(prior, 'streaming')
            if path.exists(): return path
    return ROOT/'runs'/phase/'profiles'/f"{j['index']:04d}_{j['label']}"/'summary.json'
def resources():
    mem = next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    external = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            if (p/'comm').read_text().strip() != 'pw.x': continue
            cwd = os.readlink(p/'cwd')
            if not cwd.startswith(str(ROOT)+'/'): external.append(int(p.name))
        except OSError: pass
    return dict(free_disk_GiB=shutil.disk_usage(ROOT).free/1024**3,
                available_memory_GiB=mem/1024**2, external_qe=external)

def run_jobs(jobs, phase, refresh=None):
    if not jobs and refresh is None:
        save(ROOT/f'{phase}_status.json',dict(at=now(),phase=phase,total=0,terminal=0,completed=0,failed=0,
            active=[],pending=0,status='EXECUTION_TERMINAL',reason='NO_SELECTED_OUTPUTS'))
        return
    unavailable=load(ROOT/'validation_unavailable.json') if (ROOT/'validation_unavailable.json').exists() else {}
    for j in jobs:
        if j['label'] in unavailable and not summary(j,phase).exists():
            save(summary(j,phase),dict(status='VALIDATION_UNAVAILABLE',scf_converged=False,reason=unavailable[j['label']]))
    pending = [j for j in jobs if not summary(j, phase).exists()]
    active = {}
    if not pending and refresh is None:
        rows=[load(summary(j,phase)) for j in jobs]
        save(ROOT/f'{phase}_status.json',dict(at=now(),phase=phase,total=len(jobs),terminal=len(rows),
            completed=sum(x.get('status')=='completed' for x in rows),failed=sum(x.get('status')!='completed' for x in rows),
            active=[],pending=0,status='EXECUTION_TERMINAL',reason='ALL_JOBS_ALREADY_TERMINAL'))
        return
    watching = refresh is not None
    known = {j['index'] for j in jobs}
    while pending or active or watching:
        if watching:
            discovered, watching = refresh()
            for j in discovered:
                if j['index'] in known: continue
                known.add(j['index']); jobs.append(j)
                if not summary(j, phase).exists(): pending.append(j)
        for pid, (proc, job, log) in list(active.items()):
            rc = proc.poll()
            if rc is None: continue
            log.close(); del active[pid]
            event('QE_JOB_EXIT', phase=phase, label=job['label'], pid=pid, returncode=rc,
                  summary_exists=summary(job, phase).exists())
            if not summary(job, phase).exists():
                save(summary(job, phase), dict(status='worker_failed', returncode=rc, scf_converged=False))
        res = resources()
        while pending and len(active) < CONFIG['qe']['slots'] and res['free_disk_GiB'] >= CONFIG['qe']['min_disk_gib'] and res['available_memory_GiB'] >= CONFIG['qe']['min_memory_gib'] and len(res['external_qe']) < 8:
            j = pending.pop(0)
            assert sha(j['audit_cif']) == j['cif_sha256']
            if j.get('validation_unavailable'):
                save(summary(j, phase), dict(status='VALIDATION_UNAVAILABLE', scf_converged=False,
                     reason=j['validation_unavailable']))
                continue
            row = {k:j[k] for k in ['label','jid','audit_cif','task_id']}
            runtime = ROOT/'runtime'/phase/j['label']; runtime.mkdir(parents=True, exist_ok=True)
            cmd = [PY,'-m','cise.validation.worker','--single','--code','qe','--output-dir',str(ROOT/'runs'/phase),
                   '--row-index',str(j['index']),'--row-json',json.dumps(row),'--manifest-cif-column','audit_cif',
                   '--pseudo-dir',CONFIG['qe']['pseudo_dir'],'--scf-length','20','--band-length','30',
                   '--timeout','900' if phase=='smoke' else str(CONFIG['qe']['timeout']),'--omp-threads',str(CONFIG['qe']['threads']),'--qe-version-timeout','30']
            env = dict(os.environ, PATH=str(Path(PY).parent)+':'+os.environ['PATH'],
                       PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(CONFIG['qe']['threads']),PYTHONPATH=str(CODE.parent)+os.pathsep+os.environ.get('PYTHONPATH',''),CISE_PW=CONFIG['qe']['pw'],OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='',OMPI_MCA_opal_cuda_support='false',OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,tcp')
            log = (ROOT/'logs'/f"{phase}_{j['label']}.log").open('a')
            proc = subprocess.Popen(cmd,cwd=runtime,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active[proc.pid]=(proc,j,log)
            event('QE_JOB_LAUNCHED',phase=phase,label=j['label'],pid=proc.pid)
        rows = [load(summary(j,phase)) for j in jobs if summary(j,phase).exists()]
        save(ROOT/f'{phase}_status.json',dict(at=now(),phase=phase,total=len(jobs),terminal=len(rows),
             completed=sum(x.get('status')=='completed' for x in rows),
             failed=sum(x.get('status')!='completed' for x in rows),active=[dict(pid=p,label=x[1]['label']) for p,x in active.items()],
             pending=len(pending),resources=res,watching=watching,
             status='RUNNING' if pending or active or watching else 'EXECUTION_TERMINAL'))
        if pending or active or watching: time.sleep(5)

def read_result(j, phase):
    import numpy as np
    from pymatgen.core import Structure
    from .gap_parser import parse_gap, NUM
    p=summary(j,phase); rec=load(p)
    out=dict(jid=j['jid'],label=j['label'],cif_sha256=j['cif_sha256'],status=rec.get('status'),
             summary_path=str(p),summary_sha256=sha(p),gap_eV=None,formation_eV_atom=None)
    if not rec.get('scf_converged'): return out
    assert rec['cif_sha256']==sha(j['audit_cif'])==j['cif_sha256']
    struct=Structure.from_file(j['audit_cif']); params=rec['parameters']
    assert params['xc']=='PBEsol' and params['ecutwfc_Ry']==45 and params['ecutrho_Ry']==250
    assert params['occupations']=='Gaussian 0.01 Ry' and not params['magnetism']['spin_polarized']
    scf=None
    for name in ['scf_retry2','scf_retry','scf']:
        path=p.parent/(name+'.out')
        if path.exists():
            txt=path.read_text()
            if 'JOB DONE.' in txt and 'convergence has been achieved' in txt:
                matches=re.findall(r'!\s+total energy\s*=\s*('+NUM+r')\s*Ry',txt)
                if matches: scf=path; energy=float(matches[-1].replace('D','E')); break
    assert scf is not None
    assert abs(energy*RY/len(struct)-rec['energy_eV_atom'])<1e-5
    inp=scf.with_suffix('.in').read_text()
    lattice=np.array([[float(x) for x in ln.split()] for ln in inp.split('CELL_PARAMETERS angstrom\n')[1].splitlines()[:3]])
    apos=inp.split('ATOMIC_POSITIONS crystal\n')[1].splitlines()[:len(struct)]
    assert np.allclose(lattice,struct.lattice.matrix,atol=1e-9)
    assert [ln.split()[0] for ln in apos]==[str(s.specie) for s in struct]
    assert np.allclose([[float(v) for v in ln.split()[1:4]] for ln in apos],struct.frac_coords,atol=1e-9)
    for el,h in rec['pseudopotential_sha256'].items():
        assert sha(rec['pseudopotentials'][el])==h==sha(Path(CONFIG['qe']['pseudo_dir'])/f'{el.lower()}.pbesol.UPF')
    refs=load(Path(CONFIG['qe']['reference_file']))
    missing=sorted({str(s.specie) for s in struct}-set(refs))
    if missing:out.update(formation_status='MISSING_ELEMENTAL_REFERENCE',missing_reference_elements=missing)
    else:
        reference=sum(refs[str(s.specie)]['total_elemental_Ry'] for s in struct)
        out.update(formation_eV_atom=(energy-reference)*RY/len(struct),formation_status='COMPLETED')
    out.update(geometry_verified=True,energy_verified=True,
               scf_source=str(scf),scf_sha256=sha(scf),parameters=params,
               gap_status='UNRESOLVED',reference_sha256=sha(Path(CONFIG['qe']['reference_file'])))
    try:
        gap=parse_gap(p.parent/'nscf_fixed.out',params['valence_electrons'],1,params['nbands'])
        out.update(gap_eV=gap['band_gap_eV'],gap_status='STRICT_COMPLETED',gap_evidence=gap)
    except (AssertionError,FileNotFoundError,TypeError,IndexError) as e:
        out['gap_reason']=str(e) or type(e).__name__
    if phase=='smoke':
        out.update(qetb_gap_eV=j['qetb_band_gap_eV'],qetb_formation_eV_atom=j['qetb_formation_energy_eV_atom'])
        out['formation_abs_error']=abs(out['formation_eV_atom']-out['qetb_formation_eV_atom'])
        out['gap_abs_error']=None if out['gap_eV'] is None else abs(out['gap_eV']-out['qetb_gap_eV'])
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['smoke','final']);args=ap.parse_args()
    lock=(ROOT/'dispatcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    from cise.provenance import verify_run
    verify_run()
    manifest=ROOT/f'{args.phase}_manifest.json';jobs=load(manifest)
    if args.phase=='final':
        assert load(CONFIG['qe']['smoke_evidence'])['passed'], 'QETB_SMOKE_NOT_PASSED'
        for task in CONFIG['tasks']:
            assert load(ROOT.parent/'runs'/('B' if CONFIG['method']=='cci' else 'A')/task/'status.json')['status'] == 'GENERATION_COMPLETED'
    run_jobs(jobs,args.phase)
    rows=[]
    for j in jobs:
        try:rows.append(read_result(j,args.phase))
        except Exception as exc:
            rows.append(dict(jid=j['jid'],label=j['label'],cif_sha256=j['cif_sha256'],status='AUDIT_UNRESOLVED',
                gap_eV=None,formation_eV_atom=None,audit_error=type(exc).__name__+':'+str(exc)[:300]))
    if args.phase=='smoke':
        gaps=[r['gap_abs_error'] for r in rows if r.get('gap_abs_error') is not None]
        forms=[r['formation_abs_error'] for r in rows if r.get('formation_abs_error') is not None]
        passed=len(forms)==len(jobs) and len(gaps)>=2 and max(forms)<=.2 and max(gaps)<=.05
        save(ROOT/'smoke_results.json',dict(at=now(),passed=passed,rows=rows,n_gap=len(gaps),n_form=len(forms),
             gap_MAE=sum(gaps)/len(gaps) if gaps else None,formation_MAE=sum(forms)/len(forms) if forms else None,
             scope='Source-reconstruction smoke validation on the configured reference structures; no adaptive certification.'))
        print(json.dumps(dict(smoke_passed=passed,n_gap=len(gaps),n_form=len(forms))),flush=True)
    else:
        save(ROOT.parent/'audit/final_qe_results.json',dict(at=now(),rows=rows,manifest_sha256=sha(manifest)))

if __name__=='__main__': main()
