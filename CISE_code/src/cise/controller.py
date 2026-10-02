from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import os
import subprocess
import sys
from .runtime import CONFIG,EXPERIMENT as R,CODE,save,now

def stage(value,**kw):save(R/'controller_status.json',dict(at=now(),pid=os.getpid(),stage=value,**kw))

def run_task(task):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONPATH=str(CODE.parent)+os.pathsep+os.environ.get('PYTHONPATH',''))
    with (R/f'{task}.log').open('a') as f:
        p=subprocess.Popen([sys.executable,'-m','cise.task_worker',task],env=env,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        save(R/f'{task}_launch.json',dict(pid=p.pid,at=now()));rc=p.wait()
    if rc:raise RuntimeError(task+' stopped; inspect task status and '+task+'.log')

def run():
    from .provenance import verify_run
    verify_run()
    with (R/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            stage('GENERATION_RUNNING')
            with ThreadPoolExecutor(max_workers=CONFIG['parallel_tasks']) as pool:
                fs=[pool.submit(run_task,t) for t in CONFIG['tasks']]
                for f in as_completed(fs):f.result()
            from .reporting import collect,audit
            collect();stage('GENERATION_COMPLETED')
            if CONFIG['qe']['enabled']:
                validate();metrics=audit()
                stage('COMPLETED',scientific_completion='ALL_OUTPUT_LABELS_RESOLVED' if all(m['DFT_unresolved']==0 for m in metrics) else 'UNRESOLVED_LABELS_RETAINED')
        except Exception as exc:
            from .reporting import collect
            stage('BLOCKED',error_type=type(exc).__name__,error=str(exc))
            collect(allow_partial=True)
            raise

def validate():
    if CONFIG['demo']:raise RuntimeError('Demo is not scientific input to QE')
    if not CONFIG['qe']['enabled']:raise RuntimeError('QE is disabled in this configuration')
    from .runtime import load
    from pathlib import Path
    q=CONFIG['qe'];unavailable={}
    from pymatgen.core import Structure
    for job in load(R/'qe/final_manifest.json'):
        missing=[]
        for site in Structure.from_file(job['audit_cif']):
            el=site.specie.symbol
            if not (Path(q['pseudo_dir'])/f'{el.lower()}.pbesol.UPF').exists():missing.append(el)
        if missing:unavailable[job['label']]=dict(reason='MISSING_PSEUDOPOTENTIAL',elements=sorted(set(missing)))
    save(R/'qe/validation_unavailable.json',unavailable)
    for name in ['logs','runtime','runs']:(R/'qe'/name).mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(CODE.parent)+os.pathsep+os.environ.get('PYTHONPATH',''))
    with (R/'qe/controller.log').open('a') as f:
        p=subprocess.Popen([q['python'],'-m','cise.validation.pipeline','final'],cwd=R/'qe',env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        stage('OUTPUT_QE_RUNNING',child_pid=p.pid)
        if p.wait():raise RuntimeError('QE pipeline failed; see qe/controller.log')
