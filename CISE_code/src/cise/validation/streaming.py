import fcntl
import os
from pathlib import Path

from cise.runtime import CONFIG, ROOT as RUNS
from . import pipeline as qe


def extend_manifest(jobs, rows):

    from pymatgen.core import Structure
    for row in rows:
        if row.get('status') != 'evaluated' or not row.get('output_selected'): continue
        assert row.get('phase') == 'admission' and row.get('interval_pass') is True
        h = row['cif_sha256']
        if any(j['cif_sha256'] == h for j in jobs): continue
        assert qe.sha(row['cif_path']) == h
        label = 'cise_' + h[:12]
        job = dict(index=len(jobs)+1, label=label, jid=label, audit_cif=row['cif_path'],
                   cif_sha256=h, task_id=row['task'])
        elements = {str(s.specie) for s in Structure.from_file(row['cif_path'])}
        missing = sorted(el for el in elements if not (Path(CONFIG['qe']['pseudo_dir'])/f'{el.lower()}.pbesol.UPF').exists())
        if missing: job['validation_unavailable'] = dict(reason='MISSING_PSEUDOPOTENTIAL', elements=missing)
        jobs.append(job)
    return jobs


def changed_rows(seen):
    rows = []
    for task in CONFIG['tasks']:
        for p in sorted((RUNS/task/'batches').glob('*.json')):
            stat = p.stat(); signature = (stat.st_mtime_ns, stat.st_size)
            if seen.get(p) == signature: continue
            rows.extend(qe.load(p)['rows']); seen[p] = signature
    return rows


def main():
    for name in ['logs', 'runtime', 'runs']: (qe.ROOT/name).mkdir(parents=True, exist_ok=True)
    with (qe.ROOT/'dispatcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from cise.provenance import verify_run
        verify_run()
        assert qe.load(CONFIG['qe']['smoke_evidence'])['passed'], 'QETB_SMOKE_NOT_PASSED'
        manifest = qe.ROOT/'streaming_manifest.json'
        jobs = qe.load(manifest) if manifest.exists() else []
        seen = {}

        def refresh():


            statuses = [qe.load(RUNS/t/'status.json') for t in CONFIG['tasks']]
            watching = any(s['status'] != 'GENERATION_COMPLETED' for s in statuses)
            controller = qe.ROOT.parent/'controller_status.json'
            if controller.exists() and qe.load(controller).get('stage') == 'BLOCKED': watching = False
            extend_manifest(jobs, changed_rows(seen))
            qe.save(manifest, jobs)
            return list(jobs), watching

        qe.save(qe.ROOT/'streaming_launch.json', dict(at=qe.now(), pid=os.getpid(),
                policy='CCI-selected admissions only; no QE feedback to solver; shared dispatcher lock', slots=CONFIG['qe']['slots']))
        qe.event('STREAMING_STARTED', pid=os.getpid())
        qe.run_jobs([], 'streaming', refresh=refresh)
        qe.event('STREAMING_FINISHED', jobs=len(jobs))


if __name__ == '__main__': main()
