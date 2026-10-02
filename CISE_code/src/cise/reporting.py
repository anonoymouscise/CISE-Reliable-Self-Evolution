import csv
from .runtime import CONFIG,EXPERIMENT as R,ROOT,TASKS,load,save,sha,now
from .shared.scoring import _violations_for

def task_label(task,row,evidence):
    vals={'band_gap':evidence.get('gap_eV'), 'formation_energy':evidence.get('formation_eV_atom')}
    from .shared.scoring import TASK_CONSTRAINTS
    spec=TASK_CONSTRAINTS[TASKS[task]];required=[p for p,_,_ in spec['numeric']]
    violations=_violations_for(TASKS[task],vals,dict(species=row['species']))

    if any(v[0]=='composition' or vals.get(v[0]) is not None for v in violations):return False
    if any(vals[p] is None for p in required):return None
    return True

def collect(allow_partial=False):
    members=[];metrics=[]
    for task in CONFIG['tasks']:
        run=ROOT/task;status=load(run/'status.json') if (run/'status.json').exists() else {'status':'NOT_STARTED'}
        if not allow_partial and status['status']!='GENERATION_COMPLETED':raise RuntimeError('Generation incomplete: '+task)
        its=[load(p) for p in sorted((run/'iterations').glob('*.json'))]
        if not allow_partial and len(its)!=CONFIG['iterations']:raise RuntimeError('Iteration denominator mismatch')
        if CONFIG['method']=='cci':
            if not allow_partial and status['completed_iterations']!=CONFIG['iterations']:raise RuntimeError('Incomplete CCI cycles')
            if not all(x['status']=='ITERATION_COMPLETED' and x['probe_n']>=CONFIG['cci']['probe_min_valid'] and len(x['candidates'])>=CONFIG['cci']['admission_min_valid'] for x in its):raise RuntimeError('Partial CCI iteration')
            batches=[load(p) for p in sorted((run/'batches').glob('*.json'))];records=[x for b in batches for x in b['rows']]
            calls=len(batches);probes=sum(x['status']=='probe_valid' for x in records)
        else:
            records=[x for it in its for x in it['candidates']]
            calls=sum(len(it['parse_attempts']) for it in its);probes=0
        evaluated=[x for x in records if x['status']=='evaluated']
        selected=[x for x in evaluated if x.get('proxy_pass')]
        for x in selected:
            if CONFIG['method']=='cci' and x.get('phase')!='admission':raise RuntimeError('Probe cannot be output')
            if sha(x['cif_path'])!=x['cif_sha256']:raise RuntimeError('Output CIF changed')
            members.append(x)
        (R/'results'/f'{task}_candidate_log.jsonl').write_text(''.join(__import__('json').dumps(x,allow_nan=False)+'\n' for x in records))
        metrics.append(dict(task=task,method=CONFIG['method'],demo=CONFIG['demo'],completed_iterations=len(its),
            target_iterations=CONFIG['iterations'],execution_status=status['status'],
            api_calls=calls,proposal_units=calls*2,valid_probes=probes,generated_candidates=len(evaluated),
            rejected_proposals=sum(x['status']=='rejected' for x in records),
            valid_unadmitted=sum(x['status']=='admission_valid' for x in records),
            raw_proxy_hits=sum(x.get('raw_proxy_pass',False) for x in evaluated),
            interval_pass=sum(x.get('interval_pass',False) for x in evaluated),
            abstentions=sum(x.get('abstained',False) for x in evaluated),
            outputs=len({x['cif_sha256'] for x in selected}),DFT_TP=0,DFT_FP=0,
            DFT_unresolved=len({x['cif_sha256'] for x in selected}),DFT_status='NOT_RUN'))
    members.sort(key=lambda x:(x['task'],x['candidate_id']))
    jobs={}
    for row in members:
        h=row['cif_sha256']
        if h not in jobs:
            label='cise_'+h[:12];jobs[h]=dict(index=len(jobs)+1,label=label,jid=label,audit_cif=row['cif_path'],cif_sha256=h,task_id=row['task'])
    save(R/'audit/output_memberships.json',members);save(R/'qe/final_manifest.json',list(jobs.values()))
    save(R/'audit/output_seal.json',dict(at=now(),memberships_sha256=sha(R/'audit/output_memberships.json'),manifest_sha256=sha(R/'qe/final_manifest.json')))
    write_metrics(metrics)
    return metrics

def write_metrics(metrics):
    save(R/'results/main_results.json',metrics)
    with (R/'results/main_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    rows=['# '+('DEMO — synthetic predictions, no scientific results' if CONFIG['demo'] else 'LLEMA / CISE results'),'',
          '| Task | Method | Generated candidates | Selected outputs | DFT TP | DFT FP | Unresolved / not run |',
          '|---|---|---:|---:|---:|---:|---:|']
    for m in metrics:rows.append('| '+' | '.join(str(m[k]) for k in ['task','method','generated_candidates','outputs','DFT_TP','DFT_FP','DFT_unresolved'])+' |')
    rows+=['','Generated candidates are valid evaluated admissions; probes, initialization seeds and rejected attempts are excluded.',
           'DFT TP/FP require measured evidence. No DFT is not a false positive. See CSV for calls, proposal units and completed iterations.']
    (R/'REPORT.md').write_text('\n'.join(rows)+'\n')

def audit():
    seal=load(R/'audit/output_seal.json')
    if sha(R/'audit/output_memberships.json')!=seal['memberships_sha256']:raise RuntimeError('Output membership drift')
    ev=load(R/'audit/final_qe_results.json')
    if ev['manifest_sha256']!=seal['manifest_sha256']:raise RuntimeError('QE manifest mismatch')
    byhash={e['cif_sha256']:e for e in ev['rows']};members=load(R/'audit/output_memberships.json');labels=[]
    for row in members:
        e=byhash[row['cif_sha256']]
        labels.append(dict(task=row['task'],candidate_id=row['candidate_id'],cif_sha256=row['cif_sha256'],
            full_qe_pass=task_label(row['task'],row,e),qe_status=e['status']))
    save(R/'audit/final_labels.json',labels)
    metrics=load(R/'results/main_results.json')
    for m in metrics:
        sub={x['cif_sha256']:x for x in labels if x['task']==m['task']}.values()
        tp=sum(x['full_qe_pass'] is True for x in sub);fp=sum(x['full_qe_pass'] is False for x in sub)
        m.update(DFT_TP=tp,DFT_FP=fp,DFT_unresolved=m['outputs']-tp-fp,DFT_status='AUDITED')
    write_metrics(metrics);return metrics
