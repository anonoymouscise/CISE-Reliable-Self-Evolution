import argparse
import fcntl
import hashlib
import json
import os
import pickle
import random
import time
from pathlib import Path
import numpy as np
from cise.runtime import EXPERIMENT as R, ROOT, PROTOTYPES, CONFIG, save, load, sha, now
from cise.shared.inference import Server, evaluate, score_values, fingerprint, feedback_record, persist_state
from cise.shared.inference import DUPLICATE_POLICY, initial_seen
from cise.shared.structure_edits import parent_descriptor, apply_edit
from cise.shared.prompting import build_prompt, validate_response, RESPONSE_SCHEMA
from cise.providers import request, MODEL
from cise.shared.islands import ExperienceBuffer
from .interval_bridge import correct, annotate_prompt, calibrations
from .online_dre import freeze_hash
from .context_calibration import ContextCalibration
from cise.shared.structure_features import structure_feature_values

INITIAL_SEEDS = {'wbg':'rocksalt_MgO','sse':'antifluorite_Li2O',
                        'pv':'diamond_Si'}

def features(path):
    f=structure_feature_values(path)
    if f['structure_parse_success_flag']!=1:raise ValueError('STRUCTURE_FEATURE_PARSE_FAILED')
    return {k:float(v) if np.isfinite(v) else None for k,v in f.items()}


def status(run,state,value):
    batches=[load(p) for p in sorted((run/'batches').glob('*.json'))]
    rows=[r for b in batches for r in b['rows']]
    save(run/'status.json',dict(at=now(),pid=os.getpid(),status=value,task=run.name,
        api_calls=len(batches),proposal_units=2*len(batches),iteration=state.get('iteration',0),
        completed_iterations=state.get('completed_iterations',0),target_iterations=CONFIG['iterations'],
        probe_calls=sum(b['phase']=='probe' for b in batches),admission_calls=sum(b['phase']=='admission' for b in batches),
        valid_probes=sum(r['phase']=='probe' and r['status']=='probe_valid' for r in rows),
        evaluated=sum(r['status']=='evaluated' for r in rows),
        rejected_proposals=sum(r['status']=='rejected' for r in rows),
        valid_unadmitted=sum(r['status']=='admission_valid' for r in rows),
        raw_proxy_hits=sum(r.get('proxy_hit',False) for r in rows if r['status']=='evaluated'),
        abstentions=sum(r.get('abstained',False) for r in rows),
        unique_outputs=len({r['cif_sha256'] for r in rows if r.get('output_selected')}),
        incomplete_probe_n=len(state.get('pending',{}).get('probes',[])),
        context_attempt=state.get('pending',{}).get('attempt',0)))


def context(task,state,iteration):

    buffer=state['buffer'];pm=buffer.get_prompt(iteration=iteration-1);island=pm.island_id
    memories=buffer.get_unique_examples_for_evolution(island,max_examples=20)
    seeds=state['seeds']
    pool=memories['success'][:2]+memories['failure'][:1]
    pool += [seeds[(iteration-1+j)%len(seeds)] for j in range(len(seeds))]
    parents=[];used=set()
    for row in pool:
        if row.get('abstained',False):continue
        if row['candidate_id'] in used:continue
        used.add(row['candidate_id'])
        d=parent_descriptor(row['cif_path'],task=task)
        d.update(property_values=row['property_values'],proxy_pass=row['proxy_pass'],
                 failed_constraints=row['failed_constraints'],conformal_bounds=row['conformal_bounds'])
        parents.append(d)
        if len(parents)==5:break
    feedback=dict(success=[feedback_record(r) for r in memories['success'][:3]],
        failure=[feedback_record(r) for r in memories['failure'][:3]],rejected_edits=state.get('recent_rejections',[])[-4:])
    prompt=annotate_prompt(build_prompt(task,parents,feedback,iteration),task)
    if len(prompt.encode())>60000:raise RuntimeError('PROMPT_SIZE_GUARD')
    return dict(task=task,iteration=iteration,island=island,parents=parents,prompt=prompt,
        messages=[{'role':'user','content':prompt}],attempt=0,
        buffer_sha256=hashlib.sha256(pickle.dumps(buffer)).hexdigest(),
        live_seen_sha256=freeze_hash(sorted(state['seen'])),
        duplicate_policy=DUPLICATE_POLICY,
        run_seal_sha256=sha(R/'run_seal.json'),
        generation=dict(model=MODEL,candidates_per_call=2,repair_prompt=True,
            configured_parent_pool_size=5,effective_parent_pool_size=len(parents),
            target_population='geometry-valid generator before current-task main duplicate and output selection',
            final_slots=2,adoption='first full valid batch',
            max_attempts=CONFIG['cci']['max_attempts'],
            max_probe_batches=((CONFIG['cci']['probe_min_valid']+1)//2)*CONFIG['cci']['max_attempts'],
            additional_generation='until full valid batch or attempt limit'))


def batch(task,state,ctx,phase,servers,run,weights=None):

    if freeze_hash(ctx)!=state['pending']['frozen_hash']:raise RuntimeError('CONTEXT_DRIFT')
    if hashlib.sha256(pickle.dumps(state['buffer'])).hexdigest()!=ctx['buffer_sha256']:raise RuntimeError('MEMORY_DRIFT')
    call=state['next_call'];dest=run/'batches'/f'{call:04d}.json'
    if dest.exists():return load(dest)['rows']
    for p in ctx['parents']:
        if sha(p['path'])!=p['sha256']:raise RuntimeError('PARENT_CHANGED')
    byid={p['parent_id']:p for p in ctx['parents']}
    save(run/'api'/f'{call:04d}_context.json',dict(**ctx,phase=phase,frozen_hash=state['pending']['frozen_hash']))

    py_rng=random.getstate();np_rng=np.random.get_state()
    schema=RESPONSE_SCHEMA
    try:raw,api=request(task,call,0,ctx['messages'],schema)
    finally:random.setstate(py_rng);np.random.set_state(np_rng)
    try:proposals=validate_response(json.loads(raw),schema);parse_error=None
    except Exception as e:proposals=[];parse_error=type(e).__name__
    records=[];local_seen=set()
    for ordinal in range(2):
        cid=f'cise_{task}_i{ctx["iteration"]:04d}_{phase}_{call:06d}_c{ordinal}'
        row=dict(candidate_id=cid,task=task,phase=phase,iteration=ctx['iteration'],call_sequence=call,
            candidate_ordinal=ordinal,call_id=api['call_id'],status='rejected',output_selected=False,
            duplicate_policy=ctx.get('duplicate_policy',DUPLICATE_POLICY),
            frozen_hash=state['pending']['frozen_hash'],context_attempt=ctx['attempt'])
        try:
            if ordinal>=len(proposals):raise ValueError('parse_or_missing_candidate:'+str(parse_error))
            proposal=proposals[ordinal];row['proposal']=proposal
            parent=byid.get(proposal['parent_id'])
            if parent is None:raise ValueError('unknown_parent_id')
            path=run/'cifs'/(cid+'.cif')
            if path.exists() and not Path(str(path)+'.meta.json').exists():
                path.rename(path.with_suffix('.cif.incomplete.'+str(time.time_ns())))
            if path.exists():
                meta=load(str(path)+'.meta.json')
                if meta['parent_sha256']!=parent['sha256'] or meta['substitutions']!=proposal['substitutions'] or meta['lattice_scale']!=proposal['lattice_scale']:
                    raise RuntimeError('CACHED_STRUCTURE_LINEAGE_MISMATCH')
                desc=parent_descriptor(path,task=task)
            else:desc=apply_edit(parent['path'],proposal,task,path)
            fp=fingerprint(path)
            row.update(cif_path=str(path),cif_sha256=sha(path),fingerprint=fp,features=features(path),formula=desc['formula'],
                       parent_id=proposal['parent_id'])

            if phase=='admission':
                if fp in state['seen'] or fp in local_seen:raise ValueError('duplicate_geometry_species_within_task')
                local_seen.add(fp);row['status']='admission_valid'
            else:row['status']='probe_valid'
        except (ValueError,AssertionError) as e:row['error']=str(e)[:300]
        records.append(row)
    save(dest,dict(task=task,phase=phase,call=call,iteration=ctx['iteration'],
        frozen_hash=state['pending']['frozen_hash'],rows=records,response_path=str(run/'api'/f'{call:04d}_0.json')))
    return records


def consume(state,rows):

    state['next_call']+=1


def make_calibration(task,pending):
    return ContextCalibration(task,tuple(calibrations()[task]['properties']),pending['probes'],
                              pending['context']['iteration'],pending['frozen_hash'])


def evaluate_admissions(task,rows,calibration,servers,run):
    output=[]
    for row in rows:
        if row['status']=='evaluated':output.append(row);continue
        desc=parent_descriptor(row['cif_path'],task=task)
        pred=evaluate(task,desc,servers,run,apply_interval=False)
        pred=correct(pred,score_values,calibration)
        record=dict(row);record.update(pred)

        record.update(candidate_id=row['candidate_id'],phase='admission',status='evaluated',
                      output_selected=bool(pred['proxy_pass']),frozen_hash=row['frozen_hash'])

        diagnostic={}
        for prop in calibration.properties:
            r=calibration.cutoff_without_dre(prop,row['features'],pred['raw_property_values'][prop])
            diagnostic[prop]=dict(cutoff=r.cutoff if np.isfinite(r.cutoff) else None,status=r.status,
                                 diagnostics=r.diagnostics)
        record['without_dre']=diagnostic
        path=run/'batches'/f'{row["call_sequence"]:04d}.json';b=load(path)
        b['rows']=[record if x['candidate_id']==row['candidate_id'] else x for x in b['rows']]
        save(path,b);output.append(record)
    return output


def next_attempt(pending,rows,run):
    old=pending['context'];ctx=dict(old);attempt=pending['attempt']+1
    errors=[f'candidate_{r["candidate_ordinal"]}:'+r['error'] for r in rows if r.get('error')]


    ctx.update(attempt=attempt,messages=[{'role':'user','content':old['prompt']+'\nThe previous response was rejected: '+
        json.dumps(errors)+'. Produce a fresh response matching the required JSON schema. Return two distinct valid structure edits.'}])
    pending['attempt']=attempt;pending['context']=ctx;pending['frozen_hash']=freeze_hash(ctx);pending['probes']=[]
    pending['attempt_records'].append(dict(attempt=attempt-1,valid_slots=sum(r['status']=='admission_valid' for r in rows),errors=errors))


def evolve_iterations(task,state,checkpoint,servers,run):
    target=CONFIG['iterations']
    max_attempts=CONFIG['cci']['max_attempts']
    max_probe_batches=((CONFIG['cci']['probe_min_valid']+1)//2)*max_attempts
    while state['completed_iterations']<target:
        if 'pending' not in state:
            state['iteration']=state['completed_iterations']+1
            ctx=context(task,state,state['iteration'])
            state['pending']=dict(context=ctx,frozen_hash=freeze_hash(ctx),probes=[],attempt=0,attempt_records=[],
                                  probe_batches=0,admission_batches=0)
            persist_state(checkpoint,state)
        p=state['pending']
        if p.get('admission_batches',0)>=max_attempts:
            raise RuntimeError('CCI_ADMISSION_ATTEMPTS_EXHAUSTED: no completed iteration')
        while len(p['probes'])<CONFIG['cci']['probe_min_valid']:
            if p.get('probe_batches',0)>=max_probe_batches:
                raise RuntimeError('CCI_PROBE_ATTEMPTS_EXHAUSTED: insufficient valid probes')
            rows=batch(task,state,p['context'],'probe',servers,run)
            p['probes'].extend(r for r in rows if r['status']=='probe_valid')
            p['probe_batches']=p.get('probe_batches',0)+1
            consume(state,rows);persist_state(checkpoint,state);status(run,state,'PROBING')
        calibration=make_calibration(task,p)
        save(run/'dre'/f'{state["iteration"]:04d}_{p["attempt"]:04d}.json',dict(at=now(),iteration=state['iteration'],
            frozen_hash=p['frozen_hash'],probe_ids=[r['candidate_id'] for r in p['probes']],models=calibration.logs,
            interpretation='Estimated ratio as Gibbs basis; closed-loop exploratory, no automatic coverage certificate'))
        rows=batch(task,state,p['context'],'admission',servers,run)
        valid=[r for r in rows if r['status'] in ['admission_valid','evaluated']]
        consume(state,rows)
        p['admission_batches']=p.get('admission_batches',0)+1
        if len(valid)<2:
            if p['admission_batches']>=max_attempts:
                p['attempt_records'].append(dict(attempt=p['attempt'],valid_slots=len(valid),
                    errors=[r.get('error','invalid_candidate') for r in rows if r['status']=='rejected']))
                persist_state(checkpoint,state)
                raise RuntimeError('CCI_ADMISSION_ATTEMPTS_EXHAUSTED: no completed iteration')
            next_attempt(p,rows,run);persist_state(checkpoint,state);status(run,state,'REPAIR_NEW_CONTEXT');continue
        records=evaluate_admissions(task,valid,calibration,servers,run)
        save(run/'iterations'/f'{state["iteration"]:04d}.json',dict(task=task,iteration=state['iteration'],
            frozen_hash=p['frozen_hash'],candidates=records,probe_n=len(p['probes']),
            attempts=p['attempt_records'],status='ITERATION_COMPLETED',island_id=p['context']['island']))
        for row in records:
            if not row.get('abstained',False):
                state['buffer'].register(p['context']['island'],row,{'total':row['score']},iteration=state['iteration']-1)

            state['seen'].add(row['fingerprint'])
        state['recent_rejections']=p['attempt_records'][-4:]
        state['completed_iterations']=state['iteration'];del state['pending'];persist_state(checkpoint,state)
        status(run,state,'ITERATION_COMPLETED')


def run(task):
    from cise.provenance import verify_run
    run=ROOT/task
    for folder in ['iterations','cifs','prediction_cache','api','batches','dre']:(run/folder).mkdir(parents=True,exist_ok=True)
    lock=(run/'generation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_run()
    servers=[];state={'iteration':0};checkpoint=run/'online_state.pkl'
    try:
        if checkpoint.exists():
            with checkpoint.open('rb') as f:state=pickle.load(f)
            if state.get('completed_iterations')==CONFIG['iterations']:
                status(run,state,'GENERATION_COMPLETED');return
        save(run/'status.json',dict(at=now(),status='MODEL_STARTING',task=task,pid=os.getpid()))
        servers.append(Server('mgt',run));servers.append(Server('alignn',run))
        if not checkpoint.exists():
            seeds=[evaluate(task,parent_descriptor(p['path'],task=task),servers,run)
                   for p in load(PROTOTYPES/'manifest.json') if task in p['eligible_tasks']]
            for row in seeds:row.update(output_selected=False,coverage_role='initialization_excluded')
            save(run/'seed_predictions.json',seeds)
            usable_seeds=[row for row in seeds if not row.get('abstained',False)]
            if not usable_seeds:raise RuntimeError('CCI_NO_USABLE_SEEDS: all initial intervals abstained')
            buffer=ExperienceBuffer(num_islands=5,functions_per_prompt=4,temp_init=1.,temp_period=10,
                reset_period_seconds=CONFIG['archive_reset_seconds'],max_items_per_island=5000,top_k_success=10,bottom_k_failure=10)
            base=next((row for row in usable_seeds if row['candidate_id']==INITIAL_SEEDS[task]),usable_seeds[0])
            for island in range(5):buffer.register(island,base,{'total':base['score']},iteration=0)
            state=dict(iteration=0,completed_iterations=0,next_call=1,buffer=buffer,seeds=usable_seeds,
                seen=initial_seen(seeds),duplicate_policy=DUPLICATE_POLICY,recent_rejections=[])
            persist_state(checkpoint,state)
        evolve_iterations(task,state,checkpoint,servers,run)
        status(run,state,'GENERATION_COMPLETED')
    except Exception as e:


        if checkpoint.exists():
            with checkpoint.open('rb') as f:state=pickle.load(f)
        save(run/'failure.json',dict(at=now(),error_type=type(e).__name__,error=str(e)[:400]))
        status(run,state,'GENERATION_BLOCKED')
        raise
    finally:
        for server in servers:server.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True,choices=['wbg','sse','pv'])
    args=ap.parse_args();run(args.task)
