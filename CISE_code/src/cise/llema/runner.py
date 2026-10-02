import fcntl, json, os, pickle, time
from pathlib import Path
from cise.runtime import ROOT, PROTOTYPES, CONFIG, save, load, now
from cise.shared.inference import Server, evaluate, fingerprint, feedback_record, persist_state
from cise.shared.inference import DUPLICATE_POLICY, initial_seen
from cise.shared.islands import ExperienceBuffer
from cise.shared.structure_edits import parent_descriptor, apply_edit
from cise.shared.prompting import build_prompt, validate_response, RESPONSE_SCHEMA
from cise.providers import request
from cise.provenance import verify_run

def status(run, state, status_name):
    records = [c for f in sorted((run / 'iterations').glob('*.json')) for c in load(f)['candidates']]
    save(run / 'status.json', dict(at=now(), pid=os.getpid(), status=status_name, iteration=state['iteration'],
         target_iterations=CONFIG["iterations"], requested_slots=state['iteration'] * 2,
         evaluated=sum(r['status'] == 'evaluated' for r in records),
         rejected=sum(r['status'] == 'rejected' for r in records),
         proxy_positive=sum(r.get('proxy_pass', False) for r in records)))


def generate(task):
    run = ROOT / task
    for folder in ['iterations', 'cifs', 'prediction_cache', 'api']:
        (run / folder).mkdir(parents=True, exist_ok=True)
    lock = (run / 'generation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_run()
    servers = []
    state = {'iteration': 0}
    try:
        save(run / 'status.json', dict(status='MODEL_STARTING', at=now(), iteration=0, pid=os.getpid()))
        servers.append(Server('mgt', run))
        servers.append(Server('alignn', run))
        checkpoint = run / 'state.pkl'
        if checkpoint.exists():
            with checkpoint.open('rb') as handle:
                state = pickle.load(handle)
        else:
            prototypes = load(PROTOTYPES / 'manifest.json')
            seeds = [evaluate(task, parent_descriptor(p['path'], task=task), servers, run)
                     for p in prototypes if task in p['eligible_tasks']]
            save(run / 'seed_predictions.json', seeds)
            buffer = ExperienceBuffer(num_islands=5, functions_per_prompt=4, temp_init=1.0, temp_period=10,
                                      reset_period_seconds=CONFIG["archive_reset_seconds"], max_items_per_island=5000,
                                      top_k_success=10, bottom_k_failure=10)
            initial = {'wbg': 'rocksalt_MgO', 'sse': 'antifluorite_Li2O',
                       'pv': 'diamond_Si'}[task]
            base = next(r for r in seeds if r['candidate_id'] == initial)
            for island in range(5):
                buffer.register(island, base, {'total': base['score']}, iteration=0)
            state = dict(iteration=0, buffer=buffer, seeds=seeds, seen=initial_seen(seeds),
                         duplicate_policy=DUPLICATE_POLICY, recent_rejections=[])
            persist_state(checkpoint, state)
        if state['iteration'] == CONFIG['iterations']:
            status(run, state, 'GENERATION_COMPLETED')
            return
        status(run, state, 'GENERATION_RUNNING')
        for iteration in range(state['iteration'] + 1, CONFIG['iterations'] + 1):

            buffer = state['buffer']
            dest = run / 'iterations' / f'{iteration:04d}.json'
            if dest.exists():
                result = load(dest)
                island = result['island_id']
            else:
                if 'pending' not in state:
                    prompt_memory = buffer.get_prompt(iteration=iteration - 1)
                    island = prompt_memory.island_id
                    memories = buffer.get_unique_examples_for_evolution(island, max_examples=20)

                    pool = memories['success'][:2] + memories['failure'][:1]
                    seeds = state['seeds']
                    pool += [seeds[(iteration - 1 + j) % len(seeds)] for j in range(len(seeds))]
                    selected, used = [], set()
                    for row in pool:
                        if row['candidate_id'] in used:
                            continue
                        selected.append(row)
                        used.add(row['candidate_id'])
                        if len(selected) == 5:
                            break
                    parents = []
                    for row in selected:
                        desc = parent_descriptor(row['cif_path'], task=task)
                        desc.update(property_values=row['property_values'], proxy_pass=row['proxy_pass'],
                                    failed_constraints=row['failed_constraints'])
                        parents.append(desc)
                    feedback = dict(success=[feedback_record(r) for r in memories['success'][:3]],
                                    failure=[feedback_record(r) for r in memories['failure'][:3]],
                                    rejected_edits=state['recent_rejections'][-4:])
                    prompt = build_prompt(task, parents, feedback, iteration)
                    if len(prompt.encode()) > 60000:
                        raise RuntimeError('PROMPT_SIZE_GUARD')
                    state['pending'] = dict(iteration=iteration, island_id=island, prompt=prompt, parents=parents)
                    persist_state(checkpoint, state)
                pending = state['pending']
                if pending['iteration'] != iteration:
                    raise RuntimeError('PENDING_ITERATION_MISMATCH')
                island = pending['island_id']
                prompt, parents = pending['prompt'], pending['parents']
                by_id = {p['parent_id']: p for p in parents}
                save(run / 'api' / f'{iteration:04d}_context.json', dict(prompt=prompt, parents=parents, island_id=island))
                failures, best, attempts = [], [], []
                previous = ''
                for attempt in range(CONFIG["llema"]["max_attempts"]):
                    messages = [{'role': 'user', 'content': prompt}]
                    if attempt:
                        messages += [{'role': 'assistant', 'content': previous[:16000]},
                                     {'role': 'user', 'content': 'The validator rejected the edits: ' + json.dumps(failures) +
                                      '. Return two corrected edits with all required reasoning fields.'}]
                    raw, api = request(task, iteration, attempt, messages, RESPONSE_SCHEMA)
                    previous = raw
                    failures, built = [], []
                    try:
                        proposals = validate_response(json.loads(raw))
                    except (ValueError, TypeError) as exc:
                        failures = [type(exc).__name__ + ':' + str(exc)[:300]]
                        proposals = []
                    except Exception as exc:

                        failures = ['schema_validation:' + type(exc).__name__]
                        proposals = []
                    within_attempt = set()
                    for ordinal, proposal in enumerate(proposals):
                        try:
                            parent = by_id.get(proposal['parent_id'])
                            if parent is None:
                                raise ValueError('unknown_parent_id')
                            cid = f'llema_{task}_i{iteration:04d}_c{ordinal}_a{attempt}'
                            cif = run / 'cifs' / (cid + '.cif')
                            if cif.exists() and not Path(str(cif) + '.meta.json').exists():

                                cif.rename(cif.with_suffix('.cif.incomplete.' + str(time.time_ns())))
                            if cif.exists():
                                desc = parent_descriptor(cif, task=task)
                                meta = load(str(cif) + '.meta.json')
                                assert meta['parent_sha256'] == parent['sha256']
                                assert meta['substitutions'] == proposal['substitutions'] and meta['lattice_scale'] == proposal['lattice_scale']
                            else:
                                desc = apply_edit(parent['path'], proposal, task, cif)
                            fp = fingerprint(cif)
                            if fp in state['seen'] or fp in within_attempt:
                                raise ValueError('duplicate_geometry_species_within_task')
                            within_attempt.add(fp)
                            built.append(dict(ordinal=ordinal, desc=desc, proposal=proposal))
                        except (ValueError, AssertionError) as exc:
                            failures.append(f'candidate_{ordinal}:' + str(exc)[:300])
                    attempts.append(dict(attempt=attempt, valid_slots=len(built), errors=failures, call_id=api.get('call_id')))
                    if len(built) > len(best):
                        best = built
                    if len(built) == 2:
                        break
                records = []
                for item in best:
                    row = evaluate(task, item['desc'], servers, run)
                    row.update(iteration=iteration, candidate_ordinal=item['ordinal'],
                               parent_id=item['proposal']['parent_id'], proposal=item['proposal'])
                    records.append(row)
                for ordinal in set(range(2)) - {r['candidate_ordinal'] for r in records}:
                    records.append(dict(candidate_id=f'llema_{task}_i{iteration:04d}_c{ordinal}_rejected',
                                        task=task, iteration=iteration, candidate_ordinal=ordinal,
                                        status='rejected', proxy_pass=False, errors=failures))
                result = dict(task=task, iteration=iteration, island_id=island, candidates=records,
                              parse_attempts=attempts, at=now())
                save(dest, result)
            for row in result['candidates']:
                if row['status'] == 'evaluated':
                    buffer.register(island, row, {'total': row['score']}, iteration=iteration - 1)
                    state['seen'].add(row['fingerprint'])
            state['recent_rejections'] = [a for a in result.get('parse_attempts', []) if a['errors']][-4:]
            state['iteration'] = iteration
            state.pop('pending', None)
            persist_state(checkpoint, state)
            status(run, state, 'GENERATION_COMPLETED' if iteration == CONFIG["iterations"] else 'GENERATION_RUNNING')
            print(json.dumps(dict(task=task, iteration=iteration, positive=sum(r.get('proxy_pass', False) for r in result['candidates']))), flush=True)
    except Exception as exc:
        save(run / 'failure.json', dict(at=now(), error_type=type(exc).__name__, error=str(exc)[:250], iteration=state['iteration']))
        status(run, state, 'GENERATION_BLOCKED')
        raise
    finally:
        for server in servers:
            server.close()
