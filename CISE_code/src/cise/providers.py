import fcntl
import hashlib
import json
import os
import uuid
import requests
from .runtime import CONFIG, ROOT, save, load, now

MODEL = CONFIG['llm']['model']


def key():
    value = os.environ.get(CONFIG['llm']['api_key_env'])
    if not value:
        raise RuntimeError('Missing API key environment variable: ' + CONFIG['llm']['api_key_env'])
    return value


def demo_response(task, iteration, attempt):
    context = load(ROOT / task / 'api' / f'{iteration:04d}_context.json')
    parent = context['parents'][0]
    items = []
    for i in range(2):
        factor = 1 + (0.000001 * (iteration * 6 + attempt * 2 + i + 1))
        if factor > 1.049:
            raise RuntimeError('DEMO_CALL_LIMIT')
        items.append(dict(parent_id=parent['parent_id'], substitutions=[], lattice_scale=factor,
            rules_used='isotropic lattice scaling', justification='Offline software smoke fixture; no scientific claim.',
            expected_effects=dict(band_gap='uncertain', formation_energy='uncertain'),
            failure_modes=['Synthetic predictor is not scientific evidence.', 'Geometry validity does not establish task success.']))
    return json.dumps(dict(candidates=items))


def _check_cached(record):
    http = record.get('http_status')
    if http in [401, 402, 403]:
        raise RuntimeError('API_AUTH_OR_CREDIT_BLOCKED')
    if http in [400, 404, 405, 422]:
        raise RuntimeError('API_REQUEST_CONFIGURATION_REJECTED_' + str(http))
    if record.get('status') in ['request_started', 'transport_or_decode_error']:
        raise RuntimeError('REQUEST_OUTCOME_UNKNOWN: inspect the provider request before retrying')
    return record.get('content', ''), record


def request(task, iteration, attempt, messages, schema):
    folder = ROOT / task / 'api'
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f'{iteration:04d}_{attempt}.json'
    with dest.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _request_locked(task, iteration, attempt, messages, schema, dest)


def _request_locked(task, iteration, attempt, messages, schema, dest):
    if not all(isinstance(m.get('content'), str) for m in messages):
        raise ValueError('Only text messages are supported')
    body = dict(model=MODEL, messages=messages, max_tokens=CONFIG['llm']['max_output_tokens'],
        response_format={'type': 'json_schema', 'json_schema': {'name': 'crystal_edits', 'strict': True, 'schema': schema}})
    fingerprint = hashlib.sha256(json.dumps(dict(request=body, demo=CONFIG['demo']),
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if dest.exists():
        old = load(dest)
        if old.get('request_fingerprint') != fingerprint:
            raise RuntimeError('CACHED_REQUEST_CONTEXT_MISMATCH')
        return _check_cached(old)
    credential = None if CONFIG['demo'] else key()
    record = dict(at=now(), call_id=uuid.uuid4().hex, task=task, iteration=iteration,
        attempt=attempt, request=body, demo=CONFIG['demo'], request_fingerprint=fingerprint,
        status='request_started')
    save(dest, record)
    try:
        if CONFIG['demo']:
            content = demo_response(task, iteration, attempt)
            http = 200
            generation_id = None
            usage = {}
        else:
            response = requests.post('https://openrouter.ai/api/v1/chat/completions',
                headers={'Authorization': 'Bearer ' + credential}, json=body, timeout=(20, 180))
            http = response.status_code
            if http != 200:
                record.update(status='http_error', http_status=http)
                save(dest, record)
                return _check_cached(record)
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError('Invalid response object')
            choices = payload.get('choices')
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError('Missing response choices')
            message = choices[0].get('message')
            if not isinstance(message, dict) or not isinstance(message.get('content'), str):
                raise ValueError('Missing text response')
            content = message['content']
            generation_id = payload.get('id')
            raw_usage = payload.get('usage')
            usage = ({k: v for k, v in raw_usage.items()
                      if k in ['prompt_tokens', 'completion_tokens', 'total_tokens']}
                     if isinstance(raw_usage, dict) else {})
        record.update(status='responded', http_status=http, content=content,
                      usage=usage, generation_id=generation_id)
        save(dest, record)
        return content, record
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:

        record.update(status='transport_or_decode_error', error_type=type(exc).__name__)
        save(dest, record)
        raise RuntimeError('REQUEST_OUTCOME_UNKNOWN: inspect the provider request before retrying') from None
