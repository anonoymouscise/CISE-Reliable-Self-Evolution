from functools import lru_cache
import itertools
import copy
import math
from cise.runtime import EXPERIMENT, load
from .adaptive_scale import scales
from cise.shared.structure_features import structure_feature_values
from .score_adapter import ABSOLUTE, inverse_bounds

@lru_cache(maxsize=1)
def calibrations():
    return {task:load(EXPERIMENT/'interval'/f'{task}.json') for task in ['wbg','sse','pv'] if (EXPERIMENT/'interval'/f'{task}.json').exists()}

def correct(row,score_values,current_weights=None):
    if row.get('feedback_kind'):raise ValueError('interval already applied')
    result=copy.deepcopy(row);raw=dict(row['property_values']);task=row['task'];cal=calibrations()[task]
    local=scales(row['cif_path'],raw)
    bounds={};quantiles={};test_weights={};solver_records={};abstentions=[]
    features=structure_feature_values(row['cif_path'])
    if current_weights is None:
        from .context_calibration import seed_calibration
        current_weights=seed_calibration(task,tuple(cal['properties']))
    for prop,c in cal['properties'].items():
        bounds[prop]={}
        outcome=current_weights.cutoff(prop,features,raw[prop])
        q=outcome.cutoff
        solver_records[prop]=dict(status=outcome.status,**outcome.diagnostics)
        test_weights[prop]=current_weights.ratio(prop,features)
        if outcome.status!='ok':
            q=float('inf');abstentions.append(prop+':'+outcome.status)
        elif not math.isfinite(q):abstentions.append(prop+':nonfinite_cutoff')
        if not math.isfinite(local[prop]) or local[prop]<=0 or not math.isfinite(raw[prop]):
            q=float('inf');abstentions.append(prop+':nonfinite_scale_or_prediction')
        quantiles[prop]=q if math.isfinite(q) else None
        if math.isfinite(q):
            definition=solver_records.get(prop,{}).get('score_definition',ABSOLUTE)
            inverted=inverse_bounds(raw[prop],local[prop],q,definition,c,
                                    solver_records.get(prop,{}).get('base_offsets'))
            if inverted is None:
                q=float('inf');quantiles[prop]=None
                abstentions.append(prop+':empty_or_invalid_score_inverse')
                bounds[prop]={side:None for side in c}
            else:bounds[prop].update(inverted)
        else:bounds[prop]={side:None for side in c}
        if any(v is not None and not math.isfinite(v) for v in bounds[prop].values()):
            bounds[prop]={side:None for side in bounds[prop]};quantiles[prop]=None
            abstentions.append(prop+':nonfinite_inverse_endpoint')
    if any(q is None for q in quantiles.values()):
        raw_score,raw_failures=score_values(task,raw,row['species'])
        result.update(raw_property_values=raw,raw_score=raw_score,raw_proxy_pass=not raw_failures,proxy_hit=not raw_failures,
            property_values={p:None for p in raw},score=-10.,proxy_pass=False,output_eligible=False,interval_pass=False,
            failed_constraints=abstentions or ['unbounded_conformal_interval'],feedback_kind='gibbs_conformal_interval',
            abstained=True,conformal_bounds=bounds,local_scales=local,conformal_cutoffs=quantiles,
            test_weights=test_weights,cci_solver=solver_records,
            interval_context=dict(iteration=current_weights.iteration,frozen_hash=current_weights.frozen_hash))
        return result
    names=list(bounds);corners=[]
    for values in itertools.product(*(list(bounds[p].values()) for p in names)):
        effective=dict(zip(names,values));score,failures=score_values(task,effective,row['species'])
        corners.append((score,failures,effective))
    score,failures,effective=min(corners,key=lambda x:(x[0],-len(x[1])))
    all_pass=all(not x[1] for x in corners)
    merged=[]
    for _,fs,_ in corners:
        for fail in fs:
            if fail not in merged:merged.append(fail)
    raw_score,raw_failures=score_values(task,raw,row['species'])
    result.update(raw_property_values=raw,raw_score=raw_score,raw_proxy_pass=not raw_failures,proxy_hit=not raw_failures,
        raw_failed_constraints=raw_failures,property_values=effective,score=score,proxy_pass=all_pass,output_eligible=all_pass,
        failed_constraints=merged,feedback_kind='gibbs_conformal_interval',conformal_bounds=bounds,interval_calibration=cal,local_scales=local,
        abstained=False,interval_pass=all_pass,conformal_cutoffs=quantiles,cci_solver=solver_records,
        test_weights=test_weights,
        interval_context=dict(iteration=current_weights.iteration,frozen_hash=current_weights.frozen_hash))
    return result

def annotate_prompt(prompt,task):
    old='All model values and success/failure labels in this feedback are uncorrected surrogate predictions on the exact structure:'
    new=('All property values and success/failure labels in this feedback use Gibbs test-point-corrected conformal bounds. '
         'A frozen quantile model predicts a baseline normalized-error cutoff when configured; Gibbs calibration corrects its residual score using the held-out calibration set. '
         'Each structure has its own error scale predicted by a frozen model; current-iteration unlabeled probes supply a density-ratio basis function for conditional calibration. '
         'property_values is a worst-reward interval corner, and conformal_bounds gives all calibrated endpoints. '
         'Reward is the minimum original reward over relevant corners; success requires every endpoint to satisfy the original target. '
         'For bounded band gaps both lower and upper endpoints are checked. '
         'Raw estimates are excluded from feedback. The frozen raw predictors are:')
    assert old in prompt
    return prompt.replace(old,new)
