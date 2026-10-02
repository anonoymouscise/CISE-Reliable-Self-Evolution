from functools import lru_cache
import time
import numpy as np
from cise.runtime import CONFIG, EXPERIMENT
from .online_dre import CurrentWeights, FixedCCIBasis, calibration_data
from .quantile_model import FrozenQuantileModel
from .gibbs import CutoffResult


class ContextCalibration:
    def __init__(self, task, properties, probe, iteration, frozen_hash, include_ratio=True):
        from .gibbs import GibbsCCI
        self.iteration=iteration;self.frozen_hash=frozen_hash
        self.properties=tuple(properties);self.include_ratio=include_ratio
        self.weights=CurrentWeights(task,probe,iteration,frozen_hash,properties=self.properties) if probe is not None else None
        self.logs={};self.solvers={};self.bases={};self.without={};self.quantile_models={}
        alpha=CONFIG['cci']['alpha_total']/len(self.properties)
        for prop in self.properties:
            started=time.perf_counter()
            data=calibration_data(prop)
            self.quantile_models[prop]=FrozenQuantileModel(EXPERIMENT/'interval',prop,data)
            if self.quantile_models[prop].enabled != bool(CONFIG['cci'].get('quantile_offset',False)):
                raise ValueError('Quantile-offset config and frozen score metadata disagree: '+prop)
            if self.weights is not None:
                matrix=self.weights.calibration_basis(prop,include_ratio)
                scores=self.weights.scores[prop]
                self.logs[prop]=dict(self.weights.logs[prop])
            else:
                basis=FixedCCIBasis(data['dre_source']);self.bases[prop]=basis
                matrix=basis.transform(data['rows']);scores=np.array([r['score'] for r in data['rows']])
                self.logs[prop]=dict(seed_policy='Gibbs test-point corrected base basis; no target DRE; excluded from coverage/output',cci_basis=basis.manifest())
            self.solvers[prop]=GibbsCCI(scores,matrix,alpha=alpha)
            self.logs[prop].update(alpha=alpha,calibration_n=len(scores),basis_dimension=matrix.shape[1],
                engine='gibbs_finite_unpenalized_test_corrected',
                frozen_quantile=self.quantile_models[prop].manifest(),
                preparation_seconds=time.perf_counter()-started)

    def _add_offset(self,prop,features,prediction,outcome):
        base=self.quantile_models[prop].predict(features,prediction)
        cqr=self.quantile_models[prop].score_definition == 'normalized_cqr_residual_minus_frozen_endpoints'
        total=outcome.cutoff if cqr else base+outcome.cutoff
        return CutoffResult(total,outcome.status,
            dict(outcome.diagnostics,base_quantile=base,
                 base_offsets=base if cqr else None,
                 gibbs_residual_cutoff=outcome.cutoff if np.isfinite(outcome.cutoff) else None,
                 score_definition=self.quantile_models[prop].score_definition,
                 quantile_model_sha256=self.quantile_models[prop].model_sha256,
                 inverse_mapping=('proxy + local_scale * frozen asymmetric endpoints +/- Gibbs residual cutoff'
                    if cqr else 'proxy + local_scale * total_cutoff (upper signed tail)'
                    if self.quantile_models[prop].score_definition == 'normalized_upper_residual_minus_frozen_quantile'
                    else 'proxy - local_scale * total_cutoff (lower signed tail)'
                    if self.quantile_models[prop].score_definition == 'normalized_lower_residual_minus_frozen_quantile'
                    else 'proxy +/- local_scale * (base_quantile + gibbs_residual_cutoff)')))

    def cutoff(self,prop,features,prediction=None):
        matrix=self.weights.test_basis(prop,features,self.include_ratio) if self.weights else self.bases[prop].transform([dict(features=features)])
        result=self.solvers[prop].cutoff(matrix)
        return self._add_offset(prop,features,prediction,result)

    def ratio(self,prop,features):
        return float(self.weights.models[prop].predict([dict(features=features)])[0]) if self.weights else None

    def cutoff_without_dre(self,prop,features,prediction=None):
        from .gibbs import GibbsCCI
        if self.weights is None:return self.cutoff(prop,features,prediction)
        if prop not in self.without:
            self.without[prop]=GibbsCCI(self.weights.scores[prop],self.weights.calibration_basis(prop,False),
                                      alpha=CONFIG['cci']['alpha_total']/len(self.properties))
        outcome=self.without[prop].cutoff(self.weights.test_basis(prop,features,False))
        return self._add_offset(prop,features,prediction,outcome)


@lru_cache(maxsize=8)
def seed_calibration(task,properties):
    return ContextCalibration(task,properties,None,0,'seed_initialization_without_target_DRE',False)
