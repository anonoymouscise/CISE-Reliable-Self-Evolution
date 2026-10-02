from functools import lru_cache
import hashlib
import json
import time
import copy
import numpy as np
from scipy.optimize import minimize
from cise.runtime import EXPERIMENT,load
from cise.shared.structure_features import STRUCTURE_FEATURES

def matrix(rows):
    return np.asarray([[r['features'][k] for k in STRUCTURE_FEATURES] for r in rows],dtype=float)

CCI_FEATURES = (
    'number_of_sites', 'volume_per_atom_angstrom3',
    'atomic_number_mean', 'electronegativity_mean',
)

class FixedCCIBasis:

    def __init__(self, source):
        x=self._matrix(source)
        self.median=np.asarray([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
        clean=np.where(np.isfinite(x),x,self.median)
        self.scale=np.std(clean,axis=0)
        self.scale[self.scale<1e-8]=1.

    @staticmethod
    def _matrix(rows):
        return np.asarray([[r['features'].get(k) for k in CCI_FEATURES] for r in rows],dtype=float).reshape(-1,len(CCI_FEATURES))

    def transform(self, rows):
        x=self._matrix(rows)
        return np.column_stack([np.ones(len(rows)),(np.where(np.isfinite(x),x,self.median)-self.median)/self.scale])

    def manifest(self):
        return dict(features=list(CCI_FEATURES),median=self.median.tolist(),scale=self.scale.tolist(),
                    fitted_on='fixed_property_dre_source_covariates_only',intercept=True)


def ratio_basis_diagnostics(base, ratio):
    base=np.asarray(base,dtype=float);ratio=np.asarray(ratio,dtype=float)
    fitted=base@np.linalg.lstsq(base,ratio,rcond=None)[0]
    residual=ratio-fitted
    return dict(base_rank=int(np.linalg.matrix_rank(base)),
                augmented_rank=int(np.linalg.matrix_rank(np.column_stack([base,ratio]))),
                ratio_projection_residual_l2=float(np.linalg.norm(residual)),
                ratio_projection_relative_l2=float(np.linalg.norm(residual)/max(np.linalg.norm(ratio),1e-15)),
                ratio_projection_residual_max=float(np.max(np.abs(residual))))


def ratio_summary(values, bound):
    values=np.asarray(values,dtype=float)
    total=float(values.sum());squared=float(values@values)
    return dict(n=len(values),mean=float(values.mean()),minimum=float(values.min()),maximum=float(values.max()),
                quantiles=np.quantile(values,[0,.1,.5,.9,1]).tolist(),
                lower_clip_fraction=float(np.mean(values<=1e-12)),upper_clip_fraction=float(np.mean(values>=bound-1e-10)),
                ESS=total**2/squared if squared else 0.)

class BoundedLSIF:
    def __init__(self,source,W=10.,regularization=.01):
        if not source:raise ValueError('DRE source is empty')
        if W<=0 or regularization<0:raise ValueError('Invalid DRE bound or regularization')
        self.W=W;self.regularization=regularization
        x=matrix(source);self.median=np.asarray([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
        x=np.where(np.isfinite(x),x,self.median);self.scale=np.std(x,axis=0)
        self.scale[self.scale<1e-8]=1.
        indices=sorted(range(len(source)),key=lambda i:hashlib.sha256(source[i]['id'].encode()).hexdigest())[:32]
        self.centers=((x-self.median)/self.scale)[indices]
        self.source_basis=self.basis(source)
        self.H=self.source_basis.T@self.source_basis/len(source)+2*regularization*np.eye(len(indices)+1)
        self.coefficients=np.r_[1.,np.zeros(len(indices))]

    def basis(self,rows):
        x=matrix(rows);x=(np.where(np.isfinite(x),x,self.median)-self.median)/self.scale

        distance=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(self.centers*self.centers,axis=1)[None,:]-2*x@self.centers.T,0.)
        return np.column_stack([np.ones(len(rows)),np.exp(-distance/(2*len(STRUCTURE_FEATURES)))])

    def fit(self,probe):
        if not probe:raise ValueError('DRE probe batch is empty')
        started=time.perf_counter()
        h=self.basis(probe).mean(axis=0)
        result=minimize(lambda a:.5*a@self.H@a-h@a,self.coefficients,jac=lambda a:self.H@a-h,
            method='SLSQP',bounds=[(0.,self.W)]*len(h),
            constraints=[dict(type='ineq',fun=lambda a:self.W-a.sum(),jac=lambda a:-np.ones_like(a))],
            options=dict(maxiter=500,ftol=1e-10))
        if not result.success:raise RuntimeError('LSIF_OPTIMIZATION_FAILED:'+str(result.message))
        self.coefficients=np.maximum(result.x,0.)
        if self.coefficients.sum()>self.W:self.coefficients*=self.W/self.coefficients.sum()
        return dict(method='bounded_LSIF_paper_equation_10',W=self.W,regularization=self.regularization,
            source_n=len(self.source_basis),probe_n=len(probe),centers_n=len(self.centers),
            coefficients=self.coefficients.tolist(),objective=float(result.fun),success=True,
            direction='target_over_source',feature_dimension=len(STRUCTURE_FEATURES),
            representation='source-fixed 32 RBF centers plus intercept; bandwidth squared=feature_dimension',
            source_ratio=ratio_summary(self.source_basis@self.coefficients,self.W),
            probe_ratio=ratio_summary(self.predict(probe),self.W),
            fitting_seconds=time.perf_counter()-started,optimizer_iterations=int(result.nit),
            bound_policy='nonnegative coefficients with coefficient sum <= W; final numerical clipping to [0,W]')

    def predict(self,rows):return np.clip(self.basis(rows)@self.coefficients,0.,self.W)

@lru_cache(maxsize=3)
def calibration_data(prop):return load(EXPERIMENT/'interval'/f'{prop}_residuals.json')

@lru_cache(maxsize=3)
def fixed_calibration_inputs(prop):

    data=calibration_data(prop);template=BoundedLSIF(data['dre_source'])
    basis=FixedCCIBasis(data['dre_source'])
    kernel=template.basis(data['rows']);kernel.flags.writeable=False
    base=basis.transform(data['rows']);base.flags.writeable=False
    scores=np.asarray([r['score'] for r in data['rows']],dtype=float);scores.flags.writeable=False
    return template,basis,kernel,base,scores

class CurrentWeights:
    def __init__(self,task,probe,iteration,frozen_hash,properties=None):
        self.task=task;self.iteration=iteration;self.frozen_hash=frozen_hash
        self.models={};self.weights={};self.logs={};self.bases={};self.base_matrices={};self.scores={}
        for prop in (properties if properties is not None else ['band_gap','formation_energy']):
            template,basis,kernel,base,scores=fixed_calibration_inputs(prop)
            model=copy.copy(template);model.coefficients=template.coefficients.copy();log=model.fit(probe)
            w=np.clip(kernel@model.coefficients,0.,model.W);self.models[prop]=model;self.weights[prop]=w
            log.update(weight_sum=float(w.sum()),ESS=float(w.sum()**2/(w@w)) if w@w else 0.,
                       weight_min=float(w.min()),weight_max=float(w.max()))
            self.bases[prop]=basis;self.base_matrices[prop]=base
            self.scores[prop]=scores
            log.update(calibration_ratio=ratio_summary(w,model.W),cci_basis=basis.manifest(),
                       ratio_basis=ratio_basis_diagnostics(base,w),context_hash=frozen_hash,iteration=iteration)
            self.logs[prop]=log

    def calibration_basis(self,prop,include_ratio=True):
        base=self.base_matrices[prop]
        return np.column_stack([base,self.weights[prop]]) if include_ratio else base.copy()

    def test_basis(self,prop,features,include_ratio=True):
        rows=[dict(features=features)];base=self.bases[prop].transform(rows)
        return np.column_stack([base,self.models[prop].predict(rows)]) if include_ratio else base

def freeze_hash(state):return hashlib.sha256(json.dumps(state,sort_keys=True,allow_nan=False).encode()).hexdigest()
