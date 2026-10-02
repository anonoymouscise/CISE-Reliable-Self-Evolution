from functools import lru_cache
import numpy as np
import joblib
from cise.runtime import EXPERIMENT
from cise.shared.structure_features import STRUCTURE_FEATURES, structure_feature_values

FLOORS = {'band_gap': .02, 'formation_energy': .01}

def vector(features, prediction):
    return [features[k] for k in STRUCTURE_FEATURES] + [float(prediction)]

@lru_cache(maxsize=3)
def model(prop):
    return joblib.load(EXPERIMENT/'interval'/f'{prop}_scale.joblib')

def predict_scales(prop, features, predictions):
    x=np.asarray([vector(f,p) for f,p in zip(features,predictions)],dtype=float)
    return np.maximum(np.exp(model(prop).predict(x)),FLOORS[prop])

def scales(cif, raw):
    features=structure_feature_values(cif)
    if not features['structure_parse_success_flag']:
        raise ValueError('SCALE_STRUCTURE_PARSE_FAILED')
    return {p:float(predict_scales(p,[features],[value])[0]) for p,value in raw.items()}
