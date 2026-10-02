import hashlib
from pathlib import Path

import joblib
import numpy as np

from .adaptive_scale import vector
from .score_adapter import ABSOLUTE, ABSOLUTE_QR, UPPER_QR, LOWER_QR, CQR


ADJUSTED_SCORE = ABSOLUTE_QR


class FrozenQuantileModel:


    def __init__(self, directory, prop, metadata):
        self.prop = prop
        self.score_definition = metadata.get("score_definition", "normalized_absolute_residual")


        if self.score_definition == "absolute_residual_over_local_scale":
            self.score_definition = ABSOLUTE
        filename = metadata.get("quantile_model")
        self.model = None
        self.model_sha256 = None
        self.models = None
        filenames = metadata.get("quantile_models")
        if filenames is not None:
            if self.score_definition != CQR or set(filenames) != {"lower", "upper"}:
                raise ValueError("quantile_models requires CQR lower/upper endpoints")
            self.models={};self.model_sha256={}
            expected=metadata.get("quantile_models_sha256",{})
            for side,filename in filenames.items():
                if not isinstance(filename,str) or Path(filename).name != filename:
                    raise ValueError("quantile_models must be filenames in the calibration directory")
                path=Path(directory)/filename;digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if side in expected and expected[side] != digest:
                    raise ValueError("Frozen CQR quantile model hash mismatch: "+side)
                self.models[side]=joblib.load(path);self.model_sha256[side]=digest
            filename=None
        if filename:
            if self.score_definition not in (ADJUSTED_SCORE,UPPER_QR,LOWER_QR):
                raise ValueError("Quantile model requires explicitly adjusted calibration scores")
            if not isinstance(filename,str) or Path(filename).name != filename:
                raise ValueError("quantile_model must be a filename in the calibration directory")
            path = Path(directory)/filename
            self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = metadata.get("quantile_model_sha256")
            if expected is not None and expected != self.model_sha256:
                raise ValueError("Frozen quantile model hash mismatch")
            self.model = joblib.load(path)
        elif self.score_definition in (ADJUSTED_SCORE,UPPER_QR,LOWER_QR,CQR) and self.models is None:
            raise ValueError("Adjusted calibration scores are missing their frozen quantile model")
        elif self.models is None and self.score_definition != ABSOLUTE:
            raise ValueError("Unsupported scalar score definition: "+str(self.score_definition))

    @property
    def enabled(self):
        return self.model is not None or self.models is not None

    def predict(self, features, prediction=None):
        if not self.enabled:
            return 0.0
        if prediction is None or not np.isfinite(prediction):
            raise ValueError("Frozen QR requires a finite raw proxy prediction, never a target label")
        x = np.asarray([vector(features,prediction)], dtype=float)
        if self.models is not None:
            values={side:float(np.asarray(model.predict(x)).reshape(-1)[0])
                    for side,model in self.models.items()}
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError("Frozen CQR returned a nonfinite base endpoint")
            return values
        value = float(np.asarray(self.model.predict(x)).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("Frozen QR returned a nonfinite base quantile")


        return max(0.0,value) if self.score_definition == ADJUSTED_SCORE else value

    def manifest(self):
        return dict(enabled=self.enabled, score_definition=self.score_definition,
                    quantile_model_sha256=self.model_sha256,
                    base_quantile_policy=("max(frozen_quantile_prediction,0)" if self.score_definition == ADJUSTED_SCORE
                                          else "signed_frozen_quantile_no_clipping") if self.enabled else "zero_offset_no_QR_model",
                    calibration_correction="Gibbs augmented test-point QR on signed score residuals")
