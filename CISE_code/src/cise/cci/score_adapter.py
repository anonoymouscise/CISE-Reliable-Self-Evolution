import math


ABSOLUTE = "normalized_absolute_residual"
ABSOLUTE_QR = "normalized_absolute_residual_minus_frozen_quantile"
UPPER_QR = "normalized_upper_residual_minus_frozen_quantile"
LOWER_QR = "normalized_lower_residual_minus_frozen_quantile"
CQR = "normalized_cqr_residual_minus_frozen_endpoints"


def calibration_score(truth, prediction, scale, definition, base=0.0):

    if not all(math.isfinite(float(v)) for v in (truth, prediction, scale)) or scale <= 0:
        raise ValueError("Score inputs must be finite and scale positive")
    if definition in (ABSOLUTE, ABSOLUTE_QR):
        return abs(truth-prediction)/scale-float(base)
    if definition == UPPER_QR:
        return (truth-prediction)/scale-float(base)
    if definition == LOWER_QR:
        return (prediction-truth)/scale-float(base)
    if definition == CQR:
        lower,upper=base
        return max(-lower-(truth-prediction)/scale,
                   (truth-prediction)/scale-upper)
    raise ValueError("Unsupported score definition: "+str(definition))


def inverse_bounds(prediction, scale, correction, definition, sides,
                   base_offsets=None):

    if not all(math.isfinite(float(v)) for v in (prediction, scale, correction)) or scale <= 0:
        return None
    sides=set(sides)
    if definition == CQR:
        if not base_offsets or set(base_offsets) != {'lower','upper'}:
            return None
        lo=prediction-scale*(float(base_offsets['lower'])+correction)
        hi=prediction+scale*(float(base_offsets['upper'])+correction)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            return None
        return {side:lo if side == 'lower' else hi for side in sides}
    if definition in (ABSOLUTE, ABSOLUTE_QR):

        if correction < 0:return None
        return {side:prediction-scale*correction if side == 'lower'
                else prediction+scale*correction for side in sides}
    if definition == UPPER_QR:
        if sides != {'upper'}:return None
        return {'upper':prediction+scale*correction}
    if definition == LOWER_QR:
        if sides != {'lower'}:return None
        return {'lower':prediction-scale*correction}
    return None
