from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class CutoffResult:
    cutoff: float
    status: str
    diagnostics: dict


class GibbsCCI:


    def __init__(self, scores, features, alpha=0.05):
        scores = np.asarray(scores, dtype=float).reshape(-1)
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or len(scores) != len(features) or not len(scores):
            raise ValueError("Expected nonempty scores and a matching n-by-d basis")
        if not features.shape[1] or not np.isfinite(features).all() or not np.isfinite(scores).all():
            raise ValueError("Scores and basis must be finite with at least one basis column")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        self.alpha = float(alpha)
        self.q = 1 - self.alpha

        order = np.lexsort(tuple(np.column_stack((features, scores)).T[::-1]))
        self.scores = scores[order].copy()
        self.features = features[order].copy()
        self.scores.flags.writeable = self.features.flags.writeable = False
        self.n, self.d = self.features.shape
        self.column_scale = np.maximum(np.max(np.abs(self.features), axis=0), 1.0)
        self.A = self.features / self.column_scale
        self.score_scale = max(1.0, float(np.max(np.abs(self.scores))))
        self.S = self.scores / self.score_scale
        self.rank = int(np.linalg.matrix_rank(self.A))


        self._options = {"dual_feasibility_tolerance": 1e-9,
                         "primal_feasibility_tolerance": 1e-9}

    def _test(self, test_features):
        z = np.asarray(test_features, dtype=float).reshape(-1)
        if z.shape != (self.d,) or not np.isfinite(z).all():
            raise ValueError("Test basis must be a finite vector of the fitted dimension")
        return z / self.column_scale

    def cutoff(self, test_features):

        started = time.monotonic()
        z = self._test(test_features)
        diag = {"algorithm": "gibbs_dual_boundary_lp", "alpha": self.alpha,
                "n_calibration": self.n, "basis_dimension": self.d,
                "calibration_rank": self.rank, "test_point_correction": True,
                "regularization": 0.0, "randomized": False}

        def result(value, status, **extra):
            return CutoffResult(float(value), status,
                                {**diag, **extra, "seconds": time.monotonic() - started})

        first = linprog(-self.S, A_eq=self.A.T, b_eq=-self.q * z,
                        bounds=(-self.alpha, self.q), method="highs-ipm",
                        options=self._options)
        if first.status == 2:
            return result(np.inf, "unbounded_interval", boundary_lp_status=int(first.status))
        if not first.success:
            return result(np.inf, "solver_failure", stage="dual_boundary",
                          solver_message=first.message)


        value = -float(first.fun)


        at_upper = first.x >= self.q - 1e-9
        at_lower = first.x <= -self.alpha + 1e-9
        interior = ~(at_upper | at_lower)
        inequalities = np.vstack((self.A[at_upper], -self.A[at_lower]))
        limits = np.concatenate((self.S[at_upper], -self.S[at_lower]))
        second = linprog(z, A_ub=inequalities if len(limits) else None,
                         b_ub=limits if len(limits) else None,
                         A_eq=self.A[interior] if interior.any() else None,
                         b_eq=self.S[interior] if interior.any() else None,
                         bounds=[(None, None)] * self.d,
                         method="highs", options=self._options)
        if not second.success:
            return result(np.inf, "solver_failure", stage="boundary_optimal_face",
                          solver_message=second.message)
        cutoff = float(second.fun) * self.score_scale
        dual_error = float(np.max(np.abs(self.A.T @ first.x + self.q * z)))
        residuals = self.S - self.A @ second.x
        pinball = np.sum(np.maximum(self.q * residuals, -self.alpha * residuals))
        face_error = abs(float(pinball - self.q * (z @ second.x)) - value)
        if not np.isfinite(cutoff) or dual_error > 1e-6 or face_error > 1e-6:
            return result(np.inf, "solver_failure", stage="residual_check",
                          dual_feasibility_error=dual_error, face_error=face_error)
        return result(cutoff, "ok", dual_feasibility_error=dual_error,
                      face_error=face_error, boundary_objective=value * self.score_scale,
                      lp_iterations=int(first.nit + second.nit))

    def augmented(self, test_features, assumed_score):

        z = self._test(test_features)
        if not np.isfinite(assumed_score):
            raise ValueError("The hypothetical score must be finite")
        A = np.vstack((self.A, z))
        S = np.append(self.S, float(assumed_score) / self.score_scale)
        solved = linprog(-S, A_eq=A.T, b_eq=np.zeros(self.d),
                         bounds=(-self.alpha, self.q), method="highs", options=self._options)
        if not solved.success:
            raise RuntimeError(solved.message)
        beta = -solved.eqlin.marginals
        return {"test_dual": float(solved.x[-1]),
                "test_fitted_score": float(z @ beta) * self.score_scale,
                "objective": -float(solved.fun) * self.score_scale,
                "dual_feasibility_error": float(np.max(np.abs(A.T @ solved.x)))}

