"""
NCP privacy-floor enforcement for the minimization module.

This helper turns NCP (Normalized Certainty Penalty) into an enforceable privacy floor. 
It supports either:
- an absolute floor (min_ncp), or
- a relative floor (alpha * baseline_ncp) computed from the initial solution,

with an optional "raise" mode to hard-fail if the floor is violated.
"""
from typing import Optional


class PrivacyFloorEnforcer:
    """
    The minimizer NCP score increases when generalization increases (more privacy).
    This class ensures a minimum by defining a floor and checking candidate solutions against it.

    Two ways to specify the floor:
    - Absolute: ``min_ncp`` (always takes precedence if provided)
    - Relative: ``alpha * baseline_ncp`` where baseline is captured once from the initial solution

    In "raise" mode it raises an exception on violation; otherwise in "auto" mode it only provides checks,
    so the minimizer can decide how to react (e.g., rollback to a previous state).
    """

    def __init__(
        self,
        min_ncp: Optional[float] = None,
        enforcement: str = "auto",
        alpha: Optional[float] = None
    ):
        """
        Initialize an NCP privacy-floor policy.

        :param min_ncp: Absolute minimum allowed NCP. If set, overrides alpha-based floor.
        :type min_ncp: float, optional
        :param enforcement: Enforcement mode: "auto" (report) or "raise" (error on violation).
        :type enforcement: str, optional
        :param alpha: Relative floor factor in [0, 1]. Effective floor becomes ``alpha * baseline_ncp``.
        :type alpha: float, optional
        :raises ValueError: If enforcement is invalid or alpha is outside [0, 1].
        """
        if enforcement not in ("auto", "raise"):
            raise ValueError("privacy_enforcement must be either auto or raise")
        if alpha is not None and (alpha < 0 or alpha > 1):
            raise ValueError("privacy_floor_alpha must be non-negative and between 0 and 1")
        self.min_ncp = min_ncp
        self.enforcement = enforcement
        self.alpha = alpha
        self.baseline_ncp = None

    def is_enabled(self) -> bool:
        '''
        Helper function which returns true if privacy floor mechanism is enabled
        '''
        return self.min_ncp is not None or self.alpha is not None

    def set_baseline(self, baseline_ncp):
        '''
        Sets initial baseline NCP value
        '''
        self.baseline_ncp = baseline_ncp

    def get_effective_min_ncp(self):
        '''
        Returns effective minimum NCP based on the parameters provided
        '''
        if self.min_ncp is not None:
            return self.min_ncp
        if self.alpha is not None and self.baseline_ncp is not None:
            return self.alpha * self.baseline_ncp
        return None

    def check(self, ncp_value) -> bool:
        '''
        Compares provided NCP with effective min NCP value
        '''
        effective_min_ncp = self.get_effective_min_ncp()
        if effective_min_ncp is None:
            return True
        return ncp_value >= effective_min_ncp

    def on_violation(self, ncp_value):
        '''
        If NCP falls below minimum, this method handles violation based on enforcement mode
        '''
        effective_min_ncp = self.get_effective_min_ncp()
        if effective_min_ncp is None:
            return
        if self.enforcement == "raise":
            raise ValueError("NCP privacy floor violated, ncp = " + str(ncp_value) + " min_ncp = " + str(effective_min_ncp))
