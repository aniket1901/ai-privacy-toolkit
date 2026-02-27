from typing import Optional


class PrivacyFloorEnforcer:
    """
    Wrapper for NCP privacy-floor checks
    Enforcement type is either
    - auto : Automatically increase generalization to improve NCP value
    - raise : Raise an error if NCP falls below min_ncp
    """

    def __init__(
        self,
        min_ncp: Optional[float] = None,
        enforcement: str = "auto",
        alpha: Optional[float] = None
    ):
        if enforcement not in ("auto", "raise"):
            raise ValueError("privacy_enforcement must be either auto or raise")
        if alpha is not None and (alpha < 0 or alpha > 1):
            raise ValueError("privacy_floor_alpha must be non-negative and between 0 and 1")
        self.min_ncp = min_ncp
        self.enforcement = enforcement
        self.alpha = alpha
        self.baseline_ncp = None

    def is_enabled(self) -> bool:
        return self.min_ncp is not None or self.alpha is not None

    def set_baseline(self, baseline_ncp):
        self.baseline_ncp = baseline_ncp

    def get_effective_min_ncp(self):
        # absolute floor takes precedence over relative floor.
        if self.min_ncp is not None:
            return self.min_ncp
        if self.alpha is not None and self.baseline_ncp is not None:
            return self.alpha * self.baseline_ncp
        return None

    def check(self, ncp_value) -> bool:
        effective_min_ncp = self.get_effective_min_ncp()
        if effective_min_ncp is None:
            return True
        return ncp_value >= effective_min_ncp

    def on_violation(self, ncp_value):
        effective_min_ncp = self.get_effective_min_ncp()
        if effective_min_ncp is None:
            return
        if self.enforcement == "raise":
            raise ValueError("NCP privacy floor violated, ncp = " + str(ncp_value) + " min_ncp = " + str(effective_min_ncp))
