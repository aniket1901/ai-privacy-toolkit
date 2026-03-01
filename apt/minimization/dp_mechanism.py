"""
DifferentialPrivacy helper used by the minimizer.

Wraps `diffprivlib` Laplace mechanisms to privatize scalar values (e.g., decision
tree split thresholds). Supports optional truncation so noisy outputs stay within
valid [lower, upper] bounds.
"""
from typing import Optional
from diffprivlib.mechanisms import Laplace, LaplaceTruncated


class DifferentialPrivacyMechanism:
    """
    Laplace-mechanism wrapper for (epsilon, delta)-differential privacy.
    """
    def __init__(
        self, 
        epsilon: float, 
        sensitivity: float = 1.0,
        delta: float = 0.0, 
        random_state: Optional[int] = None
    ):
        """
        :param epsilon: Privacy budget (> 0). Smaller epsilon = more noise = stronger privacy.
        :type epsilon: float
        :param sensitivity: Maximum amount the released scalar could change if one individual's record is added/removed
                            (i.e., L1 global sensitivity). (> 0).
        :type sensitivity: float, optional
        :param delta: Probability that privacy loss can exceed ε. With delta=0, we get "pure" ε-DP.
        :type delta: float, optional
        :param random_state: Optional random seed for reproducible noise.
        :type random_state: int, optional
        :raises ValueError: If epsilon <= 0, sensitivity <= 0, or delta < 0.
        """
        if epsilon <= 0:
            raise ValueError('epsilon must be > 0')
        if sensitivity <= 0:
            raise ValueError('sensitivity must be > 0')
        if delta < 0:
            raise ValueError('delta must be >= 0')

        self.epsilon = float(epsilon)
        self.sensitivity = float(sensitivity)
        self.delta = float(delta)
        self.random_state = random_state

    def privatize_value(
        self, 
        value: float, 
        lower: Optional[float] = None,
        upper: Optional[float] = None, 
        sensitivity: Optional[float] = None
    ) -> float:
        """
        Privatize a numeric value with Laplace noise.

        If ``lower`` and ``upper`` are provided, uses a truncated Laplace mechanism so the
        output stays within ``[lower, upper]``.

        :param value: Numeric value to privatize.
        :type value: float
        :param lower: Optional lower bound for truncation (must be provided with ``upper``).
        :type lower: float, optional
        :param upper: Optional upper bound for truncation (must be provided with ``lower``).
        :type upper: float, optional
        :param sensitivity: Optional per-call sensitivity override (> 0).
        :type sensitivity: float, optional
        :return: Privatized (noisy) value.
        :rtype: float
        :raises ValueError: If sensitivity <= 0, only one bound is provided, or lower >= upper.
        """
        if sensitivity is not None:
            used_sensitivity = sensitivity 
        else:
            used_sensitivity = self.sensitivity

        if used_sensitivity <= 0:
            raise ValueError('sensitivity must be > 0')

        # throw an error if only one of them is provided
        if (lower is None and upper is not None) or (lower is not None and upper is None):
            raise ValueError("provide both lower and upper bounds or neither")

        if lower is not None and upper is not None:
            if lower >= upper:
                raise ValueError('lower must be < upper')
            
            mechanism = LaplaceTruncated(
                epsilon=self.epsilon,
                delta=self.delta,
                sensitivity=used_sensitivity,
                lower=lower,
                upper=upper,
                random_state=self.random_state
            )
        else:
            mechanism = Laplace(
                epsilon=self.epsilon,
                delta=self.delta,
                sensitivity=used_sensitivity,
                random_state=self.random_state
            )
        
        return mechanism.randomise(value)
