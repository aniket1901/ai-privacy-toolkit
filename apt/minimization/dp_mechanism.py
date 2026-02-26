from typing import Optional
from diffprivlib.mechanisms import Laplace, LaplaceTruncated


class DifferentialPrivacyMechanism:
    """
    Simple laplace-mechanism implementing epsilon-differential privacy.
    """
    def __init__(
        self, 
        epsilon: float, 
        sensitivity: float = 1.0,
        delta: float = 0.0, 
        random_state: Optional[int] = None
    ):
        """
        initialize laplace-based differential privacy mechanism

        epsilon - privacy budget
        sensitivity - global sensitivity of threshold function
        delta - approximate DP parameter. For pure epsilon-DP keep it 0.0
        random_state - optional random seed for reproducibility
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

    def privatize_value(self, 
                        value: float, 
                        lower: Optional[float] = None,
                        upper: Optional[float] = None, 
                        sensitivity: Optional[float] = None) -> float:
        """
        privatize a numeric value with optional truncation bounds.

        value - numeric value to privatize.
        lower - optional lower truncation bound.
        upper - optional upper truncation bound.
        sensitivity - optional per-value sensitivity override.
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
