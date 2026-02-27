from typing import Optional, Sequence, Dict, Any
import numpy as np


class HomogeneityGuard:
    """
    Homogeneity-attack safeguard for generalized cells.

    This guard checks per-cell label distribution and can enforce:
      - min_cell_size (k-anonymity-like)
      - min_label_diversity (l-diversity)
      - min_entropy (optional)

    enforcement:
      - "auto": let minimizer merge/prune further to fix violations
      - "raise": raise an error on violation and stop
      - "warn": log violations but do not stop
    """

    def __init__(
        self,
        min_cell_size: int = 1,
        min_label_diversity: int = 1,
        min_entropy: Optional[float] = None,
        enforcement: str = "auto",
    ):
        if enforcement not in ("auto", "raise", "warn"):
            raise ValueError('enforcement must be "auto", "raise", or "warn"')
        if min_cell_size < 1:
            raise ValueError("min_cell_size must be >= 1")
        if min_label_diversity < 1:
            raise ValueError("min_label_diversity must be >= 1")
        if min_entropy is not None and min_entropy < 0:
            raise ValueError("min_entropy must be >= 0")

        self.min_cell_size = min_cell_size
        self.min_label_diversity = min_label_diversity
        self.min_entropy = min_entropy
        self.enforcement = enforcement

    def is_enabled(self):
        return self.min_cell_size > 1 or self.min_label_diversity > 1 or self.min_entropy is not None

    def _entropy(self, labels: np.ndarray):
        if labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()

        if probs.size > 0:
            entropy = float(-(probs * np.log2(probs)).sum()) 
            return max(0.0, entropy)
        else:
            return 0.0

    def cell_stats(self, labels: Sequence, indices: Sequence[int]):
        idx = list(indices)
        if len(idx) == 0:
            return {"size": 0, "distinct": 0, "entropy": 0.0}

        y = np.asarray(labels)[idx]
        distinct = len(np.unique(y))
        entropy = self._entropy(y)

        return {"size": len(idx), "distinct": distinct, "entropy": entropy}

    def check_cell(self, stats: Dict[str, Any]):
        # Return True if cell passes all constraints
        if stats["size"] < self.min_cell_size:
            return False
        if stats["distinct"] < self.min_label_diversity:
            return False
        if self.min_entropy is not None and stats["entropy"] < self.min_entropy:
            return False
        return True

    def on_violation(self, cell_id, stats: Dict[str, Any]):
        # Raise if enforcement is 'raise'. Otherwise do nothing 
        if self.enforcement == "raise":
            raise ValueError(
                "HomogeneityGuard violated for cell_id {}: size={}, distinct={}, entropy={}, required size>={}, distinct>={}, entropy>={}".format(
                    cell_id, stats["size"], stats["distinct"], stats["entropy"],
                    self.min_cell_size, self.min_label_diversity,
                    self.min_entropy if self.min_entropy is not None else "N/A"
                )
            )