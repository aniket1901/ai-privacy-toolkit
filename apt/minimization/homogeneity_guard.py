'''
Homogeneity guard wrapper used by minimizer.py to safeguard against homogeneity attacks.
Detects and enforces per-cell label homogeneity constraints.
'''
from typing import Optional, Sequence, Dict, Any
import numpy as np


class HomogeneityGuard:
    """
    The minimizer partitions records into generalized “cells”. If a cell becomes too homogeneous
    (the training labels/predictions), an attacker can infer sensitive information about 
    individuals in that cell. This guard flags cells that violate common diversity constraints.

    Supported constraints:
      - min_cell_size: Minimum number of records per cell (k-anonymity)
      - min_label_diversity: Minimum number of distinct labels per cell (l-diversity)
      - min_entropy: Minimum Shannon entropy of the label distribution per cell (optional)

    Enforcement modes:
      - "auto": violations are returned to the caller to be handled by the minimizer (e.g., merge cells / prune tree).
      - "warn": violations are detectable but not raised as exceptions.
      - "raise": raise ValueError on violation.
    """

    def __init__(
        self,
        min_cell_size: int = 1,
        min_label_diversity: int = 1,
        min_entropy: Optional[float] = None,
        enforcement: str = "auto",
    ):
        '''
        Initialize homogeneity constraints for generalized cells.

        :param min_cell_size: Minimum number of records per cell (>= 1).
        :type min_cell_size: int, optional
        :param min_label_diversity: Minimum number of distinct labels per cell (>= 1).
        :type min_label_diversity: int, optional
        :param min_entropy: Minimum Shannon entropy of labels in a cell (>= 0). If None, entropy is not enforced.
        :type min_entropy: float, optional
        :param enforcement: Enforcement mode: "auto", "warn", or "raise".
        :type enforcement: str, optional
        :raises ValueError: If parameters are out of range or enforcement mode is invalid.
        '''
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
        '''
        Helper function which returns true if homogeneity guard is enabled
        '''
        return self.min_cell_size > 1 or self.min_label_diversity > 1 or self.min_entropy is not None

    def _entropy(self, labels: np.ndarray):
        '''
        Calculates the Shannon entropy of provided label array
        '''
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
        '''
        Computes per cell label statistics and returns a dictionary with values
        '''
        idx = list(indices)
        if len(idx) == 0:
            return {"size": 0, "distinct": 0, "entropy": 0.0}

        y = np.asarray(labels)[idx]
        distinct = len(np.unique(y))
        entropy = self._entropy(y)

        return {"size": len(idx), "distinct": distinct, "entropy": entropy}

    def check_cell(self, stats: Dict[str, Any]):
        '''
        Check whether a cell satisfies all enabled constraints
        '''
        if stats["size"] < self.min_cell_size:
            return False
        if stats["distinct"] < self.min_label_diversity:
            return False
        if self.min_entropy is not None and stats["entropy"] < self.min_entropy:
            return False
        return True

    def on_violation(self, cell_id, stats: Dict[str, Any]):
        '''
        Raise an error if enforcement is 'raise'. Otherwise do nothing 
        '''
        if self.enforcement == "raise":
            raise ValueError(
                "HomogeneityGuard violated for cell_id {}: size={}, distinct={}, entropy={}, required size>={}, distinct>={}, entropy>={}".format(
                    cell_id, stats["size"], stats["distinct"], stats["entropy"],
                    self.min_cell_size, self.min_label_diversity,
                    self.min_entropy if self.min_entropy is not None else "N/A"
                )
            )