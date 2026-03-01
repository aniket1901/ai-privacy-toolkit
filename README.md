# ai-privacy-toolkit (extended minimization privacy controls)

This fork adds **four privacy mechanisms** to the *minimization* module — DP threshold privatization, a minimum NCP privacy floor, a cell-level homogeneity guard, and a novel risk-driven enforcement via membership-inference attacks. These aim to address enforcement gaps discussed in Goldsteen paper’s broader privacy-assessment context by turning privacy metrics into privacy constraints during optimization.

## New Security Features

<p align="center">
  <img src="docs/images/FeatureFlowchart.png?raw=true" width="400" title="Flowchart">
</p>
<br />

### 1) Differential Privacy (DP) for the surrogate tree release point
- **Files:** `apt/minimization/dp_mechanism.py`, `apt/minimization/minimizer.py` (`_privatize_tree_thresholds`, DP-SAFEGUARD logic in `_attach_cells_representatives`)
- **What it does:** After fitting the surrogate decision tree, we apply Laplace noise (via `diffprivlib`) to **non-leaf split thresholds** and use truncation bounds to keep noisy thresholds inside valid feature ranges. This reduces leakage from exact split points while keeping the tree usable. Since DP-noised thresholds may create empty regions, `_attach_cells_representatives` contains DP-safe fallback logic (relaxed matching + nearest-row selection) to ensure the pipeline remains stable and does not crash or create outliers through failures.

### 2) Minimum NCP privacy floor (privacy “must not fall below X”)
- **Files:** `apt/minimization/privacy_floor.py`, `apt/minimization/minimizer.py` (privacy-floor integration, snapshot/restore helpers)
- **What it does:** We enforce a minimum privacy level using the toolkit’s NCP information-loss metric as an operational proxy. A baseline NCP is computed once, then an **effective floor** is enforced either as an absolute `min_ncp` or a relative floor `min_ncp = alpha * baseline_ncp`. During the “improve accuracy” loop (feature removal from constraints) and the “improve generalization” loop (tree pruning), the algorithm **snapshots state** and rolls back if the next step would reduce NCP below the floor.

### 3) Homogeneity guard (k / l-diversity / entropy per cell)
- **Files:** `apt/minimization/homogeneity_guard.py`, `apt/minimization/minimizer.py` (`_collect_homogeneity_violations`, `_enforce_homogeneity_guard`)
- **What it does:** Homogeneity attacks occur when a generalized group (“cell”) reveals sensitive outcomes because nearly everyone in the cell shares the same label. We compute per-cell label statistics (cell size, label diversity, entropy) and enforce constraints such as `min_cell_size`, `min_label_diversity`, and optional `min_entropy`. Enforcement modes are `warn`, `raise`, or `auto` (bounded pruning/merging of cells while respecting an accuracy tolerance).

### 4) Risk-driven enforcement (membership inference attacks as a controller)
- **Files:** `apt/minimization/privacy_risk_enforcer.py`, `apt/minimization/minimizer.py` (`_enforce_privacy_risk`), uses attacks in `apt/risk/data_assessment/*`
- **What it does:** We turn the toolkit’s existing membership inference assessments into an **enforcement loop**. After producing a generalized dataset, we evaluate membership risk (classification- or KNN-based). We enforce thresholds on (a) `risk_score` (normalized ratio metric), (b) absolute member/non-member AUC caps, and (c) an explicit “no warning” quality gate. In `auto` mode the minimizer searches over additional pruning levels (bounded by `max_risk_prune_level` and `risk_accuracy_tolerance`) and rolls back to the best feasible state if constraints cannot be satisfied.

## Execution Flow (minimizer `fit()` call sequence)

- Build surrogate decision tree
- **DifferentialPrivacy:** privatize thresholds (`_privatize_tree_thresholds`)
- Build/modify cells (`_calculate_cells` → `_modify_cells`)
- Attach cell representatives (**DP-SAFEGUARD** fallback) (`_attach_cells_representatives`)
- **Homogeneity guard:** detect/enforce cell label diversity (`_enforce_homogeneity_guard`)
- **Privacy floor (NCP):** compute baseline NCP and enforce min NCP while:
  - improving generalization (`_calculate_level_cells` loop) or
  - improving accuracy (`_remove_feature_from_generalization` loop)
  - using snapshot/restore (`_snapshot_state` / `_restore_state`)
- **Risk-driven enforcement:** run membership-inference attack and enforce thresholds (`_enforce_privacy_risk`)
- Store final scores (`self.ncp.fit_score`, `self.ncp.generalizations_score`)
- Logs printed throughout for traceability: `[DP]`, `[DP-SAFEGUARD]`, `[HOMOGENEITY-GUARD]`, `[PRIVACY-FLOOR]`, `[RISK]`

## Installation and Run Instructions

> **Python requirement:** Use **Python 3.11.x** (this project is tested on 3.11; newer versions may fail to build some dependencies).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q tests/test_minimization_privacy_controls.py
```

## Run the demo notebook

After installing the package (steps above), launch Jupyter and run:

- `notebooks/minimization_privacy_controls_adult.ipynb`

This notebook reproduces the full pipeline on the Adult dataset and shows the effects of the 4 features.

## References

* Goldsteen, A., et al. Data minimization for GDPR compliance in machine learning models. *AI and Ethics*, 2021.
* Holohan, N., Antonatos, S., Braghin, S. and Mac Aonghusa, P., 2018. [The Bounded Laplace Mechanism in Differential privacy](https://doi.org/10.29012/jpc.715). *Journal of Privacy and Confidentiality 10 (1).*
* Holohan, N., Braghin, S., Mac Aonghusa, P. and Levacher, K., 2019. [Diffprivlib: the IBM Differential Privacy Library](https://arxiv.org/abs/1907.02444). *ArXiv e-prints 1907.02444 [cs.CR].*
* Machanavajjhala, A., et al. l-diversity: Privacy beyond k-anonymity. *TKDD*, 2007.
* Dwork, C., et al. Calibrating noise to sensitivity in private data analysis. *TCC*, 2006.
