# ai-privacy-toolkit: Privacy-Enforced Minimization Extension

This fork adds **four privacy mechanisms** to the *minimization* module — DP threshold privatization, a minimum NCP privacy floor, a cell-level homogeneity guard, and a novel risk-driven enforcement via membership-inference attacks. These aim to address enforcement gaps discussed in Goldsteen paper’s broader privacy-assessment context by turning privacy metrics into privacy constraints during optimization.

## New Security Features

<p align="center">
  <img src="docs/images/FeatureFlowchart.png?raw=true" width="400" title="Flowchart">
</p>
<br />

### 1. Differential Privacy (DP) on DT Thresholds
- **Files:** `apt/minimization/dp_mechanism.py`, `apt/minimization/minimizer.py` (`_privatize_tree_thresholds`, `_attach_cells_representatives`)
- This mechanism applies ε-based differential privacy by adding Laplace noise (via `diffprivlib`) to **non-leaf split thresholds** and uses truncation bounds to keep noisy thresholds inside valid ranges which reduces leakage from exact split points. Since DP-noised thresholds may create empty cells, `_attach_cells_representatives` contains DP-safe fallback logic (relaxed matching + nearest-row selection) to ensure the pipeline remains stable.

### 2. Minimum NCP Privacy Floor
- **Files:** `apt/minimization/privacy_floor.py`, `apt/minimization/minimizer.py` (privacy-floor integration, snapshot/restore helpers)
- We enforce a minimum privacy level using the toolkit’s NCP metric. A baseline NCP is computed and an **effective floor** is enforced either as an absolute `min_ncp` or a relative `min_ncp = alpha * baseline_ncp`. During accuracy/generalization improvement, we snapshot state and roll back if NCP falls below floor.

### 3. HomogeneityGuard (k-anonymity / l-diversity / entropy per cell)
- **Files:** `apt/minimization/homogeneity_guard.py`, `apt/minimization/minimizer.py` (`_collect_homogeneity_violations`, `_enforce_homogeneity_guard`)
- Homogeneity attacks happen when a generalized “cell” mostly contains one label, making the label easy to infer. We compute per-cell statistics (cell size, label diversity, entropy) and enforce constraints. Enforcement modes: `warn`, `raise`, or `auto` (bounded pruning/merging of cells while respecting an accuracy tolerance).

### 4. Risk-driven Enforcement
- **Files:** `apt/minimization/privacy_risk_enforcer.py`, `apt/minimization/minimizer.py` (`_enforce_privacy_risk`), uses attacks in `apt/risk/data_assessment/*`
- This novel feature converts the toolkit’s membership-inference assessment into an **enforcement controller**. The minimizer evaluates the current generalized output with an existing membership-inference attack and actively adjusts the solution.
We enforce thresholds on 
  - `risk_score` (normalized ratio metric)
  - `absolute member/non-member AUC` caps
  - `no_warning` an explicity quality gate. 
In `auto` mode the minimizer searches over additional pruning levels (bounded by `max_risk_prune_level` and `risk_accuracy_tolerance`) and rolls back to the best feasible state if constraints cannot be satisfied.

- **Gap addressed:** Many pipelines stop at “compute a privacy metric” and do not enforce privacy once accuracy-driven pruning/optimization begins. This mechanism closes that loop with an explicit adversary model (membership-inference), making privacy an important constraint alongside utility, not just a reported number.

## Execution Flow (minimizer `fit()` call sequence)

- Build surrogate decision tree
- **DifferentialPrivacy:** privatize thresholds (`_privatize_tree_thresholds`)
- Attach cell representatives (**DP-SAFEGUARD** fallback) (`_attach_cells_representatives`)
- **HomogeneityGuard:** detect/enforce cell label diversity (`_enforce_homogeneity_guard`)
- **PrivacyFloor (NCP):** compute baseline NCP and enforce `min_ncp`
- **Risk-driven enforcement:** run membership-inference attack and enforce thresholds (`_enforce_privacy_risk`)
- Logs printed throughout for clarity: `[DP]`, `[DP-SAFEGUARD]`, `[HOMOGENEITY-GUARD]`, `[PRIVACY-FLOOR]`, `[RISK]`

## Installation and Run Instructions

> Use **Python 3.11.x** (this project is tested on 3.11; newer versions may fail to build some dependencies).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q tests/test_minimization_privacy_controls.py # To run unit tests on the mechanism wrappers
```

## Run the demo notebook

After setup (steps above), you can launch Jupyter and run:

- `notebooks/minimization_privacy_controls_adult.ipynb`

This notebook reproduces the pipeline on the Adult dataset and shows the effects of the 4 features.

## References

* Goldsteen, A., et al. Data minimization for GDPR compliance in machine learning models. *AI and Ethics*, 2021.
* Holohan, N., Antonatos, S., Braghin, S. and Mac Aonghusa, P., 2018. [The Bounded Laplace Mechanism in Differential privacy](https://doi.org/10.29012/jpc.715). *Journal of Privacy and Confidentiality 10 (1).*
* Holohan, N., Braghin, S., Mac Aonghusa, P. and Levacher, K., 2019. [Diffprivlib: the IBM Differential Privacy Library](https://arxiv.org/abs/1907.02444). *ArXiv e-prints 1907.02444 [cs.CR].*
* Machanavajjhala, A., et al. l-diversity: Privacy beyond k-anonymity. *TKDD*, 2007.
* Dwork, C., et al. Calibrating noise to sensitivity in private data analysis. *TCC*, 2006.
