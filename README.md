# ai-privacy-toolkit (extended minimization privacy controls)

This fork extends the **minimization** module with **four security features**: **(1) Differential Privacy (DP)** at a concrete release point in the pipeline, **(2) a minimum NCP privacy floor** that prevents privacy from degrading during accuracy optimization, **(3) a homogeneity guard** to mitigate homogeneity/l-diversity failures at the cell level, and **(4) risk-driven enforcement** using membership-inference attacks already implemented in the toolkit. Together, these address a common gap in the state-of-the-art: many minimization/anonymization pipelines *measure* privacy (or report information-loss) but do not actively **enforce** privacy constraints once utility-driven optimization begins. The base toolkit is described in Goldsteen et al. (SoftwareX, 2023). DP follows the standard \((\varepsilon,\delta)\)-DP definition (Dwork et al., 2006), and homogeneity protection aligns with k-anonymity/l-diversity style failure modes (Machanavajjhala et al., 2007).

## New security features (code + purpose)

### 1) Differential Privacy (DP) for the surrogate tree release point
- **Files:** `apt/minimization/dp_mechanism.py`, `apt/minimization/minimizer.py` (`_privatize_tree_thresholds`, DP-safe logic in `_attach_cells_representatives`)
- **What it does:** After fitting the surrogate decision tree, we apply Laplace noise (via `diffprivlib`) to **non-leaf split thresholds** and use truncation bounds to keep noisy thresholds inside valid feature ranges. This reduces leakage from exact split points while keeping the tree usable. Since DP-noised thresholds may create empty regions, `_attach_cells_representatives` contains DP-safe fallback logic (relaxed matching + nearest-row selection) to ensure the pipeline remains stable and does not crash or leak outliers through failures.

### 2) Minimum NCP privacy floor (privacy “must not fall below X”)
- **Files:** `apt/minimization/privacy_floor.py`, `apt/minimization/minimizer.py` (privacy-floor integration, snapshot/restore helpers)
- **What it does:** We enforce a minimum privacy level using the toolkit’s NCP information-loss metric as an operational proxy. A baseline NCP is computed once, then an **effective floor** is enforced either as an absolute `min_ncp` or a relative floor `min_ncp = alpha * baseline_ncp`. During the “improve accuracy” loop (feature removal from constraints) and the “improve generalization” loop (tree pruning), the algorithm **snapshots state** and rolls back if the next step would reduce NCP below the floor.

### 3) Homogeneity guard (k / l-diversity / entropy per cell)
- **Files:** `apt/minimization/homogeneity_guard.py`, `apt/minimization/minimizer.py` (`_collect_homogeneity_violations`, `_enforce_homogeneity_guard`)
- **What it does:** Homogeneity attacks occur when a generalized group (“cell”) reveals sensitive outcomes because nearly everyone in the cell shares the same label. We compute per-cell label statistics (cell size, label diversity, entropy) and enforce constraints such as `min_cell_size`, `min_label_diversity`, and optional `min_entropy`. Enforcement modes are `warn`, `raise`, or `auto` (bounded pruning/merging of cells while respecting an accuracy tolerance).

### 4) Risk-driven enforcement (membership inference attacks as a controller)
- **Files:** `apt/minimization/privacy_risk_enforcer.py`, `apt/minimization/minimizer.py` (`_enforce_privacy_risk`), uses attacks in `apt/risk/data_assessment/*`
- **What it does:** We turn the toolkit’s existing membership inference assessments into an **enforcement loop**. After producing a generalized dataset, we evaluate membership risk (classification- or KNN-based). We enforce thresholds on (a) `risk_score` (normalized ratio metric), (b) absolute member/non-member AUC caps, and (c) an explicit “no warning” quality gate. In `auto` mode the minimizer searches over additional pruning levels (bounded by `max_risk_prune_level` and `risk_accuracy_tolerance`) and rolls back to the best feasible state if constraints cannot be satisfied.

## Execution flow (minimizer `fit()` call sequence)

`fit()` → encode categoricals → fit surrogate tree → **(DP)** `_privatize_tree_thresholds` → derive cells → `_attach_cells_representatives` (DP-safe representatives) → compute accuracy → **homogeneity guard** (`_enforce_homogeneity_guard`) → compute baseline NCP → **privacy floor** enforcement during pruning/feature-removal (snapshot/restore) → **risk-driven enforcement** (`_enforce_privacy_risk`) → final metrics stored in `self.ncp`. Each stage prints progress to the terminal (e.g., `[DP]`, `[PRIVACY-FLOOR]`, `[HOMOGENEITY-GUARD]`, `[RISK]`) for traceability and reproducibility.

## Reproducible run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install diffprivlib