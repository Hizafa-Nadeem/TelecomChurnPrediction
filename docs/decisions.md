# Decision log

One entry per decision that changes what the specs say. Newest last. Results behind each decision are in `reports/`.

## 1. Use plain CatBoost, no class weights, no SMOTE

- **Date:** 2026-10-09
- **Context:** `docs/development_strategy.md` Step 6 named class weights (`auto_class_weights`) as the primary way to handle the 14.1% churn rate, with SMOTE only inside CV as a comparison.
- **What was tested:** plain CatBoost, CatBoost with `auto_class_weights="Balanced"`, and CatBoost with SMOTENC inside CV, on the same 15 CV folds of the training split (`reports/baselines.md`).
- **Result:** PR-AUC 0.8994 (plain), 0.8990 (weighted), 0.8975 (SMOTENC) against a fold standard deviation of about 0.022. ROC-AUC 0.924 / 0.926 / 0.926. No variant is better in any meaningful way.
- **Decision:** use plain CatBoost with no class weights and no resampling. Do not tune `auto_class_weights` in Step 8. Drop `imbalanced-learn` use from the training path (the comparison code stays in `train.py` as the record of the experiment).
- **Why:**
  - Both PR-AUC and ROC-AUC rank customers, and neither weighting nor oversampling improved the ranking.
  - Weights inflate the predicted probabilities, so Step 9 would need to undo that distortion before the business-cost threshold can use them. Plain probabilities start closer to correct.
  - Fewer moving parts to explain, tune and serve: no synthetic data and no sampler in the training pipeline.
  - The threshold is chosen in Step 9 from expected retention value, so imbalance is handled where it matters, at the decision threshold.
- **Consequences:**
  - Probabilities come out of CatBoost closer to the true rates; calibration (Step 9) still checks them.
  - Recheck if the class balance or the data source changes (for example real operator data in v3).
  - `docs/project_spec.md` row 6 and `docs/development_strategy.md` Steps 6 and 8 were updated to match.
