# Hyperparameter tuning (Step 8)

Code: `src/churn/tune.py` (`make tune`). Results are in MLflow, experiment `telecom-churn`: a `tuning` run with one nested `trial_N` run per trial. The best parameters are saved in `reports/tuning/best_params.json` and are read by the calibration step.

## Setup

| Item | Value |
|---|---|
| Model | Plain CatBoost (decision in `reports/baselines.md`) on the shared feature pipeline |
| Data | Training split only; the test set was not read |
| Search | Optuna TPE sampler (10 random start trials, seed 42), MedianPruner |
| Trials | 30, scored by PR-AUC with 5-fold CV, 1 repeat |
| Space | `depth` 4-8, `learning_rate` 0.02-0.3 (log), `l2_leaf_reg` 1-30 (log), `iterations` 200-800 in steps of 100 |
| Final score | Best parameters re-scored with the full repeated CV (5 folds x 3 repeats = 15 fits), the same folds as the baseline |
| Cost | 2,503 s in total (42 min), about 80 s per trial |

## Result

Best trial: `depth` 5, `learning_rate` 0.124, `l2_leaf_reg` 17.4, `iterations` 200.

| | Baseline CatBoost | Tuned CatBoost |
|---|---|---|
| CV PR-AUC (15 folds) | 0.8994 +/- 0.0256 | 0.9016 +/- 0.0237 |
| CV ROC-AUC | 0.9244 +/- 0.0229 | 0.9278 +/- 0.0197 |
| Train PR-AUC | 0.9962 | 0.9718 |
| Train-vs-CV PR-AUC gap | 0.097 | 0.070 |
| Iterations | 1,000 | 200 |

## Read-out

- **No meaningful accuracy gain.** PR-AUC rises by 0.002 and ROC-AUC by 0.003, far inside the fold-to-fold spread of about 0.024. Tuning did not move the model out of the baseline's noise band.
- **The search landscape is flat.** All 30 trials scored between 0.8989 and 0.9030 (median 0.9005). No depth or iteration setting stood out; depths 4 and 5 were marginally best. The 30-trial search score of 0.903 is the best of 30 noisy results, so it is optimistic; the 15-fold re-score (0.9016) is the figure to quote.
- **What did improve:** the model overfits less (gap 0.097 to 0.070, from stronger regularization with `l2_leaf_reg` 17), and it is a fifth of the size (200 vs 1,000 iterations), so training and prediction are about 5x faster.
- **No trial was pruned.** The scores were too close for the pruner to find a clearly bad trial.

## Choices and deviations

- **30 trials instead of the spec's 100-200.** With 4 parameters and a flat landscape, more trials would mostly fit noise, and each trial costs about 80 s on CPU. The count is `tune.n_trials` in `params.yaml`.
- **`iterations` is tuned directly instead of using early stopping.** Stopping on the validation fold being scored would make the CV score optimistic.
- **Class weights are not tuned**, following the decision in `reports/baselines.md`.

## Limits

- These are CV estimates on training data. The single test-set evaluation is Step 10.
- Results depend on the seed and on the 5-fold, 1-repeat CV used in the search; a different seed could pick a different best trial with nearly the same score.
