# EDA findings (training split: 4,000 rows, 566 churners, 14.15%)

Source: `notebooks/01_eda.ipynb`. Target-based statistics use the training split only; data-quality checks use the full cleaned file. The numbers below were re-checked against `data/processed/train.csv`.

## Key findings

1. **International plan** is the strongest binary signal: 42.4% churn with the plan (n=380) vs 11.2% without (n=3,620); chi-square p = 3e-61, Cramer's V = 0.26.
2. **Service calls** are flat at about 11% for 0-3 calls, then jump: 43.7% (4), 57.1% (5), 67.9% (6). Pooled 4+ is 50.0% (n=318) vs 11.1%. Counts at 7+ calls are 11 or fewer, too small to quote.
3. **Day minutes** is the strongest numeric feature, but not monotone: churn is 11.6% / 8.4% / 22.5% across tertiles. The jump sits in the top quartile and is mostly among customers without voicemail (36.5% vs 6.0% with voicemail).
4. **Segments:** international plan x voicemail: 44.8% (plan, no voicemail), 35.9% (plan, voicemail), 13.6% (no plan, no voicemail), 4.4% (no plan, voicemail). Service calls 4+ with the plan: 67.6% (n=34); 4+ without: 47.9%; 0-3 with the plan: 39.9%; 0-3 without: 8.1%.
5. **`area_code`** has no signal (p = 0.81, V = 0.00; churn 13.6% / 14.4% / 14.2%).
6. **`state`** is weak and noisy: overall chi-square p = 9e-5, but the top rate (29.2%, state 47, n=72) is inside the chance range for the maximum of 51 groups (null 95th percentile 28.6%). States hold 41-122 training customers; 4 have fewer than 60.
7. **Redundancy:** `total_*_charge` vs `total_*_minutes` r = 1.000000 (day, eve), 0.999999 (night), 0.999993 (intl); VIF 6e4 to 1e7. After dropping the four charge columns the maximum VIF is 11.1 (`voice_mail_plan` vs `number_vmail_messages`); everything else is about 1.
8. **Data quality (full file):** no missing values, duplicates or negatives. One row (index 4902) has `voice_mail_plan` = 1 with 0 messages. Minutes can be 0, so minutes-per-call features need a zero guard. `number_vmail_messages` is 73.5% zeros.
9. **Mutual information ranking:** day charge/minutes (0.06), service calls (0.036), international plan (0.026), state (0.012).
10. **Imbalance:** no-skill PR-AUC is 0.141 and each 5-fold CV fold holds about 113 churners.

## Differences from `docs/development_strategy.md`

- "44-65% from 4 calls" holds for 4-6 calls; it cannot be stated for 7+ calls.
- "Heavy daytime use" is a top-quartile effect and mostly disappears when the customer has voicemail.
- `has_voicemail_usage` (messages > 0) duplicates `voice_mail_plan` except for the one row above, so it adds nothing.
- `account_length` shows no visible separation between classes, so buckets are not justified.
- Everything else reproduces: 14.1% churn, 42% vs 11% for the plan, r = 1.00 charge/minutes, no missing values or duplicates.

## Feeds Step 5 (features)

- Drop the four `total_*_charge` columns; compute `total_charge` only as the revenue proxy for the Step 9 cost model.
- Build `high_service_calls` (>= 4).
- Add a day-minutes x no-voicemail interaction (heavy-day-user flag) for the logistic model.
- Keep `total_minutes`, `total_calls` and the day share of minutes.
- Skip `account_length` buckets and `has_voicemail_usage`.
- Keep `number_vmail_messages` and `voice_mail_plan` (VIF 11 is mild).
- Treat `state` as a regularised, rare-level-grouped categorical (CatBoost `cat_features`; grouping or shrunken target encoding for the linear model); `area_code` is a likely drop or low-value feature.
- Guard minutes-per-call features against 0 calls.

## Feeds Step 6 (imbalance)

Class weights first; SMOTE only inside CV, as a comparison.

## Notes

- `reports/eda/profile_train.html` is the minimal ydata-profiling report (1.1 MB). The full report is 14 MB and was not committed.
