# Data Contract: Orange/BigML telecom churn

Enforced by `src/churn/validation.py` (pandera schema + leakage check) in `python -m churn.data`.

## Problem framing

| Item | Definition |
|---|---|
| Unit | One row per customer (snapshot, no date column) |
| Churn | `class` = 1 (707 of 5,000 customers, 14.1%) |
| Horizon | Not stated in the source; assumed "next billing period" |
| Retention action | Offer or call for customers flagged above the cost-optimal threshold |

## Columns (after cleaning)

| Column | Type | Allowed values | Notes |
|---|---|---|---|
| `state` | categorical (string) | "0"-"50" | Anonymous integer codes; do not interpret |
| `area_code` | categorical (string) | "408", "415", "510" | |
| `account_length` | int | >= 1 | Unit not documented |
| `international_plan` | int | 0, 1 | |
| `voice_mail_plan` | int | 0, 1 | |
| `number_vmail_messages` | int | >= 0 | Zero-inflated; > 0 matches `voice_mail_plan` = 1 in all but one row |
| `number_customer_service_calls` | int | 0-9 | Count |
| `total_{day,eve,night,intl}_minutes` | float | >= 0 | |
| `total_{day,eve,night,intl}_calls` | int | >= 0 | |
| `total_{day,eve,night,intl}_charge` | float | >= 0 | Fixed rate per minute, so redundant with minutes (dropped in Step 5) |
| `class` | int | 0, 1 | Target |

19 predictors (15 numeric, 4 categorical-like) plus the target.

## Dropped columns

| Column | Reason |
|---|---|
| `phone_number` | Identifier (unique 0-4,999, behaves like a row index). Never a feature, never logged. |

## Rules checked on every run

- Exact column set (strict schema) and the types and ranges above.
- No missing values; no duplicate rows.
- Leakage check: the run fails if any integer or text feature is unique per row, or if any single feature separates the target perfectly.

## Files

| Path | Content |
|---|---|
| `data/raw/churn.csv` | Original OpenML 40701 download. Never edited. |
| `data/interim/churn.csv` | Cleaned frame. Read it with `churn.data.load_interim` to restore categorical types. |
| `data/interim/validation_report.json` | Row count, churn rate, dropped columns, checks passed |
