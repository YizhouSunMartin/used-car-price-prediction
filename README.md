# Used Car Price Prediction

A Python machine-learning project exploring used-car valuation with a cleaned European used-car listings dataset. Based on an earlier university group project, with new reconstruction code for model training and evaluation (not the original Assignment 2 implementation).

## What it does

- Uses the cleaned Auto Prices dataset produced in Assignment 1. Original shared coursework notebooks are intentionally excluded from the public package.
- Compares a regularized linear baseline, Random Forest, and Histogram Gradient Boosting.
- Evaluates models on an 80/20 holdout split using MAE, RMSE, and R².
- **Avoids target leakage:** `Price_per_km` is calculated from the target `Price` and is explicitly excluded from training.

## Results (new runs, not the original assignment scores)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge (linear baseline) | 2384.08 | 5100.60 | 0.5951 |
| Random Forest | 1825.93 | 4763.83 | 0.6468 |
| Histogram Gradient Boosting | 2088.82 | 4891.46 | 0.6276 |

These are results from the cleaned dataset (14,995 records), with `random_state=42` and an 80/20 split. They **must not** be presented as the results of the original Assignment 2 models. Results are dataset- and environment-dependent. The best RMSE in this new experiment is from Random Forest; the original report's stronger scores may have been inflated by a target-derived feature, but this cannot be confirmed without its original training code.

## Run locally

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
python src/train.py --data data/auto_price_clean.csv
```

Place your cleaned Stage 1 Auto Price CSV at `data/auto_price_clean.csv` before running. The original group file had a `.xls` suffix but CSV content; it must be renamed to `.csv` (not converted as an Excel workbook). The data is not distributed in this public-ready repository pending permission to redistribute the course dataset.

Results are written to `results/metrics.csv` and `results/run_info.json`.

## Repository structure

- `src/train.py`: new, reproducible baseline training and evaluation.
- `notebooks/`: not included in the public package because the original notebooks belong to group coursework.
- `results/`: generated holdout metrics and run metadata.
- `docs/PROJECT_NOTES.md`: provenance, limitations, and contributions.
- `data/`: local data only, not included in the public-ready archive.

## Limitations

This dataset primarily covers European used-car listings; pricing varies by region and time. There may be duplicates, outliers, and data-quality issues. The cleaned dataset was produced as part of the original group work, while the new training code was added later. The original Assignment 2 training code was not available, so its reported results have not been independently reproduced. No cross-validation or hyperparameter search is claimed for the new experiments.

## Academic attribution

This project is based on COMP5310 group coursework at the University of Sydney. The Stage 1 notebooks and cleaned data are from group work. The original contribution statement attributes the linear regression model, report writing, code-result integration, and formatting to Yizhou Sun; the original Random Forest and Gradient Boosting work and data cleaning to Jundi Huang. The `src/train.py` implementation here is a newly written, separate version. Publish shared notebooks only with collaborators' consent and subject to university course policies.

## Before publishing

Confirm the university allows public coursework-derived solutions and ensure collaborator attribution is accurate. Do not add the original assignment PDF, shared notebooks, student IDs, or course-provided datasets without appropriate permissions.
