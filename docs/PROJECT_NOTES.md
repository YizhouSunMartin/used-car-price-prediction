# Project provenance and checks

## Relationship between assignments

- Assignment 1: data inspection, cleaning, and EDA of Auto Prices, Diabetes Diagnosis, and Forest Cover datasets.
- Assignment 2: used-car price regression model comparisons using the cleaned Auto Prices dataset.
- This repository focuses on **Auto Prices only**. It includes original Stage 1 Auto Price notebooks and newly developed model code; it does not claim to include the missing original Stage 2 model-training scripts.

## Leakage check

The original cleaned data includes `Price_per_km = Price / Mileage`. This is not known for a new car listing without first knowing the price being predicted. It must not be used to predict `Price`. The new training script uses an explicit predictor allowlist excluding this column.

## Publishing checklist

- [ ] Confirm group member permission before publishing shared notebooks or report content.
- [ ] Confirm university course policy permits publication of assignment solutions.
- [ ] Confirm redistribution license before adding any course-provided dataset to GitHub.
- [ ] Remove student IDs and other personal information from any additional documents.
- [ ] Run `python src/train.py --data data/auto_price_clean.csv` locally.
- [ ] Attribute original group work accurately.

## Next improvements

- Cross-validation and hyperparameter tuning using leakage-safe preprocessing.
- Examine outliers and possible duplicate listings.
- Add diagnostic residual plots and error analysis by car make and price range.
- Optionally build a small Streamlit prediction UI after validating model behavior.
