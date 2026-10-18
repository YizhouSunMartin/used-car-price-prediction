"""Reproducible used-car price baseline models without target leakage."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC = ['Mileage', 'Age', 'Horsepower', 'Displacement', 'Weight', 'Gears',
           'Previous_Owners', 'Inspection_New', 'Cons_Comb']
CATEGORICAL = ['Make_Model', 'Body_Type', 'Type', 'Fuel', 'Vat', 'Paint_Type',
               'Upholstery_Type', 'Gearing_Type', 'Drive_Chain', 'Age_Category']
DROP_LEAKAGE = ['Price_per_km']  # Computed using Price; NEVER use as a predictor.


def run(data_path: Path, out: Path):
    df = pd.read_csv(data_path, low_memory=False)
    if 'Price' not in df:
        raise ValueError('Input must contain a numeric Price column (cleaned Stage 1 CSV).')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Price']).copy()
    # Explicitly whitelist features; prevents accidental inclusion of target-derived columns.
    numeric = [x for x in NUMERIC if x in df]
    categorical = [x for x in CATEGORICAL if x in df]
    if not numeric:
        raise ValueError('No expected numeric features found; supply cleaned Stage 1 data.')
    X = df[numeric + categorical].copy()
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    for col in categorical:
        X[col] = X[col].astype('string').fillna('Unknown').astype(str)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric_pipe = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=False)),
                             ('scale', StandardScaler())])
    onehot = ColumnTransformer([
        ('num', numeric_pipe, numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=10, sparse_output=True), categorical),
    ])
    ordinal = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), numeric),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                               encoded_missing_value=-1), categorical),
    ])
    models = {
        'Ridge (linear baseline)': Pipeline([('preprocess', onehot), ('model', Ridge(alpha=10))]),
        'Random Forest': Pipeline([('preprocess', onehot), ('model', RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1))]),
        'Histogram Gradient Boosting': Pipeline([('preprocess', ordinal), ('model', HistGradientBoostingRegressor(
            max_iter=200, max_leaf_nodes=31, learning_rate=0.08, random_state=42))]),
    }
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        row = {'model': name, 'MAE': round(float(mean_absolute_error(y_test, pred)), 2),
               'RMSE': round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
               'R2': round(float(r2_score(y_test, pred)), 4)}
        print(row, flush=True)
        results.append(row)
    pd.DataFrame(results).to_csv(out / 'metrics.csv', index=False)
    metadata = {'rows': len(df), 'train_rows': len(X_train), 'test_rows': len(X_test),
                'numeric_features': numeric, 'categorical_features': categorical,
                'excluded_target_derived_features': DROP_LEAKAGE,
                'random_state': 42, 'test_size': 0.2,
                'note': 'New reproducible baselines; not a reproduction of the Assignment 2 reported scores.'}
    (out / 'run_info.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/auto_price_clean.csv'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    run(args.data, args.out)
