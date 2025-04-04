import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier

# Load model artifacts
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
lr_model = joblib.load("lr_model.pkl")
catboost_model = CatBoostClassifier()
catboost_model.load_model("catboost_model.cbm")

def enhance_features(df):
    df = df.copy()
    df['pressure_cut'] = df['pressure'] - 1000
    df['cloud_sunshine'] = df['cloud'] * df['sunshine'] / (np.sqrt(df['cloud']**2 + df['sunshine']**2) + 1e-8)
    df['humidity_dewpoint'] = df['humidity'] * df['sunshine'] / (np.sqrt(df['humidity']**2 + df['sunshine']**2) + 1e-8)
    df = df.rename(columns={
        'maxtemp': 'x1', 'temparature': 'x2', 'mintemp': 'x3', 'dewpoint': 'x4',
        'humidity': 'x5', 'cloud': 'x6', 'sunshine': 'x7', 'winddirection': 'x8',
        'windspeed': 'x9', 'pressure_cut': 'x10', 'cloud_sunshine': 'x11', 'humidity_dewpoint': 'x12'
    })
    df['_2_1'] = np.sqrt((df['x1'] - df['x3'])**2 + (df['x2'] - df['x4'])**2)
    df['_4_2'] = np.sqrt((df['x1'] - df['x9'])**2 + (df['x2'] - df['x10'])**2 +
                         (df['x3'] - df['x11'])**2 + (df['x4'] - df['x12'])**2)
    df['temp_range'] = df['x1'] - df['x3']
    df['humidity_pressure_ratio'] = df['x5'] / (df['pressure'] + 0.0001)
    df['wind_cloud_interaction'] = df['x9'] * df['x6']
    df['temp_dewpoint_diff'] = df['x2'] - df['x4']
    df['sunshine_humidity_ratio'] = df['x7'] / (df['x5'] + 0.0001)
    return df

def predict_rainfall(input_df):
    df = enhance_features(input_df)
    X = imputer.transform(df)
    X = scaler.transform(X)
    X = selector.transform(X)
    
    cat_pred = catboost_model.predict_proba(X)[:, 1]
    lr_pred = lr_model.predict_proba(X)[:, 1]
    final_pred = (cat_pred + lr_pred) / 2
    return np.clip(final_pred, 0.01, 0.99)
