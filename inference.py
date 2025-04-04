# inference.py

import pandas as pd
import joblib
from catboost import CatBoostClassifier

# Load trained components
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

def preprocess_input(data):
    df = data.copy()
    df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
    df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df.columns)
    return df_scaled

def predict_rainfall(input_df):
    preprocessed = preprocess_input(input_df)
    prediction = model.predict(preprocessed)
    prediction_proba = model.predict_proba(preprocessed)
    return prediction, prediction_proba
