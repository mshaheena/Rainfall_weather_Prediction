import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Load preprocessing models
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

# Load trained models
catboost_model = CatBoostClassifier()
catboost_model.load_model("catboost_model.cbm")

lr_model = joblib.load("lr_model.pkl")

def preprocess_data(df):
    """ Apply the same feature engineering as in train_model.py """
    df['pressure_cut'] = df['pressure'] - 1000
    df['cloud_sunshine'] = df['cloud'] * df['sunshine'] / (np.sqrt(df['cloud']**2 + df['sunshine']**2) + 1e-8)
    df['humidity_dewpoint'] = df['humidity'] * df['sunshine'] / (np.sqrt(df['humidity']**2 + df['sunshine']**2) + 1e-8)
    
    df = df.rename(columns={'maxtemp': 'x1', 'temperature': 'x2', 'mintemp': 'x3', 'dewpoint': 'x4',
                            'humidity': 'x5', 'cloud': 'x6', 'sunshine': 'x7', 'winddirection': 'x8',
                            'windspeed': 'x9', 'pressure_cut': 'x10', 'cloud_sunshine': 'x11', 'humidity_dewpoint': 'x12'})

    features = [col for col in df.columns if col not in ['id', 'rainfall']]
    
    # Apply transformations
    df = pd.DataFrame(imputer.transform(df[features]), columns=features)
    df = pd.DataFrame(scaler.transform(df), columns=features)
    df = selector.transform(df)
    
    return df

def predict_rainfall(df):
    """ Predicts rainfall probability using trained models """
    processed_data = preprocess_data(df)
    
    catboost_preds = catboost_model.predict_proba(processed_data)[:, 1]
    lr_preds = lr_model.predict_proba(processed_data)[:, 1]
    
    final_preds = (catboost_preds + lr_preds) / 2
    return np.clip(final_preds, 0.01, 0.99)  # Ensure predictions are within a reasonable range

if __name__ == "__main__":
    # Example usage
    test_df = pd.read_csv("test.csv")
    predictions = predict_rainfall(test_df)
    test_df["rainfall"] = predictions
    test_df[["id", "rainfall"]].to_csv("submission.csv", index=False)
    print("Predictions saved to submission.csv")
