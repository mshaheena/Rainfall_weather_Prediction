import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib

# Load preprocessing objects and models
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')
catboost_model = CatBoostClassifier()
catboost_model.load_model('catboost_model.cbm')
lr_model = joblib.load('lr_model.pkl')

# Feature engineering function (same as training)
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

# Streamlit app
st.title("Rainfall Prediction App")
st.write("Enter weather data to predict the probability of rainfall.")

# Input fields
day = st.number_input("Day", min_value=1, max_value=365, value=1)
pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)
maxtemp = st.number_input("Max Temperature (°C)", min_value=0.0, max_value=50.0, value=26.0)
temparature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=24.0)
mintemp = st.number_input("Min Temperature (°C)", min_value=0.0, max_value=50.0, value=22.0)
dewpoint = st.number_input("Dew Point (°C)", min_value=-10.0, max_value=40.0, value=20.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
cloud = st.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=75.0)
sunshine = st.number_input("Sunshine (hours)", min_value=0.0, max_value=24.0, value=3.7)
winddirection = st.number_input("Wind Direction (degrees)", min_value=0.0, max_value=360.0, value=105.0)
windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=22.0)

# Create input DataFrame
input_data = pd.DataFrame({
    'day': [day], 'pressure': [pressure], 'maxtemp': [maxtemp], 'temparature': [temparature],
    'mintemp': [mintemp], 'dewpoint': [dewpoint], 'humidity': [humidity], 'cloud': [cloud],
    'sunshine': [sunshine], 'winddirection': [winddirection], 'windspeed': [windspeed]
})

# Process input
input_processed = enhance_features(input_data)
features = [col for col in input_processed.columns if col != 'day']
input_imputed = pd.DataFrame(imputer.transform(input_processed[features]), columns=features)
input_scaled = scaler.transform(input_imputed)
input_selected = selector.transform(input_scaled)

# Predict
catboost_pred = catboost_model.predict_proba(input_imputed)[:, 1][0]
lr_pred = lr_model.predict_proba(input_selected)[:, 1][0]
final_pred = np.clip((catboost_pred + lr_pred) / 2, 0.01, 0.99)

# Display result
st.write(f"**Probability of Rainfall:** {final_pred:.2%}")
if final_pred > 0.5:
    st.write("**Prediction:** Rain is likely.")
else:
    st.write("**Prediction:** No rain expected.")