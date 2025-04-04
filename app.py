import streamlit as st
import pandas as pd
from inference import predict_rainfall

st.title("🌧️ Rainfall Prediction App")
st.markdown("Upload a CSV file with weather features to predict rainfall probability.")

uploaded_file = st.file_uploader("Choose your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.dataframe(input_data.head())

    if st.button("Predict Rainfall"):
        preds = predict_rainfall(input_data)
        input_data['Predicted_Rainfall_Probability'] = preds
        st.success("Prediction Complete!")
        st.dataframe(input_data)
