import streamlit as st
import pandas as pd
from inference import predict_rainfall

st.title("🌧️ AI Rainfall Prediction App")

st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(df.head())

    if st.button("Predict Rainfall"):
        predictions = predict_rainfall(df)
        df["rainfall"] = predictions
        st.write("### Predictions:")
        st.dataframe(df[["id", "rainfall"]])

        st.download_button(
            label="Download Predictions",
            data=df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
