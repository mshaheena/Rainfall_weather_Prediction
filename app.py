import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from inference import predict_rainfall  # Import prediction function

# Streamlit App Title
st.set_page_config(page_title="Rainfall Prediction App", layout="wide")

st.title("🌧️ Rainfall Prediction App")
st.write("Upload a CSV file with weather features to predict rainfall probability.")

# File Upload Section
uploaded_file = st.file_uploader("Choose your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)

    # Display Uploaded Data
    st.subheader("Uploaded Data Preview")
    st.write(input_data.head())

    # Show Summary Statistics
    st.subheader("📊 Dataset Summary Statistics")
    st.write(input_data.describe())

    # Heatmap of Feature Correlations
    st.subheader("🔹 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(input_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Histogram of Key Features
    st.subheader("📌 Feature Distributions")
    feature_cols = ['temperature', 'humidity', 'cloud', 'pressure', 'sunshine']

    for col in feature_cols:
        fig, ax = plt.subplots()
        sns.histplot(input_data[col], kde=True, bins=30, color="blue")
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)

    # Predict Button
    if st.button("Predict Rainfall"):
        preds = predict_rainfall(input_data)  # Get predictions

        # Display Predictions
        st.subheader("🌧️ Prediction Results with Confidence Scores")
        result_df = input_data.copy()
        result_df["Rainfall Probability"] = preds  # Add probability scores
        st.write(result_df)

        # Show bar chart of rainfall probabilities
        st.subheader("📈 Rainfall Probability Distribution")
        fig, ax = plt.subplots()
        sns.histplot(preds, bins=20, kde=True, color="red")
        plt.xlabel("Rainfall Probability")
        plt.ylabel("Frequency")
        plt.title("Distribution of Predicted Rainfall Probabilities")
        st.pyplot(fig)
