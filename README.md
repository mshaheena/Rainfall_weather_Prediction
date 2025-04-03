# 🌧️ AI-Powered Rainfall Prediction App

Welcome to the **AI-Powered Rainfall Prediction App**! This project leverages **Machine Learning (CatBoost, Logistic Regression)** to predict rainfall probability based on weather conditions.

## 🚀 Project Overview

This project includes:
✔ **Data Preprocessing & Feature Engineering**  
✔ **ML Models: CatBoost & Logistic Regression**  
✔ **Streamlit Web App for Predictions**  
✔ **Deployable on Streamlit Cloud**  

---

## 📂 Folder Structure

Rainfall_Prediction/ │── app.py # Streamlit app (Frontend UI) │── train_model.py # Model training script │── inference.py # Prediction script │── requirements.txt # Dependencies │── imputer.pkl # Saved imputer model │── scaler.pkl # Saved scaler model │── selector.pkl # Saved feature selector model │── catboost_model.cbm # Trained CatBoost model │── lr_model.pkl # Trained Logistic Regression model │── dataset/ │ ├── train.csv # Training dataset │ ├── test.csv # Test dataset │ ├── sample_submission.csv # Sample submission file

## 🔧 Installation & Setup

1️⃣ **Clone the repository**  

git clone https://github.com/mshaheena/Rainfall_Prediction.git
cd Rainfall_Prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model (optional, models already provided)
python train_model.py

4️⃣ Run the Streamlit app
streamlit run app.py

📊 How It Works
1️⃣ Train the Model
train_model.py preprocesses data, trains CatBoost & Logistic Regression, and saves the models.

2️⃣ Make Predictions
inference.py loads trained models to predict rainfall from test data.

3️⃣ Interactive Web App
app.py provides a user-friendly Streamlit UI to upload datasets and get predictions.

🎯 Future Improvements
🔹 Improve model accuracy with additional weather data
🔹 Enhance visualization in Streamlit
🔹 Add API support for real-time predictions

📞 Contact & Support
💡 Developed by: Mallela Shaheena
🔗 LinkedIn: https://www.linkedin.com/in/m-shaheena
📧 Email: mshaheena8838@gmail.com

Give a ⭐ if you found this project useful!
