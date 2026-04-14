import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD MODELS
# =========================
best_model = joblib.load('best_model_pipeline.pkl')  # pipeline
dt_model = joblib.load('dt_model.pkl')
gbr_model = joblib.load('gbr_model.pkl')

# =========================
# MODEL METRICS
# =========================
models = ['Decision Tree', 'Gradient Boosting']
mae = [33.08, 26.29]
rmse = [55.74, 43.10]

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="Bike Rental Prediction", layout="centered")

st.title("🚴 Bike Rental Prediction App")
st.write("Predict bike rental demand and compare ML models")

# =========================
# MODEL PERFORMANCE
# =========================
st.subheader("📊 Model Performance Comparison")

fig, ax = plt.subplots()

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, mae, width, label='MAE')
ax.bar(x + width/2, rmse, width, label='RMSE')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Error")
ax.legend()

st.pyplot(fig)

st.success("✅ Best Model: Gradient Boosting (used for prediction)")

# =========================
# USER INPUT
# =========================
st.subheader("🧮 Enter Input Features")

col1, col2 = st.columns(2)

with col1:
    season = st.selectbox("Season (1:Spring, 2:Summer, 3:Fall, 4:Winter)", [1,2,3,4])
    yr = st.selectbox("Year (0:2011, 1:2012)", [0,1])
    mnth = st.slider("Month", 1, 12, 6)
    holiday = st.selectbox("Holiday (0:No, 1:Yes)", [0,1])
    weekday = st.slider("Weekday (0-6)", 0, 6, 3)
    workingday = st.selectbox("Working Day (0:No, 1:Yes)", [0,1])

with col2:
    weathersit = st.selectbox("Weather (1:Clear, 2:Mist, 3:Light Snow, 4:Heavy Rain)", [1,2,3,4])
    temp = st.slider("Temperature", 0.0, 50.0, 25.0)
    atemp = st.slider("Feels Like Temp", 0.0, 50.0, 25.0)
    hum = st.slider("Humidity", 0.0, 100.0, 50.0)
    windspeed = st.slider("Windspeed", 0.0, 50.0, 10.0)

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Bike Rentals"):

    input_data = np.array([[season, yr, mnth, holiday, weekday,
                            workingday, weathersit, temp,
                            atemp, hum, windspeed]])

    prediction = best_model.predict(input_data)

    st.success(f"🚴 Predicted Bike Rentals: {int(prediction[0])}")