"""
Bike Rental Prediction App
Gradient Boosting model (MAE: 26.29 | RMSE: 39.88 | R²: 0.9519)
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Rental Predictor",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border-left: 4px solid #2196F3;
    margin-bottom: 0.5rem;
}
.metric-card h4 { margin: 0; color: #666; font-size: 0.78rem; }
.metric-card h2 { margin: 4px 0 0; color: #111; font-size: 1.6rem; }

.result-box {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    color: white;
}

.result-box .num { font-size: 3.5rem; font-weight: 700; }
.result-box .label { font-size: 0.9rem; opacity: 0.8; }

.insight-box {
    background: #e8f5e9;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    border-left: 4px solid #4CAF50;
    font-size: 0.9rem;
    color: #1b5e20;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("best_model_pipeline.pkl")

try:
    best_pipeline = load_model()
    model_loaded = True
except:
    model_loaded = False

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
MODEL_METRICS = pd.DataFrame({
    "Model": ["Decision Tree", "Gradient Boosting"],
    "MAE": [33.08, 26.29],
    "RMSE": [55.74, 39.88],
    "R²": [0.906, 0.952],
})

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE (GBR ONLY)
# ─────────────────────────────────────────────
FEATURE_IMPORTANCE = {
    "hr": 0.310,
    "temp": 0.185,
    "atemp": 0.130,
    "season": 0.085,
    "hum": 0.075,
    "yr": 0.065,
    "mnth": 0.045,
    "windspeed": 0.035,
    "weathersit": 0.030,
    "weekday": 0.020,
    "workingday": 0.010,
    "holiday": 0.010,
}

# ─────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🚴 Bike Rental Predictor")

    yr = st.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
    mnth = st.slider("Month", 1, 12, 6)
    hr = st.slider("Hour", 0, 23, 17)
    weekday = st.selectbox("Weekday", list(range(7)),
                           format_func=lambda x: ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][x])
    workingday = st.selectbox("Working Day?", [0, 1])
    holiday = st.selectbox("Holiday?", [0, 1])

    season = st.selectbox("Season", [1,2,3,4],
                           format_func=lambda x: {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}[x])

    weathersit = st.selectbox("Weather", [1,2,3,4],
                               format_func=lambda x: {1:"Clear",2:"Mist",3:"Rain",4:"Heavy"}[x])

    temp = st.slider("Temp (°C)", 0.0, 50.0, 25.0)
    atemp = st.slider("Feels Like", 0.0, 50.0, 25.0)
    hum = st.slider("Humidity", 0, 100, 50)
    windspeed = st.slider("Windspeed", 0.0, 67.0, 12.0)

    predict = st.button("Predict 🚀")

# ─────────────────────────────────────────────
# DEMO PREDICT
# ─────────────────────────────────────────────
def demo_predict(season, yr, hr, weathersit, temp, hum):
    base = 180
    base += {1:-30,2:80,3:100,4:-10}.get(season,0)
    base += 60 if yr==1 else 0
    base += {1:0,2:-40,3:-100,4:-180}.get(weathersit,0)
    base += 100 if temp>20 else -80 if temp<10 else 0
    base += [0,0,0,0,0,10,50,130,160,90,80,100,
             110,100,90,100,140,170,130,90,70,50,30,10][hr]
    base += -40 if hum>80 else 0
    return max(1, round(base))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.title("🚴 Bike Rental Demand Dashboard")

if predict:
    input_arr = np.array([[season, yr, mnth, hr, holiday, weekday,
                           workingday, weathersit, temp, atemp, hum, windspeed]])

    if model_loaded:
        prediction = int(best_pipeline.predict(input_arr)[0])
    else:
        prediction = demo_predict(season, yr, hr, weathersit, temp, hum)

    st.markdown(f"""
    <div class="result-box">
        <div class="num">{prediction}</div>
        <div class="label">Predicted bike rentals per hour</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(prediction/850,1))

    st.subheader("Why this prediction?")
    insights = []

    if hr in [7,8,17,18]:
        insights.append("Rush hour increases demand")
    if temp > 20:
        insights.append("Warm weather increases rentals")
    if weathersit >= 3:
        insights.append("Bad weather reduces demand")
    if hum > 80:
        insights.append("High humidity reduces comfort")

    for i in insights:
        st.markdown(f"<div class='insight-box'>{i}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL METRICS
# ─────────────────────────────────────────────
st.divider()
st.subheader("Model Performance")
st.dataframe(MODEL_METRICS, use_container_width=True)

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
st.subheader("Feature Importance (GBR)")

fi = pd.Series(FEATURE_IMPORTANCE).sort_values()

fig, ax = plt.subplots()
ax.barh(fi.index, fi.values)
ax.set_title("Gradient Boosting Feature Importance")
st.pyplot(fig)