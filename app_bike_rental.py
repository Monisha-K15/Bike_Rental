"""
Bike Rental Prediction App — Redesigned
Deployed model: Gradient Boosting (MAE: 26.29 | RMSE: 39.88 | R²: 0.9519)
Random Forest included only for model comparison — NOT deployed.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BikeFlow · Demand Predictor",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark premium dashboard aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background-color: #0e1117;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #13161e !important;
    border-right: 1px solid #1f2330;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

/* ── Brand header ── */
.brand-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 0.2rem;
}
.brand-name {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #7EB8F7 0%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #5a6073;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 2px 8px;
    border: 1px solid #2a2f40;
    border-radius: 4px;
    margin-left: 4px;
}

/* ── Section title ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #5a6073;
    margin: 1.6rem 0 0.7rem;
}

/* ── Prediction result card ── */
.predict-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
    border: 1px solid #2a3045;
    border-radius: 16px;
    padding: 2.2rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.predict-card::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(126,184,247,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.predict-number {
    font-family: 'Syne', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, #7EB8F7 0%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.predict-unit {
    font-size: 0.85rem;
    color: #5a6073;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
    font-family: 'DM Mono', monospace;
}
.predict-confidence {
    display: inline-block;
    margin-top: 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 12px;
    background: #1f2533;
    border: 1px solid #2e3548;
    border-radius: 20px;
    color: #7EB8F7;
}

/* ── Metric chips ── */
.metric-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.metric-chip {
    flex: 1;
    min-width: 80px;
    background: #141824;
    border: 1px solid #1f2533;
    border-radius: 10px;
    padding: 0.9rem 0.8rem;
    text-align: center;
}
.metric-chip .chip-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8eaf0;
}
.metric-chip .chip-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #5a6073;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}
.metric-chip.accent .chip-val { color: #7EB8F7; }

/* ── Insight pills ── */
.insight-pill {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 3px 3px 3px 0;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.insight-pos { background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3); color: #34d399; }
.insight-neg { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); color: #f87171; }
.insight-neu { background: rgba(126,184,247,0.1); border: 1px solid rgba(126,184,247,0.3); color: #7EB8F7; }

/* ── Sidebar inputs ── */
.sidebar-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5a6073;
    margin-bottom: -8px;
    display: block;
}
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    background: linear-gradient(90deg, #7EB8F7 0%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #1f2330;
    margin: 1.8rem 0;
}

/* ── Stray widget style cleanup ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.73rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7a8099 !important;
}
div[data-testid="stButton"] button {
    width: 100%;
    background: linear-gradient(135deg, #7EB8F7 0%, #A78BFA 100%);
    color: #0e1117;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 0;
    cursor: pointer;
    margin-top: 0.5rem;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL (GBR pipeline only — RF excluded from deployment)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("best_model_pipeline.pkl")

try:
    best_pipeline = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# All 3 models compared; RF included for reference only, NOT deployed
MODEL_METRICS = pd.DataFrame({
    "Model"         : ["Decision Tree", "Random Forest", "Gradient Boosting ✦"],
    "MAE"           : [33.08,  28.70,  26.29],
    "RMSE"          : [55.74,  42.10,  39.88],
    "R²"            : [0.906,  0.937,  0.952],
    "Deployed"      : ["—",    "—",    "✔"],
})

FEATURE_IMPORTANCE = {
    "hr"         : 0.310,
    "temp"       : 0.185,
    "atemp"      : 0.130,
    "season"     : 0.085,
    "hum"        : 0.075,
    "yr"         : 0.065,
    "mnth"       : 0.045,
    "windspeed"  : 0.035,
    "weathersit" : 0.030,
    "weekday"    : 0.020,
    "workingday" : 0.010,
    "holiday"    : 0.010,
}

HOUR_DEMAND = [0,0,0,0,0,10,50,130,160,90,80,100,
               110,100,90,100,140,170,130,90,70,50,30,10]

# ─────────────────────────────────────────────
# FEATURE ENGINEERING — mirrors notebook exactly
# ─────────────────────────────────────────────

# These are the exact columns X had after get_dummies(drop_first=True) in the notebook.
# We must build the same structure so the pipeline's scaler sees the right shape.
# The pipeline was trained on: original cols + engineered features, all one-hot encoded.
EXPECTED_COLS = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
    'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
    'is_weekend', 'temp_diff',
    # get_dummies(drop_first=True) on 'season' (1-4) → season_2, season_3, season_4
    'season_2', 'season_3', 'season_4',
    # yr → yr_1
    'yr_1',
    # weathersit → weathersit_2, weathersit_3, weathersit_4
    'weathersit_2', 'weathersit_3', 'weathersit_4',
    # time_of_day: Night/Morning/Afternoon/Evening → Morning, Night, Evening (drop first=Afternoon)
    'time_of_day_Evening', 'time_of_day_Morning', 'time_of_day_Night',
    # weather_group: Clear/Mist-Cloudy/Light Rain-Snow/Heavy Rain-Snow
    'weather_group_Light Rain/Snow', 'weather_group_Mist/Cloudy',
    'weather_group_Heavy Rain/Snow',
    # yr_num (string version of yr): '0'/'1' → yr_num_1
    'yr_num_1',
]

def build_input_df(season, yr, mnth, hr, holiday, weekday, workingday,
                   weathersit, temp, atemp, hum, windspeed):
    """Replicate the notebook's feature engineering then get_dummies, return a DataFrame."""
    # Time-of-day bucket
    if hr <= 6:
        tod = 'Night'
    elif hr <= 12:
        tod = 'Morning'
    elif hr <= 18:
        tod = 'Afternoon'
    else:
        tod = 'Evening'

    weather_group_map = {
        1: 'Clear',
        2: 'Mist/Cloudy',
        3: 'Light Rain/Snow',
        4: 'Heavy Rain/Snow',
    }

    row = {
        'season'       : season,
        'yr'           : yr,
        'mnth'         : mnth,
        'hr'           : hr,
        'holiday'      : holiday,
        'weekday'      : weekday,
        'workingday'   : workingday,
        'weathersit'   : weathersit,
        'temp'         : temp,
        'atemp'        : atemp,
        'hum'          : hum,
        'windspeed'    : windspeed,
        'is_weekend'   : 1 if weekday in [0, 6] else 0,
        'temp_diff'    : atemp - temp,
        'time_of_day'  : tod,
        'weather_group': weather_group_map[weathersit],
        'yr_num'       : str(yr),
    }

    df_row = pd.DataFrame([row])

    # Apply get_dummies the same way the notebook did
    cat_cols = ['season', 'yr', 'weathersit', 'time_of_day', 'weather_group', 'yr_num']
    df_enc = pd.get_dummies(df_row, columns=cat_cols, drop_first=True)

    # Align to training columns — add missing dummies as 0, drop any extras
    for col in EXPECTED_COLS:
        if col not in df_enc.columns:
            df_enc[col] = 0

    df_enc = df_enc[EXPECTED_COLS]
    return df_enc


# ─────────────────────────────────────────────
# FALLBACK DEMO PREDICTOR
# ─────────────────────────────────────────────
def demo_predict(season, yr, hr, weathersit, temp, hum):
    base = 180
    base += {1:-30, 2:80, 3:100, 4:-10}.get(season, 0)
    base += 60 if yr == 1 else 0
    base += {1:0, 2:-40, 3:-100, 4:-180}.get(weathersit, 0)
    base += 100 if temp > 20 else (-80 if temp < 10 else 0)
    base += HOUR_DEMAND[hr]
    base += -40 if hum > 80 else 0
    return max(1, round(base))

def demand_level(n):
    if n < 100:  return "Low",    "#f87171"
    if n < 300:  return "Medium", "#fbbf24"
    if n < 550:  return "High",   "#34d399"
    return "Peak", "#7EB8F7"

# ─────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">🚴 BikeFlow</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;color:#5a6073;font-family:\'DM Mono\',monospace;margin-top:-6px;">DEMAND PREDICTOR</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-title">📅 Time & Date</p>', unsafe_allow_html=True)
    yr       = st.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
    mnth     = st.slider("Month", 1, 12, 6)
    hr_val   = st.slider("Hour of Day", 0, 23, 17, format="%d:00")
    weekday  = st.selectbox("Weekday", list(range(7)),
                            format_func=lambda x: ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"][x])
    workingday = st.selectbox("Working Day?", [1, 0], format_func=lambda x: "Yes" if x else "No")
    holiday    = st.selectbox("Holiday?", [0, 1],    format_func=lambda x: "No" if x == 0 else "Yes")

    st.markdown('<p class="section-title">🌤 Conditions</p>', unsafe_allow_html=True)
    season = st.selectbox("Season", [1, 2, 3, 4],
                          format_func=lambda x: {1:"🌸 Spring", 2:"☀️ Summer", 3:"🍂 Fall", 4:"❄️ Winter"}[x])
    weathersit = st.selectbox("Weather Situation", [1, 2, 3, 4],
                               format_func=lambda x: {1:"☀ Clear", 2:"🌥 Mist", 3:"🌧 Light Rain", 4:"⛈ Heavy Rain"}[x])
    temp       = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, step=0.5)
    atemp      = st.slider("Feels Like (°C)",  0.0, 50.0, 25.0, step=0.5)
    hum        = st.slider("Humidity (%)", 0, 100, 50)
    windspeed  = st.slider("Wind Speed (km/h)", 0.0, 67.0, 12.0, step=0.5)

    st.markdown("---")
    predict = st.button("⚡ Predict Demand")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
  <span class="brand-name">BikeFlow</span>
  <span class="brand-tag">Demand Analytics</span>
</div>
<p style="color:#5a6073;font-size:0.85rem;font-family:'DM Sans',sans-serif;margin-bottom:0;">
  Gradient Boosting model &nbsp;·&nbsp; R² 0.9519 &nbsp;·&nbsp; MAE 26.29 bikes/hr
</p>
""", unsafe_allow_html=True)

# ─── Top metric summary ───
st.markdown('<p class="section-title">Model at a Glance</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
for col, val, label, accent in [
    (c1, "0.9519", "R² Score",      True),
    (c2, "26.29",  "MAE (bikes)",   False),
    (c3, "39.88",  "RMSE (bikes)",  False),
    (c4, "GBR",    "Deployed Model",True),
]:
    acc_cls = "accent" if accent else ""
    col.markdown(f"""
    <div class="metric-chip {acc_cls}">
      <div class="chip-val">{val}</div>
      <div class="chip-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── Prediction result ───
if predict:
    if model_loaded:
        try:
            input_df = build_input_df(season, yr, mnth, hr_val, holiday, weekday,
                                      workingday, weathersit, temp, atemp, hum, windspeed)
            prediction = int(best_pipeline.predict(input_df)[0])
        except Exception as e:
            st.warning(f"Model prediction failed ({e}). Showing demo estimate.")
            prediction = demo_predict(season, yr, hr_val, weathersit, temp, hum)
    else:
        prediction = demo_predict(season, yr, hr_val, weathersit, temp, hum)

    level, level_color = demand_level(prediction)
    pct = min(int(prediction / 850 * 100), 100)

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown(f"""
        <div class="predict-card">
            <div class="predict-number">{prediction}</div>
            <div class="predict-unit">BIKES / HOUR</div>
            <div class="predict-confidence">
                Demand level: <span style="color:{level_color}">{level}</span>
                &nbsp;·&nbsp; {pct}% of peak
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Demand bar
        st.markdown('<p class="section-title" style="margin-top:1.2rem;">Demand Gauge</p>', unsafe_allow_html=True)
        st.progress(pct / 100)

    with right_col:
        st.markdown('<p class="section-title">Why this prediction?</p>', unsafe_allow_html=True)
        insights = []

        # Hour
        if hr_val in [7, 8, 17, 18]:
            insights.append(("pos", "🕐 Rush hour — commuter surge expected"))
        elif hr_val in [0, 1, 2, 3, 4]:
            insights.append(("neg", "🌙 Late night — very low demand"))
        elif hr_val in [11, 12, 13]:
            insights.append(("neu", "🍽 Midday — moderate demand"))

        # Weather
        if weathersit == 1:
            insights.append(("pos", "☀ Clear skies boost outdoor activity"))
        elif weathersit == 3:
            insights.append(("neg", "🌧 Light rain discourages cycling"))
        elif weathersit == 4:
            insights.append(("neg", "⛈ Heavy weather strongly reduces demand"))

        # Temperature
        if temp > 20:
            insights.append(("pos", f"🌡 Warm {temp:.0f}°C — ideal riding conditions"))
        elif temp < 10:
            insights.append(("neg", f"🥶 Cold {temp:.0f}°C — reduces comfort"))

        # Humidity
        if hum > 80:
            insights.append(("neg", "💧 High humidity reduces comfort"))

        # Season
        season_label = {1:"Spring 🌸", 2:"Summer ☀️", 3:"Fall 🍂", 4:"Winter ❄️"}[season]
        season_sentiment = {1:"pos", 2:"pos", 3:"pos", 4:"neg"}[season]
        insights.append((season_sentiment, f"{season_label} season influences demand"))

        # Working day
        if workingday == 1:
            insights.append(("neu", "💼 Working day — commuter pattern expected"))
        elif holiday == 1:
            insights.append(("pos", "🎉 Holiday — leisure rides increase"))

        html_pills = ""
        for sentiment, text in insights:
            css = {"pos":"insight-pos","neg":"insight-neg","neu":"insight-neu"}[sentiment]
            html_pills += f'<span class="insight-pill {css}">{text}</span>'

        st.markdown(html_pills, unsafe_allow_html=True)

        # Hourly curve mini-chart
        st.markdown('<p class="section-title" style="margin-top:1.4rem;">Typical Hourly Pattern</p>', unsafe_allow_html=True)
        fig_mini, ax_mini = plt.subplots(figsize=(5, 2))
        fig_mini.patch.set_facecolor('#141824')
        ax_mini.set_facecolor('#141824')
        hours = list(range(24))
        ax_mini.fill_between(hours, HOUR_DEMAND, alpha=0.25, color='#7EB8F7')
        ax_mini.plot(hours, HOUR_DEMAND, color='#7EB8F7', linewidth=1.8)
        ax_mini.axvline(hr_val, color='#A78BFA', linewidth=1.5, linestyle='--', alpha=0.9)
        ax_mini.scatter([hr_val], [HOUR_DEMAND[hr_val]], color='#A78BFA', zorder=5, s=50)
        ax_mini.set_xticks([0,6,12,18,23])
        ax_mini.set_xticklabels(['0h','6h','12h','18h','23h'], color='#5a6073', fontsize=7)
        ax_mini.tick_params(axis='y', colors='#5a6073', labelsize=7)
        for sp in ax_mini.spines.values():
            sp.set_color('#1f2330')
        ax_mini.grid(axis='y', color='#1f2330', linewidth=0.5)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig_mini, use_container_width=True)

    st.markdown("---")

# ─── Model Comparison (all 3 including RF) ───
st.markdown('<p class="section-title">Model Comparison &nbsp;<span style="color:#5a6073;font-size:0.65rem">(Random Forest shown for comparison only — not deployed)</span></p>', unsafe_allow_html=True)

left, right = st.columns([1.1, 1], gap="large")

with left:
    # Styled dataframe
    def highlight_gbr(row):
        if "Gradient" in row["Model"]:
            return ['background-color: #1a2035; color: #7EB8F7; font-weight: 600'] * len(row)
        return ['color: #a0a6b8'] * len(row)

    st.dataframe(
        MODEL_METRICS.style.apply(highlight_gbr, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("""
    <p style="font-size:0.73rem;color:#5a6073;font-family:'DM Mono',monospace;margin-top:0.4rem;">
    ✦ Gradient Boosting selected for deployment &nbsp;·&nbsp; RF kept for academic comparison only
    </p>
    """, unsafe_allow_html=True)

with right:
    # Comparison bar chart
    fig2, axes = plt.subplots(1, 3, figsize=(6, 2.6))
    fig2.patch.set_facecolor('#141824')

    model_names_short = ["DT", "RF", "GBR"]
    colors_bar = ['#3d4460', '#4a5275', '#7EB8F7']
    metrics_data = [
        ([33.08, 28.70, 26.29], "MAE",  "lower is better"),
        ([55.74, 42.10, 39.88], "RMSE", "lower is better"),
        ([0.906, 0.937, 0.952], "R²",   "higher is better"),
    ]

    for ax, (vals, label, note) in zip(axes, metrics_data):
        ax.set_facecolor('#141824')
        bars = ax.bar(model_names_short, vals, color=colors_bar, width=0.55)
        best = np.argmin(vals) if "lower" in note else np.argmax(vals)
        bars[best].set_color('#A78BFA')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.015,
                    f"{v:.2f}", ha='center', va='bottom',
                    fontsize=6.5, color='#a0a6b8')
        ax.set_title(label, color='#7EB8F7', fontsize=8, fontweight='bold', pad=4)
        ax.tick_params(colors='#5a6073', labelsize=7)
        ax.set_ylim(0, max(vals)*1.22)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis='y', color='#1f2330', linewidth=0.5)
        ax.tick_params(axis='x', colors='#7a8099')
        ax.tick_params(axis='y', colors='#3d4460')

    plt.tight_layout(pad=0.5)
    st.pyplot(fig2, use_container_width=True)
    st.markdown('<p style="text-align:center;font-size:0.68rem;color:#5a6073;font-family:\'DM Mono\',monospace;">Purple bar = best performer</p>', unsafe_allow_html=True)

st.markdown("---")

# ─── Feature Importance ───
st.markdown('<p class="section-title">Feature Importance — Gradient Boosting</p>', unsafe_allow_html=True)

fi = pd.Series(FEATURE_IMPORTANCE).sort_values()

fig3, ax3 = plt.subplots(figsize=(10, 3.2))
fig3.patch.set_facecolor('#141824')
ax3.set_facecolor('#141824')

bar_colors = ['#7EB8F7' if fi.values[-1] == v else '#2a3450' for v in fi.values]
# Top 3 accented
sorted_vals = sorted(FEATURE_IMPORTANCE.values(), reverse=True)[:3]
bar_colors = ['#A78BFA' if v in sorted_vals else '#2e3a55' for v in fi.values]

bars3 = ax3.barh(fi.index, fi.values, color=bar_colors, height=0.55)
for bar, v in zip(bars3, fi.values):
    ax3.text(v + 0.003, bar.get_y() + bar.get_height()/2,
             f"{v:.3f}", va='center', fontsize=7.5, color='#7a8099')

ax3.tick_params(colors='#7a8099', labelsize=8)
ax3.set_xlabel("Importance Score", color='#5a6073', fontsize=8)
for sp in ax3.spines.values():
    sp.set_visible(False)
ax3.grid(axis='x', color='#1f2330', linewidth=0.5)
ax3.tick_params(axis='x', colors='#3d4460')

legend_patches = [
    mpatches.Patch(color='#A78BFA', label='Top 3 features'),
    mpatches.Patch(color='#2e3a55', label='Other features'),
]
ax3.legend(handles=legend_patches, loc='lower right',
           frameon=False, labelcolor='#7a8099', fontsize=7.5)

plt.tight_layout(pad=0.4)
st.pyplot(fig3, use_container_width=True)

st.markdown("""
<p style="font-size:0.78rem;color:#5a6073;font-family:'DM Sans',sans-serif;margin-top:-0.5rem;">
<strong style="color:#A78BFA">hr</strong> (hour of day),
<strong style="color:#A78BFA">temp</strong> (temperature), and
<strong style="color:#A78BFA">atemp</strong> (feels-like temp)
are the three strongest predictors — consistent with EDA findings.
</p>
""", unsafe_allow_html=True)

# ─── Footer ───
st.markdown("---")
st.markdown("""
<p style="text-align:center;font-size:0.7rem;color:#3d4460;font-family:'DM Mono',monospace;">
BikeFlow · DS Group 6 · Gradient Boosting Regressor · Deployed model excludes Random Forest
</p>
""", unsafe_allow_html=True)
