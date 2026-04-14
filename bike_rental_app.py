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
    .metric-card h4 { margin: 0; color: #666; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card h2 { margin: 4px 0 0; color: #111; font-size: 1.6rem; }
    .result-box {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .result-box .num { font-size: 3.5rem; font-weight: 700; line-height: 1; }
    .result-box .label { font-size: 0.9rem; opacity: 0.8; margin-top: 6px; }
    .insight-box {
        background: #e8f5e9;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border-left: 4px solid #4CAF50;
        font-size: 0.9rem;
        color: #1b5e20;
        margin-top: 0.5rem;
    }
    .badge-best {
        background: #e8f5e9; color: #2e7d32;
        padding: 2px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
    }
    .badge-base {
        background: #f3f3f3; color: #666;
        padding: 2px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (GBR pipeline only)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipeline = joblib.load("best_model_pipeline.pkl")
    return pipeline


try:
    best_pipeline = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ─────────────────────────────────────────────
# METRICS & STATIC DATA
# ─────────────────────────────────────────────
MODEL_METRICS = pd.DataFrame({
    "Model" : ["Decision Tree", "Random Forest", "Gradient Boosting"],
    "MAE"   : [33.08, 26.13, 26.29],
    "RMSE"  : [55.74, 42.98, 39.88],
    "R²"    : [0.906, 0.944, 0.952],
})

# Feature importance from GBR (static — no model needed)
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

# Random Forest feature importance (static — for comparison, no model needed)
RF_FEATURE_IMPORTANCE = {
    "hr"         : 0.298,
    "temp"       : 0.178,
    "atemp"      : 0.125,
    "season"     : 0.079,
    "hum"        : 0.082,
    "yr"         : 0.070,
    "mnth"       : 0.050,
    "windspeed"  : 0.038,
    "weathersit" : 0.033,
    "weekday"    : 0.022,
    "workingday" : 0.013,
    "holiday"    : 0.012,
}

# ─────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🚴 Bike Rental Predictor")
    st.caption("Gradient Boosting • R² 0.952")
    st.divider()

    st.subheader("📅 Time & Date")
    yr        = st.selectbox("Year", options=[0, 1],
                              format_func=lambda x: "2011" if x == 0 else "2012")
    mnth      = st.slider("Month", 1, 12, 6,
                           format="%d",
                           help="1 = January … 12 = December")
    hr        = st.slider("Hour of Day", 0, 23, 17,
                           help="0 = midnight, 17 = 5pm")
    weekday   = st.selectbox(
        "Weekday",
        options=list(range(7)),
        index=3,
        format_func=lambda x: ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][x],
    )
    workingday = st.selectbox("Working Day?", [0, 1],
                               format_func=lambda x: "Yes" if x else "No")
    holiday    = st.selectbox("Public Holiday?", [0, 1],
                               format_func=lambda x: "Yes" if x else "No")

    st.divider()
    st.subheader("🌤 Weather")
    season     = st.selectbox(
        "Season",
        options=[1, 2, 3, 4],
        index=1,
        format_func=lambda x: {1:"🌸 Spring", 2:"☀️ Summer",
                                3:"🍂 Fall",   4:"❄️ Winter"}[x],
    )
    weathersit = st.selectbox(
        "Weather Condition",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "☀️  Clear / Few Clouds",
            2: "🌥  Mist / Cloudy",
            3: "🌧  Light Rain / Snow",
            4: "⛈  Heavy Rain / Ice",
        }[x],
    )
    temp       = st.slider("Temperature (°C)",    0.0, 50.0, 25.0, 0.5)
    atemp      = st.slider("Feels-Like Temp (°C)", 0.0, 50.0, 25.0, 0.5)
    hum        = st.slider("Humidity (%)",         0,   100,  50)
    windspeed  = st.slider("Windspeed (km/h)",     0.0, 67.0, 12.0, 0.5)

    st.divider()
    predict_btn = st.button("🚀  Predict Rental Demand", type="primary",
                             use_container_width=True)


# ─────────────────────────────────────────────
# HELPER: demo prediction fallback
# ─────────────────────────────────────────────
def demo_predict(season, yr, hr, weathersit, temp, hum):
    base = 180
    base += {1: -30, 2: 80, 3: 100, 4: -10}.get(season, 0)
    base += 60 if yr == 1 else 0
    base += {1: 0, 2: -40, 3: -100, 4: -180}.get(weathersit, 0)
    if temp < 10:
        base -= 80
    elif temp > 20:
        base += 100
    hour_curve = [0, 0, 0, 0, 0, 10, 50, 130, 160, 90, 80, 100,
                  110, 100, 90, 100, 140, 170, 130, 90, 70, 50, 30, 10]
    base += hour_curve[hr]
    if hum > 80:
        base -= 40
    return max(1, round(base))


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.title("Bike Rental Demand Dashboard")
tab_pred, tab_compare, tab_insights = st.tabs(
    ["🎯 Predict", "📊 Model Comparison", "🔍 Feature Insights"]
)


# ── TAB 1: PREDICT ───────────────────────────
with tab_pred:

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    for col, label, value, unit in [
        (kpi1, "Best Model",  "GBR",    "Gradient Boosting"),
        (kpi2, "MAE",         "26.29",  "bikes avg error"),
        (kpi3, "RMSE",        "39.88",  "root mean sq error"),
        (kpi4, "R² Score",    "0.9519", "variance explained"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <h4>{label}</h4>
            <h2>{value}</h2>
            <p style="margin:0;font-size:0.75rem;color:#999;">{unit}</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Build input array
    input_arr = np.array([[
        season, yr, mnth, holiday, weekday,
        workingday, weathersit, temp, atemp, hum, windspeed
    ]])

    if model_loaded:
        prediction = int(best_pipeline.predict(input_arr)[0])
    else:
        prediction = demo_predict(season, yr, hr, weathersit, temp, hum)

    pct = min(prediction / 850 * 100, 100)

    if prediction < 100:
        tier, tier_color = "Low demand", "#f44336"
    elif prediction < 300:
        tier, tier_color = "Moderate demand", "#ff9800"
    elif prediction < 550:
        tier, tier_color = "High demand", "#4CAF50"
    else:
        tier, tier_color = "Peak demand", "#2196F3"

    col_res, col_ctx = st.columns([1, 1])

    with col_res:
        st.markdown(f"""
        <div class="result-box">
            <div class="num">{prediction:,}</div>
            <div class="label">predicted bike rentals / hour</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"**Demand level:** <span style='color:{tier_color};font-weight:600'>{tier}</span>",
                    unsafe_allow_html=True)
        st.progress(int(pct))
        st.caption(f"{prediction} out of ~850 typical max")

    with col_ctx:
        st.markdown("#### Why this prediction?")

        insights = []
        if hr in [7, 8, 17, 18, 19]:
            insights.append("🕐 Rush hour — commuter peak window")
        elif hr < 6 or hr > 22:
            insights.append("🌙 Late night — very low baseline demand")

        if temp > 20:
            insights.append(f"🌡 Warm temp ({temp}°C) boosts ridership")
        elif temp < 8:
            insights.append(f"🥶 Cold temp ({temp}°C) suppresses demand")

        if weathersit == 1:
            insights.append("☀️ Clear skies — optimal riding conditions")
        elif weathersit >= 3:
            insights.append("🌧 Adverse weather significantly reduces demand")

        if season in [2, 3]:
            insights.append("🍂 Summer/Fall — peak rental season")
        elif season == 4:
            insights.append("❄️ Winter — demand typically at yearly low")

        if yr == 1:
            insights.append("📈 2012 baseline is ~30% higher than 2011")

        if hum > 80:
            insights.append(f"💧 High humidity ({hum}%) reduces comfort")

        if not insights:
            insights.append("📊 Average conditions → moderate demand predicted")

        for ins in insights[:5]:
            st.markdown(f'<div class="insight-box">{ins}</div>',
                        unsafe_allow_html=True)

    if not model_loaded:
        st.warning(
            "⚠️ `best_model_pipeline.pkl` not found. "
            "Showing **demo predictions**. Place the `.pkl` file in the same "
            "directory as this app to use the real model.",
            icon="⚠️",
        )


# ── TAB 2: MODEL COMPARISON ──────────────────
with tab_compare:

    st.subheader("All Models — Performance Metrics")

    styled = MODEL_METRICS.style\
        .highlight_min(subset=["MAE", "RMSE"], color="#c8e6c9")\
        .highlight_max(subset=["R²"],          color="#c8e6c9")\
        .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}"})\
        .set_properties(**{"text-align": "center"})

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # ── Bar Charts: MAE / RMSE / R² ──
    colors = ["#ef9a9a", "#90caf9", "#a5d6a7"]
    bar_kw = dict(color=colors, edgecolor="white", linewidth=0.8)

    c1, c2 = st.columns(2)

    with c1:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        fig.patch.set_alpha(0)

        for ax, metric, ylabel in [
            (axes[0], "MAE",  "MAE (bikes)"),
            (axes[1], "RMSE", "RMSE (bikes)"),
        ]:
            bars = ax.bar(MODEL_METRICS["Model"], MODEL_METRICS[metric], **bar_kw)
            best_idx = MODEL_METRICS[metric].idxmin()
            bars[best_idx].set_edgecolor("#2e7d32")
            bars[best_idx].set_linewidth(2)
            for bar, val in zip(bars, MODEL_METRICS[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(range(len(MODEL_METRICS["Model"])))
            ax.set_xticklabels(MODEL_METRICS["Model"], rotation=20, ha="right", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
        fig2.patch.set_alpha(0)
        bars2 = ax2.bar(MODEL_METRICS["Model"], MODEL_METRICS["R²"], **bar_kw)
        best_r2 = MODEL_METRICS["R²"].idxmax()
        bars2[best_r2].set_edgecolor("#2e7d32")
        bars2[best_r2].set_linewidth(2)
        for bar, val in zip(bars2, MODEL_METRICS["R²"]):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001, f"{val:.4f}",
                     ha="center", va="bottom", fontsize=8)
        ax2.set_ylabel("R² Score", fontsize=9)
        ax2.set_ylim(0.85, 0.97)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax2.set_xticks(range(len(MODEL_METRICS["Model"])))
        ax2.set_xticklabels(MODEL_METRICS["Model"], rotation=20, ha="right", fontsize=8)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.divider()

    # ── Radar / Spider comparison ──
    st.subheader("📡 Multi-Metric Radar Comparison")

    # Normalise metrics for radar (higher = better for all, so invert error metrics)
    max_mae  = max(MODEL_METRICS["MAE"])
    max_rmse = max(MODEL_METRICS["RMSE"])

    radar_data = {
        "R²"            : MODEL_METRICS["R²"].tolist(),
        "MAE (inverted)": [1 - v / max_mae  for v in MODEL_METRICS["MAE"]],
        "RMSE (inverted)": [1 - v / max_rmse for v in MODEL_METRICS["RMSE"]],
    }
    categories = list(radar_data.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    model_colors_radar = ["#ef9a9a", "#90caf9", "#a5d6a7"]
    fig_r, ax_r = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig_r.patch.set_alpha(0)

    for idx, model_name in enumerate(MODEL_METRICS["Model"]):
        values = [radar_data[cat][idx] for cat in categories]
        values += values[:1]
        ax_r.plot(angles, values, "o-", linewidth=2,
                  color=model_colors_radar[idx], label=model_name)
        ax_r.fill(angles, values, alpha=0.12, color=model_colors_radar[idx])

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(categories, fontsize=9)
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_r.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax_r.set_title("Normalised Model Performance\n(higher = better for all axes)",
                   fontsize=10, fontweight="bold", pad=20)
    plt.tight_layout()

    _, radar_col, _ = st.columns([1, 2, 1])
    with radar_col:
        st.pyplot(fig_r, use_container_width=True)
    plt.close(fig_r)

    st.divider()
    st.subheader("Model Analysis")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Decision Tree** <span class='badge-base'>Baseline</span>",
                    unsafe_allow_html=True)
        st.markdown("""
- Simple, interpretable structure
- High variance — prone to overfitting
- Lowest R² and worst error metrics
- Useful as a benchmark only
""")
    with col_b:
        st.markdown("**Random Forest** <span class='badge-base'>Ensemble</span>",
                    unsafe_allow_html=True)
        st.markdown("""
- Averages hundreds of trees
- Reduces variance vs single DT
- Slight train/test R² gap → mild overfit
- Strong overall but not best here
""")
    with col_c:
        st.markdown("**Gradient Boosting** <span class='badge-best'>✓ Best</span>",
                    unsafe_allow_html=True)
        st.markdown("""
- Sequentially corrects prior errors
- Best test R²: **0.9519**
- CV R²: **0.9462 ± 0.0013** (very stable)
- Selected for production predictions
""")


# ── TAB 3: FEATURE INSIGHTS ──────────────────
with tab_insights:

    st.subheader("Feature Importance — GBR vs Random Forest")
    st.caption("Both models agree on the top drivers; GBR concentrates importance more on `hr` and `temp`.")

    fi_gbr = pd.Series(FEATURE_IMPORTANCE).sort_values()
    fi_rf  = pd.Series(RF_FEATURE_IMPORTANCE)
    fi_rf  = fi_rf.reindex(fi_gbr.index)  # same order

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig3.patch.set_alpha(0)

    bar_colors_gbr = ["#a5d6a7" if v >= 0.15 else
                      "#90caf9" if v >= 0.06 else "#e0e0e0"
                      for v in fi_gbr.values]
    bar_colors_rf  = ["#a5d6a7" if v >= 0.15 else
                      "#90caf9" if v >= 0.06 else "#e0e0e0"
                      for v in fi_rf.values]

    # GBR
    bars_gbr = axes3[0].barh(fi_gbr.index, fi_gbr.values,
                              color=bar_colors_gbr, edgecolor="white")
    for bar, val in zip(bars_gbr, fi_gbr.values):
        axes3[0].text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                      f"{val * 100:.1f}%", va="center", fontsize=9)
    axes3[0].set_xlabel("Importance Score", fontsize=10)
    axes3[0].set_xlim(0, 0.40)
    axes3[0].set_title("Gradient Boosting", fontweight="bold", fontsize=11)
    axes3[0].spines[["top", "right"]].set_visible(False)
    axes3[0].grid(axis="x", alpha=0.3)

    # RF
    bars_rf = axes3[1].barh(fi_rf.index, fi_rf.values,
                             color=bar_colors_rf, edgecolor="white")
    for bar, val in zip(bars_rf, fi_rf.values):
        axes3[1].text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                      f"{val * 100:.1f}%", va="center", fontsize=9)
    axes3[1].set_xlabel("Importance Score", fontsize=10)
    axes3[1].set_xlim(0, 0.40)
    axes3[1].set_title("Random Forest", fontweight="bold", fontsize=11)
    axes3[1].spines[["top", "right"]].set_visible(False)
    axes3[1].grid(axis="x", alpha=0.3)

    fig3.suptitle("Feature Importance Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.caption("Green = high importance (>15%) · Blue = medium (6–15%) · Grey = low (<6%)")

    # ── Difference bar chart ──
    st.divider()
    st.subheader("📊 Importance Difference: GBR − Random Forest")

    diff = fi_gbr - fi_rf
    diff_sorted = diff.sort_values()
    diff_colors = ["#ef9a9a" if v < 0 else "#a5d6a7" for v in diff_sorted.values]

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    fig4.patch.set_alpha(0)
    bars4 = ax4.barh(diff_sorted.index, diff_sorted.values,
                     color=diff_colors, edgecolor="white")
    ax4.axvline(x=0, color="#333", linewidth=1)
    for bar, val in zip(bars4, diff_sorted.values):
        ax4.text(val + (0.001 if val >= 0 else -0.001),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:+.3f}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=8)
    ax4.set_xlabel("GBR Importance − RF Importance", fontsize=10)
    ax4.set_title("Where the models disagree most", fontweight="bold")
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)
    st.caption("Green bars = GBR weights this feature more · Red bars = RF weights it more")

    st.divider()
    st.subheader("Key EDA Findings")

    findings = [
        ("🕐 Commuter Pattern",
         "`hr` is the single most important feature in both models. Demand peaks at **8am** and **5–6pm** "
         "on working days — classic commuter behaviour. Weekends show a flat midday curve."),
        ("🌡 Temperature Drives Demand",
         "`temp` and `atemp` together account for **~31%** of prediction in GBR. Warm days (20–35°C) "
         "generate the highest rentals. Cold weather (<10°C) cuts demand significantly."),
        ("🌧 Weather Condition",
         "Clear weather sees **3–4×** the rentals of heavy rain. Weather sits 4th in importance "
         "but has the largest single-factor suppression effect."),
        ("📈 Year-on-Year Growth",
         "`yr` carries moderate importance. 2012 demand was ~30% higher than 2011, confirming "
         "the bike-share system's rapid adoption and network growth."),
        ("👥 Two User Segments",
         "Registered users dominate weekday/commuter demand. Casual users spike on weekends "
         "and holidays. These two groups require different marketing and fleet strategies."),
        ("🍂 Seasonal Peaks",
         "Fall > Summer > Spring > Winter in average demand. Fleet and staffing should be "
         "scaled accordingly — winter requires fewer bikes and maintenance focus."),
    ]

    for title, body in findings:
        with st.expander(title, expanded=False):
            st.markdown(body)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Bike Rental Predictor · DS Group 6 · "
    "Models: Decision Tree, Random Forest, Gradient Boosting · "
    "Dataset: UCI Bike Sharing"
)
