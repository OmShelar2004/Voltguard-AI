import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# ─── Page Config ───
st.set_page_config(
    page_title="VoltGuard AI – EV Battery Health Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    /* ── Import Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 40%, #0a0f1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #e2e8f0 !important;
    }

    /* ── Header ── */
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
        margin-bottom: 1.5rem;
    }

    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }

    .main-header .tagline {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        letter-spacing: 0.03em;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.6));
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }

    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.1);
    }

    .metric-card .label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .metric-card .sub {
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* ── Verdict Banner ── */
    .verdict-banner {
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        backdrop-filter: blur(12px);
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }

    .verdict-banner .verdict-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .verdict-banner .verdict-rec {
        font-size: 0.95rem;
        opacity: 0.85;
    }

    /* ── Section Headers ── */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Info Card ── */
    .info-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.6), rgba(30, 41, 59, 0.4));
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.7;
    }

    /* ── Sidebar Styling ── */
    .sidebar-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.2rem;
        letter-spacing: 0.02em;
    }

    .sidebar-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    /* ── Streamlit Overrides ── */
    .stSlider > div > div > div > div {
        background-color: #38bdf8 !important;
    }

    div[data-testid="stNumberInput"] input {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.03em !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
    }

    /* ── Hide Streamlit defaults ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15, 23, 42, 0.6);
        border-radius: 10px;
        padding: 8px 20px;
        color: #94a3b8;
        border: 1px solid rgba(56, 189, 248, 0.1);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.3), rgba(124, 58, 237, 0.3)) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "ev_battery_model.pkl")
    return joblib.load(model_path)


@st.cache_data
def load_dataset():
    csv_path = os.path.join(os.path.dirname(__file__), "ev_battery_dataset.csv")
    return pd.read_csv(csv_path)


model = load_model()
df = load_dataset()

# ─── Header ───
st.markdown("""
<div class="main-header">
    <h1>⚡ VoltGuard AI</h1>
    <p class="tagline">Intelligent EV Battery Health Prediction & Monitoring</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar Inputs ───
with st.sidebar:
    st.markdown('<p class="sidebar-title">🔋 Battery Parameters</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">Adjust the values to analyze battery health</p>', unsafe_allow_html=True)

    st.markdown("---")

    charge_cycles = st.number_input(
        "⚡ Charge Cycles",
        min_value=0,
        max_value=5000,
        value=1000,
        step=50,
        help="Total number of charge-discharge cycles completed."
    )

    fast_charging = st.slider(
        "🔌 Fast Charging Frequency (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.5,
        help="Percentage of total charges done using fast charging."
    )

    temperature = st.slider(
        "🌡️ Avg Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=25.0,
        step=0.5,
        help="Average operating temperature of the battery."
    )

    aggression = st.slider(
        "🏎️ Driving Aggression Index",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="0 = very calm driving, 1 = very aggressive driving."
    )

    st.markdown("---")
    predict_btn = st.button("🚀 Analyze Battery Health", use_container_width=True)

# ─── Prediction Logic ───
if predict_btn:
    sample = pd.DataFrame({
        "Charge_Cycles": [charge_cycles],
        "Fast_Charging_Frequency_%": [fast_charging],
        "Avg_Temperature_C": [temperature],
        "Driving_Aggression_Index": [aggression],
    })

    degradation = model.predict(sample)[0]
    battery_health = max(0, min(100, 100 - degradation))

    # Verdict
    if battery_health >= 90:
        verdict, emoji, rec = "Excellent", "🟢", "Battery is in excellent condition. Keep up the great maintenance!"
        verdict_bg = "linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(52, 211, 153, 0.08))"
        verdict_border = "rgba(16, 185, 129, 0.4)"
        verdict_color = "#34d399"
        gauge_color = "#10b981"
    elif battery_health >= 80:
        verdict, emoji, rec = "Healthy", "🟢", "Battery is healthy. Continue normal usage patterns."
        verdict_bg = "linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(52, 211, 153, 0.08))"
        verdict_border = "rgba(16, 185, 129, 0.4)"
        verdict_color = "#34d399"
        gauge_color = "#10b981"
    elif battery_health >= 70:
        verdict, emoji, rec = "Monitor", "🟡", "Monitor battery health regularly. Consider reducing fast charging."
        verdict_bg = "linear-gradient(135deg, rgba(234, 179, 8, 0.15), rgba(250, 204, 21, 0.08))"
        verdict_border = "rgba(234, 179, 8, 0.4)"
        verdict_color = "#fbbf24"
        gauge_color = "#eab308"
    elif battery_health >= 60:
        verdict, emoji, rec = "Needs Attention", "🟠", "Reduce fast charging frequency and avoid extreme temperatures."
        verdict_bg = "linear-gradient(135deg, rgba(249, 115, 22, 0.15), rgba(251, 146, 60, 0.08))"
        verdict_border = "rgba(249, 115, 22, 0.4)"
        verdict_color = "#fb923c"
        gauge_color = "#f97316"
    else:
        verdict, emoji, rec = "Replace Soon", "🔴", "Battery replacement is strongly recommended in the near future."
        verdict_bg = "linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(248, 113, 113, 0.08))"
        verdict_border = "rgba(239, 68, 68, 0.4)"
        verdict_color = "#f87171"
        gauge_color = "#ef4444"

    # Risk
    if degradation <= 20:
        risk, risk_color = "Low Risk", "#10b981"
    elif degradation <= 40:
        risk, risk_color = "Moderate Risk", "#eab308"
    elif degradation <= 60:
        risk, risk_color = "High Risk", "#f97316"
    else:
        risk, risk_color = "Critical Risk", "#ef4444"

    # ─── Results ───

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Battery Health</div>
            <div class="value" style="color: {verdict_color};">{battery_health:.1f}%</div>
            <div class="sub">Overall condition</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Degradation</div>
            <div class="value" style="color: #f87171;">{degradation:.1f}%</div>
            <div class="sub">Capacity loss</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Risk Level</div>
            <div class="value" style="color: {risk_color};">{risk.split(' ')[0]}</div>
            <div class="sub">{risk}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Verdict</div>
            <div class="value" style="color: {verdict_color};">{emoji}</div>
            <div class="sub">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)

    # Verdict Banner
    st.markdown(f"""
    <div class="verdict-banner" style="background: {verdict_bg}; border: 1px solid {verdict_border};">
        <div class="verdict-title" style="color: {verdict_color};">{emoji} {verdict}</div>
        <div class="verdict-rec" style="color: {verdict_color};">{rec}</div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Charts ───
    st.markdown('<div class="section-header">📊 Detailed Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Health Gauge", "📈 Input Breakdown", "📉 Dataset Insights"])

    with tab1:
        col_gauge, col_info = st.columns([3, 2])

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=battery_health,
                number={"suffix": "%", "font": {"size": 48, "family": "Inter", "color": "#e2e8f0"}},
                delta={"reference": 100, "decreasing": {"color": "#f87171"}, "increasing": {"color": "#10b981"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#334155",
                             "tickfont": {"color": "#64748b"}},
                    "bar": {"color": gauge_color, "thickness": 0.3},
                    "bgcolor": "rgba(15, 23, 42, 0.3)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 60], "color": "rgba(239, 68, 68, 0.1)"},
                        {"range": [60, 70], "color": "rgba(249, 115, 22, 0.1)"},
                        {"range": [70, 80], "color": "rgba(234, 179, 8, 0.1)"},
                        {"range": [80, 100], "color": "rgba(16, 185, 129, 0.1)"},
                    ],
                    "threshold": {
                        "line": {"color": "#38bdf8", "width": 3},
                        "thickness": 0.8,
                        "value": battery_health,
                    },
                },
                title={"text": "Battery Health Score", "font": {"size": 16, "color": "#94a3b8"}},
            ))

            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=30, r=30, t=60, b=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_info:
            # Estimated range impact
            est_range_factor = battery_health / 100.0
            base_range_km = 400  # Assume 400 km original range

            st.markdown(f"""
            <div class="info-card">
                <p style="font-weight: 600; color: #38bdf8; margin-bottom: 0.8rem; font-size: 1rem;">
                    📋 Health Summary
                </p>
                <p>🔋 <strong>Capacity Retained:</strong> {battery_health:.1f}%</p>
                <p>📉 <strong>Total Degradation:</strong> {degradation:.1f}%</p>
                <p>🚗 <strong>Est. Range:</strong> ~{base_range_km * est_range_factor:.0f} km / {base_range_km} km</p>
                <p>⚡ <strong>Charge Cycles:</strong> {charge_cycles:,}</p>
                <p>🌡️ <strong>Avg Temp:</strong> {temperature}°C</p>
                <p style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(56, 189, 248, 0.15);">
                    <strong style="color: {verdict_color};">{emoji} {verdict}</strong> — {risk}
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        # Radar chart for inputs
        categories = ["Charge Cycles", "Fast Charging", "Temperature", "Aggression"]
        # Normalize values to 0-100 scale for visualization
        norm_cycles = min(charge_cycles / 3000 * 100, 100)
        norm_fast = fast_charging
        norm_temp = min(max((temperature + 10) / 70 * 100, 0), 100)
        norm_aggr = aggression * 100

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=[norm_cycles, norm_fast, norm_temp, norm_aggr],
            theta=categories,
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.15)',
            line=dict(color='#38bdf8', width=2),
            marker=dict(size=8, color='#38bdf8'),
            name='Your Battery'
        ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor="rgba(100, 116, 139, 0.2)",
                    tickfont=dict(color="#64748b", size=10),
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 116, 139, 0.2)",
                    tickfont=dict(color="#94a3b8", size=12),
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=60, r=60, t=40, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("""
        <div class="info-card" style="margin-top: 0.5rem;">
            <p style="color: #94a3b8; font-size: 0.85rem;">
                📌 The radar chart shows your input parameters normalized to a 0–100 scale.
                Higher values in each axis indicate greater stress on the battery.
                Aim to keep all axes low for optimal battery longevity.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_hist = px.histogram(
                df, x="Battery_Degradation_%", nbins=40,
                color_discrete_sequence=["#818cf8"],
                labels={"Battery_Degradation_%": "Battery Degradation (%)"},
            )
            fig_hist.update_layout(
                title=dict(text="Degradation Distribution", font=dict(color="#e2e8f0", size=14)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(100,116,139,0.15)", tickfont=dict(color="#94a3b8"),
                           title_font=dict(color="#94a3b8")),
                yaxis=dict(gridcolor="rgba(100,116,139,0.15)", tickfont=dict(color="#94a3b8"),
                           title_font=dict(color="#94a3b8")),
                height=350,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            # Add a vertical line for user's prediction
            fig_hist.add_vline(
                x=degradation, line_dash="dash", line_color="#f87171", line_width=2,
                annotation_text=f"Your Battery ({degradation:.1f}%)",
                annotation_font=dict(color="#f87171", size=11),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with chart_col2:
            fig_scatter = px.scatter(
                df, x="Charge_Cycles", y="Battery_Degradation_%",
                color="Fast_Charging_Frequency_%",
                color_continuous_scale="viridis",
                opacity=0.5,
                labels={
                    "Charge_Cycles": "Charge Cycles",
                    "Battery_Degradation_%": "Degradation (%)",
                    "Fast_Charging_Frequency_%": "Fast Charge %"
                },
            )
            fig_scatter.update_layout(
                title=dict(text="Cycles vs Degradation", font=dict(color="#e2e8f0", size=14)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(100,116,139,0.15)", tickfont=dict(color="#94a3b8"),
                           title_font=dict(color="#94a3b8")),
                yaxis=dict(gridcolor="rgba(100,116,139,0.15)", tickfont=dict(color="#94a3b8"),
                           title_font=dict(color="#94a3b8")),
                height=350,
                margin=dict(l=40, r=20, t=50, b=40),
                coloraxis_colorbar=dict(
                    tickfont=dict(color="#94a3b8"),
                    title=dict(font=dict(color="#94a3b8")),
                ),
            )
            # Highlight user's point
            fig_scatter.add_trace(go.Scatter(
                x=[charge_cycles], y=[degradation],
                mode="markers",
                marker=dict(size=14, color="#f87171", symbol="star", line=dict(width=2, color="#fff")),
                name="Your Battery",
                showlegend=True,
            ))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Correlation heatmap
        st.markdown('<div class="section-header">🔗 Feature Correlations</div>', unsafe_allow_html=True)
        corr = df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            labels=dict(color="Correlation"),
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(tickfont=dict(color="#94a3b8", size=10)),
            yaxis=dict(tickfont=dict(color="#94a3b8", size=10)),
            coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

else:
    # ─── Welcome / Landing State ───
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="value" style="color: #38bdf8;">🔋</div>
            <div class="label" style="margin-top: 0.5rem;">Smart Prediction</div>
            <div class="sub">ML-powered battery degradation analysis using Random Forest models trained on real EV data.</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="value" style="color: #818cf8;">📊</div>
            <div class="label" style="margin-top: 0.5rem;">Visual Insights</div>
            <div class="sub">Interactive gauges, radar charts, and dataset visualizations for deep health analysis.</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="value" style="color: #c084fc;">⚡</div>
            <div class="label" style="margin-top: 0.5rem;">Instant Results</div>
            <div class="sub">Get health verdict, risk assessment, and actionable recommendations in seconds.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card" style="text-align: center;">
        <p style="font-size: 1.1rem; color: #94a3b8;">
            👈 <strong style="color: #38bdf8;">Adjust the parameters</strong> in the sidebar and click
            <strong style="color: #818cf8;">Analyze Battery Health</strong> to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem 0 0.5rem; color: #475569; font-size: 0.8rem;">
    Built with ❤️ by <strong style="color: #64748b;">VoltGuard AI</strong> &nbsp;•&nbsp;
    Powered by Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)
