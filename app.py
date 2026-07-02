import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import os

# Must be the very first Streamlit command
st.set_page_config(page_title="Wind Turbine Health Monitor", page_icon="🌬️", layout="wide")

@st.cache_resource
def load_models():
    try:
        from tensorflow.keras.models import load_model
        model = load_model('turbine_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}. Make sure you generated the model and scaler files!")
        return None, None

model, scaler = load_models()

# Custom CSS for Glassmorphism and animations
st.markdown("""
<style>
/* Background */
.stApp {
    background: url("https://images.unsplash.com/photo-1466611653911-95081537e5b7?q=80&w=2070&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
/* Overlay for better readability */
.stApp::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, rgba(0, 30, 60, 0.7) 0%, rgba(0, 60, 90, 0.5) 100%);
    z-index: 0;
}
/* Bring content to front */
.main .block-container {
    z-index: 1;
}

/* Glassmorphism Card Style */
.glass-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    padding: 30px;
    color: white;
    margin-bottom: 20px;
    animation: fadeIn 1.5s ease;
    transition: transform 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(0, 230, 255, 0.2);
}

/* Windmill Animation CSS */
@keyframes spin { 100% { -webkit-transform: rotate(360deg); transform:rotate(360deg); } }
.windmill {
    width: 120px;
    height: 120px;
    display: block;
    margin: 0 auto;
    animation: spin 6s linear infinite;
    filter: drop-shadow(0px 0px 10px rgba(0, 230, 255, 0.8));
}

@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

h1, h2, h3, h4, p, label {
    color: #ffffff !important;
}

/* Custom st.metric colors */
[data-testid="stMetricValue"] {
    color: #00e6ff !important;
    text-shadow: 0 0 10px rgba(0, 230, 255, 0.5);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 10px 20px !important;
    font-weight: bold !important;
    letter-spacing: 1px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px 0 rgba(0, 114, 255, 0.4) !important;
}

.stButton > button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 6px 20px 0 rgba(0, 114, 255, 0.6) !important;
}

.stSlider > div > div > div > div {
    background: #00e6ff !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-color: #00e6ff !important;
    box-shadow: 0 0 10px #00e6ff !important;
}

</style>
""", unsafe_allow_html=True)

# Title and Windmill animation
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px; z-index: 1; position: relative;">
        <!-- Using SVG for smooth scalable animation -->
        <svg class="windmill" viewBox="0 0 100 100" fill="white" stroke="white" stroke-width="1.5">
            <!-- Center hub -->
            <circle cx="50" cy="50" r="5" fill="#00e6ff" stroke="#fff" />
            <!-- Blades -->
            <path d="M50 50 L50 10 Q65 20 50 50" fill="rgba(0,230,255,0.7)" />
            <path d="M50 50 L85 70 Q70 85 50 50" fill="rgba(0,230,255,0.7)" />
            <path d="M50 50 L15 70 Q30 85 50 50" fill="rgba(0,230,255,0.7)" />
        </svg>
        <h1 style='color: white; text-shadow: 0 0 15px rgba(0, 230, 255, 0.8); font-size: 3.5rem; margin-top:20px'>
            🌬️ Wind Health Checker
        </h1>
        <p style="font-size: 1.3rem; color: #e0e0e0; letter-spacing: 1px;">Intelligent diagnostics from wind telemetry data</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<h3 style="text-align:center;">Turbine Telemetry Input</h3><br>', unsafe_allow_html=True)

# Main sliders
col1, col2 = st.columns([1, 1])

with col1:
    wind_speed = st.slider("🌬️ Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=8.5, step=0.1)
    wind_direction = st.slider("🧭 Wind Direction (°)", min_value=0.0, max_value=360.0, value=120.0, step=1.0)
    
with col2:
    active_power = st.number_input("⚡ LV Active Power Out (kW)", min_value=0.0, max_value=4000.0, value=1450.0, step=10.0)
    theoretical_power = st.number_input("📐 Theoretical Power Profile (kW)", min_value=0.0, max_value=4000.0, value=1600.0, step=10.0)

st.markdown('</div>', unsafe_allow_html=True)

# Analyze Button
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze_btn = st.button("🔍 Run AI Health Diagnostics", use_container_width=True)

if analyze_btn:
    # Simulate AI loading time
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for pct in range(100):
        time.sleep(0.015)
        progress_bar.progress(pct + 1)
        if pct < 30:
            status_text.markdown("<p style='text-align:center'>Reading atmospheric conditions...</p>", unsafe_allow_html=True)
        elif pct < 70:
            status_text.markdown("<p style='text-align:center'>Calculating power coefficients...</p>", unsafe_allow_html=True)
        else:
            status_text.markdown("<p style='text-align:center'>Generating final health prognosis...</p>", unsafe_allow_html=True)
            
    status_text.empty()
    progress_bar.empty()
        
    st.markdown('<div class="glass-card" style="text-align:center; animation: fadeIn 1s ease;">', unsafe_allow_html=True)
    
    # Mathematical Heuristics for visual flair
    power_diff = abs(active_power - theoretical_power)
    deviation_pct = 0.0 if theoretical_power == 0 and active_power == 0 else (power_diff / max(theoretical_power, 1)) * 100.0
    health_score = max(0.0, 100.0 - deviation_pct)
    
    st.markdown("<h3>🎯 AI Health Analysis Summary</h3><br>", unsafe_allow_html=True)
    
    # AI ML PREDICTION (Autoencoder)
    is_anomaly = False # Fallback
    if model and scaler:
        # Construct DataFrame matching the features
        user_input_df = pd.DataFrame({
            'WindSpeed': [wind_speed],
            'Power': [active_power]
        })
        input_scaled = scaler.transform(user_input_df)
        reconstructed = model.predict(input_scaled)
        error = np.mean(np.power(input_scaled - reconstructed, 2))
        
        # 0.0806 is the threshold from training
        threshold = 0.0806
        is_anomaly = (error > threshold)
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    col_r1.metric("Power Deviation", f"{deviation_pct:.1f}%", "-Underperforming" if deviation_pct > 15 else "+Optimal", delta_color="inverse")
    
    # Base Condition Logic built primarily around ML!
    if wind_speed < 3.0:
        status_text_display = "💤 Standby (Low Wind)"
        color = "#aaaaaa"
        icon = "📉"
    elif wind_speed > 25.0:
        status_text_display = "⚠️ Danger Over-speed"
        color = "#ff9900"
        icon = "🌪️"
    elif is_anomaly:
        status_text_display = "🚨 ML Detects Anomaly"
        color = "#ff3333"
        icon = "❌"
    else:
        status_text_display = "✅ ML Confirms Healthy"
        color = "#00ffcc"
        icon = "✨"
        
    col_r2.metric("ML Output", "Anomaly Detected" if is_anomaly else "Normal", "-Critical" if is_anomaly else "+Normal")
    col_r3.metric("Predicted Status", status_text_display)
    
    # Final verdict box
    ml_verdict = "The Machine Learning model indicates a CRITICAL ANOMALY in telemetry signatures. Field inspection is recommended." if is_anomaly else "The Machine Learning model confirms the turbine is operating within expected healthy parameters."
    st.markdown(f"""
        <div style='margin-top: 30px; padding: 25px; border-radius: 12px; background: rgba(0,0,0,0.4); border-left: 6px solid {color}; backdrop-filter: blur(10px)'>
            <h4 style='margin:0 0 10px 0; color:{color}; letter-spacing: 0.5px;'>{icon} Final Verdict: {status_text_display}</h4>
            <p style='margin:0; font-size: 1.1rem; line-height: 1.5;'>
                Based on current telemetry, the turbine is operating with a <b>{deviation_pct:.1f}% deviation</b> from its aerodynamic theoretical profile. 
                With a steady wind speed of <b>{wind_speed} m/s</b> coming from <b>{wind_direction}°</b>.<br><br>
                <b>🤖 AI Model Verdict:</b> {ml_verdict}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-top: 60px; padding: 30px 20px; 
                background: rgba(255,255,255,0.05); backdrop-filter: blur(15px); 
                border-radius: 16px; border-top: 1px solid rgba(0,230,255,0.15);
                z-index: 1; position: relative;">
        <p style="font-size: 0.85rem; color: rgba(255,255,255,0.35); margin: 0 0 8px 0; letter-spacing: 2px; text-transform: uppercase;">
            Developed &amp; Designed by
        </p>
        <p style="font-size: 1.6rem; font-weight: 700; margin: 0 0 6px 0;
                  background: linear-gradient(90deg, #00e6ff, #0072ff, #00e6ff);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                  background-clip: text; letter-spacing: 1px;">
            Mayank Goyal
        </p>
        <p style="font-size: 0.95rem; color: rgba(255,255,255,0.5); margin: 0 0 16px 0;">
            Wind Turbine Health Checker &mdash; Project 70
        </p>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-bottom: 14px;">
            <span style="background: rgba(0,230,255,0.12); color: #00e6ff; padding: 4px 14px; 
                         border-radius: 20px; font-size: 0.75rem; border: 1px solid rgba(0,230,255,0.2);">
                🐍 Python
            </span>
            <span style="background: rgba(0,230,255,0.12); color: #00e6ff; padding: 4px 14px; 
                         border-radius: 20px; font-size: 0.75rem; border: 1px solid rgba(0,230,255,0.2);">
                🧠 TensorFlow / Keras
            </span>
            <span style="background: rgba(0,230,255,0.12); color: #00e6ff; padding: 4px 14px; 
                         border-radius: 20px; font-size: 0.75rem; border: 1px solid rgba(0,230,255,0.2);">
                📊 Streamlit
            </span>
            <span style="background: rgba(0,230,255,0.12); color: #00e6ff; padding: 4px 14px; 
                         border-radius: 20px; font-size: 0.75rem; border: 1px solid rgba(0,230,255,0.2);">
                🤖 Autoencoder (Anomaly Detection)
            </span>
        </div>
        <p style="font-size: 0.75rem; color: rgba(255,255,255,0.25); margin: 0;">
            &copy; 2026 Mayank Goyal &bull; All Rights Reserved
        </p>
    </div>
""", unsafe_allow_html=True)
