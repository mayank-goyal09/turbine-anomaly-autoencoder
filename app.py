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
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply font to Streamlit elements */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
}

/* Background image & gradient overlay */
.stApp {
    background: url("https://images.unsplash.com/photo-1500485035595-cbe6f645feb1?q=80&w=2074&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
/* Cloud-like glass overlay */
.stApp::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, rgba(15, 34, 64, 0.65) 0%, rgba(98, 172, 222, 0.45) 100%);
    z-index: 0;
}
/* Bring content to front */
.main .block-container {
    z-index: 1;
}

/* Glassmorphism Card Style */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(25px) saturate(140%);
    -webkit-backdrop-filter: blur(25px) saturate(140%);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 12px 40px 0 rgba(135, 206, 250, 0.1);
    padding: 30px;
    color: white;
    margin-bottom: 25px;
    animation: fadeIn 1.2s cubic-bezier(0.25, 1, 0.5, 1);
    transition: all 0.4s cubic-bezier(0.25, 1, 0.5, 1);
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 50px 0 rgba(0, 191, 255, 0.25);
    border: 1px solid rgba(0, 191, 255, 0.4);
}

/* Windmill Animation CSS */
@keyframes spin { 
    100% { -webkit-transform: rotate(360deg); transform:rotate(360deg); } 
}
.windmill {
    width: 130px;
    height: 130px;
    display: block;
    margin: 0 auto;
    filter: drop-shadow(0px 0px 15px rgba(0, 191, 255, 0.85));
    transition: filter 0.3s ease;
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
    color: #00e5ff !important;
    text-shadow: 0 0 12px rgba(0, 229, 255, 0.5);
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] > div > p {
    color: #e0f7fa !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(90deg, #00bfff 0%, #007acc 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 12px 28px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px;
    transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1) !important;
    box-shadow: 0 4px 15px 0 rgba(0, 191, 255, 0.3) !important;
}

.stButton > button:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 6px 22px 0 rgba(0, 191, 255, 0.6) !important;
    background: linear-gradient(90deg, #00d2ff 0%, #0099ff 100%) !important;
}

/* Slider Track & Active state */
.stSlider > div > div > div > div {
    background: #00bfff !important;
}
.stSlider [data-baseweb="slider"] {
    background: rgba(255,255,255,0.2) !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-color: #00bfff !important;
    box-shadow: 0 0 12px #00bfff !important;
}

</style>
""", unsafe_allow_html=True)

# Create a placeholder at the top for the dynamic header + windmill
header_placeholder = st.empty()

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<h3 style="text-align:center; font-weight: 600; letter-spacing: 0.5px;">Turbine Telemetry Input Panel</h3><br>', unsafe_allow_html=True)

# Main sliders
col1, col2 = st.columns([1, 1])

with col1:
    wind_speed = st.slider("🌬️ Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=8.5, step=0.1)
    wind_direction = st.slider("🧭 Wind Direction (°)", min_value=0.0, max_value=360.0, value=120.0, step=1.0)
    
with col2:
    active_power = st.number_input("⚡ LV Active Power Out (kW)", min_value=0.0, max_value=4000.0, value=1450.0, step=10.0)
    theoretical_power = st.number_input("📐 Theoretical Power Profile (kW)", min_value=0.0, max_value=4000.0, value=1600.0, step=10.0)

st.markdown('</div>', unsafe_allow_html=True)

# Calculate spin duration based on current wind speed dynamically
if wind_speed == 0:
    animation_style = "animation: none;"
else:
    # Scale: higher wind = faster rotation (lower animation-duration)
    spin_duration = max(0.3, 15.0 / wind_speed)
    animation_style = f"animation: spin {spin_duration:.2f}s linear infinite;"

# Render the dynamic header into the placeholder
header_placeholder.markdown(f"""
    <div style="text-align: center; margin-bottom: 30px; z-index: 1; position: relative;">
        <!-- SVG windmill animation dynamic spin -->
        <svg class="windmill" style="{animation_style}" viewBox="0 0 100 100" fill="white" stroke="white" stroke-width="1.5">
            <!-- Center hub -->
            <circle cx="50" cy="50" r="5" fill="#00e5ff" stroke="#fff" />
            <!-- Blades -->
            <path d="M50 50 L50 10 Q65 20 50 50" fill="rgba(135, 206, 250, 0.8)" />
            <path d="M50 50 L85 70 Q70 85 50 50" fill="rgba(135, 206, 250, 0.8)" />
            <path d="M50 50 L15 70 Q30 85 50 50" fill="rgba(135, 206, 250, 0.8)" />
        </svg>
        <h1 style='color: white; text-shadow: 0 0 18px rgba(0, 191, 255, 0.8); font-size: 3.5rem; margin-top:20px; font-weight: 700;'>
            🌬️ Wind Health Checker
        </h1>
        <p style="font-size: 1.3rem; color: #e0f7fa; letter-spacing: 1.5px; font-weight: 300;">Intelligent diagnostics from wind telemetry data</p>
    </div>
""", unsafe_allow_html=True)

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
        
    st.markdown('<div class="glass-card" style="text-align:center; animation: fadeIn 1.0s ease;">', unsafe_allow_html=True)
    
    # Mathematical Heuristics for visual flair
    power_diff = abs(active_power - theoretical_power)
    deviation_pct = 0.0 if theoretical_power == 0 and active_power == 0 else (power_diff / max(theoretical_power, 1)) * 100.0
    health_score = max(0.0, 100.0 - deviation_pct)
    
    st.markdown("<h3 style='font-weight:600; letter-spacing:0.5px;'>🎯 AI Health Analysis Summary</h3><br>", unsafe_allow_html=True)
    
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
        color = "#b0bec5"
        icon = "📉"
    elif wind_speed > 25.0:
        status_text_display = "⚠️ Danger Over-speed"
        color = "#ffb74d"
        icon = "🌪️"
    elif is_anomaly:
        status_text_display = "🚨 ML Detects Anomaly"
        color = "#ff5252"
        icon = "❌"
    else:
        status_text_display = "✅ ML Confirms Healthy"
        color = "#00e5ff"
        icon = "✨"
        
    col_r2.metric("ML Output", "Anomaly Detected" if is_anomaly else "Normal", "-Critical" if is_anomaly else "+Normal")
    col_r3.metric("Predicted Status", status_text_display)
    
    # Final verdict box
    ml_verdict = "The Machine Learning model indicates a CRITICAL ANOMALY in telemetry signatures. Field inspection is recommended." if is_anomaly else "The Machine Learning model confirms the turbine is operating within expected healthy parameters."
    st.markdown(f"""
        <div style='margin-top: 30px; padding: 25px; border-radius: 16px; background: rgba(0,0,0,0.35); border-left: 6px solid {color}; backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.08); border-left-width: 6px; text-align: left;'>
            <h4 style='margin:0 0 12px 0; color:{color}; font-weight:600; letter-spacing: 0.5px;'>{icon} Final Verdict: {status_text_display}</h4>
            <p style='margin:0; font-size: 1.1rem; line-height: 1.6; color: #eceff1 !important;'>
                Based on current telemetry, the turbine is operating with a <b>{deviation_pct:.1f}% deviation</b> from its aerodynamic theoretical profile. 
                With a steady wind speed of <b>{wind_speed} m/s</b> coming from <b>{wind_direction}°</b>.<br><br>
                <b>🤖 AI Model Verdict:</b> {ml_verdict}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer credits
st.markdown("""
    <div style="text-align: center; margin-top: 60px; padding: 30px 20px; 
                background: rgba(255,255,255,0.04); backdrop-filter: blur(15px); 
                border-radius: 20px; border-top: 1px solid rgba(0,191,255,0.15);
                z-index: 1; position: relative;">
        <p style="font-size: 0.85rem; color: rgba(255,255,255,0.4); margin: 0 0 8px 0; letter-spacing: 2px; text-transform: uppercase;">
            Developed &amp; Designed by
        </p>
        <p style="font-size: 1.6rem; font-weight: 700; margin: 0 0 6px 0;
                  background: linear-gradient(90deg, #00e5ff, #007acc, #00e5ff);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                  background-clip: text; letter-spacing: 1px;">
            Mayank Goyal
        </p>
        <p style="font-size: 0.95rem; color: rgba(255,255,255,0.55); margin: 0 0 16px 0;">
            Wind Turbine Health Checker &mdash; Project 70
        </p>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-bottom: 14px;">
            <span style="background: rgba(0,229,255,0.12); color: #00e5ff; padding: 4px 14px; 
                         border-radius: 20px; font-size: 0.75rem; border: 1px solid rgba(0,229,255,0.25);">
                🐍 Python
            </span>
            <span style="background: rgba(0,229,255,0.12); color: #00e5ff; padding: 4px 14px; 
                         border-radius: 20px; font-size: 0.75rem; border: 1px solid rgba(0,229,255,0.25);">
                🧠 TensorFlow / Keras
            </span>
            <span style="background: rgba(0,229,255,0.12); color: #00e5ff; padding: 4px 14px; 
                         border-radius: 20px; font-size: 0.75rem; border: 1px solid rgba(0,229,255,0.25);">
                📊 Streamlit
            </span>
            <span style="background: rgba(0,229,255,0.12); color: #00e5ff; padding: 4px 14px; 
                         border-radius: 20px; font-size: 0.75rem; border: 1px solid rgba(0,229,255,0.25);">
                🤖 Autoencoder (Anomaly Detection)
            </span>
        </div>
        <p style="font-size: 0.75rem; color: rgba(255,255,255,0.3); margin: 0;">
            &copy; 2026 Mayank Goyal &bull; All Rights Reserved
        </p>
    </div>
""", unsafe_allow_html=True)

