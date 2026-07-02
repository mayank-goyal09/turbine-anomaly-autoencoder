# 🌬️ Wind Turbine Health Checker (Anomaly Detection)

A machine learning application that detects operational anomalies in wind turbines by monitoring real-time telemetry data. The system uses a Deep Autoencoder neural network trained with TensorFlow/Keras on historical healthy operation data to recognize when a turbine's performance deviates from normal behavior.

---

## 🎯 Project Overview
Wind turbines are complex mechanical systems operating in harsh, variable environments. Early detection of anomalies prevents catastrophic failures and reduces maintenance costs. 

This repository implements:
1. **Autoencoder-based Anomaly Detection**: Neural network model trained solely on healthy turbine telemetry (Wind Speed and Active Power output) to learn normal operating dynamics.
2. **Interactive Streamlit Dashboard**: A user-friendly web interface with animations, custom styling, and sliders for real-time telemetry inputs to run diagnostics on demand.
3. **Robust Data Preprocessing**: Standard scaling and data filtering techniques to optimize input variables.

---

## 📊 Telemetry Features
- **Wind Speed (m/s)**: The velocity of wind driving the turbine.
- **LV Active Power Out (kW)**: Actual electrical power generated.
- **Theoretical Power Curve (kW)**: Manufacturer-specified expected power output for the given wind speed.
- **Wind Direction (°)**: The direction of incoming wind.

---

## 🤖 How the AI Model Works
An **Autoencoder** is trained to reconstruct input features (`WindSpeed` and `Power`) that match a healthy turbine profile. 
- During inference, if the input data represents normal, healthy operation, the reconstruction error (Mean Squared Error) remains extremely low.
- If the turbine is underperforming or behaving abnormally (e.g., due to blade damage or mechanical friction), the reconstruction error increases.
- An anomaly is flagged when the error exceeds a calculated threshold of **`0.0806`**.

---

## 🚀 Getting Started

### 📋 Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Git

### 🔧 Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/mayank-goyal09/turbine-anomaly-autoencoder.git
   cd turbine-anomaly-autoencoder
   ```

2. Install dependencies:
   ```bash
   pip install streamlit pandas numpy tensorflow scikit-learn joblib plotly nbformat
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📁 File Structure
- `app.py`: Streamlit dashboard code with modern glassmorphism styling and windmills animations.
- `main.py` / `main.ipynb`: Model training scripts/notebook that pre-process data, build, train and export the Autoencoder.
- `turbine_model.h5`: Trained Keras model weights.
- `scaler.pkl`: Fitted MinMaxScaler for telemetry features.
- `T1.csv`: Historical wind turbine telemetry dataset.
- `.gitignore`: Configured to exclude system and temporary files.

---

Developed by **Mayank Goyal**
