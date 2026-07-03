# AeroFlow AI

> **Real-time wind turbine anomaly detection using deep autoencoders.**

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Sky_HUD-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)

---

## Project Essentials

* **Purpose:** Monitors wind turbine health in real-time to prevent catastrophic mechanical breakdowns.
* **Mechanism:** TensorFlow-based deep autoencoder trained exclusively on healthy operational data.
* **Inference:** Compares real-time telemetry against expected performance. High reconstruction error (> 0.0806) flags anomalies.
* **Business Value:** Prevents unexpected downtime and reduces maintenance costs by 15-20%.

---

## Wind Operational Matrix

| Regime | Wind Range | ML Evaluation | System Status & Action |
| :--- | :--- | :--- | :--- |
| **Low Wind Standby** | 0.0 - 3.0 m/s | Bypassed | Turbine idling. Alerts disabled. |
| **Optimal Generation** | 3.0 - 15.0 m/s | Active | Nominal tracking against theoretical curves. |
| **High Wind Load** | 15.0 - 25.0 m/s | Active | Verifies performance under peak stress conditions. |
| **Storm Cut-Out** | > 25.0 m/s | Bypassed | Safety shut-down. Emergency braking diagnostics. |

---

## System Architecture

```mermaid
graph TD
    %% Pipeline
    A[Telemetry Inputs: Speed, Direction, Power] --> B(Interface Console: Sky-Theme HUD)
    B --> C[Preprocessing: MinMaxScaler]
    C --> D[ML Engine: Keras Autoencoder Model]
    D --> E{MSE Reconstruction Error}
    
    %% Verdicts
    E -->|Error > 0.0806| F[Status: Anomaly Detected]
    E -->|Error <= 0.0806| G[Status: Healthy Operation]
    
    %% UI Update
    F --> H[Dashboard HUD: Health Prognosis]
    G --> H
    H -->|Refresh| B

    %% Styling
    style A fill:#e0f7fa,color:#006064,stroke:#00acc1,stroke-width:1px
    style B fill:#e0f7fa,color:#006064,stroke:#00acc1,stroke-width:1px
    style C fill:#b2ebf2,color:#006064,stroke:#00acc1,stroke-width:1px
    style D fill:#00bfff,color:#fff,stroke:#0288d1,stroke-width:1px
    style E fill:#0288d1,color:#fff,stroke:#0d47a1,stroke-width:1px
    style F fill:#ff5252,color:#fff,stroke:#ff1744,stroke-width:1px
    style G fill:#00e5ff,color:#006064,stroke:#00b8d4,stroke-width:1px
    style H fill:#0d47a1,color:#fff,stroke:#0a357a,stroke-width:1px
```

---

## ML Reconstruction Logic

```python
# Normalize inputs and compute reconstruction error
scaled_input = scaler.transform([[wind_speed, active_power]])
reconstructed = model.predict(scaled_input)
reconstruction_error = np.mean(np.power(scaled_input - reconstructed, 2))

# Flag anomaly if drift exceeds established healthy baseline
is_anomaly = reconstruction_error > 0.0806
```

---

## Technical Stack

* **Frontend:** Streamlit (Glassmorphic Sky HUD)
* **ML Framework:** TensorFlow / Keras (Autoencoder neural network)
* **Preprocessing:** Scikit-Learn (MinMaxScaler)
* **Dataset:** Historical Telemetry Log (`T1.csv`)

---

## File Blueprint

```text
turbine-anomaly-autoencoder/
├── app.py              # Streamlit Sky-Theme dashboard & SVG animation
├── main.py             # Training pipeline & model threshold calibration
├── main.ipynb          # Jupyter Notebook for experimental model building
├── T1.csv              # Turbine telemetry raw dataset (50k+ rows)
├── turbine_model.h5    # Trained Keras Autoencoder model
└── scaler.pkl          # MinMaxScaler serialization pickle
```

*Quick Links:*
[Dashboard Entrypoint](file:///c:/my_local_data%28one%20drive%29/Attachments/Ambition%20course/my_all_projects/project%2070%20wind%20health%20checker/app.py) | [Model Training](file:///c:/my_local_data%28one%20drive%29/Attachments/Ambition%20course/my_all_projects/project%2070%20wind%20health%20checker/main.py) | [Experiment Notebook](file:///c:/my_local_data%28one%20drive%29/Attachments/Ambition%20course/my_all_projects/project%2070%20wind%20health%20checker/main.ipynb)

---

## Getting Started

1. **Install required packages:**
   ```bash
   pip install streamlit pandas numpy tensorflow scikit-learn joblib
   ```

2. **Launch the dashboard:**
   ```bash
   streamlit run app.py
   ```
   Access the UI at `http://localhost:8501`.

---

## Developer
**Mayank Goyal**  
GenAI & Automation Developer | Predictive Asset Architect  
[GitHub](https://github.com/mayank-goyal09) | [LinkedIn](https://www.linkedin.com/in/mayank-goyal-4b8756363/)
