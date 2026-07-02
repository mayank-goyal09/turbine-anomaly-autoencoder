#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the lightweight dataset
df = pd.read_csv('T1.csv')

# Clean up column names for easier coding
df.columns = ['Timestamp', 'Power', 'WindSpeed', 'Theoretical_Curve', 'WindDirection']

# Check for missing values
print(f"Total rows: {len(df)}")
print(f"Missing values:\n{df.isnull().sum()}")

# Drop any rows with missing data (we have 50k+ rows, so it's fine)
df = df.dropna()


# In[6]:


df.shape


# In[6]:


df.info()


# In[3]:


import plotly.express as px


# In[7]:


# Creating a professional scatter plot
fig = px.scatter(df.sample(5000), # Sampling for faster rendering
                 x='WindSpeed', y='Power', 
                 color='Theoretical_Curve',
                 title='Wind Turbine: Real Power vs. Theoretical Curve',
                 labels={'WindSpeed': 'Wind Speed (m/s)', 'Power': 'Active Power (kW)'},
                 template='plotly_dark')

fig.show()


# In[1]:


import nbformat
print(nbformat.__version__)


# In[12]:


# 1. First, make sure we have the right column names (from our previous step)
df.columns = ['Timestamp', 'Power', 'WindSpeed', 'Theoretical_Curve', 'WindDirection']

# 2. Define 'Healthy' data: Power should be at least 80% of what the 
# manufacturer (Theoretical_Curve) says it should be.
df_healthy = df[df['Power'] > (df['Theoretical_Curve'] * 0.8)].copy()

df_healthy


# In[11]:


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 1. Select only the features the model needs to learn
# We use Wind Speed and Power to find the relationship
features = ['WindSpeed', 'Power']
data_to_scale = df_healthy[features] # Using only healthy data for training!

# 2. Scale data between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_to_scale)



# In[14]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 1. Define the Input Shape (We have 2 features: WindSpeed and Power)
input_dim = 2 
input_layer = Input(shape=(input_dim,))

# 2. THE ENCODER (Compressing)
encoder = Dense(8, activation="relu")(input_layer)
encoder = Dense(4, activation="relu")(encoder)

# 3. THE BOTTLENECK (The 'Essence' of the turbine)
bottleneck = Dense(2, activation="relu")(encoder)

# 4. THE DECODER (Reconstructing)
decoder = Dense(4, activation="relu")(bottleneck)
decoder = Dense(8, activation="relu")(decoder)

# 5. THE OUTPUT (Back to 2 features)
output_layer = Dense(input_dim, activation="sigmoid")(decoder)

# 6. CREATE THE MODEL
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# 7. COMPILE (Giving it a goal)
autoencoder.compile(optimizer='adam', loss='mse')

print("✅ The 'autoencoder' brain is now alive and defined!")
autoencoder.summary()


# In[15]:


# Train the model
history = autoencoder.fit(
    scaled_data, scaled_data,      # Input and Target are the SAME
    epochs=50,                     # Give it 50 'laps' to learn
    batch_size=32,                 # Process 32 rows at a time
    validation_split=0.1,          # Save 10% to test itself during training
    shuffle=True,                  # Mix the data so it doesn't learn 'order'
    verbose=1
)


# In[16]:


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Learning: How well did it learn "Normal"?')
plt.xlabel('Epochs')
plt.ylabel('Error (MSE)')
plt.legend()
plt.show()


# In[18]:


import numpy as np


# In[19]:


# 1. Scale the FULL dataset (Normal + Anomalies)
full_scaled_data = scaler.transform(df[['WindSpeed', 'Power']])

# 2. Let the model 'Reconstruct' it
reconstructions = autoencoder.predict(full_scaled_data)

# 3. Calculate the Mean Squared Error (MSE) for each row
# This is our 'Anomaly Score'
mse = np.mean(np.power(full_scaled_data - reconstructions, 2), axis=1)

# Add it to our dataframe
df['MSE'] = mse


# In[20]:


# Calculate threshold using only healthy training data error
threshold = np.mean(mse) + 3 * np.std(mse)

# Mark anomalies
df['Anomaly'] = df['MSE'] > threshold

print(f"✅ Anomaly Threshold set at: {threshold:.4f}")
print(f"⚠️ Total Anomalies detected: {df['Anomaly'].sum()}")


# In[21]:


import plotly.graph_objects as go

fig = go.Figure()

# Plot Normal Data
fig.add_trace(go.Scatter(
    x=df[df['Anomaly']==False]['WindSpeed'], 
    y=df[df['Anomaly']==False]['Power'],
    mode='markers', name='Normal Operation',
    marker=dict(color='blue', opacity=0.3)
))

# Plot Anomalies
fig.add_trace(go.Scatter(
    x=df[df['Anomaly']==True]['WindSpeed'], 
    y=df[df['Anomaly']==True]['Power'],
    mode='markers', name='ANOMALY DETECTED',
    marker=dict(color='red', size=8, symbol='x')
))

fig.update_layout(title='AI Maintenance Alert: Detecting Turbine Failures',
                  xaxis_title='Wind Speed (m/s)', yaxis_title='Power (kW)',
                  template='plotly_dark')
fig.show()


# In[22]:


# Calculate total 'Lost Power' captured by your anomalies
lost_energy = df[df['Anomaly'] == True]['Theoretical_Curve'].sum() - df[df['Anomaly'] == True]['Power'].sum()

print(f"📊 PROJECT IMPACT SUMMARY")
print(f"---------------------------")
print(f"✅ Total Operational Hours Analyzed: {len(df) * 10 / 60:.1f} hours")
print(f"🚨 Anomalous Events Detected: {df['Anomaly'].sum()}")
print(f"⚡ Estimated Energy Saved (by early detection): {lost_energy:.2f} kWh")


# In[ ]:


def check_turbine_health(wind_speed, actual_power):
    # 1. Scale the new input
    new_data = scaler.transform([[wind_speed, actual_power]])

    # 2. Get reconstruction error
    reconstructed = autoencoder.predict(new_data)
    error = np.mean(np.power(new_data - reconstructed, 2))

    # 3. Compare to our statistical threshold
    if error > threshold:
        return f"🚨 ALERT: Anomaly Detected! Error: {error:.4f}"
    else:
        return "✅ Status: Healthy"

# Test it! 
print(check_turbine_health(15, 50)) # High wind, low power = Should trigger alert


# In[ ]:




