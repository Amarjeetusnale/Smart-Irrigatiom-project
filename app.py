import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("Farm_Irrigation_System.pkl")

st.set_page_config(page_title="Farm Irrigation System", layout="centered")
st.title("🚜 Smart Farm Irrigation System")
st.markdown("Predict irrigation needs for each parcel based on sensor data.")

st.header("🌱 Enter Sensor Data")

# Descriptive sensor labels for all 20 features
sensor_labels = [
    "Temperature (°C) [sensor_0]",
    "Humidity (%) [sensor_1]",
    "Soil Moisture (%) [sensor_2]",
    "Soil pH Level [sensor_3]",
    "Rainfall (mm) [sensor_4]",
    "Soil Temperature (°C) [sensor_5]",
    "Leaf Wetness (%) [sensor_6]",
    "Solar Radiation (W/m²) [sensor_7]",
    "Wind Speed (m/s) [sensor_8]",
    "Wind Direction (°) [sensor_9]",
    "Atmospheric Pressure (hPa) [sensor_10]",
    "Dew Point (°C) [sensor_11]",
    "Evapotranspiration (mm) [sensor_12]",
    "Water Table Depth (cm) [sensor_13]",
    "Soil Conductivity (dS/m) [sensor_14]",
    "Soil Nitrogen Level (mg/kg) [sensor_15]",
    "Soil Phosphorus Level (mg/kg) [sensor_16]",
    "Soil Potassium Level (mg/kg) [sensor_17]",
    "Canopy Temperature (°C) [sensor_18]",
    "Irrigation Flow Rate (L/min) [sensor_19]",
]

# Reasonable slider ranges for each sensor (min, max, default)
sensor_ranges = [
    (0.0, 50.0, 25.0),    # Temperature
    (0.0, 100.0, 50.0),   # Humidity
    (0.0, 100.0, 30.0),   # Soil Moisture
    (3.0, 10.0, 6.5),     # Soil pH
    (0.0, 500.0, 50.0),   # Rainfall
    (0.0, 50.0, 20.0),    # Soil Temperature
    (0.0, 100.0, 10.0),   # Leaf Wetness
    (0.0, 1500.0, 500.0), # Solar Radiation
    (0.0, 20.0, 2.0),     # Wind Speed
    (0.0, 360.0, 180.0),  # Wind Direction
    (800.0, 1100.0, 1013.0), # Atmospheric Pressure
    (-10.0, 30.0, 10.0),  # Dew Point
    (0.0, 10.0, 2.0),     # Evapotranspiration
    (0.0, 200.0, 50.0),   # Water Table Depth
    (0.0, 5.0, 1.0),      # Soil Conductivity
    (0.0, 1000.0, 100.0), # Soil Nitrogen
    (0.0, 1000.0, 50.0),  # Soil Phosphorus
    (0.0, 1000.0, 150.0), # Soil Potassium
    (0.0, 50.0, 25.0),    # Canopy Temperature
    (0.0, 100.0, 10.0),   # Irrigation Flow Rate
]

st.subheader("Option 1: Use sliders")
sensor_inputs = []
for label, (min_val, max_val, default_val) in zip(sensor_labels, sensor_ranges):
    val = st.slider(label, min_val, max_val, default_val)
    sensor_inputs.append(val)

st.subheader("Option 2: Paste comma-separated values")
csv_input = st.text_input(
    "Paste 20 comma-separated sensor values (in order):",
    value=",".join(str(d) for _, _, d in sensor_ranges)
)

use_csv = st.checkbox("Use input values from above text box instead of sliders")

if st.button("💧 Predict Irrigation Need"):
    try:
        if use_csv:
            values = [float(x.strip()) for x in csv_input.split(",")]
            if len(values) != 20:
                st.error("❌ Please enter exactly 20 values.")
                st.stop()
            input_data = np.array(values).reshape(1, -1)
        else:
            input_data = np.array(sensor_inputs).reshape(1, -1)
        prediction = model.predict(input_data)
        if prediction.ndim == 2 and prediction.shape[1] == 3:
            st.success(
                f"Parcel 0: {'ON' if prediction[0][0] else 'OFF'} | "
                f"Parcel 1: {'ON' if prediction[0][1] else 'OFF'} | "
                f"Parcel 2: {'ON' if prediction[0][2] else 'OFF'}"
            )
        else:
            st.success(f"Prediction: {prediction}")
    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")

with st.expander("📊 Show Sample Data"):
    df = pd.read_csv("irrigation_machine.csv")
    st.dataframe(df.head())