import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -------------------------
# Load Dataset & Train Model
# -------------------------
data = pd.read_csv("weather_dataset.csv")

X = data[['Temperature', 'Humidity', 'WindSpeed']]
y = data['Rain']

model = LogisticRegression()
model.fit(X, y)

# -------------------------
# UI Design
# -------------------------
st.set_page_config(page_title="Weather AI", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
    🌦️ AI Weather Prediction System
    </h1>
""", unsafe_allow_html=True)

st.write("### Enter Weather Parameters")

# Input Section
col1, col2 = st.columns(2)

with col1:
    temp = st.slider("🌡️ Temperature (°C)", 15, 45, 30)

with col2:
    humidity = st.slider("💧 Humidity (%)", 30, 100, 70)

wind = st.slider("🌬️ Wind Speed (km/h)", 0, 25, 10)

st.divider()

# -------------------------
# Prediction Button
# -------------------------
if st.button("🔍 Predict Weather"):

    prediction = model.predict([[temp, humidity, wind]])
    prob = model.predict_proba([[temp, humidity, wind]])

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.success("🌧️ Rain Expected")
    else:
        st.info("☀️ No Rain Expected")

    st.write(f"**Confidence:** {max(prob[0])*100:.2f}%")

    st.progress(int(max(prob[0])*100))

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Built using Machine Learning + Streamlit")
