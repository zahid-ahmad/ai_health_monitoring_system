import streamlit as st
import pandas as pd
import os
from datetime import datetime
from transformers import pipeline

# -------------------------------
# Load a small Hugging Face QA model
# -------------------------------
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_model = load_qa_model()

# -------------------------------
# Data Storage
# -------------------------------
DATA_FILE = "health_data.csv"
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["DateTime", "Heart Rate", "BP Systolic", "BP Diastolic", "Oxygen", "Temperature"])

# -------------------------------
# App Layout
# -------------------------------
st.title("🏥 AI-Powered Virtual Health Assistant")

tabs = st.tabs(["📊 Health Log", "📈 Trends", "🤖 Chatbot Assistant"])

# -------------------------------
# Tab 1: Health Log
# -------------------------------
with tabs[0]:
    with st.form("health_form"):
        col1, col2 = st.columns(2)
        with col1:
            heart_rate = st.number_input("❤️ Heart Rate (bpm)", min_value=30, max_value=200, value=72)
            bp_systolic = st.number_input("🩸 BP - Systolic", min_value=80, max_value=200, value=120)
        with col2:
            bp_diastolic = st.number_input("🩸 BP - Diastolic", min_value=40, max_value=130, value=80)
            oxygen = st.number_input("🫁 Oxygen (%)", min_value=50, max_value=100, value=98)
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=30.0, max_value=43.0, value=36.8, step=0.1)
        
        submitted = st.form_submit_button("Submit Vitals")

    if submitted:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = {"DateTime": now, "Heart Rate": heart_rate, "BP Systolic": bp_systolic,
                    "BP Diastolic": bp_diastolic, "Oxygen": oxygen, "Temperature": temperature}
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.success("✅ Data submitted successfully!")

    st.subheader("📊 Your Health Log")
    st.dataframe(df)

# -------------------------------
# Tab 2: Trends
# -------------------------------
with tabs[1]:
    st.subheader("📈 Trends Over Time")
    if not df.empty:
        st.line_chart(df.set_index("DateTime")[["Heart Rate", "BP Systolic", "BP Diastolic", "Oxygen", "Temperature"]])
    else:
        st.info("No data available yet.")

# -------------------------------
# Tab 3: Chatbot
# -------------------------------
with tabs[2]:
    st.subheader("🤖 Health Assistant Chatbot")

    user_q = st.text_input("Ask me anything about health (e.g., What does high BP mean?)")
    if user_q:
        context = """
        High blood pressure (hypertension) increases the risk of heart disease and stroke. 
        Normal blood pressure is around 120/80 mmHg. 
        Oxygen saturation below 95% can be concerning. 
        A healthy body temperature is about 36.5–37.5 °C. 
        Heart rate between 60–100 bpm is normal for adults.
        """
        result = qa_model(question=user_q, context=context)
        st.write("💬 Assistant:", result["answer"])
