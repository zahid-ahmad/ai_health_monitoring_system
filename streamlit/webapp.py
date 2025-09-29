# health_assistant.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

DATA_FILE = "health_data.csv"

# Load existing data
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["Date", "Time", "Blood Pressure", "Heart Rate", "Glucose", "Mood", "Symptoms"])

st.title("🩺 AI-Powered Virtual Health Assistant")
st.write("Log your daily health data for remote monitoring.")

# Sidebar form for data entry
with st.sidebar.form("data_entry_form"):
    st.subheader("Enter Today's Health Data")
    
    # Vitals
    bp_systolic = st.number_input("Blood Pressure (Systolic)", min_value=80, max_value=200, value=120)
    bp_diastolic = st.number_input("Blood Pressure (Diastolic)", min_value=50, max_value=130, value=80)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
    glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=400, value=100)
    
    # Other health info
    symptoms = st.text_area("Symptoms (if any)", placeholder="e.g., headache, dizziness, cough...")
    mood = st.slider("Mood (1 = Very Low, 10 = Excellent)", 1, 10, 7)
    
    submitted = st.form_submit_button("Submit")

# Save data with timestamp
if submitted:
    now = datetime.now()
    new_entry = {
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Blood Pressure": f"{bp_systolic}/{bp_diastolic}",
        "Heart Rate": heart_rate,
        "Glucose": glucose,
        "Mood": mood,
        "Symptoms": symptoms
    }
    
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)  # Save to file
    st.success("✅ Data submitted successfully!")

# Display records
if not df.empty:
    st.subheader("📊 Your Health Log")
    st.dataframe(df)

    # Trend chart
    st.subheader("📈 Trends Over Time")
    df_plot = df.copy()
    df_plot["DateTime"] = pd.to_datetime(df_plot["Date"] + " " + df_plot["Time"])
    df_plot.set_index("DateTime", inplace=True)

    fig, ax = plt.subplots()
    df_plot[["Heart Rate", "Glucose", "Mood"]].plot(ax=ax, marker="o")
    plt.xlabel("Date & Time")
    plt.ylabel("Values")
    st.pyplot(fig)
