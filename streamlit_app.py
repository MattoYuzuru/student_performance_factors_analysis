import os
import time

import dotenv
import pandas as pd
import requests
import streamlit as st

dotenv.load_dotenv()

data = pd.read_csv('StudentPerformanceFactors.csv')

st.title("🎓 Predict Your Exam Score")

with st.form("exam_score"):
    attendance = st.slider("📅 Attendance (%)", 0, 100)
    hours_studied = st.slider("📚 Hours Studied per Week", 0, 50)
    prev_score = st.slider("📊 Previous Score (%)", 0, 100)
    tutoring_sessions = st.slider("👩‍🏫 Tutoring Sessions per Week", 0, 10)
    physical_activity = st.slider("🏃‍ Physical Activity per Week (Hours)", 0, 10)
    access_to_resources = st.select_slider(
        "📖 Access to Learning Resources",
        options=["Low", "Medium", "High"],
    )
    parental_involvement = st.select_slider(
        "👨‍👩‍👧 Parental Involvement",
        options=["Low", "Medium", "High"],
    )

    submit = st.form_submit_button("📈 Predict Score!")

if submit:
    status_placeholder = st.empty()

    fun_messages = [
        "📄 Sharpening your pencil...",
        "🔍 Reviewing your notes...",
        "🎯 Setting up the exam environment..."
    ]

    for msg in fun_messages:
        status_placeholder.info(msg)
        time.sleep(1.5)

    data_to_send = {
        "attendance": attendance,
        "hours_studied": hours_studied,
        "prev_score": prev_score,
        "tutoring_sessions": tutoring_sessions,
        "physical_activity": physical_activity,
        "access_to_resources": access_to_resources,
        "parental_involvement": parental_involvement,
    }

    try:
        response = requests.post(f"{os.getenv('API_PATH')}/predict", json=data_to_send)
        response.raise_for_status()
        status_placeholder.empty()
        st.success("🎉 Prediction Successful!")
        st.markdown(
            f'''<div style="padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9;">
        <h3 style="color: #4CAF50;">
        📈 Predicted Exam Score: {response.text}</h3>
            </div>''', unsafe_allow_html=True
        )

    except requests.exceptions.RequestException as e:
        status_placeholder.empty()
        st.error("🚨 Failed to get a prediction. Please try again.")
        st.code(str(e))
