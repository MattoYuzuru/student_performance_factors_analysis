import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import requests
import os
import dotenv

dotenv.load_dotenv()

data = pd.read_csv('StudentPerformanceFactors.csv')
st.title("Hello")

with st.form("exam_score"):
    attendance = st.slider("Attendance", 0, 100)
    hours_studied = st.slider("Hours Studied", 0, 50)
    prev_score = st.slider("Previous Score", 0, 100)
    tutoring_sessions = st.slider("Tutoring Sessions", 0, 10)
    physical_activity = st.slider("Physical Activity", 0, 10)
    access_to_resources = st.select_slider("Access to Resources", options=["Low", "Medium", "High"])
    parental_involvement = st.select_slider("Parental Involvement", options=["Low", "Medium", "High"])

    submit = st.form_submit_button("Predict Score!")


if submit:
    data_to_send = {
        "attendance": attendance,
        "hours_studied": hours_studied,
        "prev_score": prev_score,
        "tutoring_sessions": tutoring_sessions,
        "physical_activity": physical_activity,
        "access_to_resources": access_to_resources,
        "parental_involvement": parental_involvement,
    }

    response = requests.post(f"{os.getenv("API_PATH")}/predict", json=data_to_send)

    st.text(response.text)
