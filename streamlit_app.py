import os
import time

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from streamlit import cache_data



@cache_data
def load_and_preprocess_data():
    data = pd.read_csv('StudentPerformanceFactors.csv')
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    columns_to_map = ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities']
    changing = {'Yes': True, 'No': False}
    for col in columns_to_map:
        data[col] = data[col].map(changing)

    return data


@cache_data
def additional_columns(data):
    data['Attendance_Rough'] = pd.cut(
        data['Attendance'],
        bins=[0, 70, 85, 100],
        labels=['Low', 'Medium', 'High'],
        right=True
    )

    data['Study_Hours_Rough'] = pd.cut(
        data['Hours_Studied'],
        bins=[0, 10, 20, 50],
        labels=['Low', 'Medium', 'High']
    )
    return data


@cache_data
def generate_correlation_matrix(data):
    categorical_columns = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                           'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                           'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Gender',
                           'Parental_Education_Level', 'Distance_from_Home']

    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    correlation_matrix = data_encoded.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .75},
        annot=False

    )
    plt.title("Correlation Matrix")

    st.header("📈 Correlations with Exam Score")
    exam_score_correlation = correlation_matrix['Exam_Score'].sort_values(ascending=False)
    st.write(exam_score_correlation)

    return fig


@cache_data
def plot_feature_importance(data):
    tmp_data = data.copy()
    categorical_columns = [
        "Access_to_Resources", "Parental_Involvement", "Peer_Influence", "Family_Income",
        "Parental_Education_Level", "Distance_from_Home", "Motivation_Level", "Teacher_Quality",
        "Learning_Disabilities", "Extracurricular_Activities", "School_Type", "Gender", "Internet_Access"
    ]
    for col in categorical_columns:
        tmp_data[col] = LabelEncoder().fit_transform(tmp_data[col])

    X = tmp_data.drop("Exam_Score", axis=1)
    y = tmp_data["Exam_Score"]

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    factor_importance = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    factor_importance.sort_values().plot(kind='barh', ax=ax)
    plt.title("Factor Importance")
    return fig


@cache_data
def plot_distributions(data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(data=data, x='Exam_Score', alpha=0.8, color='#c505a7', ax=axes[0, 0])
    axes[0, 0].set_title('Exam Score Distribution')

    sns.histplot(data=data, x='Previous_Scores', alpha=0.8, color='#7617ac', bins=15, ax=axes[0, 1])
    axes[0, 1].set_title('Previous Scores Distribution')

    sns.histplot(data=data, x='Attendance', alpha=0.8, color='#68ecdb', ax=axes[1, 0])
    axes[1, 0].set_title('Attendance Distribution')

    sns.histplot(data=data, x='Hours_Studied', alpha=0.8, color='#9780fb', ax=axes[1, 1])
    axes[1, 1].set_title('Hours Studied Distribution')

    plt.tight_layout()
    return fig


@cache_data
def pie_chart(data):
    tdata = data['Attendance_Rough'].value_counts()

    labels = tdata.index
    values = tdata.values

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_traces(textinfo='percent+label', hole=0.4)  # Add optional styling for better visualization
    fig.update_layout(title_text="Attendance Rough Distribution", title_x=0.5)

    return fig


@cache_data
def scatterplot_3d(data):
    sampled_data2 = data.sample(n=500, random_state=52)
    fig2 = px.scatter_3d(
        sampled_data2,
        x='Attendance',
        y='Hours_Studied',
        z='Exam_Score',
        color='Exam_Score',
        size='Exam_Score',
        title="3D Plot: Key Factors vs Exam Score",
        labels={'Attendance': 'Attendance (%)', 'Hours_Studied': 'Study Hours', 'Exam_Score': 'Exam Score'}
    )

    return fig2


@cache_data
def pairplot(data):
    sampled_data1 = data.sample(n=500, random_state=52)

    fig = sns.pairplot(
        sampled_data1,
        vars=[
            'Exam_Score',
            'Hours_Studied',
            'Previous_Scores',
            'Attendance',
            'Tutoring_Sessions'
        ],
        hue='Attendance_Rough',
        palette='viridis'
    )
    return fig


@cache_data
def other_plot1(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=data, x='Hours_Studied', y='Exam_Score', ax=ax)
    ax.set_xticks(np.arange(0, 50, 5))
    ax.set_title('Hours Studied Distribution')
    return fig


@cache_data
def other_plot2(data):
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(
        data=data,
        x='Sleep_Hours',
        y='Attendance',
        hue='Distance_from_Home',
        palette='viridis',
        dodge=True,
        ax=ax
    )
    ax.set_title('Attendance vs Sleep Hours by Distance from Home')
    ax.legend(title='Distance from Home')
    return fig


@cache_data
def other_plot3(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=data, x='Attendance', y='Exam_Score', hue='Access_to_Resources', ax=ax)
    ax.set_yticks(np.arange(60, 81, 2))
    st.pyplot(fig)
    return fig


@cache_data
def other_plot4(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(
        data=data,
        x='Access_to_Resources',
        y='Exam_Score',
        hue='Attendance_Rough',
        ax=ax
    )
    ax.set_yticks(np.arange(50, 101, 5))
    ax.set_title('Impact of Access to Resources by Attendance')
    ax.set_xlabel('Access to Resources')
    ax.set_ylabel('Exam Score')

    return fig


@cache_data
def other_plot5(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x='Tutoring_Sessions',
        y='Exam_Score',
        data=data,
        palette='Set3',
        hue='Tutoring_Sessions',
        ax=ax
    )
    ax.set_yticks(np.arange(50, 101, 5))
    ax.set_xlabel('Tutoring Sessions')
    ax.set_ylabel('Exam Score')
    ax.set_title('Number of Tutoring Sessions vs Exam Score')
    return fig


@cache_data
def other_plot6(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=data,
        x="Parental_Involvement",
        y="Exam_Score",
        hue="Parental_Education_Level",
        ax=ax
    )
    ax.set_xlabel('Parental Involvement')
    ax.set_ylabel('Exam Score')
    ax.set_yticks(np.arange(50, 101, 2))
    ax.set_title('Parental Involvement vs Exam Score by Education Level')
    return fig


def main():
    data = load_and_preprocess_data()

    st.title("📊 Data Analysis and Visualization")

    st.header("🔍 Correlation Heatmap")
    correlation_matrix = generate_correlation_matrix(data)
    st.pyplot(correlation_matrix)

    st.header("🌟 Feature Importance")
    feature_importance = plot_feature_importance(data)
    st.pyplot(feature_importance)

    st.header("📊 Key Distributions")
    distributions = plot_distributions(data)
    st.pyplot(distributions)

    data = additional_columns(data)

    st.header("🏫 Attendance fraction")
    pie = pie_chart(data)
    st.plotly_chart(pie)

    st.header("🌌 3D Visualization")
    scatter_3d = scatterplot_3d(data)
    st.plotly_chart(scatter_3d)

    st.header("📶 Pairplot")
    pair = pairplot(data)
    st.pyplot(pair)  # !

    st.header("Other plots")

    plot1 = other_plot1(data)
    st.pyplot(plot1)

    plot2 = other_plot2(data)
    st.pyplot(plot2)

    plot3 = other_plot3(data)
    st.pyplot(plot3)

    plot4 = other_plot4(data)
    st.pyplot(plot4)

    plot5 = other_plot5(data)
    st.pyplot(plot5)

    plot6 = other_plot6(data)
    st.pyplot(plot6)


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
                </div>'''
                , unsafe_allow_html=True
            )

        except requests.exceptions.RequestException as e:
            status_placeholder.empty()
            st.error("🚨 Failed to get a prediction. Please try again.")
            st.code(str(e))


if __name__ == "__main__":
    main()
