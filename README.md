# Student Performance Factors Analysis

## Project Overview

This project is part of my studies at HSE's Data Science and Business Analytics (DSBA) program. The analysis explores the **Student Performance Factors** dataset from Kaggle, focusing on the various factors that may influence student outcomes. The objective is to uncover meaningful insights and patterns within the data that could help in understanding what factors contribute most to student success.

**Dataset**: [Student Performance Factors on Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Analysis Summary](#analysis-summary)
- [Contributions](#contributions)
- [License](#license)

## Dataset Description

The dataset includes various student-related features, such as demographic information, academic background, and other factors. These attributes can provide insights into potential performance outcomes.

### Columns info
1) `Hours_Studied` - Number of hours spent studying per week.
2) `Attendance` - Percentage of classes attended.
3) `Parental_Involvement` - Level of parental involvement in the student's education (Low, Medium, High).
4) `Access_to_Resources` - Availability of educational resources (Low, Medium, High).
5) `Extracurricular_Activities` - Participation in extracurricular activities (Yes, No).
6) `Sleep_Hours` - Average number of hours of sleep per night.
7) `Previous_Scores` - Scores from previous exams.
8) `Motivation_Level` - Student's level of motivation (Low, Medium, High).
9) `Internet_Access` - Availability of internet access (Yes, No).
10) `Tutoring_Sessions` - Number of tutoring sessions attended per month.
11) `Family_Income` - Family income level (Low, Medium, High).
12) `Teacher_Quality`	- Quality of the teachers (Low, Medium, High).
13) `School_Type` - Type of school attended (Public, Private).
14) `Peer_Influence` - Influence of peers on academic performance (Positive, Neutral, Negative).
15) `Physical_Activity` - Average number of hours of physical activity per week.
16) `Learning_Disabilities` - Presence of learning disabilities (Yes, No).
17) `Parental_Education_Level` - Highest education level of parents (High School, College, Postgraduate).
18) `Distance_from_Home` - Distance from home to school (Near, Moderate, Far).
19) `Gender` - Gender of the student (Male, Female).
20) `Exam_Score` - Final exam score.

### Objective
The goal is to identify significant predictors of student performance using exploratory data analysis and possibly some machine learning techniques.

## Installation and Setup

To get started, clone this repository and ensure you have Jupyter Notebook installed along with necessary libraries such as `pandas`, `numpy`, `matplotlib`, and `seaborn`.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MattoYuzuru/student_performance_factors_analysis.git
   ```

2. **Navigate to the project folder:**
   ```bash
   cd student_performance_factors_analysis
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook main.ipynb
   ```

## Usage

1. **Load the Data**: The notebook provides code to load and inspect the dataset.
2. **Run Analysis Cells**: Execute each cell sequentially to understand and analyze the data step-by-step. The notebook includes sections for:
   - Data Cleaning and Preprocessing
   - Exploratory Data Analysis (EDA)
   - Feature Engineering
   - Predictive Modeling (if applicable)

3. **Visualizations**: Run the visualization cells to generate insights into data distributions and correlations.

## Analysis Summary

This section will include a summary of key findings from the analysis, including:
- Significant correlations between study habits and performance
- Visualization of the impact of absences on grades
- Exploration of demographic influences on student success

## Contributions

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions for improving the analysis or visualizations.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.