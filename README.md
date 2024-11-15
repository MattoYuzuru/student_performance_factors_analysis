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

### Features
Some key features in the dataset include:
- `gender`: Gender of the student
- `age`: Age of the student
- `studytime`: Time dedicated to studying
- `absences`: Number of school absences
- `G1`, `G2`, `G3`: Grades in three different periods
- And many more...

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