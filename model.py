import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


data = pd.read_csv('StudentPerformanceFactors.csv')

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

columns_to_map = ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities']
changing = {'Yes': True, 'No': False}
for col in columns_to_map:
    data[col] = data[col].map(changing)


features = ['Attendance', 'Hours_Studied', 'Previous_Scores',
            'Tutoring_Sessions', 'Physical_Activity',
            'Access_to_Resources', 'Parental_Involvement']
target = 'Exam_Score'

X = data[features]
y = data[target]

categorical_features = ['Access_to_Resources', 'Parental_Involvement']
numerical_features = ['Attendance', 'Hours_Studied', 'Previous_Scores',
                      'Tutoring_Sessions', 'Physical_Activity']


numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

model = RandomForestRegressor(n_estimators=100, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

pipeline.fit(X, y)
