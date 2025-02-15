import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from model import pipeline, clean_data

app = FastAPI()


class StudentInfo(BaseModel):
    attendance: int
    hours_studied: int
    prev_score: int
    tutoring_sessions: int
    physical_activity: int
    access_to_resources: str
    parental_involvement: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/data-sample")
def get_data_sample(n: int = 5):
    data = clean_data(pd.read_csv("StudentPerformanceFactors.csv"))
    sample_data = data.sample(n=n).to_dict(orient="records")
    return {"sample": sample_data}


@app.post("/predict")
def predict(data: StudentInfo):
    tmp = pd.read_json(data.model_dump_json(), orient="index").T
    tmp.rename(
        {"attendance": "Attendance",
         "hours_studied": "Hours_Studied",
         "prev_score": "Previous_Scores",
         "tutoring_sessions": "Tutoring_Sessions",
         "physical_activity": "Physical_Activity",
         "access_to_resources": "Access_to_Resources",
         "parental_involvement": "Parental_Involvement", }, inplace=True, axis="columns",
    )

    prediction = pipeline.predict(tmp)
    return prediction[0]
