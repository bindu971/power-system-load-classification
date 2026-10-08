# ==========================================
# SIMPLE PREDICTION API (FastAPI)
# ==========================================
# Receives one raw electricity reading and returns the predicted load type.
# The reading goes through the same cleaning and feature steps as in training,
# then the saved model (models/final_model.pkl) makes the prediction.
#
# Run from the project folder:
#   uvicorn app.app:app --reload
# Then open http://127.0.0.1:8000/docs to try it in the browser.

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.preprocessing import clean_data
from src.feature_engineering import add_features
from src.predict import load_model, predict_load_type


app = FastAPI(title="Power System Load Type Prediction")

# Load the trained model once when the API starts, not on every request
model = load_model()


# FastAPI uses this class to check the request body.
# Every field is required and must have the correct type.
# NSM is not needed: it is calculated from date_time, the same as in training.
class Reading(BaseModel):
    date_time: str          # format: day-month-year hour:minute, e.g. "13-12-2018 14:30"
    usage_kwh: float
    lagging_kvarh: float
    leading_kvarh: float
    co2: float
    lagging_pf: float
    leading_pf: float


@app.get("/")
def health_check():
    # Simple check to confirm the API is running
    return {"status": "ok"}


@app.post("/predict")
def predict(reading: Reading):
    # Put the reading into a one-row data frame, like one row of the dataset
    reading_data = pd.DataFrame([reading.dict()])

    # Apply the same cleaning and feature engineering used in training
    reading_data = clean_data(reading_data)
    reading_data = add_features(reading_data)

    # Predict with the saved model and return the label as plain text
    prediction = predict_load_type(model, reading_data)[0]
    return {"load_type": prediction}
