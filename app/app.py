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

from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing import clean_data
from src.feature_engineering import add_features
from src.predict import load_model, predict_load_type


# Same date format as the dataset, e.g. "11-12-2018 18:00"
DATE_FORMAT = "%d-%m-%Y %H:%M"

app = FastAPI(title="Power System Load Type Prediction")

# Load the trained model once when the API starts, not on every request
model = load_model()


# FastAPI uses this class to check the request body.
# Every field is required. Field(...) sets the allowed range, so impossible values
# (negative energy, power factor above 100%) are rejected with a 422 error.
# NSM is not needed: it is calculated from date_time, the same as in training.
class Reading(BaseModel):
    date_time: str = Field(description="Format DD-MM-YYYY HH:MM, e.g. 11-12-2018 18:00. "
                                       "Like the dataset, 00:00 means midnight at the end of that day.")
    usage_kwh: float = Field(ge=0)
    lagging_kvarh: float = Field(ge=0)
    leading_kvarh: float = Field(ge=0)
    co2: float = Field(ge=0)
    lagging_pf: float = Field(ge=0, le=100)
    leading_pf: float = Field(ge=0, le=100)


@app.get("/")
def health_check():
    # Simple check to confirm the API is running
    return {"status": "ok"}


@app.post("/predict")
def predict(reading: Reading):
    # Check the date format first, so a wrong date gives a clear 422 error
    # instead of crashing the server (500 error)
    try:
        datetime.strptime(reading.date_time, DATE_FORMAT)
    except ValueError:
        raise HTTPException(status_code=422,
                            detail="date_time must use the format DD-MM-YYYY HH:MM, e.g. 11-12-2018 18:00")

    # Put the reading into a one-row data frame, like one row of the dataset
    reading_data = pd.DataFrame([reading.model_dump()])

    # Apply the same cleaning and feature engineering used in training
    reading_data = clean_data(reading_data)
    reading_data = add_features(reading_data)

    # Predict with the saved model and return the label as plain text
    prediction = predict_load_type(model, reading_data)[0]
    return {"load_type": str(prediction)}
