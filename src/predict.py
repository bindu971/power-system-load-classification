# ==========================================
# MAKE PREDICTIONS WITH THE SAVED MODEL
# ==========================================
# Loads models/final_model.pkl (created by src/train_model.py)
# and predicts the load type for prepared feature data.
#
# "Prepared feature data" = a data frame that has the columns in FEATURES
# (the output of clean_data() + add_features()).
#
# Run an example from the project folder:
#   python -m src.predict

from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import load_data, clean_data
from src.feature_engineering import add_features, FEATURES, TARGET


# Path to the saved model, built from this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "load_data.csv"


def load_model():
    # Load the trained model.
    # We use the saved model to make predictions on new data without retraining.
    return joblib.load(MODEL_PATH)


def predict_load_type(model, feature_data):
    # Keep only the feature columns, in the same order used during training.
    # The pipeline inside the model fills missing values with the training medians.
    return model.predict(feature_data[FEATURES])


if __name__ == "__main__":
    model = load_model()

    # Example: prepare the December readings (the test month) the same way as in training
    df = add_features(clean_data(load_data(DATA_PATH)))
    test_data = df[df["month"] == 12]

    # Pick 8 random December readings (same rows as the example in the notebook)
    example_rows = test_data.sample(8, random_state=42)

    example_predictions = pd.DataFrame({
        "date_time": example_rows["date_time"],
        "usage_kwh": example_rows["usage_kwh"],
        "actual": example_rows[TARGET],
        "predicted": predict_load_type(model, example_rows),
    })
    print(example_predictions.to_string(index=False))
