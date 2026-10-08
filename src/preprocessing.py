# ==========================================
# DATA LOADING AND CLEANING
# ==========================================
# These are the same cleaning steps used in notebooks/Load_Type_Classification.ipynb.
# Missing values are NOT filled here. They are filled by the imputer inside the
# model pipeline, so the median is learned from training data only.

import pandas as pd


# Original column names -> shorter names used in the rest of the project
COLUMN_NAMES = {
    "Date_Time": "date_time",
    "Usage_kWh": "usage_kwh",
    "Lagging_Current_Reactive.Power_kVarh": "lagging_kvarh",
    "Leading_Current_Reactive_Power_kVarh": "leading_kvarh",
    "CO2(tCO2)": "co2",
    "Lagging_Current_Power_Factor": "lagging_pf",
    "Leading_Current_Power_Factor": "leading_pf",
    "NSM": "nsm",
    "Load_Type": "load_type",
}


def load_data(file_path):
    # Load the raw electricity readings from the CSV file.
    return pd.read_csv(file_path)


def clean_data(df):
    # Rename columns to short names. rename() returns a new data frame,
    # so the original raw data is not changed.
    df = df.rename(columns=COLUMN_NAMES)

    # Convert text dates to real datetimes.
    # The format is day-month-year, so we give it explicitly.
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%Y %H:%M")

    # One timestamp appears twice in the dataset and the second copy is corrupted.
    # We keep the first copy. This must happen BEFORE sorting,
    # so the original file order decides which copy is kept.
    df = df.drop_duplicates(subset="date_time", keep="first")

    # Sort by time because the train/test split and cross-validation are time-based.
    df = df.sort_values("date_time").reset_index(drop=True)

    # NSM = seconds from midnight. Many stored values are wrong or missing,
    # so we recalculate it from the timestamp, which is reliable.
    df["nsm"] = df["date_time"].dt.hour * 3600 + df["date_time"].dt.minute * 60

    # Power factor is a percentage (0-100). Values above 100 are invalid,
    # so we mark them as missing and let the imputer fill them later.
    df.loc[df["lagging_pf"] > 100, "lagging_pf"] = None
    df.loc[df["leading_pf"] > 100, "leading_pf"] = None

    return df
