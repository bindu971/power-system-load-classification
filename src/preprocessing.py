# ==========================================
# DATA LOADING AND CLEANING
# ==========================================
# These are the same cleaning steps used in notebooks/Load_Type_Classification.ipynb.
#
# Note: about 4-10% of usage / reactive power values look corrupted (roughly 2x too high).
# The notebook shows the evidence. Treating them did not improve cross-validation,
# so they are kept as they are (see the notebook, sections 5 and 10).
#
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

    # Midnight readings are written with the date of the day they END.
    # In the file, "01-01-2018 23:45" is followed by "01-01-2018 00:00" and then "02-01-2018 00:15",
    # so "01-01-2018 00:00" is really midnight at the start of 2 January.
    # We add one day to every 00:00 timestamp BEFORE sorting and before creating date features.
    is_midnight = (df["date_time"].dt.hour == 0) & (df["date_time"].dt.minute == 0)
    df.loc[is_midnight, "date_time"] = df.loc[is_midnight, "date_time"] + pd.Timedelta(days=1)

    # One timestamp appears twice in the dataset and the second copy is corrupted.
    # We keep the first copy. This must happen BEFORE sorting,
    # so the original file order decides which copy is kept.
    df = df.drop_duplicates(subset="date_time", keep="first")

    # Sort by time because cross-validation is time-based.
    # After the midnight fix, every row is exactly 15 minutes after the previous one.
    df = df.sort_values("date_time").reset_index(drop=True)

    # NSM = seconds from midnight. Many stored values are wrong or missing,
    # so we recalculate it from the timestamp, which is reliable.
    df["nsm"] = df["date_time"].dt.hour * 3600 + df["date_time"].dt.minute * 60

    # Power factor is a percentage (0-100). Values above 100 are invalid,
    # so we mark them as missing and let the imputer fill them later.
    df.loc[df["lagging_pf"] > 100, "lagging_pf"] = None
    df.loc[df["leading_pf"] > 100, "leading_pf"] = None

    return df
