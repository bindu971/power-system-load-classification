# ==========================================
# FEATURE ENGINEERING
# ==========================================
# Same features as in notebooks/Load_Type_Classification.ipynb.

# Columns the model uses, in this exact order
FEATURES = [
    "usage_kwh",
    "lagging_kvarh",
    "leading_kvarh",
    "co2",
    "lagging_pf",
    "leading_pf",
    "nsm",
    "day_of_week",
    "is_weekend",
    "is_winter",
]

# Column we want to predict
TARGET = "load_type"

# EDA on the TRAINING period (Jan-Nov) showed that Jan, Feb and Nov follow a different
# daily load schedule from Mar-Oct. December is treated as winter because it sits
# between November and January (an assumption, not something learned from December).
WINTER_MONTHS = [11, 12, 1, 2]


def add_features(df):
    # Work on a copy so the input data frame is not changed
    df = df.copy()

    # Day of week: 0 = Monday, 6 = Sunday
    df["day_of_week"] = df["date_time"].dt.dayofweek

    # Month is used to create the winter flag.
    # It is NOT a model feature, because December never appears in training.
    df["month"] = df["date_time"].dt.month

    # Weekends have much lower load, so we give the model a simple weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 1 for winter months, 0 for the rest of the year
    df["is_winter"] = df["month"].isin(WINTER_MONTHS).astype(int)

    return df
