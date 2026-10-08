# ==========================================
# TRAIN, SELECT, EVALUATE AND SAVE THE MODEL
# ==========================================
# Same methodology as notebooks/Load_Type_Classification.ipynb:
#   1. Train period = January-November, test period = December.
#   2. Compare models with TimeSeriesSplit cross-validation on Jan-Nov only.
#   3. Select the model with the best cross-validation macro F1.
#   4. Train it on all of Jan-Nov and evaluate it once on December.
#   5. Save the trained model to models/final_model.pkl.
#
# Run from the project folder:
#   python -m src.train_model

from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

from src.preprocessing import load_data, clean_data
from src.feature_engineering import add_features, FEATURES, TARGET


# Paths are built from this file's location, so the script works from any folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "load_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.pkl"

# Fixed random seed so results are the same every run
RANDOM_STATE = 42

# Class order used in the report (from lowest to highest load)
LABELS = ["Light_Load", "Medium_Load", "Maximum_Load"]


def main():
    # ------------------------------------------
    # Load, clean and add features
    # ------------------------------------------
    raw_data = load_data(DATA_PATH)
    df = clean_data(raw_data)
    df = add_features(df)
    print("Rows after cleaning:", len(df))

    # ------------------------------------------
    # Time-based split: December is the test set
    # ------------------------------------------
    is_december = df["month"] == 12
    train_data = df[~is_december]
    test_data = df[is_december]

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]
    print("Train rows (Jan-Nov):", len(X_train))
    print("Test rows (December):", len(X_test))

    # ------------------------------------------
    # Models to compare
    # ------------------------------------------
    # Each model is a pipeline. The imputer (and scaler) are inside the pipeline
    # so they are fitted only on the training part of each cross-validation fold.
    # Only logistic regression is scaled, because tree models do not need scaling.
    models = {
        "Baseline (most frequent)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression (no scaling)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "Logistic Regression (scaled)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }

    # ------------------------------------------
    # Model selection with TimeSeriesSplit (Jan-Nov only)
    # ------------------------------------------
    # Each fold trains on earlier data and validates on the period right after it.
    # December is NOT used here.
    cv = TimeSeriesSplit(n_splits=5)

    cv_results_list = []
    for model_name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"])
        cv_results_list.append({
            "model": model_name,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_f1_macro": scores["test_f1_macro"].mean(),
            "cv_f1_macro_std": scores["test_f1_macro"].std(),
        })

    model_results = pd.DataFrame(cv_results_list).round(3)
    model_results = model_results.sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)
    print()
    print("Cross-validation results (Jan-Nov):")
    print(model_results.to_string())

    # The model with the highest cross-validation macro F1 is selected
    best_model_name = model_results.loc[0, "model"]
    print()
    print("Selected model:", best_model_name)

    # ------------------------------------------
    # Train the final model on all of Jan-Nov
    # ------------------------------------------
    final_model = models[best_model_name]
    final_model.fit(X_train, y_train)

    # ------------------------------------------
    # Evaluate once on December
    # ------------------------------------------
    predictions = final_model.predict(X_test)

    print()
    print("December test results:")
    print("Accuracy:         ", round(accuracy_score(y_test, predictions), 3))
    print("Precision (macro):", round(precision_score(y_test, predictions, average="macro"), 3))
    print("Recall (macro):   ", round(recall_score(y_test, predictions, average="macro"), 3))
    print("F1-score (macro): ", round(f1_score(y_test, predictions, average="macro"), 3))
    print()
    print(classification_report(y_test, predictions, labels=LABELS, digits=3))

    # ------------------------------------------
    # Save the trained model
    # ------------------------------------------
    # The saved model is the one trained on Jan-Nov, so it matches the reported results.
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print("Model saved to:", MODEL_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
