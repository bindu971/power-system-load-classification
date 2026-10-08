# Power System Load Type Classification

A machine learning project that predicts the load type of a power system
(`Light_Load`, `Medium_Load` or `Maximum_Load`) from 15-minute electricity readings.

---

## 1. Problem Statement

Predict the `Load_Type` of a power system from historical data.
The task covers data preprocessing, exploratory data analysis, feature engineering,
model selection and model evaluation. As required, the **last month of data is used as the test set**
to check how well the model generalises to recent, unseen data.
(Full text: `Problem statement.docx`.)

## 2. Objective

Build a 3-class classifier for `Load_Type` and evaluate it on the last month (December 2018)
using accuracy, precision, recall, F1-score and the confusion matrix.

## 3. Dataset

File: `data/load_data.csv`. 35,041 rows, 9 columns, one reading every 15 minutes during 2018.

| Column | Meaning |
|---|---|
| `Date_Time` | Timestamp (day-month-year hour:minute) |
| `Usage_kWh` | Energy consumption (kWh) |
| `Lagging_Current_Reactive.Power_kVarh` | Lagging reactive power (kVarh) |
| `Leading_Current_Reactive_Power_kVarh` | Leading reactive power (kVarh) |
| `CO2(tCO2)` | CO2 |
| `Lagging_Current_Power_Factor` | Lagging power factor (%) |
| `Leading_Current_Power_Factor` | Leading power factor (%) |
| `NSM` | Number of seconds from midnight |
| `Load_Type` | **Target**: `Light_Load`, `Medium_Load`, `Maximum_Load` |

Class distribution (all data): Light_Load 18,073 (51.6%), Medium_Load 9,696 (27.7%), Maximum_Load 7,272 (20.8%).

## 4. Data Quality Issues

All found by running the notebook:

| Issue | Count |
|---|---|
| Missing values | Every numeric column (e.g. `Usage_kWh`: 1,559, `Leading_Current_Power_Factor`: 1,471, `CO2`: 455) |
| `NSM` not matching its timestamp | 3,414 (some values above 86,400, which is impossible) + 455 missing |
| Lagging power factor above 100% | 3,225 |
| Leading power factor above 100% | 2,685 |
| Duplicate timestamp (corrupted second copy) | 1 |
| Rows out of time order (midnight readings) | 365 |
| Exact duplicate rows | 0 |

## 5. Data Cleaning

- **Duplicate timestamp:** the second copy has a missing usage value and a power factor of 217, so the first copy is kept.
  This is done *before* sorting so the file order decides which copy stays.
- **Sorting:** rows are sorted by time, because the split and cross-validation are time-based.
- **NSM:** recalculated from the timestamp (`hour * 3600 + minute * 60`), because the timestamp is reliable.
- **Power factor above 100%:** set to missing, because a percentage cannot be above 100.
- **Missing values:** filled with the **median of the training data only**, using a `SimpleImputer` inside the model pipeline.

## 6. Feature Engineering

EDA showed that load type follows a strong time schedule:
00:00-08:59 is always `Light_Load`, weekends are mostly `Light_Load`,
and the hours of `Maximum_Load` are different in winter (Jan, Feb, Nov) than in March-October.

Features used by the model:

| Feature | Source |
|---|---|
| `usage_kwh`, `lagging_kvarh`, `leading_kvarh`, `co2`, `lagging_pf`, `leading_pf` | Original measurements (cleaned) |
| `nsm` | Seconds from midnight (recalculated) |
| `day_of_week` | From timestamp (0 = Monday) |
| `is_weekend` | 1 for Saturday/Sunday |
| `is_winter` | 1 for Nov, Dec, Jan, Feb |

`month` is **not** a feature: December never appears in training, so the model could not learn from it.
December is treated as winter because it sits between November and January.

## 7. Train/Test Strategy

- **Train:** January-November 2018 (32,064 rows)
- **Test:** December 2018 (2,976 rows)

A random split is not used because it would train the model on readings from after the test period.

## 8. Models Compared

All models use the same pipeline: median imputer → (scaler) → classifier.

- Baseline (always predicts the most frequent class)
- Logistic Regression without scaling
- Logistic Regression with `StandardScaler` (logistic regression is sensitive to feature scale)
- Decision Tree
- Random Forest (200 trees)
- Gradient Boosting (`HistGradientBoostingClassifier`)

Tree models are not scaled, because tree splits do not depend on feature scale.

## 9. Model Selection

Models are compared with **5-fold `TimeSeriesSplit` cross-validation on January-November only**.
Each fold trains on earlier data and validates on the period after it.
The model with the highest **cross-validation macro F1** is selected.
Macro F1 is used because the classes are imbalanced and it gives each class equal weight.

| Model | CV accuracy | CV macro F1 | CV macro F1 std |
|---|---|---|---|
| **Gradient Boosting** | 0.898 | **0.880** | 0.145 |
| Random Forest | 0.890 | 0.871 | 0.141 |
| Decision Tree | 0.886 | 0.869 | 0.142 |
| Logistic Regression (scaled) | 0.744 | 0.689 | 0.078 |
| Logistic Regression (no scaling) | 0.676 | 0.617 | 0.079 |
| Baseline (most frequent) | 0.501 | 0.223 | 0.005 |

The large standard deviation comes from the first fold, which trains only on January-February
(winter schedule) and validates on March-April (non-winter schedule).

**Selected model: Gradient Boosting.** Random Forest and Decision Tree are close behind.

## 10. Final Results

The selected model was trained on January-November and evaluated **once** on December.

| Metric (December) | Value |
|---|---|
| Accuracy | 0.955 |
| Precision (macro) | 0.958 |
| Recall (macro) | 0.940 |
| F1-score (macro) | 0.949 |

Per class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Light_Load | 0.949 | 0.975 | 0.962 | 1,744 |
| Medium_Load | 0.967 | 0.946 | 0.956 | 704 |
| Maximum_Load | 0.958 | 0.900 | 0.928 | 528 |

Confusion matrix (rows = actual, columns = predicted):

| | Light | Medium | Maximum |
|---|---|---|---|
| **Light** | 1700 | 23 | 21 |
| **Medium** | 38 | 666 | 0 |
| **Maximum** | 53 | 0 | 475 |

Most errors are `Medium_Load` / `Maximum_Load` readings predicted as `Light_Load`.

## 11. Data Leakage Prevention

1. **Time-based split:** January-November for training, December for testing.
2. **No December data in model selection:** models are compared with `TimeSeriesSplit` on January-November only.
3. **Preprocessing fitted on training data only:** the imputer and scaler are inside a `Pipeline`,
   so they are re-fitted on the training part of every fold.
4. **December is used once**, for the final evaluation.
5. **No target leakage:** no feature is created from `Load_Type`. Time features are allowed because
   the time of a reading is always known when predicting.

## 12. Project Structure

```
assignment-2/
├── Problem statement.docx         # original assignment
├── data/
│   └── load_data.csv              # dataset
├── notebooks/
│   └── Load_Type_Classification.ipynb   # full analysis (main explanation of the project)
├── src/
│   ├── preprocessing.py           # load and clean the data
│   ├── feature_engineering.py     # create features, feature list
│   ├── train_model.py             # compare models, select, evaluate, save model
│   └── predict.py                 # load saved model and predict
├── models/
│   └── final_model.pkl            # trained Gradient Boosting pipeline (Jan-Nov)
├── app/
│   └── app.py                     # FastAPI prediction API
├── requirements.txt
└── README.md
```

The notebook and the `src` scripts follow the same methodology and produce the same results.

## 13. How to Install

Tested with **Python 3.7**. Package versions are pinned because the saved model must be loaded with the
same scikit-learn version it was trained with.

```bash
cd assignment-2
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 14. How to Run

Run all commands from the `assignment-2` folder.

**Notebook**
```bash
jupyter notebook notebooks/Load_Type_Classification.ipynb
```

**Train the model** (prints CV results and December results, saves `models/final_model.pkl`)
```bash
python -m src.train_model
```

**Predict with the saved model** (8 example December readings)
```bash
python -m src.predict
```

**Start the API**
```bash
uvicorn app.app:app --reload
```
Then open http://127.0.0.1:8000/docs.

## 15. Example Prediction

Request (a real December reading from the dataset; its actual load type is `Maximum_Load`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"date_time": "11-12-2018 18:00", "usage_kwh": 52.62, "lagging_kvarh": 3.71,
       "leading_kvarh": 1.98, "co2": 0.01, "lagging_pf": 99.3, "leading_pf": 99.8}'
```

Response:

```json
{"load_type": "Maximum_Load"}
```

The API applies the same cleaning and features as training (NSM is calculated from `date_time`,
power factors above 100 are treated as missing).

## 16. Technologies Used

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter, joblib, FastAPI, Uvicorn.

## 17. Conclusion

- Load type mainly follows a time schedule, and that schedule changes in winter.
  Adding the `is_winter` flag was the key modelling decision.
- Tree-based models clearly beat the baseline and logistic regression in cross-validation.
  Gradient Boosting had the best cross-validation macro F1 and reached 0.955 accuracy and
  0.949 macro F1 on December.
- **Limitations:** only one year of data and one test month; December is assumed to follow the winter schedule.
