
# 🧭 ML Workflow (EDA → Preprocessing → Modelling → Evaluation)

---

## **1. Exploratory Data Analysis (EDA)**

**1.1 General overview**

* `df.info()`, `df.describe()`, `df.isna().sum()` → check missing values, datatypes, ranges.
* Count unique values per categorical column.

**1.2 Target distribution (y = "value")**

* Plot time series of `value` per country (`Aufenthaltsland`).
* Plot seasonal decomposition (`statsmodels.tsa.seasonal_decompose`).
* Boxplots by `Monat`, `Quartal`, `Saison`.

**1.3 Feature-target relationship**

* Group by season, country, accommodation type (`NACEr2`) → mean overnight stays.
* Correlation heatmap (numeric features only: `Monat`, `Lag_1`, `Lag_12`, `MA3`, etc.).

**1.4 Outliers & anomalies**

* Check sudden spikes/drops per country.
* Consider pandemic years (2020–2021) separately.

---

## **2. Feature Engineering**

✅ Already present:

* **Time-based**: `Monat`, `Quartal`, `Saison`, `Jahr`, `Month_cycl_sin`, `Month_cycl_cos`.
* **Rolling / lagged features**: `MA3`, `MA6`, `MA12`, `Lag_1`, `Lag_3`, `Lag_12`.
* **Interactions**: `Aufenthaltsland_Saison`, `NACEr2_Saison`, `Land_Monat`.

🚀 Additional ideas:

* Public holidays, school holidays, major events (external dataset).
* Weather indicators (avg temperature, snow, etc. if available).
* Pandemic dummy variables (2020–2021).
* Growth rates (`value / lag_12 - 1`).

---

## **3. Preprocessing**

* **Categorical variables**:

  * `OneHotEncoder` (scikit-learn) for:
    `Aufenthaltsland`, `Quartal`, `Saison`, `Aufenthaltsland_Saison`, `NACEr2`, `NACEr2_Saison`, `Land_Monat`.
  * Alternatively: **TargetEncoding** (for high-cardinality features like `Land_Monat`).

* **Numerical variables**:

  * Standard scaling (e.g., `StandardScaler` or `MinMaxScaler`) for continuous features (`MA3`, `Lag_1`, etc.).
  * Cyclical encoding already done for `Monat`.

* **Time split**:

  * **Do NOT random split**. Use **time-based train-test split** (e.g., train 2012–2022, validate 2023, test 2024–2025).

---

## **4. ML Modelling**

### **4.1 Traditional ML (tabular approach)**

Models:

* **Tree-based**: `RandomForestRegressor`, `XGBoost`, `LightGBM`, `CatBoost`.
* **Linear models**: `Ridge/Lasso`, `ElasticNet`.
* **Ensemble**: stacking regressor (trees + linear).

Pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Preprocessing
categorical = ["Aufenthaltsland", "Quartal", "Saison", "NACEr2", "Land_Monat"]
numerical = ["Lag_1", "Lag_3", "Lag_12", "MA3", "MA6", "MA12", "Month_cycl_sin", "Month_cycl_cos"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numerical)
])

# Model
model = Pipeline([
    ("prep", preprocess),
    ("reg", RandomForestRegressor(n_estimators=300, random_state=42))
])
```

---

### **4.2 Time Series / Forecasting models**

For **explicit forecasting per country**:

* `statsmodels`: SARIMAX (handles seasonality, country dummies).
* `Prophet`: good for strong seasonality, holidays.
* `darts` / `sktime`: modern Python frameworks for time-series ML.

---

## **5. Model Evaluation**

* Metrics: `RMSE`, `MAE`, `MAPE` (tourism data is scale-dependent → MAPE is intuitive).
* Evaluate per country (`groupby Aufenhaltsland`) and overall.
* Residual plots: check if seasonality still remains.

---

## **6. Deployment / Monitoring**

* Automate monthly update pipeline.
* Add explainability (SHAP values for tree models).
* Monitor performance drift over time.

---

# 📋 Final Checklist

### **EDA**

* [ ] Data summary, missing values, duplicates
* [ ] Seasonal decomposition (trend/seasonality/residual)
* [ ] Outlier analysis

### **Feature Engineering**

* [ ] Lag features, rolling averages
* [ ] Interactions (Residency × Season, etc.)
* [ ] Growth rates & external features

### **Preprocessing**

* [ ] OneHotEncoder for categoricals
* [ ] Scaling for numerical
* [ ] Time-based split

### **Modelling**

* [ ] Baseline (Naive forecast = lag\_12)
* [ ] ML models (RF, XGB, CatBoost)
* [ ] Time-series models (SARIMAX, Prophet)

### **Evaluation**

* [ ] RMSE, MAE, MAPE per country
* [ ] Residual diagnostics

