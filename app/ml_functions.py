import pandas as pd
from config import load_config


# Load default config
cfg = load_config("config.json")

# Leite aus den historischen Daten Features ab und definiere eine Zielvariable:
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt Trainingsfeatures und eine Zielvariable (steigt/fällt am nächsten Tag).
    """
    df["return"] = df["value"].pct_change()
    df["MA3"] = df["value"].rolling(3).mean()
    df["MA12"] = df["value"].rolling(12).mean()
    df["target"] = (df["return"].shift(-1) > 0).astype(int)  # increase
    df.dropna(inplace=True)
    return df

# Wähle ein geeignetes Modell (z.B. RandomForestClassifier, LogisticRegression). Trainiere, evaluiere und verwende es zur Vorhersage:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(df: pd.DataFrame):
    """
    Trainiert ein Klassifikationsmodell und gibt es zusammen mit dem Test-Score zurück.
    """
    X = df[["return", "MA5", "MA20"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test-Accuracy: {acc:.2f}")
    return model

# Prognose für den letzten Tag:
def predict_move(model, df: pd.DataFrame) -> bool:
    """
    Erzeugt eine Vorhersage für die letzte Zeile des DataFrames.
    
    Returns:
        True, wenn Modell einen Wertanstieg erwartet, sonst False.
    """
    last_row = df[["return", "MA3", "MA12"]].iloc[-1].values.reshape(1, -1)
    pred = model.predict(last_row)[0]
    return bool(pred)

def predict_move_proba(model, df: pd.DataFrame) -> float:
    """
    Returns the probability that the value (overnight stays) will rise in the next month.

    Args:
        model: Trained classification model.
        df: Feature-engineered DataFrame.

    Returns:
        Probability (0.0–1.0) that the next month's value will be higher.
    """
    last_row = df[["return", "MA3", "MA12"]].iloc[-1].values.reshape(1, -1)
    proba = model.predict_proba(last_row)[0]  # [prob_down, prob_up]
    return float(proba[1])  # probability of "up"

# ------------------------------
# 
# ------------------------------
def ridge_regression_baseline(df: pd.DataFrame, land: str ="DE", nace: str = "551") -> pd.DataFrame:
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

    # 1) Take one slice (Country + NACEr2), monthly sorted
    df_slice = (
        df[(df["Geopolitische_Meldeeinheit"]==land) & (df["NACEr2"]==nace)]
        .sort_values(["Jahr","Monat"])
        .reset_index(drop=True)
    )

    # 2) Features (adjust to your exact column names)
    feat_default = ["MA3","MA6","MA12","Lag_1","Lag_3","Lag_12","pch_sm","pch_sm_19",
            "NACEr2_Saison", "Land_Saison", "Aufenthaltsland_Saison",
            "Month_cycl_sin","Month_cycl_cos","Jahr", "pandemic_dummy"]
    feat = cfg.get("features_ridge_model", feat_default)
    target = "value"

    # Drop rows with NA features from initial lags/MAs
    data = df_slice.dropna(subset=feat+[target]).copy()

    # 3) Train/validation split for a quick check (e.g., last 12 months)
    train = data.iloc[:-12].copy()
    valid = data.iloc[-12:].copy()

    model = Ridge(alpha=1.0)
    model.fit(train[feat], train[target])

    pred = model.predict(valid[feat])
    print("MAE:", mean_absolute_error(valid[target], pred))
    print("MAPE:", mean_absolute_percentage_error(valid[target], pred))

    # 4) Refit on full data & forecast next 3 months (example)
    model.fit(data[feat], data[target])

    # Build a small helper to roll forward features for the next months:
    def month_cyc(m):  # m=1..12
        import math
        return math.sin(2*math.pi*m/12), math.cos(2*math.pi*m/12)

    future_rows = []
    last = df_slice.copy()  # we’ll append forecasts to compute future lags/MAs

    for step in range(3):  # next season = next 3 months
        # figure next Jahr/Monat
        last_year, last_month = int(last["Jahr"].iloc[-1]), int(last["Monat"].iloc[-1])
        next_year = last_year + 1 if last_month == 12 else last_year
        next_month = 1 if last_month == 12 else last_month + 1

        # compute future features from last (which includes actuals + prior forecasts)
        # NOTE: implement helpers to compute Lag_1, Lag_3, Lag_12, MA3, MA6, MA12 from `last`
        # and Month_cycl_sin/cos from next_month; then:
        sin_m, cos_m = month_cyc(next_month)
        feat_row = {
            "Lag_1": last["value"].iloc[-1],
            "Lag_3": last["value"].iloc[-3] if len(last)>=3 else last["value"].iloc[-1],
            "Lag_12": last["value"].iloc[-12] if len(last)>=12 else last["value"].iloc[-1],
            "MA3": last["value"].tail(3).mean() if len(last)>=3 else last["value"].mean(),
            "MA6": last["value"].tail(6).mean() if len(last)>=6 else last["value"].mean(),
            "MA12": last["value"].tail(12).mean() if len(last)>=12 else last["value"].mean(),
            "Month_cycl_sin": sin_m,
            "Month_cycl_cos": cos_m,
            "Jahr": next_year,
        }
        yhat = float(model.predict(pd.DataFrame([feat_row])[feat]))
        future_rows.append({"Jahr": next_year, "Monat": next_month, "value": yhat})
        last = pd.concat([last, pd.DataFrame([{"Jahr": next_year, "Monat": next_month, "value": yhat}] )], ignore_index=True)

    future_df = pd.DataFrame(future_rows)
    print(future_df)
    
    