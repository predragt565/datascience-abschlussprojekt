import pandas as pd
import numpy as np
import streamlit as st
# import datetime as dt
# from zoneinfo import ZoneInfo
# import logging
# from pathlib import Path
# from typing import Dict, List, Any
# from urllib.parse import urlparse, parse_qs
# import requests
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error


# logger = logging.getLogger("log-alerts")

# Core functions start here

# -------------------------------
# Show data distribution skewness
# -------------------------------
def show_distribution(
    dataset: pd.DataFrame,
    columns_list,
    rows: int,
    cols: int,
    title: str = "Distributions"
):
    # Subplots erstellen:
    
    fig_skew = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[str(c) for c in columns_list]   # str(c).upper() - gives ability to prepare the title's words in various ways
    )
    
    for i, col in enumerate(columns_list):
        r = i // cols + 1   # Berechnet die Zeilenummer des Subplots z.B. cols=3, i=0, 1, 2, 0//3=0 (+1)=1, 3//3=1 (+1)=2
        c = i % cols + 1
        
        # Nur numerische Werte verwenden:
        df_cleaned = pd.to_numeric(dataset[col], errors="coerce").dropna()
        
        # Histogram zeichen:
        fig_skew.add_trace(go.Histogram(
            x=df_cleaned,
            nbinsx=min(50, max(10, int(np.sqrt(len(df_cleaned))))),
            showlegend=False
        ),
            row=r,
            col=c
        )
        
        # Skewness berechnen:
        skew = float(df_cleaned.skew())
        fig_skew.layout.annotations[i].text = f"{col} (Skewness: {skew:.2f})"
        fig_skew.layout.annotations[i].text = f"{col} (Skewness: {skew:.2f})<br>{'Rechts-schief' if skew > 0 else ('Links-schief' if skew < 0 else 'Symmetrisch')}"

     
    fig_skew.update_layout(
        title=title,
        template="plotly_white",
        bargap=0.05,
        height=max(400, 280 * rows),
        width=max(500, 320 * cols)
    )
    
    
    for ax in fig_skew["layout"]:
        fig_skew.update_yaxes(title_text="Häufigkeit")
    
    
    return fig_skew

# ------------------------------
# Transform skewness of num cols
# ------------------------------
def log_transform_and_skewness(df, numeric_columns, show_skewness=False):
    # Kopie:
    df_transfored = df.copy()
    
    # Transformation anwenden:
    df_transfored[numeric_columns] = np.log1p(df_transfored[numeric_columns])
    
    # Skewness vorher und nacher:
    before_skew = df[numeric_columns].skew()
    after_skew = df_transfored[numeric_columns].skew()
    
    # Skewness anzeigen:
    if show_skewness:
        skew_df = pd.DataFrame({
            "Before": before_skew,
            "After": after_skew          
        })
        print(skew_df)
    
    return df_transfored

# -------------------
# Evaluate Regression
# -------------------
def evaluate_regression(
    model,
    X_train, y_train,
    X_test, y_test,
    *, # Ab hier nur noch Keyword-Argumente
    cv_splits: int = 5,   # Anzahl der Fold
    cv_shuffle: bool = True,     # Daten vor Aufteilung mischen
    cv_random_state=42,
    scoring: str = "r2",    # Angabe der Metrik, fur die Cross-Validation
    return_print: bool = True
):
    # 1. Cross-Validation:
    cv = KFold(n_splits=cv_splits, shuffle=cv_shuffle, random_state=cv_random_state)     # Strategie der Aufteilung
    cv_scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv)    # C-V durchführen
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores, ddof=1))
    
    # 2. Modell fitten und Test-Vorhersage:
    fitted = clone(model).fit(X_train, y_train)
    y_pred = fitted.predict(X_test)
    
    # 3. Metriken:
    r2_test = float(r2_score(y_test, y_pred))
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    # 4. Korrigierte Bestimmtheitsmaß
    # 4.1 Anzahl der Beobachtungen im Testset:
    n_test = X_test.shape[0]
    # 4.2 Anzahl der Features (Prädikatoren):
    p = X_test.shape[1]
    # 4.3 Nenner der Formel:
    denom = n_test - p - 1
    # 4.4 Berechnung:
    if denom <= 0:
        adj_r2_test = np.nan
    else:
        adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / denom      # formula under, in the next Markdown cell 
        adj_r2_test = float(adj_r2_test)
    
    # 5. Metriken zusammenfassen:
    result={
        "R2_test": r2_test,
        "Adj_R2_test": adj_r2_test,
        "RMSE_test": rmse_test,
        "CV_mean": cv_mean,
        "CV_std": cv_std,
        "CV_metric": scoring,
        "n_test": n_test,
        "p_features": p,
        "cv_splits": cv_splits
    }
    
    if return_print:
        print(f"RMSE (Test): {result['RMSE_test']:.4f}")
        print(f"R2: {result['R2_test']:.4f}")
        print(f"Adjusted R2 (Test): {result['Adj_R2_test']:.4f}" if np.isfinite(result['Adj_R2_test']) else "Adjusted R2 (Test: n/a)")
        print(f"CV-{scoring.upper()} (Train) Mittelwert: {result['CV_mean']:.4f} | Std: {result['CV_std']:.4f} | Folds: {cv_splits}")
    
    return result


# --------------------------
# Skewness Summary table
# --------------------------
def skewness_summary(
    df: pd.DataFrame,
    numeric_columns: list = None,
    transform: str = None,
    show: bool = True
) -> pd.DataFrame:
    """
    Create a skewness summary table for numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    numeric_columns : list, optional
        List of numeric columns to evaluate. If None, selects all numeric columns automatically.
    transform : str, optional
        Apply transformation before skewness calculation:
        - "log1p"  : np.log1p (applied only to columns with all values >= 0)
        - "yeojohnson" : sklearn PowerTransformer (handles negatives too)
        - None     : no transformation (default)
    show : bool, default=True
        If True, prints basic info in the console (for debugging).
    
    Returns
    -------
    pd.DataFrame
        Table with skewness values before and (if transform given) after transformation.
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is empty or None.")
        
        # auto-detect numeric columns
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            raise ValueError("No numeric columns found to calculate skewness.")
        
        df_num = df[numeric_columns].copy()
        
        # baseline skewness
        before_skew = df_num.skew(numeric_only=True)
        result = pd.DataFrame({"Skewness_before": before_skew})
        
        # optional transformation
        if transform is not None:
            if transform.lower() == "log1p":
                after_skews = {}
                for col in numeric_columns:
                    series = df_num[col].dropna()
                    if (series < 0).any():
                        after_skews[col] = np.nan  # mark invalid for log1p
                    else:
                        after_skews[col] = np.log1p(series).skew()
                result["Skewness_after_log1p"] = pd.Series(after_skews)
            
            elif transform.lower() == "yeojohnson":
                from sklearn.preprocessing import PowerTransformer
                pt = PowerTransformer(method="yeo-johnson")
                try:
                    df_trans = pd.DataFrame(
                        pt.fit_transform(df_num),
                        columns=numeric_columns,
                        index=df_num.index
                    )
                    result["Skewness_after_YJ"] = df_trans.skew(numeric_only=True)
                except Exception as e:
                    if show:
                        print(f"Yeo–Johnson transform failed: {e}")
                    result["Skewness_after_YJ"] = np.nan
            
            else:
                raise ValueError(f"Unknown transform: {transform}")
        
        result = result.sort_values("Skewness_before", key=lambda x: x.abs(), ascending=False)
        
        if show:
            print("Skewness summary calculated successfully.")
        
        return result.round(3)
    
    except Exception as e:
        print(f"Error in skewness_summary: {e}")
        return pd.DataFrame()


# --------------------------
# Skewness Summary table display formatting
# --------------------------
def render_skew_table(skew_table: pd.DataFrame, title: str = "Skewness-Übersicht") -> None:
    if skew_table is None or skew_table.empty:
        st.info("Keine Skewness-Werte verfügbar.")
        return

    def _badge(v):
        if pd.isna(v): return "—"
        a = abs(v)
        if a < 1:   return "🟢"
        if a < 2:   return "🟡"
        return "🔴"

    df_disp = skew_table.copy().rename_axis("Feature").reset_index()

    # only the real skew columns (before adding badges)
    skew_cols = [c for c in df_disp.columns if c.lower().startswith("skew")]

    # add badge columns
    for col in skew_cols:
        df_disp[f"{col}_badge"] = df_disp[col].apply(_badge)

    # reorder nicely: Feature + (badge, value) pairs
    new_cols = ["Feature"]
    for col in skew_cols:
        new_cols.append(f"{col}_badge")
        new_cols.append(col)

    df_disp = df_disp[new_cols]

    st.markdown(f"### {title}")
    st.data_editor(
        df_disp,
        hide_index=True,
        disabled=True,
        column_config={
            "Feature": st.column_config.Column("Feature", width="medium"),
        },
        width="stretch",
    )
