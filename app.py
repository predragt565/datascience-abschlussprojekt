import streamlit as st
import csv
import io
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests # for JSON download
from data.estat_load_data import eurostat_url, load_prepare_data, validate_required_columns_json, validate_required_columns_csv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

from joblib import dump
from scipy import stats
from scipy.stats import skew
import hashlib
import json

# Setting up the global colour palette template:
import plotly.io as pio

# Example themes: "plotly_dark", "plotly_white", "ggplot2", "simple_white"
pio.templates.default = "ggplot2"

# Define a global blue-green colorway for all charts
pio.templates["ggplot2"].layout.colorway = [
    "#066e69",  # Soft teal / ocean
    "#17becf",  # Cyan / turquoise
    "#2ca02c",  # Vibrant green
    "#66c2a5",  # Soft teal
    "#005f73",  # Deep teal / ocean
    "#8dd3c7"   # Light aqua
]
# für Heatmap Farbenpalette verwenden
CUSTOM_CONTINUOUS=[
        "#8dd3c7",  # Light aqua (low values)
        "#4ea88b",  # Soft teal (mid-range)
        "#0ea8b9",  # Cyan / turquoise
        # "#2ca02c",  # Vibrant green
        "#066e69",  # Deep ocean teal (high values)
        "#005f73"  # Deep teal / ocean
    ]


# MAIN TITLE
st.set_page_config(
    page_title="EUSTAT Studio Pred",
    layout="wide"
    )

# Fix: apply left margin to block container, so main content shifts
# st.markdown("""
#     <style>
#         /* Push main content to the right */
#         .block-container {
#             margin-left: 32rem; /* match or slightly exceed sidebar width */
#         }

#         /* Optional: ensure sidebar stays in flow */
#         [data-testid="stSidebar"] {
#             position: relative !important;
#         }

#         /* Optional: hide sidebar toggle hamburger */
#         [data-testid="collapsedControl"] {
#             display: none;
#         }
#     </style>
# """, unsafe_allow_html=True)


# --------------------------------
# Helper
# --------------------------------

# Initialize all the session_state keys
for key, default in {
    "df": pd.DataFrame(),
    "df_from_json": False,
    "uploaded_filename": None,
    "df_filtered": None,
    "cols_for_outlier": [],
    "outlier_method": None,
    "mask": None,
    "show_normalized": False,
    "model_trained": False,
    "last_trained_state": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Define a cached fetch JSON URL function to avoid repetative calls
@st.cache_data(show_spinner=True)
def fetch_eurostat_data(url: str):
    resp = requests.get(url)
    resp.raise_for_status()
    return load_prepare_data(resp.json())

@st.cache_data(show_spinner=False)  # !!! IMPORTANT when using with Streamlit !
def read_csv(file, sep):
    return pd.read_csv(file, sep=sep)

def detect_outliers_iqr(df: pd.DataFrame, cols, factor=1.5):
    # Start mit "leerer" Series (False=kein Ausreßer):
    mask = pd.Series(False, index=df.index)     # index is used to set a size for the mask to match size of df
    # Quartile berechnen:
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        outliers = (df[col] < lower) | (df[col] > upper)
        mask = mask | outliers
    
    return mask

def detect_outliers_z(df: pd.DataFrame, cols, threshold=3):
    """
    Vectorized Z-Score outlier detection using NumPy.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        cols (list): List of numerical columns to check for outliers
        threshold (float): Z-score threshold (default=3.0)

    Returns:
        pd.Series: Boolean mask where True = outlier row
    """
    data = df[cols].astype(float)   # Extract numerical data

    # Compute mean & std per column, ignoring NaNs
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0)
    stds[stds == 0] = 1.0   # Avoid division by zero → set std=1 where std=0 to prevent NaNs
    z_scores = (data - means) / stds    # Compute Z-scores for all columns simultaneously
    outlier_matrix = np.abs(z_scores) > threshold   # Mark where absolute Z-score > threshold
    mask = outlier_matrix.any(axis=1)   # Combine across all columns → True if any column is an outlier
    # Return as a Pandas Series aligned to df.index
    return pd.Series(mask, index=df.index)


def detect_outliers_iforest(df, cols, contamination=0.05, n_estimators=200, random_state=42):
    """
    Detects outliers using IsolationForest.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        cols (list): List of numerical columns to check for outliers
        contamination (float): Proportion of outliers expected (default=0.05)
        random_state (int): Random seed for reproducibility

    Returns:
        pd.Series: Boolean mask where True = outlier
    """
    # Drop NaNs
    df_clean = df[cols].dropna()
    # Initialize the IsolationForest model
    clf = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        max_samples="auto"
    )
    # Fit the model on the selected columns -> with df converted to ndarray
    labels = clf.fit_predict(df_clean.to_numpy())
    # Create mask aligned to the original DataFrame index
    mask = pd.Series(False, index=df.index)
    mask.loc[df_clean.index] = labels == -1

    return mask


def infer_column_types(df: pd.DataFrame):
    """
    Infers and separates the numeric and categorical columns in a given pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame 
        for which column types need to be determined.

    Returns:
        tuple:
            - numeric_cols (list): (e.g., int, float).
            - categorical_cols (list): (e.g., object, category, bool).
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.to_list()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
    return numeric_cols, categorical_cols

def get_feature_names_from_ct(ct: ColumnTransformer, input_df: pd.DataFrame):
    """Ermittle die finalen Feature-Namen nach dem ColumnTransformer."""
    feature_names = []
    for name, transformer, cols in ct.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps"):
            # Pipeline
            last = list(transformer.named_steps.values())[-1]
            if hasattr(last, "get_feature_names_out"):
                fn = last.get_feature_names_out(cols)
            else:
                fn = cols
        else:
            # Direkter Transformer
            if hasattr(transformer, "get_feature_names_out"):
                fn = transformer.get_feature_names_out(cols)
            else:
                fn = cols
        feature_names.extend(fn)
    return list(feature_names)

# Modell einbauen und als Objekt zurückgeben:
def build_model(model_name: str):
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=alpha, random_state=random_state)
    elif model_name == "Lasso":
        model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
    elif model_name == "ElasticNet":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    elif model_name == "RandomForestRegressor":
        # Mapping der Auswahl auf gültige Werte
        max_d = None if max_depth == 0 else max_depth
        mfc = None if max_features_choice == "Alle (None)" else max_features_choice
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_d,
            max_features=mfc,
            random_state=random_state
        )
    elif model_name == "GradientBoostingRegressor":
        # Mapping der Auswahl auf gültige Werte
        max_d = None if max_depth == 0 else max_depth
        mfc = None if max_features_choice == "Alle (None)" else max_features_choice
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_features=max_features_choice
        )
    return model

def compute_model_hash(df, model_name, test_size, random_state,
                       cols_for_outlier, apply_skew_correction_global):
    """
    Computes a stable hash of key model-related parameters and data shape.
    Used to detect when the UI or data has changed so we can warn the user
    about outdated metrics or plots.
    """
    try:
        hash_input = {
            "shape": df.shape,
            "columns": list(df.columns),
            "model_name": model_name,
            "test_size": test_size,
            "random_state": random_state,
            "cols_for_outlier": cols_for_outlier,
            "apply_skew_correction_global": apply_skew_correction_global
        }
        hash_str = json.dumps(hash_input, sort_keys=True)
        return hashlib.md5(hash_str.encode("utf-8")).hexdigest()
    except Exception:
        return None

def reset_app_state():
    """Clears all session state variables to reset the app."""
    keys_to_keep = ["uploaded_filename"]  # Keep current filename only
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]


# --------------------------------
# App Header - Main Title
# --------------------------------

st.markdown("### 🛏️ Eurostat Tourist Overnight Stays 2012-2025 (EU10) by Pred")
st.markdown("#### Interaktive Analyse von EDA zu ML-Vorhersage")

# --------------------------------
# Sidebar (Datei & Einstellungen)
# --------------------------------

# DONE: Add an optional checkbox to call a function in 'estat_load_data.py' to fetch JSON file and preprocess features
# instead of loading the file; fetched file is assigned to the 'upload' object and passed further on to the next block

# Upload CSV-Datei
st.sidebar.header("1) Daten hochladen")
use_json = st.sidebar.checkbox("Eurostat JSON von URL laden", value=False)
upload = st.sidebar.file_uploader(
    "CSV-Datei auswählen", 
    type=["csv"]
    )

# Default separator option
possible_separators = [",", ";", "\t", "|"]
detected_sep = ","

# --- JSON load path (optional, leaves CSV path untouched) ---
if use_json and "df_from_json" not in st.session_state:
    try:        
        url = eurostat_url()  # 🔗 get URL
        df_json = fetch_eurostat_data(url)  # 🔗 cached fetch
        
        st.session_state.df = df_json
        st.session_state.df_from_json = True
        st.session_state.uploaded_filename = "eurostat_json"  # used later for model filename
        # st.success(f"Eurostat JSON geladen | {df_json.shape[0]} Zeilen x {df_json.shape[1]} Spalten")
    except Exception as e:
        st.error(f"Fehler beim Laden der JSON-Datei: {e}")
        st.stop()

# --- Reset session state if JSON was previously selected but is now unchecked ---
if not use_json and st.session_state.get("df_from_json", False):
    st.session_state.df = pd.DataFrame()
    st.session_state.df_from_json = False
    # for key in ["df", "df_from_json"]:
    #     if key in st.session_state:
    #         del st.session_state[key]
    st.session_state.uploaded_filename = None


# --- CSV path (only if not using JSON) ---
# --- Detect file changes and reset session if needed ---
if upload is not None and not use_json:
    # --- Detect if user uploaded a NEW or DIFFERENT file ---
    previous_filename = st.session_state.get("uploaded_filename", None)
    
    if use_json:
        current_filename = "eurostat_json"
    else:
        current_filename = upload.name if upload is not None else None

    # --- Reset session if user switched source or file ---
    if previous_filename is not None and current_filename is not None and previous_filename != current_filename:
        # New file uploaded → reset the entire app state EXCEPT filename
        for key in list(st.session_state.keys()):
            del st.session_state[key]

    # Save the current filename into session_state
    if current_filename is not None:
        st.session_state.uploaded_filename = current_filename

    # Read a small sample from the uploaded file for delimiter detection
    sample = upload.read(2048).decode("utf-8", errors="ignore")
    upload.seek(0)
    
    # Try to detect the delimiter using csv.Sniffer
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=possible_separators)
        detected_sep = dialect.delimiter
    except:
        detected_sep = "," # Fallback
        
    # Show only the detected separator in the selectbox
    sep = st.sidebar.selectbox("CSV-Trenner (erkannt)", [detected_sep])
elif not use_json:
    # CSV removed or not provided; if current source was CSV, clear stale df
    if upload is None and not st.session_state.get("df_from_json", False):
        if "df" in st.session_state:
            st.session_state.df = pd.DataFrame()
            # del st.session_state["df"]
        st.session_state.uploaded_filename = None
    
    sep = st.sidebar.selectbox("CSV-Trenner", possible_separators, index=0)

# Daten laden in ein DataFrame (JSON always has priority!)
try:
    if use_json:
        # JSON selected → ignore CSV completely
        url = eurostat_url()
        df_json = fetch_eurostat_data(url)
        df = validate_required_columns_json(df_json)
        st.session_state.df = df
        st.session_state.df_from_json = True
        st.session_state.uploaded_filename = "eurostat_json"

    elif upload is not None:
        # Only use CSV if JSON is not selected
        df_csv = read_csv(file=upload, sep=sep)
        df = validate_required_columns_csv(df_csv)
        st.session_state.df = df
        st.session_state.df_from_json = False
        st.session_state.uploaded_filename = upload.name

    else:
        # No data source → clear df
        if "df" in st.session_state:
            st.session_state.df = pd.DataFrame()
            # del st.session_state["df"]
        st.sidebar.error("No data source found. Please select JSON or upload a CSV file.")

except Exception as e:
    st.sidebar.error(f"Error while reading CSV/JSON: {e}")
    if "df" in st.session_state:
        st.session_state.df = pd.DataFrame()
        # del st.session_state["df"]


# Erfolgsmeldung:  
if not st.session_state.df.empty:
    df = st.session_state.df
    filename = st.session_state.get("uploaded_filename", "Unbekannt")
    if st.session_state.get("df_from_json", False):
        st.sidebar.success(f"Datei geladen: **Eurostat JSON** | From: {df.shape[0]} Zeilen x {df.shape[1]} Spalten")
    else:
        st.sidebar.success(f"CSV Datei geladen: **{filename}** | From: {df.shape[0]} Zeilen x {df.shape[1]} Spalten")
# else:
    # st.sidebar.info("⚠️ Keine Daten geladen – bitte CSV hochladen oder JSON aktivieren")



# --- Show the rest of the sidebar only if df is valid ---
if not st.session_state.df.empty:   
    st.sidebar.header("2) Zielvariable & Split")
    test_size = st.sidebar.slider(
        "Testgröße (%)", 
        5, 40, 20, step=1
        ) / 100 # for the percentual value of test size
    # st.sidebar.write(f"Testgröße: {test_size}")

    random_state = st.sidebar.number_input(
        "Random State",
        min_value=0,
        value=42,
        step=1
    )

    st.sidebar.header("3) ML-Modell einstellen")
    model_name = st.sidebar.selectbox(
        "Algorithmus",
        [
            "LinearRegression",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "RandomForestRegressor",
            "GradientBoostingRegressor"
        ],
        index=4
    )

    # Hyperparameter, modellabhängigkeit anzeigen:
    if model_name == "LinearRegression":
        pass

    if model_name == "Ridge":
        alpha = st.sidebar.number_input(
            "alpha (Ridge)",
            min_value=0.0,
            max_value=1.0,
            step=0.1
            # format="%.2f"
        )
    elif model_name == "Lasso":
        alpha = st.sidebar.number_input(
            "alpha (Lasso)",
            min_value=0.0,
            value=0.001,
            step=0.001,
            format="%.3f"
        )

    elif model_name == "ElasticNet":
        alpha = st.sidebar.number_input(
            "alpha (ElasticNet)",
            min_value=0.0,
            value=0.001,
            step=0.001,
            format="%.3f"
        )
        l1_ratio = st.sidebar.slider(
            "l1_ratio",
            0.0,
            1.0,
            0.5,
            0.05
        )
    elif model_name == "RandomForestRegressor":
        n_estimators = st.sidebar.slider(
            "n_estimators",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Anzahl der Bäume im Ensemble:\n"
                "- Höher = stabilere Vorhersagen aber langsames Training\n"
                "- Zu niedrig = Modell kann unterfitten\n"
                "- Typischer Bereich: 100-300"
        )
        max_depth = st.sidebar.slider(
            "max_depth (None=0)",
            0, 50, 0, 1,
            help="Max. Tiefe der Bäume:\n"
                "- 0 = Keine Begrenzung\n"
                "- Kleinere Werte = flachere Bäume, oft besser generalisierend\n"
                "- Höhere Werte = tiefere Bäume, können Overfitting verursachen" 
        )
        max_features_choice = st.sidebar.selectbox(
            "max_features",
            ["sqrt", "log2", "Alle (None)"],
            help="Bestimmt, wie viele Features pro Split zufällig in Betracht gezogen werden\n"
                "- 'sqrt': Wurzel ausa der Feature-Anzahl\n"
                "- 'log2': Log2 der Feature-Anzahl\n"
                "- 'Alle (None)': Alle Features\n\n"
                "Kleinere Werte machen die Bäume zufälliger, verringern Korrelation und können Ovrfitting reduzieren"
        )
    elif model_name == "GradientBoostingRegressor":
        n_estimators = st.sidebar.slider(
            "n_estimators",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Anzahl der Bäume im Ensemble:\n"
        )
        learning_rate = st.sidebar.number_input(
            "learning_rate",
            min_value=0.01,
            value=0.1,
            step=0.01,
            format="%.2f"
        )
        max_depth = st.sidebar.slider(
            "max_depth",
            1, 10, 3, 1
        )
        max_features_choice = st.sidebar.selectbox(
            "max_features",
            ["sqrt", "log2", "Alle (None)"],
            help="Bestimmt, wie viele Features pro Split zufällig in Betracht gezogen werden\n"
                "- 'sqrt': Wurzel ausa der Feature-Anzahl\n"
                "- 'log2': Log2 der Feature-Anzahl\n"
                "- 'Alle (None)': Alle Features\n\n"
                "Kleinere Werte machen die Bäume zufälliger, verringern Korrelation und können Ovrfitting reduzieren"
        )

    # Sidebar Option: Apply Skewness Correction Globally
    # --------------------------------
    apply_skew_correction_global = st.sidebar.checkbox(
        "Globale Skewness-Korrektur (für Modelltraining)",
        value=False,
        help="Wendet Yeo-Johnson-Transformation auf numerische Features an, bevor das Modell trainiert wird.  \n"
            "Empfohlen bei starker Schiefe (|Skewness| > 1)."
    )




# --------------------------------
# Main Section
# --------------------------------

# Prüfen ob Datenquelle vorhanden ist (CSV ODER JSON)
if st.session_state.df.empty:
    # no data at all
    st.info("Lade eine CSV hoch (links in der Sidebar) oder aktiviere 'Eurostat JSON von URL laden', um zu starten")
else:
    df = st.session_state.df

    # --- If validation passes, show rest of page ---
    # st.success(f"Daten geladen: {df.shape[0]} Zeilen x {df.shape[1]} Spalten")

    # --- TABS per layout diagram --- #
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Korrelation-Matrix",
        "🔎 Explorative Analyse",
        "🚨 Ausreißer-Erkennung",
        "🚀 ML Modell Trainieren",
        "📊 Vorhersage & Visualisierungen"
    ])
# DONE: Responsive Correlation heatmap with slider filters implemented
    with tab1:
        # Übersicht der 
        with st.expander("Heat map", expanded=True):
            group_slider1 = df["Geopolitische_Meldeeinheit"].unique()
            group_slider2 = ["NACEr2", "Aufenthaltsland"]
            group_slider3 = []
            group_slider4 = ["Frühling", "Sommer", "Herbst", "Winter"]

            # numeric features only
            num_cols = [
                "value", "Monat", "Jahr", "pch_sm",
                "Month_cycl_sin", "Month_cycl_cos",
                "MA3", "MA6", "MA12",
                "Lag_1", "Lag_3", "Lag_12"
            ]
            
            c1, c2, c3 ,c4 = st.columns(4)
            
            with c1:
                choice1 = st.select_slider(f"Choose Country out of {len(group_slider1)}", options=group_slider1)
                # Build index column name dynamically
                choice1_idx_col = "Geopolitische_Meldeeinheit_Idx"
                # Get the corresponding index value
                choice1_idx_val = df[df["Geopolitische_Meldeeinheit"] == choice1][choice1_idx_col].iloc[0] if not df[df["Geopolitische_Meldeeinheit"] == choice1].empty else None

            with c2:
                choice2 = st.select_slider(f"Choose Feature Group out of {len(group_slider2)}", options=group_slider2)
                # st.write(f"You selected: {choice2}")
            
            with c3:
                group_slider3 = sorted(df[choice2].dropna().unique())
                choice3 = st.select_slider(f"Choose Feature out of {len(group_slider3)}", options=group_slider3)
                
                # Build index column name dynamically
                choice3_idx_col = f"{choice2}_Idx"

                # Get the corresponding index value (assuming 1-to-1 mapping)
                choice3_idx_val = df[df[choice2] == choice3][choice3_idx_col].iloc[0] if not df[df[choice2] == choice3].empty else None
                # st.write(f"You selected: {choice3} | {choice3_idx_val}")

            with c4:
                choice4 = st.select_slider(f"Choose Season out of {len(group_slider4)}", options=group_slider4)
                # st.write(f"You selected: {choice3}")
            
            # Plot a heatmap based on selected choice
            slider2_map = {
                "Geopolitische_Meldeeinheit": "Land_Saison", 
                "NACEr2": "NACEr2_Saison", 
                "Aufenthaltsland": "Aufenthaltsland_Saison"
            }
            composite_column1 = slider2_map.get("Geopolitische_Meldeeinheit", "Unknown_Group")
            composite_column2 = slider2_map.get(choice2, "Unknown_Group")
            composite_value1 = f"{choice1_idx_val}_{choice4}"
            composite_value2 = f"{choice3_idx_val}_{choice4}"

            # --- Filter the DataFrame ---
            df_filtered = df[
                (df[composite_column1] == composite_value1) &
                (df[composite_column2] == composite_value2)
            ]
                
            # --- Show heatmap if data exists ---
            if not df_filtered.empty:
                corr = df_filtered[num_cols].corr().round(5)
                fig11 = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    title=f"Correlation heatmap: {composite_column2} = {composite_value2} & {composite_column1} = {composite_value1}",
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1
                )
                fig11.update_traces(textfont=dict(size=10))
                fig11.update_layout(    
                    xaxis=dict(tickfont=dict(size=12)),
                    yaxis=dict(tickfont=dict(size=12))
                )
                st.plotly_chart(fig11, width='stretch')
                
                if st.button("💾 Save currently displayed correlation to CSV"):
                    corr.to_csv(f"data/correlation_heatmap/correlation_{choice1}_{choice2}_{choice3}_{choice4}.csv")

                
                with st.expander("Tabellarisch übersicht"):
                    st.dataframe(df_filtered)
                
            else:
               st.warning(f"No data for selection: `{composite_column1}` = `{composite_value1}` and `{composite_column2}` = `{composite_value2}`")

# TODO: Continue from here - Line chart feature
    
    with tab2:
        # Hinweise
        st.caption("Hinweis: Kategorie Spalten werden automatish One-Hot-encodiert; numerische werden skaliert (optiona je nach Modell)")
        with st.expander("Quellndaten (erste 50 Zeilen)", expanded=False):
            st.dataframe(df.head(50))

        # Zielspalte wählen:
        st.subheader("Zielvariable")
        num_cols = df.select_dtypes(include=["number"]).columns.to_list()
        target_col = st.selectbox("Zielvariable (nur numerisch)", num_cols)

        # Einfache EDA:
        with st.expander("🔎 Explorative Analyse", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Numerische Übersicht**")
                # num_cols = df.select_dtypes(include=["number"]).columns.to_list()
                if len(num_cols) > 0:
                    st.dataframe(df[num_cols].describe().T, width='stretch') # .T pivots columns/rows
                else:
                    st.info("Keine numerische Spalten gefunden")
                # st.write(num_cols)
            with c2:
                st.markdown("**Fehlende Werte je Spalte**")    
                miss = df.isna().sum().sort_values(ascending=True)
                st.dataframe(miss.to_frame("missing"), width='stretch')
                
            c3, c4 = st.columns(2)
            # Histogram:
            with c3:
                if target_col in num_cols:
                    fig1 = px.histogram(
                        df,
                        x=target_col,
                        nbins=50,
                        title="Verteilung der Zielvariable"
                    )
                    fig1.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig1, width='stretch')
            # Korrelationsmatrix:
            with c4:
                if len(num_cols) > 1:
                    corr = df[num_cols].corr(numeric_only=True)
                    fig2 = px.imshow(
                        corr,
                        text_auto=False,
                        title="Korrelationsmatrix",
                        color_continuous_scale=CUSTOM_CONTINUOUS
                    )
                    fig2.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig2, width='stretch')
            
            # Scatterplot:
            st.markdown("**Relation zum Target**")
            all_cols = df.columns.to_list()
            feature_candidates = [c for c in all_cols if c !=target_col]
            show_cols = st.multiselect(
                "Features für Scatter-Plot auswählen",
                feature_candidates,
                feature_candidates[:4]
                )
            tabs = st.tabs([f"📑 {c}" for c in show_cols] or ["Hinweis"])
            
            if show_cols:
                for tab, col in zip(tabs, show_cols):
                    with tab:
                        x_is_num = pd.api.types.is_numeric_dtype(df[col])
                        y_is_num = pd.api.types.is_numeric_dtype(df[target_col])
                        if x_is_num and y_is_num:
                            try:
                                fig3 = px.scatter(
                                    df, x=col, y=target_col, 
                                    trendline="ols", title=f"{col} vs. {target_col}"
                                    )
                            except Exception:
                                fig3 = px.scatter(
                                    df, x=col, y=target_col, 
                                    title=f"{col} vs. {target_col}"
                                    )
                        elif not x_is_num and y_is_num:
                            fig3 = px.box(
                                df, x=col, y=target_col, points="all",
                                title=f"{col} (kateg.) vs. {target_col} (num.)"
                            )
                        elif x_is_num and not y_is_num:
                            fig3 = px.box(
                                df, x=target_col, y=col, points="all",
                                title=f"{target_col} (kateg.) vs. {col} (num.)"
                            )
                        else:
                            st.warning(f"Weder '{col}' noch '{target_col}' ist numerisch, Plot wird übersprungen!")
                            
                        st.plotly_chart(fig3, width='stretch')
            
            else:
                tabs[0].write("Wähle Features aus, um Scatter-/Box-Plots zu sehen")
            # ---- END original EDA block ----

        # --------------------------------
        # Skewness-Analyse der numerischen Features (Plotly)
        # --------------------------------

        with st.expander("📈 Skewness der numerischen Features", expanded=False):
            # st.subheader("📈 Skewness der numerischen Features")

            if len(num_cols) == 0:
                st.info("Keine numerischen Spalten verfügbar für Skewness-Analyse.")
            else:
                # Correct formula for rows
                rows = (len(num_cols) + 1) // 2

                # Force Streamlit to display this BEFORE columns
                st.markdown(f"**Diagrammzeilen:** {rows}")
                st.markdown("---")  # optional visual separator for clarity
                
                col1, col2 = st.columns(2)
                plot_cols = [col1, col2]

                for idx, col in enumerate(num_cols):
                    current_col = plot_cols[idx % 2]
                    with current_col:
                        df_col = df[col].dropna()
                        sk = df_col.skew()              # -> bias=False
                        # sk_stat = stats.skew(df_col)  # -> bias=True
                        n = len(df_col)

                        fig4 = px.histogram(
                            df,
                            x=col,
                            nbins=40,
                            title = f"{col} | Skew = {sk:.4f} | n={n}"
                        )
                        fig4.update_layout(
                            height=300,
                            margin=dict(l=10, r=10, t=40, b=10)
                        )
                        st.plotly_chart(fig4, width='stretch')

    with tab3:
        # --------------------------------
        # Outlier Detection (with skewness handling)
        # --------------------------------

        with st.expander("🚨 Ausreißer-Erkennung", expanded=True):
            st.markdown("Erkennung auf Basis **selbst gewählter** numerischer Features")
            default_cols = [c for c in num_cols if c != target_col]
            
            # 1) Auswahl der Spalten für die Erkennung
            cols_for_outlier = st.multiselect(
                "Spalten für die Ausreßer-Erkennung",
                options=default_cols,
                default=default_cols[:5] if len(default_cols) > 5 else default_cols[:3],
                help="Diese Spalten fließen in die Detektion ein (IQR/Z-Score/IsolationForest)."
            )
            if cols_for_outlier != st.session_state.cols_for_outlier:
                st.session_state.cols_for_outlier = cols_for_outlier
            
            # st.session_state.cols_for_outlier = cols_for_outlier  # <-- persist

            # Always define defaults to prevent NameErrors
            mask = pd.Series([False] * len(df))       # Default: no rows are outliers
            method = None                            # Default: no method selected
            apply_transform = False
            transformed_successfully = False
            show_normalized = False
            df_proc = df.copy()
            df_for_plot = df.copy()
            
            if not cols_for_outlier:
                st.info("Bitte mindestens eine numerische Feature-Spalte auswählen")
                mask = pd.Series([False] * len(df), index=df.index)  # Safe default mask
                # st.stop()   # prevent plotting when empty - too agressive
            else:
                # Auswahl der Methode
                method = st.radio(
                    "Methode wählen:",
                    ["IQR", "Z-Score", "IsolationForest"],
                    horizontal=True
                )
                st.session_state.outlier_method = method  # <-- persist

                df_proc = df.copy()
                df_for_plot = df
                apply_transform = False
                transformed_successfully = False

                # ----------------------------
                # Z-Score: Optional Skewness-Korrektur - Nur Visualisierung
                # ----------------------------
                if method == "Z-Score":
                    cleaned_df = df[cols_for_outlier].dropna()
                    skew_values = cleaned_df.apply(skew)
                    highly_skewed = skew_values[skew_values.abs() > 1]

                    if not highly_skewed.empty:
                        st.warning(
                            f"⚠️ Hohe Skewness entdeckt in: "
                            f"{', '.join(highly_skewed.index)} "
                            f"(|skew| > 1.0). Transformation empfohlen!"
                        )
                        apply_transform = st.checkbox(
                            "🔄 Skewness-Korrektur nur für die Visualisierung (Yeo-Johnson)",
                            value=True,
                            help="Wendet eine Yeo-Johnson-Transformation an, um schiefe Verteilungen zu stabilisieren.  \n"
                                "Diese Transformation wird nur für die Visualisierung und Z-Score-Ausreißererkennung verwendet, "
                                "nicht für das Modelltraining."
                        )
                    else:
                        st.info("✅ Keine stark schiefen Verteilungen gefunden — Transformation nicht erforderlich.")

                # ----------------------------
                # Berechnung durchführen
                # ----------------------------
                if method == "IQR":
                    factor = st.slider("IQR-Faktor", 1.0, 3.0, 1.5, 0.1, format="%.1f")
                    mask = detect_outliers_iqr(df, cols_for_outlier, factor)
                    st.session_state.mask = mask  # <-- persist
                    df_for_plot = df

                elif method == "Z-Score":
                    threshold = st.slider("Z-Score-Schwelle", 2.0, 5.0, 3.0, 0.1, format="%.1f")

                    # Transformation, falls gewünscht
                    if apply_transform:
                        try:
                            pt = PowerTransformer(method="yeo-johnson")
                            df_proc[cols_for_outlier] = pt.fit_transform(df_proc[cols_for_outlier])
                            st.success("✅ Yeo-Johnson-Transformation **nur für Visualisierung** erfolgreich angewendet!")
                            transformed_successfully = True
                        except Exception as e:
                            st.error(f"Fehler bei Transformation: {e}")
                            transformed_successfully = False

                    # Optionaler Toggle für Visualisierung
                    if "show_normalized" not in st.session_state:
                        st.session_state.show_normalized = False
                    
                    show_normalized = st.checkbox(
                        "📐 Normalisierte Ansicht anzeigen",
                        value=False,
                        help="Zeigt Scatterplots und Boxplots mit transformierten Werten an"
                    ) if apply_transform and transformed_successfully else False

                    # Maskenberechnung — immer von den korrekten Daten
                    if apply_transform and transformed_successfully:
                        mask = detect_outliers_z(df_proc, cols_for_outlier, threshold)
                        st.session_state.mask = mask  # <-- persist
                        df_for_plot = df_proc.copy() if show_normalized else df
                        # Zielspalte immer aus Rohdaten übernehmen
                        df_for_plot[target_col] = df[target_col]
                    else:
                        mask = detect_outliers_z(df, cols_for_outlier, threshold)
                        st.session_state.mask = mask  # <-- persist
                        df_for_plot = df

                else:
                    cont = st.slider("IsolationForest: Contamination", 0.01, 0.20, 0.05, 0.01)
                    mask = detect_outliers_iforest(df, cols_for_outlier, contamination=cont, random_state=random_state)
                    st.session_state.mask = mask  # <-- persist
                    df_for_plot = df

            # ----------------------------
            # Zusammenfassung der Ausreißerinformation
            # ----------------------------
            n_outlier = int(mask.sum())
            st.write(f"Gefundene Ausreßer: **{n_outlier}** von {len(df)} Zeilen ({n_outlier/len(df):.1%})")

            # ----------------------------
            # Visualisierung (immer anzeigen, selbst bei 0 Ausreißern)
            # ----------------------------
            x_feat = st.selectbox(
                "Feature für die Visualisierung (X-Achse)",
                options=default_cols,
                index=0
            )

            fig_out = px.scatter(
                df_for_plot,
                x=x_feat,
                y=target_col,
                color=mask.map({
                    True: "Ausreßer",
                    False: "Normal"
                }),
                title=f"Ausreißer-Visualisierung: {x_feat} vs. {target_col}",
                labels={x_feat: x_feat, target_col: target_col},
                color_discrete_map={"Ausreßer": "#EF553B", "Normal": "#19D3f3"}
            )
            st.plotly_chart(fig_out, width='stretch')

            # ----------------------------
            # Boxplots optional anzeigen
            # ----------------------------
            if st.checkbox("Boxplots der ausgewählten Spalten anzeigen"):
                if not cols_for_outlier:
                    st.warning("⚠️ Bitte zuerst mindestens eine Spalte auswählen, um Boxplots anzuzeigen.")
                else:
                    points_mode = st.selectbox(
                        "Welche Punkte sollen angezeigt werden?",
                        options=["Alle Punkte", "Nur Ausreißer", "Keine Punkte"],
                        index=0
                    )
                    mapping = {
                        "Alle Punkte": "all",
                        "Nur Ausreißer": "outliers",
                        "Keine Punkte": False
                    }

                # ✅ Sync boxplot dataset with scatterplots
                if method == "Z-Score" and show_normalized and apply_transform and transformed_successfully:
                    df_box = df_proc.copy()
                    df_box[target_col] = df[target_col]  # Keep Y in original scale
                else:
                    df_box = df

                # Plot boxplots using the correct dataset
                for col in cols_for_outlier:
                    fig_box = px.box(
                        df_box,
                        y=col,
                        points=mapping[points_mode],
                        title=f"Boxplot - {col}"
                    )
                    st.plotly_chart(fig_box, width='stretch')

        # ----------------------------
        # Ausreißer entfernen (optional)
        # ----------------------------
        if st.checkbox("Ausreißer aus den Daten entfernen (für das Training)"):
            filtered_df = df.loc[~mask].copy()
            st.session_state.df_filtered = filtered_df
            st.success(f"Neue Datenform: {filtered_df.shape[0]} Zeilen x {filtered_df.shape[1]} Spalten")
        else:
            st.session_state.df_filtered = df.copy()

        st.write(f"Datenform für das Training: {st.session_state.df_filtered.shape[0]} Zeilen x {st.session_state.df_filtered.shape[1]} Spalten")

    with tab4:
        # --------------------------------
        # Outlier Detection
        # --------------------------------

        # --------------------------------
        # Optional: Apply global skewness correction for modelling
        # --------------------------------
        if apply_skew_correction_global:
            try:
                pt_global = PowerTransformer(method="yeo-johnson")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = pt_global.fit_transform(df[numeric_cols])
                st.success("✅ Globale Yeo-Johnson-Skewness-Korrektur erfolgreich angewendet!")
            except Exception as e:
                st.error(f"Fehler bei der globalen Transformation: {e}")

        # Erstellen von Feature-Matrix und Label-Vektor:
        df_train = st.session_state.get("df_filtered", df) # pass a saved filtered dataframe from Ausreißer section
        st.markdown(f"Ausreißer-entfernte Datenform: {df_train.shape[0]} Zeilen x {df_train.shape[1]} Spalten")
        
        # --- Outlier settings from Tab 2 ---
        cols_for_outlier = st.session_state.get("cols_for_outlier", [])
        method = st.session_state.get("outlier_method", None)
        mask = st.session_state.get("mask", pd.Series(False, index=df.index))

        X = df.drop(columns=[target_col], axis=1)
        y = df[target_col]



        # Spaltentypen bestimmen:
        num_cols_all, cat_cols_all = infer_column_types(X)
        # st.write("num_cols_all: ", num_cols)
        # st.write("cat_cols_all: ", cat_cols_all)

        # Preprocessing
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True))
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols_all),
                ("cat", categorical_transformer, cat_cols_all)
            ],
            remainder="passthrough"     # optional "drop"
        )

        # Modell erstellen
        model = build_model(model_name)

        # Pipeline zusammenbauen
        pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", model)
            ]
        )

        # --------------------------
        # Modelltrainieren
        # --------------------------

        # Check if we already have stored results in the session
        if "model_trained" not in st.session_state:
            st.session_state.model_trained = False

        # Beginnt erst nach dem Drücken der Taste
        if st.button("🚀 Modell trainieren"):
            
            # Train/Test-Split einbauen
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
                )

            # Modelltraining
            with st.spinner("Trainiere Modell..."):
                pipe.fit(X_train, y_train)

            # Vorhersagen erstellen 'y_tran_pred' und 'y_test_pred'
            y_train_pred = pipe.predict(X_train)
            y_test_pred = pipe.predict(X_test)

            # Performance-Metriken berechnen
            def metrics_block(y_true, y_pred):
                r2 = r2_score(y_true, y_pred)
                rmse = root_mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                return r2, rmse, mae

            train_r2, train_rmse, train_mae = metrics_block(y_train, y_train_pred)
            test_r2, test_rmse, test_mae = metrics_block(y_test, y_test_pred)

            # Alle Variablen in 'session_state' speichern
            st.session_state.update({
                "model_trained": True,
                "pipe": pipe,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "y_train_pred": y_train_pred,
                "y_test_pred": y_test_pred,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "preprocessor": preprocessor
            })
            
            # Save uploaded filename separately in session_state (for later comparison)
            if "upload" in locals() and upload is not None:
                st.session_state.uploaded_filename = upload.name
            
            # Store a unique hash representing the current data + model setup AT TRAINING TIME only!
            # Speichert den aktuellen Zustand der Trainingskonfiguration
            st.session_state.last_trained_state = {
                "df_shape": tuple(df.shape),
                "model_name": str(model_name),
                "test_size": float(test_size),
                "random_state": int(random_state),
                "apply_skew_correction_global": bool(apply_skew_correction_global),
                "cols_for_outlier": sorted([str(c) for c in cols_for_outlier]),
                # Store the filename to detect future changes in dataset
                "uploaded_filename": st.session_state.get("uploaded_filename", None),

                # Hyperparameters (convert to safe formats)
                "n_estimators": int(locals().get("n_estimators", 0)) if locals().get("n_estimators") is not None else None,
                "max_depth": int(locals().get("max_depth", 0)) if locals().get("max_depth") is not None else None,
                "max_features_choice": str(locals().get("max_features_choice", "")) if locals().get("max_features_choice") is not None else None,
                "learning_rate": float(locals().get("learning_rate", 0.0)) if locals().get("learning_rate") is not None else None,

                # Outlier detection options
                "outlier_method": str(locals().get("method", "")),
                "z_threshold": float(locals().get("threshold", 0.0)) if locals().get("threshold") is not None else None,
                "iqr_factor": float(locals().get("factor", 0.0)) if locals().get("factor") is not None else None,
                "apply_transform": bool(locals().get("apply_transform", False)),
            }

            
        # --------------------------
        # 📊 Performance-Anzeige
        # --------------------------

        if "train_r2" in st.session_state:    
            st.subheader("📊 Performance")
            
            # --- Detect outdated model results ---
            # Create a dict of the CURRENT UI + data settings
            current_state = {
                "df_shape": df.shape,
                "model_name": model_name,
                "test_size": test_size,
                "random_state": random_state,
                "apply_skew_correction_global": apply_skew_correction_global,
                "cols_for_outlier": sorted(cols_for_outlier) if cols_for_outlier else [],
                "uploaded_filename": st.session_state.get("uploaded_filename", None),

                # RandomForest-specific
                "n_estimators": locals().get("n_estimators", None),
                "max_depth": locals().get("max_depth", None),
                "max_features_choice": locals().get("max_features_choice", None),

                # GradientBoosting-specific
                "learning_rate": locals().get("learning_rate", None),
            }
            
                # Outlier detection options
                # Only include method-specific params if at least one feature is selected
            if cols_for_outlier:
                current_state.update({
                    "outlier_method": locals().get("method", None),
                    "z_threshold": locals().get("threshold", None),
                    "iqr_factor": locals().get("factor", None),
                    "apply_transform": locals().get("apply_transform", None),
                })
            else:
                # Set stable defaults when no features are selected
                current_state.update({
                    "outlier_method": None,
                    "z_threshold": None,
                    "iqr_factor": None,
                    "apply_transform": False,
                })
            
            # Normalize states before comparison
            def normalize_state(state: dict, cols: list) -> dict:
                """Ensure consistent defaults & ignore stale outlier params when no cols are selected"""
                normalized = dict(state)  # shallow copy
                
                # When NO features are selected, force defaults
                if not cols:
                    normalized["outlier_method"] = None
                    normalized["z_threshold"] = None
                    normalized["iqr_factor"] = None
                    normalized["apply_transform"] = False
                
                # Fill missing defaults safely
                for key, default in {
                    "outlier_method": None,
                    "z_threshold": None,
                    "iqr_factor": None,
                    "apply_transform": False,
                }.items():
                    if key not in normalized:
                        normalized[key] = default

                return normalized

            trained_state = normalize_state(
                st.session_state.get("last_trained_state", {}), cols_for_outlier
            )
            current_state = normalize_state(current_state, cols_for_outlier)

            # Check if last trained state differs from the current UI state
            # Show warning if training state doesn't match current settings
            if trained_state != current_state:
                st.warning(
                    "⚠️ Hinweis: Die unten angezeigten Ergebnisse stammen aus einem vorherigen Training "
                    "und spiegeln möglicherweise **nicht** die aktuellen Änderungen wider. "
                    "Bitte trainiere das Modell erneut."
                )
            
            # Use saved values
            train_r2 = st.session_state.train_r2
            test_r2 = st.session_state.test_r2
            train_rmse = st.session_state.train_rmse
            test_rmse = st.session_state.test_rmse
            train_mae = st.session_state.train_mae
            test_mae = st.session_state.test_mae
            y_test = st.session_state.y_test
            y_test_pred = st.session_state.y_test_pred

            # ------------------------------
            # Metric-Anzeige
            # ------------------------------
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Train R²**", help=(
                    "R² (Bestimmtheitsmaß) für die Trainingsdaten.\n"
                    "- 1.0 = perfekte Anpassung\n"
                    "- 0 = keine Erklärungskraft\n\n"
                    "⚠️ Ein sehr hohes Train R² bei deutlich niedrigerem Test R² kann auf Overfitting hindeuten."
                ))
                st.metric(" ", f"{train_r2:.3f}")

            with c2:
                st.markdown("**Test R²**", help=(
                    "R² (Bestimmtheitsmaß) für die Testdaten.\n"
                    "Je näher an 1.0, desto besser erklärt das Modell die Varianz unbekannter Daten.\n\n"
                    "👉 Dies ist die wichtigste Kennzahl zur Beurteilung der Generalisierungsfähigkeit."
                ))
                st.metric(" ", f"{test_r2:.3f}")

            with c3:
                st.markdown("**Generalization Gap ΔR²**", help=(
                    "Differenz zwischen Train- und Test-R².\n"
                    "Ein kleiner Wert = gute Generalisierung.\n"
                    "Ein großer positiver Wert = Overfitting."
                ))
                st.metric(" ", f"{(train_r2 - test_r2):.3f}")

            c4, c5, c6 = st.columns(3)
            with c4:
                st.markdown("**Test RMSE**", help=(
                    "Root Mean Squared Error (Wurzel des mittleren quadratischen Fehlers).\n"
                    "- Misst die durchschnittliche Abweichung zwischen Vorhersage und Realität.\n"
                    "- Größere Fehler werden stärker bestraft (Quadrat).\n"
                    "👉 Je kleiner, desto besser."
                ))
                st.metric(" ", f"{test_rmse:.4f}")

            with c5:
                st.markdown("**Test MAE**", help=(
                    "Mean Absolute Error (mittlerer absoluter Fehler).\n"
                    "- Misst die durchschnittliche Abweichung (in Originaleinheiten).\n"
                    "- Weniger empfindlich gegenüber Ausreißern als RMSE.\n"
                    "👉 Je kleiner, desto besser."
                ))
                st.metric(" ", f"{test_mae:.4f}")

            with c6:
                st.markdown("**Train RMSE**", help=(
                    "Root Mean Squared Error auf den Trainingsdaten.\n"
                    "- Gibt an, wie gut das Modell die Trainingsdaten angepasst hat.\n"
                    "👉 Vergleiche mit Test-RMSE: große Unterschiede deuten auf Overfitting hin."
                ))
                st.metric(" ", f"{train_rmse:.4f}")


            # Cross-Validation durchführen
            with st.expander("📦 K-Fold Cross-Validation"):
                k = st.slider("K (Folds)", 3, 10, 5, 1)
                
                # 1) If user clicks button → calculate CV and store results
                if st.button("CV starten"):
                    try:
                        # Spinner while K-Fold runs
                        with st.spinner("K-Fold durchführen..."):
                            cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
                            scores = cross_val_score(st.session_state.pipe, X, y, scoring="r2", cv=cv, n_jobs=-1)

                        # Ergebnisse und Figure persistent speichern
                        st.session_state.cv_scores = scores
                        st.session_state.cv_k = k
                        st.session_state.cv_fig = px.box(
                            y=scores,
                            points="all",
                            title="CV R² Scores"
                        )

                    except Exception as e:
                        st.error(f"Fehler bei der Cross-Validation: {e}")

                # 2) Display persisted results if they exist
                if "cv_scores" in st.session_state:
                    try:
                        st.info(f"Zeige gespeicherte Ergebnisse für K={st.session_state.cv_k}")
                        st.write(
                            f"R² Mittelwert: **{st.session_state.cv_scores.mean():.3f}**  |  "
                            f"Std: **{st.session_state.cv_scores.std():.3f}**"
                        )

                        # Zeige gespeichertes Diagramm, falls vorhanden
                        if "cv_fig" in st.session_state:
                            st.plotly_chart(st.session_state.cv_fig, width='stretch')
                    except Exception as e:
                        st.warning(f"Konnte gespeicherte Cross-Validation-Ergebnisse nicht laden: {e}")

                # old part of the code block
                    #     st.write(f"R² Mittelwert: **{scores.mean():.3f}**  |  Std: **{scores.std():.3f}**")
                    #     fig = px.box(y=scores, points="all", title="CV R² Scores")
                    #     st.session_state.cv_fig = fig
                    #     st.plotly_chart(fig, width='stretch')
                    # except Exception as e:
                    #     st.error(f"Fehler bei der Cross-Validation: {e}")


            # ------------------------
            # Visualisierungen (Actual vs. Predicted, Residuum-Plot) 
            # (Persist via session_state)
            # ------------------------

        if (
            "y_test" in st.session_state 
            and "y_test_pred" in st.session_state 
            and "pipe" in st.session_state 
            and "X_train" in st.session_state
        ):
            st.subheader("🖼️ Plotly-Visualisierungen")
            
            # Restore necessary variables from session_state
            y_test = st.session_state.y_test
            y_test_pred = st.session_state.y_test_pred
            pipe = st.session_state.pipe
            X_train = st.session_state.X_train

            # 1) Actual vs Predicted (Test)
            fig_ap = go.Figure()
            fig_ap.add_trace(go.Scatter(x=y_test, y=y_test_pred, mode="markers", name="Predictions"))
            miny = min(y_test.min(), y_test_pred.min())
            maxy = max(y_test.max(), y_test_pred.max())
            fig_ap.add_trace(go.Scatter(x=[miny, maxy], y=[miny, maxy], mode="lines", name="Ideal (y=x)"))
            fig_ap.update_layout(title="Actual vs. Predicted (Test)", xaxis_title="Actual", yaxis_title="Predicted")
            st.plotly_chart(fig_ap, width='stretch')

            # 2) Residuen-Plot (Test)
            resid = y_test_pred - y_test
            fig_res = px.scatter(x=y_test_pred, y=resid, labels={"x": "Predicted", "y": "Residual"},
                                title="Residuen vs. Vorhersage (Test)")
            fig_res.add_hline(y=0)
            st.plotly_chart(fig_res, width='stretch')

            # 3) Residuen-Verteilung
            fig_hist = px.histogram(resid, nbins=50, title="Residuen-Histogramm (Test)")
            st.plotly_chart(fig_hist, width='stretch')

            # 4) Feature Importance / Koeffizienten (wenn verfügbar)
            with st.expander("🧠 Feature-Wichtigkeit / Koeffizienten"):
                st.markdown(
                    "ℹ️ **Hinweis:** "
                    "Bei baumbasierten Modellen (z. B. Random Forest, Gradient Boosting) "
                    "wird die Wichtigkeit anhand der Feature-Splits berechnet. "
                    "Bei linearen Modellen (z. B. Ridge, Lasso, ElasticNet) "
                    "werden die Regressionskoeffizienten angezeigt."
                )
                try:
                    # Hole Variablen aus st.session_state, falls verfügbar
                    if "preprocessor" in st.session_state and "X_train" in st.session_state and "pipe" in st.session_state:
                        preprocessor = st.session_state.preprocessor
                        X_train = st.session_state.X_train
                        pipe = st.session_state.pipe
                        
                        # Feature-Namen nach Preprocessing extrahieren
                        feature_names = get_feature_names_from_ct(preprocessor, X_train)
                        model_step = pipe.named_steps["model"]

                        if hasattr(model_step, "feature_importances_"):
                            importances = model_step.feature_importances_
                            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
                            fig_imp = px.bar(imp_df.head(40), x="importance", y="feature", orientation="h", title="Feature Importances (Top 40)")
                            st.plotly_chart(fig_imp, width='stretch')
                            st.dataframe(imp_df, width='stretch')
                        elif hasattr(model_step, "coef_"):
                            coefs = np.ravel(model_step.coef_)
                            coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs}).sort_values("coefficient", key=np.abs, ascending=False)
                            fig_coef = px.bar(coef_df.head(40), x="coefficient", y="feature", orientation="h", title="Koeffizienten (Top 40, nach Betrag sortiert)")
                            st.plotly_chart(fig_coef, width='stretch')
                            st.dataframe(coef_df, width='stretch')
                        else:
                            st.info("Dieses Modell stellt keine Feature-Importances/Koeffizienten bereit.")
                    else:
                        st.info("ℹTrainiere zuerst ein Modell, um die Feature-Wichtigkeiten zu sehen.")
                except Exception as e:
                    st.warning(f"Konnte Feature-Wichtigkeiten nicht extrahieren: {e}")

            # ------------------------
            # Inferenz auf neuen Daten
            # ------------------------

            with st.expander("🛠️ Neue Daten vorhersagen"):
                pred_file = st.file_uploader("CSV mit **denselben Feature-Spalten** (ohne Target) hochladen", type=["csv"], key="pred")
                if pred_file:
                    try:
                        pred_df = pd.read_csv(pred_file, sep=sep)
                        preds = pipe.predict(pred_df)
                        out = pred_df.copy()
                        out["prediction"] = preds
                        st.dataframe(out.head(50), width='stretch')
                        # Download
                        csv_bytes = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Vorhersagen als CSV herunterladen", data=csv_bytes, file_name="regression_predictions.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Vorhersage fehlgeschlagen: {e}")


        # -----------------
        # Modell speichern
        # -----------------

        with st.expander("💾 Trainiertes Modell speichern", expanded=True):
            if st.button("Als .joblib exportieren"):
                buf = io.BytesIO()
                dump(pipe, buf)
                buf.seek(0)

                # Use uploaded filename if available, else fallback to 'model'
                uploaded_file = st.session_state.get("uploaded_filename", "dataset")
                # Remove file extension from uploaded filename, keep only base name
                base_name = os.path.splitext(uploaded_file)[0]

                st.download_button(
                    label="Download model.joblib",
                    data=buf,
                    file_name=f"{base_name}_regression_{model_name}.joblib",
                    mime="application/octet-stream",
                )

    # --------------------------
    # 📊 Vorhersage
    # --------------------------
    
    with tab5:
        st.subheader("Vorhersage & Visualisierungen")
        st.info("To be developed")


# Footer
# st.caption("Made with ❤️")
st.caption("🚧 Under Construction / Noch im Bau", width="stretch")
