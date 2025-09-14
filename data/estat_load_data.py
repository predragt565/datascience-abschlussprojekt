import requests
import pandas as pd
import numpy as np

import sys
import json
from IPython.display import display


# ----------------------------------------------------------- #
# ---  Main project file for loading and preaparing data  --- #
# ---                                                     --- #
# ----------------------------------------------------------- #

def eurostat_url():
    """
    Returns a full URL path to the EUSTAT JSON Databuilder Tool  
    with predefined filters:  
    * Time period from: 2012-01
    * Time period to: present
    * The EU country: DK, DE, EL, ES, HR, IT, PT, FI, SE, NO
    * Visitor type: DOM, FOR
    * Accommodation NACEr2 category: I551, I552, I553
    * Dataset language: De (Deutsch)
    """
    # URL of the JSON data
    url_domain = "https://ec.europa.eu/eurostat/"
    url_site = "api/dissemination/statistics/1.0/data/tour_occ_nim"
    url_qry_base = "?format=JSON"
    url_qry_period_from = "&sinceTimePeriod=2012-01"
    url_qry_period_to = ""
    url_qry_geo = "&geo=DK&geo=DE&geo=EL&geo=ES&geo=HR&geo=IT&geo=PT&geo=FI&geo=SE&geo=NO"
    url_qry_unit = "&unit=NR&unit=PCH_SM&unit=PCH_SM_19"
    url_qry_resid = "&c_resid=DOM&c_resid=FOR"
    url_qry_nace = "&nace_r2=I551&nace_r2=I552&nace_r2=I553"
    url_qry_lang = "&lang=de"

    url = url_domain + url_site + url_qry_base + url_qry_period_from + url_qry_period_to + url_qry_geo + url_qry_unit + url_qry_resid + url_qry_nace + url_qry_lang
    return url

def validate_required_columns_json(df):
    """
    Validate if required columns are present in the DataFrame based on JSON URL fetch.
    Raises ValueError if any are missing.
    """
    required_columns = {
        "Aufenthaltsland_Idx",
        "NACEr2_Idx",
        "Geopolitische_Meldeeinheit_Idx",
        "JahrMonat",
        "value"
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Bitte wähle eine geeignete JSON-Datei aus.\nFehlende erforderliche Spalten: {missing}")
    return df

def validate_required_columns_csv(df):
    """
    Validate if required columns are present in the DataFrame based on loaded CSV file.
    Raises ValueError if any are missing.
    """
    required_columns = {
        "Aufenthaltsland_Idx",
        "Aufenthaltsland",
        "NACEr2_Idx",
        "NACEr2",
        "Geopolitische_Meldeeinheit_Idx",
        "Geopolitische_Meldeeinheit",
        "JahrMonat",
        "value",
        "pch_sm",
        "pch_sm_19",
        "month",
        "pch_sm_12",
        "Monat",
        "Quartal",
        "Saison",
        "Jahr",
        "Month_cycl_sin",
        "Month_cycl_cos",
        "MA3",
        "MA6",
        "MA12",
        "Lag_1",
        "Lag_3",
        "Lag_12",
        "Aufenthaltsland_Saison",
        "NACEr2_Saison",
        "Land_Monat",
        "Land_Saison",
        "pandemic_dummy"
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Bitte laden eine geeignete CSV-Datei hoch.\nFehlende erforderliche Spalten: {missing}")
    return df


def load_prepare_data(json_obj):
    """
    Load and preprocess Eurostat JSON object into prepared DataFrame df_anzahl.
    """
    # URL of the JSON data
    # url_domain = "https://ec.europa.eu/eurostat/"
    # url_site = "api/dissemination/statistics/1.0/data/tour_occ_nim"
    # url_qry_base = "?format=JSON"
    # url_qry_period_from = "&sinceTimePeriod=2012-01"
    # url_qry_period_to = ""
    # url_qry_geo = "&geo=DK&geo=DE&geo=EL&geo=ES&geo=HR&geo=IT&geo=PT&geo=FI&geo=SE&geo=NO"
    # url_qry_unit = "&unit=NR&unit=PCH_SM&unit=PCH_SM_19"
    # url_qry_resid = "&c_resid=DOM&c_resid=FOR"
    # url_qry_nace = "&nace_r2=I551&nace_r2=I552&nace_r2=I553"
    # url_qry_lang = "&lang=de"

    # url = url_domain + url_site + url_qry_base + url_qry_period_from + url_qry_period_to + url_qry_geo + url_qry_unit + url_qry_resid + url_qry_nace + url_qry_lang

    # --- Input validation ---
    if json_obj is None or not isinstance(json_obj, dict):
        raise ValueError("Invalid input: 'json_obj' must be a non-null JSON object (dict).")
    
    if json_obj:
        # --- Use input object ---
        data = json_obj
    else:
        # --- Download and parse the JSON---
        response = requests.get(url) # BUG:
        data = response.json()

    
    dims = data['dimension']
    values = data['value']
    
    # --- Extract dimension metadata --- #
    dim_order = data['id']  # dimension order
    dim_sizes = data['size']  # sizes of dimensions
    dim_labels = {dim: dims[dim]['label'] for dim in dim_order}
    dim_categories = {dim: dims[dim]['category']['label'] for dim in dim_order}
    dim_category_keys = {dim: list(dims[dim]['category']['label'].keys()) for dim in dim_order}

    # TODO: Print dimension overviews in table format     # <- move to Streamlit UI
    # print("DIMENSION OVERVIEW:\n")
    # for dim in dim_order:
    #     label = dim_labels[dim]
    #     categories = dim_categories[dim]
        
    #     df = pd.DataFrame(list(categories.items()), columns=[f"{label} ID", f"{label} Label"])
    #     print(f"Dimension: {label}")
        # print(df.to_string(index=False))
        # display(df) # Use with IPython imported module only
        # print("\n" + "-" * 40 + "\n")


    # --- Flatten values --- #
    records = []
    for idx, val in values.items():
        idx = int(idx)
        indexes = []
        remainder = idx
        for size in reversed(dim_sizes):
            indexes.append(remainder % size)
            remainder //= size
        indexes.reverse()

        # Map to dimension keys and values
        row = {}
        for i, dim in enumerate(dim_order):
            keys = dim_category_keys[dim]
            key = keys[indexes[i]]
            label = dim_categories[dim][key]
            # dim_name = dim_labels[dim]

            # Column names
            if dim == 'freq':
                row['Zeitliche_Frequenz_Idx'] = key
                row['Zeitliche_Frequenz'] = label
            elif dim == 'c_resid':
                row['Aufenthaltsland_Idx'] = key
                row['Aufenthaltsland'] = label
            elif dim == 'unit':
                row['Maßeinheit_Idx'] = key
                row['Maßeinheit'] = label
            elif dim == 'nace_r2':
                row['NACEr2_Idx'] = key
                row['NACEr2'] = label
            elif dim == 'geo':
                row['Geopolitische_Meldeeinheit_Idx'] = key
                row['Geopolitische_Meldeeinheit'] = label
            elif dim == 'time':
                # row['Zeit_Idx'] = key
                row['JahrMonat'] = label
        row['value'] = val
        records.append(row)

    # Create DataFrame
    df = pd.DataFrame(records)
    
    # --- Validate columns before further processing ---
    df = validate_required_columns_json(df)
    # missing = required_columns - set(df.columns)
    # if missing:
    #     raise ValueError(f"Missing required columns: {missing}")


    # Export to CSV - before adding additional features
    # csv_filename = "estat_tour_overnight_stays_2012-2025_eu10_de.csv"
    # df.to_csv(csv_filename, index=False)
    # print(f"Exported to {csv_filename}")
    # print(df.tail(30))

    # df["Maßeinheit_Idx"].value_counts()

    # --- Separate records based on type of unit in "Maßeinheit_Idx" --- #
    mask_anzahl = df["Maßeinheit_Idx"] == "NR"
    mask_pch_sm = df["Maßeinheit_Idx"] == "PCH_SM"
    mask_pch_sm_19 = df["Maßeinheit_Idx"] == "PCH_SM_19"
    df_anzahl = df[mask_anzahl].copy()  # independent copy - no. of overnight stays
    df_pch_sm = df[mask_pch_sm].copy()    # independent copy - percentage change MoM (YoY)
    df_pch_sm_19 = df[mask_pch_sm_19].copy()    # independent copy - percentage change MoM (vs 2019)

    # --- Add percentage change columns MoM, YoY, MoBaseYear back to the main dataframe 'df_anzahl' --- #
    # Define join keys
    keys = ["Aufenthaltsland_Idx", "NACEr2_Idx", "Geopolitische_Meldeeinheit_Idx", "JahrMonat"]

    # Select + rename value column from df_pch_sm
    df_pch_sm_sel = df_pch_sm[keys + ["value"]].rename(columns={"value": "pch_sm"})

    # Select + rename value column from df_pch_sm_19
    df_pch_sm_19_sel = df_pch_sm_19[keys + ["value"]].rename(columns={"value": "pch_sm_19"})

    # Merge step by step
    df_merged = (
        df_anzahl
        .merge(df_pch_sm_sel, on=keys, how="left")
        .merge(df_pch_sm_19_sel, on=keys, how="left")
    )

    df_anzahl = df_merged.copy()

    # --- Add a new column 'pch_sm_12' --- #
    # - Calculate percentage change same month over base year 2012
    # Define the grouping columns
    group_cols = ["Aufenthaltsland_Idx", "NACEr2_Idx", "Geopolitische_Meldeeinheit_Idx"]

    # # Extract month part to align with base year
    df_anzahl["month"] = df_anzahl["JahrMonat"].str[-2:]

    # Create a reference DataFrame with 2012 values
    base = (
        df_anzahl[df_anzahl["JahrMonat"].str.startswith("2012")]
        .loc[:, group_cols + ["month", "value"]]
        .rename(columns={"value": "base_value"})
    )

    # Merge base year values on group + month
    df_12 = df_anzahl.merge(base, on=group_cols + ["month"], how="left")

    # Calculate percentage change from base
    df_12["pch_sm_12"] = np.where(
        df_12["base_value"].notnull() & (df_12["base_value"] != 0),
        round(((df_12["value"] - df_12["base_value"]) / df_12["base_value"]) * 100, 2),
        np.nan
    )
    df_anzahl["pch_sm_12"] = df_12["pch_sm_12"].copy()

    # display(df_anzahl)


    # ---  Add additional Feature columnms  --- #
    # 1. Monat (month as number, regex)
    df_anzahl["Monat"] = df_anzahl["JahrMonat"].str.extract(r"-(\d{2})").astype(int)

    # 2. Quartal (map months to quarters)
    month_to_quarter = {
        1: "1", 2: "1", 3: "1",
        4: "2", 5: "2", 6: "2",
        7: "3", 8: "3", 9: "3",
        10: "4", 11: "4", 12: "4"
    }
    df_anzahl["Quartal"] = df_anzahl["Monat"].map(month_to_quarter)

    # 3. Season (die Saison/ die Jahreszeit)
    month_to_season = {
        1: "Winter", 2: "Winter", 3: "Frühling",
        4: "Frühling", 5: "Frühling", 6: "Sommer",
        7: "Sommer", 8: "Sommer", 9: "Herbst",
        10: "Herbst", 11: "Herbst", 12: "Winter"
    }
    df_anzahl["Saison"] = df_anzahl["Monat"].map(month_to_season)

    # 4. Jahr (year as number, regex)
    df_anzahl["Jahr"] = df_anzahl["JahrMonat"].str.extract(r"(\d{4})").astype(int)

    # 5. Cyclical encoding of month
    df_anzahl["Month_cycl_sin"] = np.sin(2 * np.pi * df_anzahl["Monat"] / 12)
    df_anzahl["Month_cycl_cos"] = np.cos(2 * np.pi * df_anzahl["Monat"] / 12)

    # 6. Moving averages (gleitender Durchschnitt)
    df_anzahl = df_anzahl.sort_values(["Geopolitische_Meldeeinheit_Idx", 
                        "Aufenthaltsland_Idx",
                        "NACEr2_Idx",
                        "JahrMonat"]) # chronological order

    df_anzahl["MA3"] = (
        df_anzahl.groupby(["Geopolitische_Meldeeinheit_Idx", 
                    "Aufenthaltsland_Idx",
                    "NACEr2_Idx"])["value"]
        .transform(lambda x: x.shift(1).rolling(3).mean().fillna(0).astype(int))
    )

    df_anzahl["MA6"] = (
        df_anzahl.groupby(["Geopolitische_Meldeeinheit_Idx", 
                    "Aufenthaltsland_Idx",
                    "NACEr2_Idx"])["value"]
        .transform(lambda x: x.shift(1).rolling(6).mean().fillna(0).astype(int))
    )

    df_anzahl["MA12"] = (
        df_anzahl.groupby(["Geopolitische_Meldeeinheit_Idx", 
                    "Aufenthaltsland_Idx",
                    "NACEr2_Idx"])["value"]
        .transform(lambda x: x.shift(1).rolling(12).mean().fillna(0).astype(int))
    )

    # 7. Lags des Targets (lag_1, lag_3, lag_12):
    df_anzahl = df_anzahl.sort_values(["Geopolitische_Meldeeinheit_Idx",
                        "Aufenthaltsland_Idx",
                        "NACEr2_Idx",
                        "JahrMonat"])

    # Gruppe definieren
    grp = df_anzahl.groupby(["Geopolitische_Meldeeinheit_Idx",
                    "Aufenthaltsland_Idx",
                    "NACEr2_Idx"])

    # Lags berechnen
    for L in [1, 3, 12]:
        df_anzahl[f"Lag_{L}"] = grp["value"].shift(L).fillna(0).astype(int)

    # 8. Residency × Saison (Inländer vs. Ausländer):
    df_anzahl["Aufenthaltsland_Saison"] = df_anzahl["Aufenthaltsland_Idx"] + "_" + df_anzahl["Saison"].astype(str)

    # 9. Unterkunft × Saison:
    df_anzahl["NACEr2_Saison"] = df_anzahl["NACEr2_Idx"] + "_" + df_anzahl["Saison"].astype(str)

    # 10. Land × Monat:
    df_anzahl["Land_Monat"] = df_anzahl["Geopolitische_Meldeeinheit_Idx"] + "_" + df_anzahl["Monat"].astype(str)

    # 10. Land × Saison:
    df_anzahl["Land_Saison"] = df_anzahl["Geopolitische_Meldeeinheit_Idx"] + "_" + df_anzahl["Saison"].astype(str)

    # 11. Pandemic (Covid19) Maske
    # Define pandemic period (adjust dates as needed)
    pandemic_start = "2020-03"
    pandemic_end   = "2023-04"

    df_anzahl["JahrMonat"] = pd.to_datetime(df_anzahl["JahrMonat"])   # <- use later to convert to DateTime type

    df_anzahl["pandemic_dummy"] = (
        (df_anzahl["JahrMonat"] >= pandemic_start) &
        (df_anzahl["JahrMonat"] <= pandemic_end)
    ).astype(int)

    # display(df_anzahl)

    # DONE: --- Check NaN values --- #    <- move this block to Streamlit UI
    # df_anzahl.info()
    # na_counts = df_anzahl.isna().sum()
    # na_counts = na_counts[na_counts > 0]
    # na_counts = na_counts.to_frame(name="NaN count")
    # na_counts

    # --- Replace all NaN values with zero (0) - in 'pch_sm', 'pch_sm_19' , 'pch_sm_12' --- #
    cols_to_fix = ["pch_sm", "pch_sm_19", "pch_sm_12"]
    df_anzahl[cols_to_fix] = df_anzahl[cols_to_fix].fillna(0)
    # df_anzahl.info()
    
    # --- Remove all columns with 1 unique value only (Aufenthaltsland- _Idx, Zeitliche_Frequenz- _Idx, Maßeinheit- _Idx)

    # DONE: --- Identify categorical (object/string) columns --- #    <- move this block to Streamlit UI
    cat_cols = df_anzahl.select_dtypes(include=["object"]).columns

    # Count unique values per categorical column
    unique_counts = df_anzahl[cat_cols].nunique().sort_values(ascending=False)

    # Keep only categorical columns with more than 1 unique value
    valid_cat_cols = unique_counts[unique_counts > 1].index

    # Drop the rest
    df_anzahl = df_anzahl.drop(columns=[col for col in cat_cols if col not in valid_cat_cols])

    # ----------------------- #
    
    # print("Unique values per categorical column:\n")
    # print(unique_counts)

    # for col in cat_cols:
    #     print(f"{col}: {df_anzahl[col].nunique()} unique values\n")
    #     print(df_anzahl[col].value_counts().head(10))  # show top 10 categories
    #     print("-" * 50 + "\n")

    # Export to CSV - including extended features - DISABLED
    # csv_filename = "data/estat_tour_overnight_stays_2012-2025_eu10_de_ext.csv"
    # df_anzahl.to_csv(csv_filename, index=False)
    # print(f"JSON converted Dataframe exported to {csv_filename}")
    # print(df_anzahl.tail(10))

    return df_anzahl

# --- END of this part: all new feature columns added, NaN values replaced --- #
# --- --- --- --- --- #