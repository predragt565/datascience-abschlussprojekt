## Abschlussprojekt - Data Science - Overnight stays in the EU 2012-2025

### Task List:

### **IMPORTANT!:**
- check grouping level for calculating moving average - DONE

### Part 1 - Data preparation

1. Set a URL schema to fetch data from the Eurostat website Tool - DONE  

2. Download and parse JSON dataset into Pandas DataFrame, extract dimensions  - DONE  

3. Flatten values, map dimension keys and values for the Features (categories) - DONE  

4. Add additional statistical categories:   - DONE
   - Target lag 
   - Season by residency 
   - Season by accommodation type 
   - Season by country
   - Month by country

5. Handle the missing values - DONE

6. ### !! Important !! - Handling Pandemic period (2020-03 through 2023-04)
* keep the full dataset but mark the pandemic period with a binary variable. - DONE
* Then, make sure the train/test split does not accidentally mix in “pandemic months”.
* **Time split**:

  * **Do NOT random split**. Use **time-based train-test split** (e.g., train 2012–2020, validate 2023, test 2024–2025). 


### Part 2 - Explorative data analysis in Streamlit UI

1. Set up a sidebar option to chose between uploading a CSV file or linking to Eurostat URL JSON link - DONE
   - estat_load_data.py - handles transformation of JSON object into DataFrame, automates feature addition from Part 1
   - 

2. Set the Main section tabs:  
   - 2.1 Correlation Heatmap
   - 2.2 Explorative Analysis
   - 2.3 Ausreißer-Erkennung
   - 2.4 ML Modell Trainineren
   - 2.5 Vorhersage & Visualisierung

   2.1.
   - User-controlled filters:
      * Country (Geopolitische Meldeeinheit)
      * Accommodation type (NACEr2 EUSTAT categories)
      * Season
   - Responsive correlation-heatmap
   - Export correlation table button

   2.2. 
   - Value distribution
   - Skewness over numeric features
   - Categorical features - category distibution, normalized

   2.3.
   - Outlier identification and exclusion methods
   - Visual display of normalized or raw values
   - Optional filter selector to exclude outlier values from the model training

   2.4.
   - ML Modell Training module - In progress
   - ML Performance indicators (R2, RMSE, MAE)
   - Actual vs. predicted
   - Residual vs. predicted
   - Feature importance ditribution
   - Save the trained model to Pickle file


   - Assessment of classification models (ml_05 ROC, AUC lesson) - to be implemented

---

## Which columns to use

### Numeric features (value distributions, skewness)

* `value`
* `pch_sm`, `pch_sm_19`, `pch_sm_12`
* `Monat`, `Jahr`
* `Month_cycl_sin`, `Month_cycl_cos`
* `MA3`, `MA6`, `MA12`
* `Lag_1`, `Lag_3`, `Lag_12`
* `pandemic_dummy` (binary, but technically numeric)

### Categorical features (normalized category distribution)

* `Zeitliche_Frequenz_Idx`, `Zeitliche_Frequenz`
* `Aufenthaltsland_Idx`, `Aufenthaltsland`
* `Maßeinheit_Idx`, `Maßeinheit`
* `NACEr2_Idx`, `NACEr2`
* `Geopolitische_Meldeeinheit_Idx`, `Geopolitische_Meldeeinheit`
* `Quartal`, `Saison`
* `Aufenthaltsland_Saison`, `NACEr2_Saison`, `Land_Monat`, `Land_Saison`

(`JahrMonat` is time-series, better for line plots, not histograms.)

---

### 🔑 Notes

* For **numeric columns**: use `px.histogram` (or KDE if needed).
* For **categorical columns**: `value_counts(normalize=True)` then plot with `px.bar`.
* Skewness gives a quick numeric indicator whether a variable is symmetric (`≈0`), right-skewed (`>0`), or left-skewed (`<0`).

---



### Part 3 - MAchine learning

1. Prepare target and Features for ML model tranining - IMPORTANT! - include the top 3 correlation features

2. Set up basic ML models

3. Train a model, save as pickle, use later with UI for prediction instead of training it every time (ml_01 lesson)

 


### Part 4 - UI Features

1. set up config.json and config.py for initial parameters

2. set up core functions

3. set up ML functions

4. set up logging

5. set up utils


