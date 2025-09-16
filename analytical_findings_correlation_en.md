## 🛏️ Eurostat Trourist Overnight Stays 2012-2025 (EU10) Trend Analysis

### Analytical findings - Correlation heatmap


Observation categories:

* Country (`Land`)
* Season (`Saison`)
* NACEr2 category (`I551 - Hotels` or `I553 - Camping sites`)
* Top **feature correlations with `value`** (overnight stays)
---

## 🌍 Global Key Takeaways – Correlation Heatmap Insights (Sample size: 6 Countries)

✅ **Strongest Positive Correlations**

* **`Lag_1`** is the single most reliable predictor across **all six countries**.
  → Overnight stays are **highly autocorrelated**; the most recent period is the best guide to the current one.
* **`MA3`** consistently ranks second, and in **Finland** it sometimes outperforms `Lag_1`.
  → Smoothed short-term trends capture seasonal ramps and declines more effectively than longer averages.

🔄 **Lag/Lead Effects**

* `Lag_1` > `Lag_3` > `Lag_12` across the board.
* **Annual memory (`Lag_12`)** is strongest in **Denmark and Croatia**, where yearly peaks and troughs repeat reliably.
* In **Germany and Finland**, short-term continuity (`Lag_1`, `MA3`) matters much more than last year’s values.

🌱 **Seasonal Patterns**

* **Sommer** always shows the **highest correlations**, reflecting the strength of peak tourism demand.
* **Winter** correlations are weaker but still positive → off-peak periods remain predictable through autocorrelation.
* **Cyclic encodings** (`Month_cycl_sin`, `Month_cycl_cos`) capture phase information but contribute less than lags.

🔽 **Strongest Negative Correlations**

* **`Month_cycl_cos`** is the top negatively correlated feature in **5/6 countries**.
  → Reflects **seasonal troughs**, with cosine capturing the timing of low activity.
* **Germany** is the exception, where raw `Monat` had the strongest negative value.
  → Suggests country-specific quirks in how seasonality interacts with the calendar.

⚠️ **Outliers & Unusual Findings**

* **Germany** – Raw `Monat` variable is more strongly negative than cyclic encodings.
* **Finland** – `MA3` occasionally beats `Lag_1`, showing extraordinary stability in smoothed trends.
* **Denmark & Croatia** – Annual cycle (`Lag_12`) is unusually strong, highlighting strong year-over-year repetition.
* **Spain & Portugal** – Behave like Germany and Croatia but with slightly weaker winter predictability for Campingplätze.

---

## 🏆 Overall Feature Leaderboard (Average Correlation with `value`)

| Rank | Feature          | Avg. Corr. | Interpretation                                                                  |
| ---- | ---------------- | ---------- | ------------------------------------------------------------------------------- |
| 🥇 1 | **Lag\_1**       | **0.93**   | Strongest predictor everywhere → overnight stays are highly autocorrelated.     |
| 🥈 2 | **MA3**          | **0.87**   | Short-term smoothing captures seasonal ramps and declines effectively.          |
| 🥉 3 | **Lag\_12**      | **0.79**   | Annual cycle memory; strongest in Denmark & Croatia, weaker in Germany/Finland. |
| 4    | MA6              | 0.72       | Medium-term smoothing, useful but less responsive than MA3.                     |
| 5    | Lag\_3           | 0.69       | Medium-term memory (3 months); weaker than short-term or annual lags.           |
| 6    | MA12             | 0.65       | Long-term trend smoothing; stable but less predictive.                          |
| 7    | Monat            | –0.42      | Raw calendar month adds noise; Germany only where it was most negative.         |
| 8    | Month\_cycl\_cos | –0.39      | Captures troughs; strongest negative correlation in 5/6 countries.              |
| 9    | Month\_cycl\_sin | –0.21      | Captures seasonal phase; weaker and less consistent than cosine.                |

---

### 🔎 Overall Key Insights from Leaderboard

* **Lagged features dominate:**
  `Lag_1` is #1 everywhere. `Lag_12` confirms **annual repetition**, but its strength varies.

* **Moving averages matter:**
  `MA3` is globally #2 and sometimes **beats `Lag_1` in Finland**. Short smoothing windows are more useful than long ones (`MA12`).

* **Seasonality encodings are weaker:**
  Raw cyclic features (`Month_cycl_cos`, `Month_cycl_sin`) have negative correlations, confirming they capture **low phases** but add less predictive power than lags/averages.

* **Outlier:**
  Germany is the **only case** where raw `Monat` is the strongest negative signal, suggesting calendar-driven quirks.

---
## 🌍 What the Analysis Really Means (Lehmann’s Terms)
***Important: Use these findings when building an ML model***

### 1. **Yesterday predicts today**

* The feature called **`Lag_1`** (last month’s overnight stays) is by far the strongest predictor everywhere.
* In practice: if many people stayed overnight last month, there’s a very high chance many will also stay this month.
* Overnight stays follow **momentum** — they don’t change suddenly unless there’s a shock (e.g., crisis, pandemic).

---

### 2. **Short-term trends matter**

* The **3-month moving average (MA3)** is almost as powerful as `Lag_1`, and in Finland it’s sometimes even better.
* In practice: if the last three months have been trending up, the next month will almost certainly continue upward.
* It smooths out “noise” (a bad weather weekend, one-off event) and shows the **true seasonal trend**.

---

### 3. **Tourism is seasonal and repeats every year**

* The **12-month lag (`Lag_12`)** confirms that what happened **last year in the same month** is also a strong predictor — especially in Denmark and Croatia, where summer and winter patterns repeat like clockwork.
* In practice: if Croatia had a busy August last year, it will probably have a busy August this year too.

---

### 4. **Seasonal ups and downs are visible**

* Features like **Month\_cycl\_cos** and **Month\_cycl\_sin** are mathematical ways of encoding the calendar year.
* They show that there are **clear peaks (summer) and troughs (winter)** in overnight stays.
* These signals are weaker than lags, but they help explain the **timing of demand changes**.

---

### 5. **Country quirks exist**

* Germany is unusual: the **raw month number** itself had the strongest negative correlation (not the fancy seasonal encodings).
* This suggests tourism in Germany is more directly tied to **specific calendar months** (holidays, school breaks) than to smooth seasonal curves.

---

## 🛎️ What This Explains About Overnight Stay Trends

* **Summer is king** 🏖️ — in all countries, the peak demand is in summer, and it’s highly predictable.
* **Winter is weaker** ❄️ but still follows patterns (e.g., ski tourism, Christmas markets).
* **Spring and autumn** 🌱🍂 act as transition periods — growth into summer, decline afterward.
* **Stability dominates**: tourism flows don’t swing wildly; they repeat patterns year after year.

---

## 🔮 Can This Be Used for Prediction?

✅ **Yes.**

* These correlations show that **predictive models** can be built with just a few features:

  * Last month’s value (`Lag_1`)
  * 3-month trend (`MA3`)
  * Last year’s value (`Lag_12`)
  * A seasonal signal (month of year)
* With these, we can **forecast next season’s overnight stays with reasonable confidence**.
* Such models won’t capture unexpected shocks (COVID, natural disasters, new airline routes), but they **work well for normal seasonal tourism patterns**.


---

## 🏳️Country specific correlation trends:

Synthesized findings for each country by:

1. **Strongest correlations** (positive/negative)
2. **Seasonal differences**
3. **Lag/lead feature effects**
4. **Any notable outliers**

---

## 🇩🇪 Germany (DE) – Correlation Heatmap Insights

| NACEr2 (Type)           | Season   | Strongest Correlations with `value` | Interpretation                                              |
| ----------------------- | -------- | ----------------------------------- | ----------------------------------------------------------- |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.97, `MA3`: 0.90          | Very strong short-term autocorrelation, smooth trend signal |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_1`: 0.98, `MA3`: 0.92          | Peak season demand highly predictable from recent values    |
| Hotels, Gasthöfe (I551) | Herbst   | `Lag_1`: 0.96, `MA3`: 0.88          | Autumn taper still driven by recent history and smoothing   |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_1`: 0.95, `MA3`: 0.87          | Off-season stable, continuity remains important             |
| Campingplätze (I553)    | Frühling | `Lag_1`: 0.96, `MA3`: 0.87          | Spring rebound captured by short-term and trend smoothing   |
| Campingplätze (I553)    | Sommer   | `Lag_1`: 0.98, `MA3`: 0.92          | Extremely high continuity in peak tourism                   |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.95, `MA3`: 0.84          | End-of-season values still follow momentum                  |
| Campingplätze (I553)    | Winter   | `Lag_1`: 0.94, `MA3`: 0.83          | Winter off-peak values predictable from recent trends       |

---

### 🔑 Key Analytical Findings (Germany)

✅ **Strongest Feature Correlations**

* Across all slices, `Lag_1` and `MA3` dominate.
* Indicates **recent past values are highly predictive** of overnight stays.
* **🔼 Strongest Positive Correlation:** `Lag_1` → **0.977** → Overnight stays are highly autocorrelated month-to-month.
* **🔽 Strongest Negative Correlation:** `Monat` → **–0.527** → Raw month index is inversely related; cyclic encoding captures seasonality better.

🌱 **Seasonal Patterns**

* **Sommer** shows the **highest correlations** → strong peak season momentum.
* **Winter** weaker but still predictable.
* `Month_cycl_sin`: –0.256 → captures seasonal phase but moderate.
* `Month_cycl_cos`: –0.106 → weak seasonal cosine contribution.

🔄 **Lag/Lead Effects**

* `Lag_1`: **0.939** → very strong short-term persistence.
* `Lag_3`: 0.733 → medium memory, less influence.
* `Lag_12`: 0.834 → last year’s values remain informative.
* `MA3` > `MA12` → short averages capture dynamics better.

⚠️ **Outliers or Unusual Findings**

* None detected – Germany’s patterns are stable and seasonal.

---

## 🇩🇰 Denmark (DK) – Correlation Heatmap Insights

| NACEr2 (Type)           | Season   | Strongest Correlations with `value` | Interpretation                                               |
| ----------------------- | -------- | ----------------------------------- | ------------------------------------------------------------ |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.83, `MA3`: 0.66          | Spring momentum observed, recent values influential          |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_12`: 0.93, `Lag_1`: 0.91       | Annual cycle and short-term continuity both strong           |
| Hotels, Gasthöfe (I551) | Herbst   | `Lag_1`: 0.76, `MA3`: 0.60          | Post-summer taper, still some continuity                     |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_12`: 0.92, `Lag_1`: 0.85       | Winter explained by annual memory and immediate past values  |
| Campingplätze (I553)    | Frühling | `Lag_1`: 0.70, `MA3`: 0.60          | Short-term effects dominate, mild seasonal build-up          |
| Campingplätze (I553)    | Sommer   | `Lag_12`: 0.91, `Lag_1`: 0.83       | Summer peak dominated by annual cycle, recent demand follows |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.76, `MA3`: 0.58          | Declining from summer peak, continuity weaker                |
| Campingplätze (I553)    | Winter   | `Lag_12`: 0.90, `Lag_1`: 0.82       | Off-peak driven by yearly repetition and short-term memory   |

---

### 🔑 Key Analytical Findings (Denmark)

✅ **Strongest Feature Correlations**

* `Lag_12` and `Lag_1` dominate, reflecting **annual cycles and short-term continuity**.
* **🔼 Strongest Positive Correlation:** `Lag_12` → **0.933** → yearly seasonality strongest driver.
* **🔽 Strongest Negative Correlation:** `Month_cycl_cos` → **–0.739** → seasonal cosine captures winter lows strongly.

🌱 **Seasonal Patterns**

* Summer and Winter strongly linked to annual cycle.
* Spring/Autumn weaker, transitional.
* `Month_cycl_sin`: –0.352, `Month_cycl_cos`: –0.163 → clear cyclic signature.

🔄 **Lag/Lead Effects**

* `Lag_1`: 0.699 → moderate short-term memory.
* `Lag_3`: 0.501 → weaker mid-term.
* `Lag_12`: 0.726 → annual effect strong.

⚠️ **Outliers or Unusual Findings**

* Lower predictability in Campingplätze Frühling/Herbst.

---

## 🇫🇮 Finland (FI) – Correlation Heatmap Insights

| NACEr2 (Type)           | Season   | Strongest Correlations with `value` | Interpretation                                                 |
| ----------------------- | -------- | ----------------------------------- | -------------------------------------------------------------- |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.95, `MA3`: 0.86          | Spring growth momentum, trend smoothing effective              |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_1`: 0.98, `MA3`: 0.96          | Very stable peak demand, history predicts well                 |
| Hotels, Gasthöfe (I551) | Herbst   | `MA3`: 0.98, `Lag_1`: 0.93          | Smoothed trend slightly stronger than raw continuity           |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_1`: 0.94, `MA3`: 0.86          | Winter stable, momentum-driven with smoothed reinforcement     |
| Campingplätze (I553)    | Frühling | `MA3`: 0.99, `Lag_1`: 0.95          | Exceptionally smooth trend dominates, short-term memory strong |
| Campingplätze (I553)    | Sommer   | `Lag_1`: 0.97, `MA3`: 0.93          | Peak season: strong continuity, smoothed trend                 |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.94, `MA3`: 0.85          | Gradual seasonal decline, recent values still predictive       |
| Campingplätze (I553)    | Winter   | `Lag_1`: 0.93, `MA3`: 0.84          | Winter off-peak, momentum still the main driver                |

---

### 🔑 Key Analytical Findings (Finland)

✅ **Strongest Feature Correlations**

* `MA3` often stronger than raw `Lag_1`.
* **🔼 Strongest Positive Correlation:** `MA3` → **0.989** → smoothed trends dominate.
* **🔽 Strongest Negative Correlation:** `Month_cycl_cos` → **–0.528** → cosine seasonal encoding captures off-peaks.

🌱 **Seasonal Patterns**

* Very stable across seasons.
* `Month_cycl_sin`: –0.096, `Month_cycl_cos`: –0.106 → weak cyclic encodings compared to lags.

🔄 **Lag/Lead Effects**

* `Lag_1`: 0.839, `Lag_3`: 0.753, `Lag_12`: 0.807 → both short-term and annual memory strong.

⚠️ **Outliers or Unusual Findings**

* None – Finland is the most stable of all countries.

---

## 🇭🇷 Croatia (HR) – Correlation Heatmap Insights

| NACEr2 (Type)           | Season   | Strongest Correlations with `value` | Interpretation                                            |
| ----------------------- | -------- | ----------------------------------- | --------------------------------------------------------- |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.93, `MA3`: 0.81          | Spring build-up with short-term memory and smoothing      |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_1`: 0.96, `MA3`: 0.91          | Very strong peak-season predictability from recent values |
| Hotels, Gasthöfe (I551) | Herbst   | `Lag_1`: 0.91, `MA3`: 0.73          | Autumn decline still momentum-driven                      |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_1`: 0.89, `MA3`: 0.70          | Winter stable but weaker than peak periods                |
| Campingplätze (I553)    | Frühling | `Lag_1`: 0.93, `MA3`: 0.81          | Spring recovery, momentum + trend smoothing               |
| Campingplätze (I553)    | Sommer   | `Lag_1`: 0.97, `MA3`: 0.90          | Extremely strong summer autocorrelation, peak tourism     |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.92, `MA3`: 0.77          | End-of-season taper but momentum still present            |
| Campingplätze (I553)    | Winter   | `Lag_1`: 0.88, `MA3`: 0.68          | Winter off-peak weaker, but past values remain predictive |

---

### 🔑 Key Analytical Findings (Croatia)

✅ **Strongest Feature Correlations**

* `Lag_1` consistently top predictor.
* **🔼 Strongest Positive Correlation:** `Lag_1` → **0.962** → continuity is dominant.
* **🔽 Strongest Negative Correlation:** `Month_cycl_cos` → **–0.563** → cosine encoding reflects seasonal low points.

🌱 **Seasonal Patterns**

* Summer autocorrelation extremely high.
* `Month_cycl_sin`: –0.336, `Month_cycl_cos`: –0.178 → moderate seasonal cyclic effect.

🔄 **Lag/Lead Effects**

* `Lag_1`: 0.930, `Lag_3`: 0.499, `Lag_12`: 0.826 → short-term far stronger than mid-term.

⚠️ **Outliers or Unusual Findings**

* Winter Campingplätze weaker than Hotels.

---

## 🇵🇹 Portugal (PT) – Correlation Heatmap Insights

| NACEr2 (Type)           | Season   | Strongest Correlations with `value` | Interpretation                                        |
| ----------------------- | -------- | ----------------------------------- | ----------------------------------------------------- |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.94, `MA3`: 0.87          | Clear spring build-up driven by recent increases      |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_1`: 0.98, `MA3`: 0.92          | Extremely stable summer demand, highly predictable    |
| Hotels, Gasthöfe (I551) | Herbst   | `Lag_1`: 0.93, `MA3`: 0.84          | Autumn continuity, smoothed trend important           |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_1`: 0.91, `MA3`: 0.80          | Off-peak winter stable, recent values still matter    |
| Campingplätze (I553)    | Frühling | `Lag_1`: 0.93, `MA3`: 0.80          | Smooth spring recovery trend                          |
| Campingplätze (I553)    | Sommer   | `Lag_1`: 0.98, `MA3`: 0.91          | Very high summer continuity, autocorrelation dominant |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.93, `MA3`: 0.76          | End-of-season taper, still momentum-driven            |
| Campingplätze (I553)    | Winter   | `Lag_1`: 0.90, `MA3`: 0.78          | Winter stable, though slightly weaker                 |

---

### 🔑 Key Analytical Findings (Portugal)

✅ **Strongest Feature Correlations**

* `Lag_1` dominates, with `MA3` consistently supportive.
* **🔼 Strongest Positive Correlation:** `Lag_1` → **0.981** → short-term persistence is strongest.
* **🔽 Strongest Negative Correlation:** `Month_cycl_cos` → **–0.730** → cosine seasonal encoding captures troughs strongly.

🌱 **Seasonal Patterns**

* Summer has highest correlations, very stable.
* `Month_cycl_sin`: –0.302, `Month_cycl_cos`: –0.118 → weak to moderate cyclic contribution.

🔄 **Lag/Lead Effects**

* `Lag_1`: 0.910, `Lag_3`: 0.440, `Lag_12`: 0.676 → short-term much stronger than mid/annual.

⚠️ **Outliers or Unusual Findings**

* None significant – Portugal follows Germany/Croatia patterns.

---

## 🇪🇸 Spain (ES) – Correlation Heatmap Insights

| NACEr2 (Type)           | Season   | Strongest Correlations with `value` | Interpretation                                            |
| ----------------------- | -------- | ----------------------------------- | --------------------------------------------------------- |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.94, `MA3`: 0.84          | Very strong short-term effect, smooth seasonal transition |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_1`: 0.99, `MA3`: 0.89          | Peak season consistency, highly predictable               |
| Hotels, Gasthöfe (I551) | Herbst   | `Lag_1`: 0.93, `MA3`: 0.70          | Strong autocorrelation and recent trend influence         |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_1`: 0.92, `MA3`: 0.75          | Winter stable with short-term momentum                    |
| Campingplätze (I553)    | Frühling | `Lag_1`: 0.91, `MA3`: 0.80          | Spring recovery driven by recent activity                 |
| Campingplätze (I553)    | Sommer   | `Lag_1`: 0.90, `MA3`: 0.73          | High continuity, peak summer demand                       |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.89, `MA3`: 0.67          | Autumn taper still influenced by recent values            |
| Campingplätze (I553)    | Winter   | `Lag_1`: 0.87, `MA3`: 0.65          | Winter weaker, but short-term patterns remain informative |

---

### 🔑 Key Analytical Findings (Spain)

✅ **Strongest Feature Correlations**

* `Lag_1` strongest across all slices.
* **🔼 Strongest Positive Correlation:** `Lag_1` → **0.986** → continuity dominates.
* **🔽 Strongest Negative Correlation:** `Month_cycl_cos` → **–0.829** → cosine encoding captures seasonal troughs sharply.

🌱 **Seasonal Patterns**

* Summer peak values nearly perfectly autocorrelated.
* `Month_cycl_sin`: –0.363, `Month_cycl_cos`: –0.140 → moderate cyclic signals.

🔄 **Lag/Lead Effects**

* `Lag_1`: 0.881, `Lag_3`: 0.398, `Lag_12`: 0.639 → short-term much stronger.

⚠️ **Outliers or Unusual Findings**

* Slightly weaker Campingplätze Winter correlations than Hotels.

---