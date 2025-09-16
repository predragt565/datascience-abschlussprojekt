## 🛏️ Eurostat Trourist Overnight Stays 2012-2025 (EU10) Trend Analysis

### Analytical findings - the Feature-Skewness

### 🔎 What the full skewness analysis shows

1. **Before transformation (raw data):**
   * Skewness is consistently high across many features.  
   * Several features are **extremely right-skewed**, with skewness values well above +10, and the most extreme (`pch_sm`) reaching over **+42**.  
   * In total, **10 out of 15 numeric features** exceed the |2| threshold, marking them as highly problematic for direct ML model training.  
   * This confirms that heavy-tailed distributions are a **systematic dataset issue**, not limited to a few columns.

2. **After log1p transformation:**
   * For strictly positive features, skewness was greatly reduced.  
   * Many previously extreme features (e.g., Lag features) flipped to moderate **left-skew** (around -1.7 to -2.6).  
   * Only **4 features** remain outside the acceptable range (|skew| ≥ 2) after log1p.  
   * For some features with negative values (e.g., percentage change variables), `log1p` could not be applied and those columns remain untreated in the current transformation. These features should instead be processed using **Yeo–Johnson** in a dedicated step to ensure consistent transformation across the dataset.

---

### ⚖️ Interpretation

* **Raw data**: Strongly unsuitable for linear or distance-based ML models due to severe skewness.  
* **Log1p transformed data**: Substantially improved. Most features are now closer to symmetric, but a subset remains either untreated (due to negatives) or still moderately skewed.  
* **Overcorrection pattern**: As seen in the lagged features, heavy right-skew often becomes moderate left-skew after log1p. This is acceptable and generally still improves model stability.

---

### ⚠️ Dataset quality considerations

* **Skewness issue is systemic**: Nearly all numeric features are affected.  
* **Transformation coverage**: Features with negatives (e.g., percentage changes) were excluded from log1p, and still show very high skewness. These will require an alternative transformation such as **Yeo–Johnson**.  
* **Residual skewness**: A handful of features remain outside the |2| range even after log1p. These may still benefit from Yeo–Johnson or Winsorizing.  
* **Outliers**: Still not directly assessed. Outlier analysis (IQR, Z-score, IsolationForest) will complement skewness correction.

---

### 📝 Recommendations

1. **Apply log1p transformation to all strictly positive features.**
   * Already shown to reduce skewness significantly in lag-based variables.
   * Retain these transformed versions for model training.

2. **Handle features with negatives separately.**
   * For percentage change variables and similar, apply **Yeo–Johnson** instead of log1p.
   * This will ensure transformation coverage across the full feature set.

3. **Re-check skewness after mixed transformation strategy.**
   * Ideal skewness is between -1 and +1.  
   * Pay particular attention to the 4 features still outside |2| after log1p.

4. **Perform outlier detection after skewness correction.**
   * Use IQR or Z-score for simpler features, or IsolationForest for multivariate detection.
   * Carefully validate whether extreme points are genuine events or noise.

5. **Scale or standardize features for ML models.**
   * Especially important for linear and distance-based algorithms.
   * Less critical for tree-based models, but variance stabilization is still beneficial.

---

### 📊 About data sufficiency

The extended analysis confirms that **skewness is a dataset-wide issue**.  
Screenshots captured the problem in a subset of features, but the full table demonstrates that virtually every numeric variable is affected.  
A consistent transformation pipeline is therefore necessary before proceeding with ML.

---

✅ **Summary finding**:  
The dataset exhibits **systemic right-skewness**, with extreme values in several features (up to skewness > 40).  
Applying `log1p` substantially reduces skewness for positive features, but leaves negative-valued features untreated.  
A mixed strategy (`log1p` for positives, `Yeo–Johnson` for negatives) is recommended, followed by outlier detection and scaling.  
This will yield a much more balanced and reliable dataset for downstream ML model training.
