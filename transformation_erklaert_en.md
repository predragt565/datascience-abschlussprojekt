
## Two Separate Goals

1. **Skewness correction (feature transformation)**  
   - Goal: make numeric distributions more symmetric, stabilize variance.  
   - Typical methods: `log1p`, Box-Cox, Yeo–Johnson.  
   - Effect: improves performance of models sensitive to distribution shape (linear regression, Lasso, Ridge, ElasticNet, etc.).

2. **Outlier detection (row filtering)**  
   - Goal: remove (or down-weight) entire rows that are extreme compared to the rest.  
   - Methods: IQR, Z-score, IsolationForest.  
   - Effect: prevents a few extreme observations from dominating training.

---

### Should both be used?  
Yes — but they are not interchangeable, and the order matters:

1. **Apply skewness transformation first (log1p or Yeo–Johnson).**  
   - This reduces heavy tails, makes outlier detection more reliable.  
   - Example: If Z-score is applied on raw skewed data, the mean/std dev are distorted by long tails → too many points falsely flagged as outliers.

2. **Then run outlier detection (IQR, Z-score, or IsolationForest).**  
   - Now the distributions are closer to normal → thresholds (like ±3 std dev) make sense.  
   - Rows can be removed or kept depending on domain knowledge.

---

### Which transformation should be used for ML pipeline?
- **Log1p**: Simple, works well for positive-only features (like counts, monetary values, durations).  
- **Yeo–Johnson**: More flexible (handles 0 and negatives). If the dataset sometimes has negative/zero values, Yeo–Johnson is safer.  

Since features often represent counts/aggregates (≥ 0), **log1p is sufficient and interpretable**.  
There is no need to combine log1p and Yeo–Johnson. Select one method.

---

### Recommended Workflow

1. Apply **log1p** (or Yeo–Johnson if only one transformation is required).  
2. Run **outlier detection** on transformed data (choose one method):  
   - Z-score → simple, works well after skew correction.  
   - IQR → robust, less sensitive to distribution assumptions.  
   - IsolationForest → suitable for complex/high-dimensional datasets, but less interpretable.  
3. Train ML model on the **cleaned + transformed dataset**.  

---

### Final Remarks
- Skewness transformation and outlier detection solve different problems.  
- Both should be used, in sequence:  
  - Transformation (log1p or Yeo–Johnson) → first.  
  - Outlier detection (Z-score / IQR / IF) → second.  
- For datasets with right-skewed positive counts, **log1p + Z-score** is often an effective and efficient choice.  
- Outlier removal should be guided by context: extreme peaks may be valid events and can hold predictive value.
