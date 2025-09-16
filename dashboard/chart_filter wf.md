```
              ┌───────────────┐
              │   DataFrame   │
              └───────┬───────┘
                      │
       choice21 → pick feature group (col name, e.g. "NACEr2")
                      │
                      ▼
       choice22 → pick value from df[choice21]
                      │
        map via f"{choice21}_Idx" → choice22_idx_val
                      │
                      ▼
     Filter df where df[choice22_idx_col] == choice22_idx_val
                      │
                      ▼
       choice23 → pick country or "Alle Länder"
          │
     ┌────┴─────────────┐
     │                  │
"Alle Länder"       Specific country
(no filter)        map to Geopolitische_Meldeeinheit_Idx
     │                  │
     └───────┬──────────┘
             │
             ▼
       Filtered DataFrame
             │
             ▼
     Aggregation & Rolling Averages
```