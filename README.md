# Identify Customer Segments

> An unsupervised learning pipeline that clusters German demographic data to identify which population segments are most likely to respond to a mail-order sales campaign — built for Bertelsmann Arvato Analytics using real-world proprietary data.

**Verdict: target the established, financially secure, middle-aged homeowner segment.** Two clusters (1 and 4) are overrepresented in the customer base by a factor of **1.7–1.8×** versus the general population; two others (0 and 3) are underrepresented by roughly **0.4×** and profile as young, mobile, urban, low-income individuals with no apparent affinity for mail-order retail.

---

## Results

### Cluster distribution: customers vs. general population

| Cluster | Population % | Customers % | Ratio | Signal |
| :--- | ---: | ---: | ---: | :--- |
| **0** | **12.3%** | **5.1%** | **0.41** | ❌ Underrepresented — avoid |
| **1** | **8.7%** | **15.2%** | **1.75** | ✅ Overrepresented — core target |
| 2 | 9.4% | 11.3% | 1.20 | Neutral |
| **3** | **11.2%** | **4.8%** | **0.43** | ❌ Underrepresented — avoid |
| **4** | **10.8%** | **18.7%** | **1.73** | ✅ Overrepresented — core target |
| 5 | 9.1% | 9.5% | 1.04 | Neutral |
| 6 | 10.3% | 8.6% | 0.83 | Neutral |
| 7 | 8.9% | 10.4% | 1.17 | Neutral |
| 8 | 9.8% | 11.1% | 1.13 | Neutral |
| 9 | 9.5% | 5.3% | 0.56 | Weakly underrepresented |

### Principal component interpretations

| Component | Positive pole | Negative pole | Interpretation |
| :--- | :--- | :--- | :--- |
| PC1 | `HH_EINKOMMEN_SCORE`, `FINANZ_HAUSBAUER`, `PLZ8_ANTG1` | `FINANZ_MINIMALIST`, `MOBI_REGIO`, `ANZ_PERSONEN` | Financial security & home ownership ↔ low financial interest & mobility |
| PC2 | `MOVEMENT` (avantgarde), `GREEN_AVANTGARDE`, `SEMIO_KULT` | `DECADE` (1940s/50s), `SEMIO_TRADV`, `SEMIO_REL` | Avantgarde / younger generational cohort ↔ traditional / older cohort |
| PC3 | `EWDICHTE`, `INNENSTADT`, `BALLRAUM` | `KBA05_ANTG1`, `PLZ8_ANTG1`, `WOHNLAGE` | Dense urban / commercial proximity ↔ low-density rural residential |

The full weight tables for all 30 components are embedded in the notebook's Step 2.3 cells.

---

## Pipeline

```txt
AZDIAS (891,211 × 85)                     CUSTOMERS (191,652 × 85)
  │                                          │
  ├─ NaN re-encoding (feat_info + specials)  ├─ clean_data() ──────────────┐
  ├─ Drop 6 high-missingness cols (>20%)     │                             │
  ├─ Split rows on threshold (≤10 missing)   │  .transform() only — no re-fit
  │    ├─ retained:  735,714 rows            │                             │
  │    └─ outliers:  155,497 rows            │                             │
  ├─ Drop 12 multi-level categoricals        │                             │
  ├─ Re-encode OST_WEST_KZ (O→0, W→1)        │                             │
  ├─ Engineer DECADE + MOVEMENT              │                             │
  │    from PRAEGENDE_JUGENDJAHRE            │                             │
  ├─ Engineer WEALTH + LIFE_STAGE            │                             │
  │    from CAMEO_INTL_2015                  │                             │
  └─ Final shape: 735,714 × 67              ─┘                             │
       │                                                                   │
       ├─ SimpleImputer (median)  ───────────────────────────────────────  ┤
       ├─ StandardScaler ────────────────────────────────────────────────  ┤
       ├─ PCA (n_components=30, captures 85.3% of variance) ─────────────  ┤
       └─ KMeans (k=10, random_state=42) ────────────────────────────────  ┘
            └─ azdias_labels                    └─ customers_labels
```

| Choice | Setting |
| :--- | :--- |
| Missing-value threshold (columns) | >20% → drop (6 columns removed) |
| Missing-value threshold (rows) | >10 missing values per row (~17.4% of data set aside) |
| Categorical handling | Binary numeric: kept; `OST_WEST_KZ` (string): re-encoded; multi-level (12 cols): dropped |
| Mixed feature engineering | `PRAEGENDE_JUGENDJAHRE` → `DECADE` + `MOVEMENT`; `CAMEO_INTL_2015` → `WEALTH` + `LIFE_STAGE` |
| Final feature count | 67 |
| Imputation | `SimpleImputer(strategy='median')` |
| Scaling | `StandardScaler` (μ=0, σ=1) |
| Dimensionality reduction | PCA, 30 components — elbow in cumulative variance curve at ~85% |
| Clustering | KMeans, k=10 — elbow in within-cluster distance curve between k=8 and k=12 |

---

## Getting Started

**Requirements:** Python 3.8+, scikit-learn, pandas, numpy, matplotlib, seaborn, jupyter (developed on Python 3.10.13, scikit-learn 1.3).

```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
jupyter notebook Identify_Customer_Segments.ipynb
```

Run it headless (note: the AZDIAS subset is ~230 MB and must be present in the working directory — it is not auto-downloaded due to Arvato's data agreement):

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  Identify_Customer_Segments.ipynb
```

### Re-applying the fitted pipeline to new data

```python
import pickle, pandas as pd

# load fitted objects (serialized after notebook execution)
imputer  = pickle.load(open("imputer.pkl",  "rb"))
scaler   = pickle.load(open("scaler.pkl",   "rb"))
pca      = pickle.load(open("pca.pkl",      "rb"))
kmeans   = pickle.load(open("kmeans.pkl",   "rb"))

new_df_clean   = clean_data(pd.read_csv("new_data.csv", sep=";"))
new_imputed    = imputer.transform(new_df_clean)
new_scaled     = scaler.transform(new_imputed)
new_pca        = pca.transform(new_scaled)
new_labels     = kmeans.predict(new_pca)
```

---

## Project Structure

```txt
.
├── Identify_Customer_Segments.ipynb      # main deliverable — executed, 0 errors
├── Udacity_AZDIAS_Subset.csv             # general population demographics (not redistributed)
├── Udacity_CUSTOMERS_Subset.csv          # mail-order customer demographics (not redistributed)
├── AZDIAS_Feature_Summary.csv            # feature metadata (85 rows × 4 cols)
├── Data_Dictionary.md                    # feature encoding reference
├── docs/
│   ├── 1. Project Introduction.md        # course context
│   ├── 2. Project Overview.md            # business scenario and requirements
│   ├── 3. Project Details.md             # step-by-step task outline
│   ├── 4. Arvato Terms and Conditions.md # data usage agreement
│   └── 5. Rubric.md                      # grading criteria
└── LICENSE.md                            # MIT
```

> **Data notice:** `Udacity_AZDIAS_Subset.csv` and `Udacity_CUSTOMERS_Subset.csv` are proprietary to AZ Direct GmbH / Arvato Financial Solutions and are governed by a separate terms-of-use agreement. They are not included in this repository and must be deleted within two weeks of project completion per the Arvato data agreement.

---

## Recommendation (tl;dr)

Focus mailout spend on **clusters 1 and 4**: individuals who score high on financial security and home ownership (`FINANZ_HAUSBAUER`, `HH_EINKOMMEN_SCORE`), are from mainstream generational cohorts of the 1970s–80s (`DECADE`, `MOVEMENT`), and live in low-density residential areas (`KBA05_ANTG1`, `PLZ8_ANTG1`). These two clusters together account for **33.9% of the customer base while representing only 19.5% of the general population** — a combined lift of ~1.74×. Clusters 0 and 3, which profile as younger, mobile, urban, low-income individuals, should be explicitly excluded from campaign targeting. The full segment profiles, PCA weight tables, and cluster comparison plots are embedded in the notebook's Step 3 cells.

---

## License

MIT — see [LICENSE.md](LICENSE.md).
