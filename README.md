# Identify Customer Segments

> An unsupervised learning pipeline that clusters German demographic data to identify which population segments are most likely to respond to a mail-order sales campaign — built for Bertelsmann Arvato Analytics using real-world proprietary data.

**Verdict: target clusters 3 and 4.** Together they represent only **18.8% of the general population** yet account for **50.4% of all customers** — a combined lift of ~2.7×. The strongest non-targets (clusters 1, 5 and 8) collectively cover ~26.7% of the population but only ~3.1% of customers.

---

## Results

### Cluster distribution: customers vs. general population

| Cluster | Population % | Customers % | Ratio | Signal |
| :--- | ---: | ---: | ---: | :--- |
| **3** | **7.6%** | **24.8%** | **3.27** | ✅ Core target — green-avantgarde, high-income, settled men |
| **4** | **11.2%** | **25.6%** | **2.28** | ✅ Core target — young women, 1980s–90s cohort, dreamy/idealistic |
| 9 | 11.6% | 15.2% | 1.31 | Mildly overrepresented |
| 2 | 7.4% | 9.0% | 1.21 | Neutral |
| 7 | 12.1% | 8.3% | 0.69 | Neutral |
| 0 | 11.0% | 6.8% | 0.62 | Neutral |
| **6** | **12.4%** | **7.3%** | **0.59** | ❌ Weakly underrepresented |
| **8** | **8.2%** | **1.5%** | **0.19** | ❌ Strongly underrepresented — event-oriented young men |
| **1** | **8.2%** | **0.7%** | **0.09** | ❌ Strongly underrepresented — young women, no financial engagement |
| **5** | **10.3%** | **0.8%** | **0.08** | ❌ Strongly underrepresented — young men, no financial engagement |

### Principal component interpretations

| Component | Variance | Positive pole | Negative pole | Interpretation |
| :--- | ---: | :--- | :--- | :--- |
| PC1 | 17.2% | `MOBI_REGIO`, `KBA05_ANTG1`, `PLZ8_ANTG1`, `FINANZ_MINIMALIST` | `PLZ8_ANTG3/4`, `PLZ8_BAUMAX`, `WEALTH`, `ORTSGR_KLS9` | Settled low-rise suburban/financially active ↔ affluent dense-urban apartment dweller |
| PC2 | 13.1% | `ALTERSKATEGORIE_GROB`, `FINANZ_VORSORGER`, `SEMIO_ERL` | `DECADE`, `SEMIO_REL`, `SEMIO_TRADV`, `FINANZ_SPARER` | Traditional cautious older households ↔ secular non-saving modern youth |
| PC3 | 9.2% | `ANREDE_KZ` (female), `SEMIO_KAEM/DOM/KRIT` (low) | `SEMIO_VERT/SOZ/FAM/KULT` (high affinity), `FINANZ_MINIMALIST` | Calm detached female personality ↔ assertive pragmatic male personality |

The full weight tables for all 30 components are embedded in the notebook's Step 2.3 cells.

---

## Pipeline

```txt
AZDIAS (891,221 × 85)                     CUSTOMERS (191,652 × 85)
  │                                          │
  ├─ NaN re-encoding (feat_info + specials)  ├─ clean_data() ──────────────┐
  ├─ Drop 6 high-missingness cols (>20%)     │                             │
  │    TITEL_KZ 99.8%, AGER_TYP 77.0%,      │  .transform() only — no re-fit
  │    KK_KUNDENTYP 65.6%, KBA05_BAUMAX 53.5%│                            │
  │    GEBURTSJAHR 44.0%, ALTER_HH 34.8%     │                            │
  ├─ Split rows on missingness threshold     │                            │
  │    ├─ retained:  792,287 rows (88.9%)    │                            │
  │    └─ set aside:  98,934 rows (11.1%)    │                            │
  ├─ Drop 15 multi-level categoricals        │                            │
  ├─ Re-encode OST_WEST_KZ (O→0, W→1)       │                            │
  ├─ Engineer DECADE + MOVEMENT              │                            │
  │    from PRAEGENDE_JUGENDJAHRE            │                            │
  ├─ Engineer WEALTH + LIFE_STAGE            │                            │
  │    from CAMEO_INTL_2015                  │                            │
  └─ Final shape: 792,287 × 66             ─┘                            │
       │                                                                  │
       ├─ SimpleImputer (median)  ──────────────────────────────────────  ┤
       ├─ StandardScaler ─────────────────────────────────────────────── ┤
       ├─ PCA (n_components=30, 87.47% of variance) ──────────────────── ┤
       └─ KMeans (k=10, random_state=42, n_init=10) ────────────────────  ┘
            └─ azdias_labels                    └─ customers_labels
```

| Choice | Setting |
| :--- | :--- |
| Missing-value threshold (columns) | >20% → drop (6 columns removed; next highest was `KKK` at 17.7%) |
| Missing-value threshold (rows) | Rows above threshold set aside as outlier subset (11.1% of data) |
| Categorical handling | Binary numeric: kept; `OST_WEST_KZ` (O/W): re-encoded 0/1; multi-level (15 cols): dropped |
| Mixed feature engineering | `PRAEGENDE_JUGENDJAHRE` → `DECADE` + `MOVEMENT`; `CAMEO_INTL_2015` → `WEALTH` + `LIFE_STAGE` |
| Final feature count | 66 |
| Imputation | `SimpleImputer(strategy='median')` — robust to the skewed count distributions in this dataset |
| Scaling | `StandardScaler` (μ=0, σ=1) |
| PCA | 30 components — 80% variance at 22, **87.5% at 30**, 90% at 34 |
| Clustering | KMeans, k=10 — elbow in within-cluster distance curve between k=8 (36.5) and k=10 (34.9) |

---

## Getting Started

**Requirements:** Python 3.8+, scikit-learn, pandas, numpy, matplotlib, seaborn, jupyter (developed on Python 3.14.7, scikit-learn 1.6).

```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
jupyter notebook Identify_Customer_Segments.ipynb
```

Run it headless (note: the AZDIAS and CUSTOMERS subsets must be in the working directory — they are not auto-downloaded due to Arvato's data agreement):

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  Identify_Customer_Segments.ipynb
```

### Re-applying the fitted pipeline to new data

```python
import pickle, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# load fitted objects (serialized after notebook execution)
imputer = pickle.load(open("imputer.pkl",  "rb"))
scaler  = pickle.load(open("scaler.pkl",   "rb"))
pca     = pickle.load(open("pca.pkl",      "rb"))
kmeans  = pickle.load(open("kmeans.pkl",   "rb"))

# pipeline: clean → align columns → .transform() only at every step
new_clean     = clean_data(pd.read_csv("new_data.csv", sep=";"))
new_clean     = new_clean[azdias_final.columns]  # align to training columns
new_imputed   = imputer.transform(new_clean)
new_scaled    = scaler.transform(new_imputed)
new_pca       = pca.transform(new_scaled)
new_labels    = kmeans.predict(new_pca)
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

> **Data notice:** `Udacity_AZDIAS_Subset.csv` and `Udacity_CUSTOMERS_Subset.csv` are proprietary to AZ Direct GmbH / Arvato Financial Solutions and governed by a separate terms-of-use agreement. They are not included in this repository and must be deleted within two weeks of project completion per the Arvato data agreement.

---

## Recommendation (tl;dr)

Focus mailout spend on **clusters 3 and 4**. Cluster 3 (ratio 3.27×) profiles as affluent, settled, mostly male, green-avantgarde customers who are actively financially engaged — high household income (`HH_EINKOMMEN_SCORE` ≈ 2.5, where 1 = highest), strong investor and home-builder affinity (`FINANZ_ANLEGER` ≈ 1.5, `FINANZ_HAUSBAUER` ≈ 1.9), low mobility, and near-uniform avantgarde movement membership (`GREEN_AVANTGARDE` ≈ 0.97). Cluster 4 (ratio 2.28×) profiles as young women from the 1980s–90s cohorts (`DECADE` ≈ 85.5, `ALTERSKATEGORIE_GROB` ≈ 1.8) who are dreamy and idealistic rather than rational, dutiful or traditional (low `SEMIO_TRADV/PFLICHT/RAT` affinity, high `SEMIO_VERT`). Mail spend should be suppressed on clusters 5, 1 and 8 (combined ratio < 0.2, together ~26.7% of the population but ~3.1% of customers): financially disengaged young men and women with experience-oriented rather than household-shopping lifestyles are very unlikely to respond to a catalog campaign. The full segment centroid profiles, PC weight tables, and cluster proportion plots are embedded in the notebook's Step 3 cells.

---

## License

MIT — see [LICENSE.md](LICENSE.md).
