# Identify Customer Segments

> An unsupervised learning pipeline that clusters German demographic data to identify which population segments over-index in a mail-order customer base — built for Bertelsmann Arvato Analytics using real-world proprietary data. The segments below are audience *hypotheses* to validate against campaign response data.

**Verdict: cluster 3 is the promising audience hypothesis.** It represents only **6.8% of the general population** yet accounts for **24.8% of all customers** — a 3.67× lift. The under-represented clusters 5, 1, 4 and 8 (ratios 0.09–0.19) are lower-priority hypotheses: together ~33.7% of the population but only ~4.1% of customers. A fourth group also sits above the line — the set-aside high-missingness stratum (label −1: 11.1% of population vs. 26.5% of customers, ratio 2.39) — but it is a data-completeness finding, not a demographic segment. All shares are computed over the full datasets (891,221 and 191,652 rows) and each proportion column sums to 1.0.

---

## Results

### Cluster distribution: customers vs. general population

| Cluster | Population % | Customers % | Ratio | Signal |
| :--- | ---: | ---: | ---: | :--- |
| **3** | **6.8%** | **24.8%** | **3.67** | ✅ Core audience hypothesis — green-avantgarde, high-income, settled men |
| **−1 (set-aside)** | **11.1%** | **26.5%** | **2.39** | ⚠ High-missingness stratum — data-completeness finding, no centroid |
| 9 | 10.3% | 14.3% | 1.39 | Mildly overrepresented |
| 2 | 6.6% | 9.0% | 1.36 | Neutral |
| 7 | 10.7% | 8.1% | 0.75 | Neutral (mildly underrepresented) |
| 0 | 9.8% | 6.3% | 0.65 | Neutral (mildly underrepresented) |
| 6 | 11.1% | 7.1% | 0.64 | Neutral (mildly underrepresented) |
| **8** | **7.3%** | **1.4%** | **0.19** | ❌ Lower-priority hypothesis — event-oriented young men |
| **4** | **10.0%** | **1.2%** | **0.12** | ❌ Lower-priority hypothesis — young women, 1980s–90s, dreamy/non-investing |
| **1** | **7.3%** | **0.7%** | **0.10** | ❌ Lower-priority hypothesis — event-oriented young men |
| **5** | **9.2%** | **0.8%** | **0.09** | ❌ Lower-priority hypothesis — young women, no financial engagement |

### Principal component interpretations

| Component | Variance | Positive pole | Negative pole | Interpretation |
| :--- | ---: | :--- | :--- | :--- |
| PC1 | 17.2% | `MOBI_REGIO`, `KBA05_ANTG1`, `PLZ8_ANTG1`, `KBA05_GBZ`, `FINANZ_MINIMALIST` | `PLZ8_ANTG3/4`, `PLZ8_BAUMAX`, `WEALTH` (high = *poor*), `ORTSGR_KLS9` | Settled, financially active low-rise suburban/small-town ↔ lower-income dense-urban apartment dwellers in large cities |
| PC2 | 13.1% | `ALTERSKATEGORIE_GROB`, `FINANZ_VORSORGER` (high = *weak* preparedness affinity), `SEMIO_ERL`, `SEMIO_LUST`, `RETOURTYP_BK_S` | `DECADE`, `SEMIO_REL`, `SEMIO_TRADV`, `SEMIO_PFLICHT`, `FINANZ_SPARER` | Older conventional cohort (age contrast — **not** higher financial preparedness) ↔ secular, non-traditional, non-saving 1990s youth |
| PC3 | 9.2% | `ANREDE_KZ` (female); high dreamful/social/family/cultural affinity; low combative/dominant/critical/rational affinity | Male; low dreamful/social/family/cultural affinity; high combative/dominant/critical/rational affinity | Socially warm, culturally engaged ↔ assertive-pragmatic personality contrast |

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
  ├─ Row quality over QUALITY_COLUMNS        ├─ Row quality: same 79 cols, │
  │    (79 cols, pre-engineering)            │    same threshold, mask     │
  │    same 15-missing threshold             │    aligned to original index│
  │    ├─ retained:  792,287 rows (88.9%)    │    ├─ 140,898 → pipeline    │
  │    └─ set aside:  98,934 rows (11.1%)    │    └─ 50,754 set aside(26.5%)│
  ├─ Drop 13 multi-level categoricals        │                            │
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
                 (+98,934 set-aside)               (+50,754 set-aside)
                        └── both rejoin step 3.3 as segment −1 (each column sums to 1.0)
```

| Choice | Setting |
| :--- | :--- |
| Missing-value threshold (columns) | >20% → drop (6 columns removed; next highest was `KKK` at 17.7%) |
| Missing-value threshold (rows) | 15 missing over the explicit 79-column `QUALITY_COLUMNS` list (post-1.1.2 drop, pre-engineering) — applied to **both** datasets: 11.1% of population rows and 26.5% of customer rows set aside as segment −1 |
| Categorical handling | Binary numeric: kept; `OST_WEST_KZ` (O/W): re-encoded 0/1; multi-level (13 cols): dropped |
| Mixed feature engineering | `PRAEGENDE_JUGENDJAHRE` → `DECADE` + `MOVEMENT`; `CAMEO_INTL_2015` → `WEALTH` + `LIFE_STAGE` |
| Final feature count | 66 |
| Imputation | `SimpleImputer(strategy='median')` — robust to the skewed count distributions in this dataset |
| Scaling | `StandardScaler` (μ=0, σ=1) |
| PCA | 30 components — 80% variance at 22, **87.5% at 30**, 90% at 34 |
| Clustering | KMeans, k=10 — elbow in within-cluster distance curve between k=8 (36.5) and k=10 (34.9) |

---

## Getting Started

**Requirements:** Python 3.10+, scikit-learn, pandas, numpy, matplotlib, seaborn, jupyter (developed on Python 3.14.7, scikit-learn 1.9.1, pandas 3.0.6).

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

Treat **cluster 3 (ratio 3.67×)** as the promising audience hypothesis for the next mailout: affluent, settled, mostly male, green-avantgarde customers who are actively financially engaged — high household income (`HH_EINKOMMEN_SCORE` ≈ 2.5, where 1 = highest), strong investor and home-builder affinity (`FINANZ_ANLEGER` ≈ 1.5, `FINANZ_HAUSBAUER` ≈ 1.9), low mobility, and near-uniform avantgarde movement membership (`GREEN_AVANTGARDE` ≈ 0.97). Clusters 5 (0.09×), 1 (0.10×), 4 (0.12×) and 8 (0.19×) are lower-priority hypotheses — together ~33.7% of the population but only ~4.1% of existing customers — and their demotion reflects who has bought, not a prediction of who will respond, so it should be confirmed against campaign response data. The set-aside −1 stratum (2.39×, 26.5% of customer records) is a data-completeness question rather than a target: compare `customers_high` against `azdias_outliers` before deciding it is a coherent niche. The PC interpretations above apply sign-plus-coding logic (e.g. `FINANZ_VORSORGER`'s positive loading means *weaker* preparedness affinity on the inverted scale). The full segment centroid profiles, PC weight tables, and cluster proportion plots are embedded in the notebook's Step 3 cells.

---

## License

MIT — see [LICENSE.md](LICENSE.md).
