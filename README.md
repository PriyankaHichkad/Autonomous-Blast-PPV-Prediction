---
TITLE: Autonomous-Blast-PPV-Prediction
AUTHOR: Priyanka Rajeev Hichkad
---

> **Research Declaration**  
> This is a real-life, original research project — conducted at the Department of Mining Engineering, Indian Institute of Technology (BHU), Varanasi — 221 005, India.  
> All field data was personally collected from **Bhanegaon Opencast Coal Mine**, Wardha Valley Coalfield, Maharashtra, India.  
> The framework, models, synthetic data generation pipeline, and autonomous monitoring system were designed and implemented from scratch as part of a B.Tech final year project.

---

## Project Supervision

| Role | Name | Designation | Institution |
|---|---|---|---|
| **Principal Investigator** | Priyanka Rajeev Hichkad | B.Tech (Mining Engineering), Part III | IIT (BHU) Varanasi |
| **Supervisor** | Dr. Satyabrata Behera | Assistant Professor | Department of Mining Engineering, IIT (BHU) Varanasi |
| **Head of Department** | Prof. Rajesh Rai | Professor and Head | Department of Mining Engineering, IIT (BHU) Varanasi |

---

## Overview

Blasting is the primary method of rock fragmentation in surface coal mines. Every detonation generates seismic waves that propagate through the rock mass and are quantified as **Peak Particle Velocity (PPV)** in mm/s. Exceeding the safe PPV limit prescribed by **IS 6922 (1973)** poses risks of structural damage to nearby buildings and is a regulatory violation enforced by the **Directorate General of Mines Safety (DGMS), India**.

Existing predictive methods — principally the **USBM empirical scaled-distance law** (Duvall and Fogelson, 1962) — rely on only two variables and are calibrated once, never updated. As a mine excavates through different geological formations, these static models silently degrade in accuracy.

This project proposes and fully implements an **Autonomous Machine Learning Framework** that:

- Combines the **USBM physics law** with a **machine learning residual corrector** into a hybrid architecture
- Generates **physics-consistent synthetic data** simulating three geological zone transitions
- Trains and benchmarks **nine models** including classical ML, deep learning, and fuzzy logic
- Deploys an **autonomous monitoring system** using EWMA control charts that detects model drift and retrains without human intervention
- Serves predictions through a **Streamlit web application** with **Google Sheets cloud storage**

> **Live Application:** [autonomous-blast-ppv-prediction.streamlit.app](https://autonomous-blast-ppv-prediction-hmlswetfr6vryhnaq5zpeu.streamlit.app)

---

## Repository Structure

```
ppv-prediction/
│
├── part1_eda.py              # Data loading, feature engineering, fake data, EDA plots
├── part2_models.py           # Synthetic data generation and all model training
├── part3_ui.py               # Streamlit web application (4-tab UI + cloud storage)
├── model_utils.py            # Shared HybridPPVModel class (critical for pickle loading)
├── hybrid_gbr_model.py       # Alternative Hybrid Physics + Gradient Boosting model
│
├── data/
│   ├── real_blast_data.xlsx           # Raw field data from Bhanegaon mine
│   └── clean_field_data.xlsx          # Preprocessed data
│
├── results/
│   ├── real_field_data.csv            # 44 validated real blast records
│   ├── combined_with_synthetic.csv    # 644 records (real + synthetic)
│   ├── model_benchmark.csv            # All nine model metrics
│   ├── hybrid_gbr_summary.json        # Hybrid GBR model summary
│   └── live_blast_log.csv             # Runtime prediction log (auto-generated)
│
├── models/
│   └── hybrid_ppv_model.pkl           # Trained Hybrid RF model (auto-generated)
│
├── plots/
│   ├── Fig3_1_univariate_correlation.png
│   ├── Fig3_2_scatter_vs_ppv.png
│   ├── Fig3_3_loglog_vs_ppv.png
│   ├── Fig3_4_scaled_distance_vs_ppv.png
│   ├── Fig3_5_distance_frequency.png
│   ├── Fig3_6_correlation_heatmap.png
│   ├── Fig3_7_ppv_distribution.png
│   ├── Fig3_8_real_vs_fake.png
│   ├── Fig4_1_physics_model.png
│   ├── Fig4_2_synthetic_distribution.png
│   ├── Fig4_3_model_benchmark.png
│   ├── Fig4_4_actual_vs_predicted.png
│   ├── Fig4_5_error_distribution.png
│   ├── Fig4_6_prediction_timeline.png
│   ├── Fig4_7_feature_importance.png
│   └── Fig4_8_train_test_r2.png
│
├── credentials/
│   └── google_credentials.json        # Google Sheets service account (not committed)
│
├── requirements.txt
└── README.md
```

---

## Field Data

Data was collected personally from **Bhanegaon Opencast Coal Mine**, Wardha Valley Coalfield, Maharashtra, India over a one-month monitoring campaign (October–November 2025).

| Parameter | Description |
|---|---|
| **Mine** | Bhanegaon Opencast Coal Mine, Maharashtra |
| **Coalfield** | Wardha Valley Coalfield |
| **Seams monitored** | Seam Top-5, Seam-3, Seam 4 Floor, Seam 5 Floor, 40 Coal Top, Seam Top-3, Seam 3 Floor |
| **Blast events** | 10 unique blast events |
| **Vibrometers** | 5 per blast (IDs: 20697, 20698, 15335, 15336, 23314) |
| **Raw records** | 60 |
| **Valid records** | 44 (Distance + PPV both present) |
| **PPV range** | 1.354 – 17.340 mm/s |
| **PPV mean** | 3.948 mm/s (CV = 79.2%) |
| **IS 6922 exceedances** | 10 records (22.7%) exceeded 5 mm/s safe limit |

### Input Columns

| Column | Unit | Description |
|---|---|---|
| `Date` | dd.mm.yy | Date of blast event |
| `Blast_No` | — | Sequential blast reference |
| `No_of_Holes` | count | Total holes charged with explosive |
| `Explosion` | kg | Total explosive mass per event |
| `Per_Hole` | kg | Charge per hole — **this is Q in the USBM formula** |
| `Depth` | m | Drilled hole depth |
| `No_of_Rows` | count | Number of hole rows |
| `Spacing` | m | Distance between adjacent holes |
| `Seam_location` | — | Seam/overburden type |
| `Vibrometer` | ID | Monitoring instrument identifier |
| `Distance` | m | Blast-to-vibrometer distance (D) |
| `PPV` | mm/s | Peak Particle Velocity — **target variable** |
| `Frequency` | Hz | Dominant vibration frequency |

---

## Methodology

### Feature Engineering

All engineered features are physically derived from blast vibration theory:

| Feature | Formula | Physical Basis |
|---|---|---|
| `Q` | `= Per_Hole` | Max charge per delay (USBM) |
| `TQ` | `= No_of_Holes × Q` | Total charge (IS 6922) |
| `SD` | `= D / √Q` | USBM scaled distance |
| `SD_TQ` | `= D / √TQ` | IS 6922 scaled distance |
| `Stemming` | `= (2/3) × Depth` | Inert column length |
| `Ch_length` | `= (1/3) × Depth` | Explosive column length |
| `log(SD), log(D), log(Q), log(TQ), log(N), log(Depth)` | Natural logs | Linearise power-law relationships |

### Physics Model

The USBM empirical power law is used as the physics baseline, calibrated with literature values for Bhanegaon mine:

```
PPV_physics = k × SD^n = k × (D / √Q)^n
```

| Constant | Value | Range (Indian Coal Mines) | Source |
|---|---|---|---|
| `k` (site transmission constant) | **650** | 200 – 1100 | Pal Roy (1993) |
| `n` (attenuation exponent) | **−1.4** | −1.2 to −1.6 | Pal Roy (1993) |

> Both `k` and `n` are **always non-zero** by physical definition.

### Synthetic Data Generation

To overcome data scarcity (44 real records), 600 synthetic records were generated across **three geological zones** simulating geological transitions the mine encounters during excavation:

| Zone | k value | Interpretation | Records |
|---|---|---|---|
| **Zone A** | 650 (baseline) | Current site conditions — hard Gondwana sandstone | 200 |
| **Zone B** | 845 (×1.30) | Harder rock — more energy transmitted | 200 |
| **Zone C** | 468 (×0.72) | Softer/weathered overburden — higher attenuation | 200 |

```
PPV_synthetic = k_zone × SD^n × ε,    ε ~ Normal(1.0, 0.16)
```

Noise CV of 16% is calibrated from Hao and Wu (2005) field measurement uncertainty.  
**Combined training dataset: 44 real + 600 synthetic = 644 records.**

### Hybrid Model Architecture

The proposed model combines physics and ML:

```
PPV_final = PPV_physics + RF(Residual_features)
          = k × SD^n   + RF(X)
```

The **physics component** ensures physically consistent baseline predictions.  
The **Random Forest residual component** learns geological corrections not captured by the formula — primarily blast scale effects (No. of Holes) and seam-specific variability.

---

## Results

### Nine-Model Benchmark (20% Independent Test Set, n = 129 records)

| Model | Test R² | MAE (mm/s) | RMSE (mm/s) | MAPE (%) |
|---|---|---|---|---|
| M01 — Physics Only (USBM) | 0.6437 | 0.5494 | 1.2689 | 22.78 |
| M02 — Random Forest | 0.5081 | 0.6971 | 1.4909 | 29.01 |
| M03 — Gradient Boosting | 0.4809 | 0.7166 | 1.5315 | 28.63 |
| M04 — Extra Trees | 0.5011 | 0.7023 | 1.5015 | 29.49 |
| M05 — SVR (RBF) | 0.4665 | 0.7583 | 1.5527 | 32.32 |
| M06 — K-Nearest Neighbours | 0.4011 | 0.7802 | 1.6450 | 36.03 |
| M07 — Ridge Regression | 0.5613 | 0.6264 | 1.4079 | 26.55 |
| M08 — MLP Neural Network | 0.4475 | 0.8311 | 1.5801 | 37.87 |
| **M09 — Hybrid RF (Proposed) ★** | **0.6467** | **0.5722** | **1.2635** | **23.75** |

> ★ Proposed model achieves the **lowest RMSE** and **lowest MAPE** of all nine models.  
> Training R² = **0.923** — confirms the model has learned the underlying physics correctly.

### Key Findings

1. **Total Charge TQ** is a stronger univariate predictor of PPV (R² = 0.226) than Scaled Distance SD (R² ≈ 0.000) in multi-seam datasets. Blast management should monitor total explosive energy alongside per-hole scaled distance.

2. **No. of Holes** is the top feature in the RF residual model. The USBM formula ignores it entirely — a blast with 160 holes fires four times the total energy of one with 40 holes at the same scaled distance. The residual model corrects for this.

3. **MLP Neural Network** performed worst of all models — confirming that deep learning requires far larger datasets than the 515 training records available here. Ensemble methods are the correct choice for mine-scale blast vibration datasets.

---

## Autonomous Monitoring System

After every new blast record is submitted, the system:

1. Computes prediction error (MAPE)
2. Updates the **EWMA control chart**: `z_t = λ × MAPE_t + (1 − λ) × z_{t−1}` (λ = 0.20)
3. Flags drift when **4 consecutive predictions exceed 25% MAPE** or EWMA exceeds UCL
4. Automatically retrains on the **most recent 100 records** from a sliding window buffer
5. Saves and reloads the updated model — **zero human intervention required**

This is the first implementation of an autonomous retraining system for blast vibration prediction in an Indian coal mine context.

---

## Streamlit Application

The web application has four tabs:

| Tab | Function |
|---|---|
| **🎯 Predict PPV** | Enter blast design parameters → instant hybrid model prediction + IS 6922 safety zone gauge |
| **📋 Log New Blast** | Enter actual measured PPV → saved to local CSV + Google Sheets simultaneously |
| **📊 Live Monitor** | EWMA control chart, actual vs predicted trace, drift status |
| **🔬 Analytics** | All EDA and model figures, nine-model benchmark table, physics constant explanation |

---

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ppv-blast-prediction.git
cd ppv-blast-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Part 1 — EDA and Feature Engineering

```bash
python part1_eda.py
```

Outputs: `results/real_field_data.csv`, `results/combined_real_fake.csv`, all EDA figures in `plots/`

### 4. Run Part 2 — Model Training

```bash
python part2_models.py
```

Outputs: `models/hybrid_ppv_model.pkl`, `results/model_benchmark.csv`, model result figures in `plots/`

> **Important:** Always run `part2_models.py` to generate the model file before launching the app.  
> The model is saved using the `HybridPPVModel` class defined in `model_utils.py`.  
> Both `part2_models.py` and `part3_ui.py` import from `model_utils.py` — this prevents pickle serialisation errors.

### 5. Run Part 3 — Streamlit Application

```bash
streamlit run part3_ui.py
```

### 6. (Optional) Alternative Hybrid GBR Model

```bash
python hybrid_gbr_model.py
```

Trains a Hybrid Physics + Gradient Boosting model and produces a separate comparison.

---

## ☁️ Google Sheets Setup (Cloud Storage)

To enable automatic cloud storage of blast records:

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → Create a project
2. Enable **Google Sheets API** and **Google Drive API**
3. Create a **Service Account** → Download JSON credentials
4. Save the credentials file to `credentials/google_credentials.json`
5. Create a Google Sheet → Share it with the service account email (Editor access)
6. In `part3_ui.py`, set:
   ```python
   GSHEET_ID = 'your_actual_sheet_id_here'
   ```
7. Restart the app

Without credentials, the app runs in local CSV-only mode — all core functionality is retained.

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.4
matplotlib>=3.7
seaborn>=0.13
scipy>=1.11
streamlit>=1.32
gspread>=6.0
google-auth>=2.28
openpyxl>=3.1
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Known Issue and Fix — Pickle Model Loading

**Error:**
```
AttributeError: Can't get attribute 'HybridPPVModel'
on <module '__main__' from 'part3_ui.py'>
```

**Root Cause:** When `part2_models.py` runs as `__main__`, Python records the class as `__main__.HybridPPVModel` in the pickle file. When `part3_ui.py` loads it, `__main__` refers to `part3_ui.py` — where the class does not exist.

**Fix (already implemented):** `HybridPPVModel` is defined in `model_utils.py` — a neutral, always-importable module. Both `part2_models.py` and `part3_ui.py` import from there. Pickle records `model_utils.HybridPPVModel`, which resolves correctly from any script.

**If you encounter this error:** Delete `models/hybrid_ppv_model.pkl` and re-run `python part2_models.py`.

---

## References

| Type | Reference |
|---|---|
| Report | Duvall, W.I. and Fogelson, D.E. (1962). USBM RI 5968. U.S. Bureau of Mines. |
| Report | Pal Roy, P. (1993). Putting ground vibration predictors into practice. *Colliery Guardian*, 241(2), 63–67. |
| Standard | IS 6922 (1973). Bureau of Indian Standards, New Delhi. |
| Report | DGMS (2013). Technical Circular No. 7. Directorate General of Mines Safety. |
| Journal | Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. |
| Journal | Friedman, J.H. (2001). Gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. |
| Journal | Monjezi, M. et al. (2011). ANN prediction of PPV. *Tunnelling and Underground Space Technology*, 26(1), 46–52. |
| Journal | Zhang, H. et al. (2020). RF for blast vibration. *Applied Sciences*, 10(3). |
| Journal | Hao, H. and Wu, C. (2005). Blast-induced ground motion. *Soil Dynamics and Earthquake Engineering*, 25(1), 39–53. |
| Journal | Montgomery, D.C. (2012). *Statistical Quality Control: A Modern Introduction*. Wiley. |
| Journal | Brzezinski, D. and Stefanowski, J. (2014). Concept drift. *IEEE TNNLS*, 25(1), 81–94. |
| Journal | Page, E.S. (1954). Continuous inspection schemes. *Biometrika*, 41(1–2), 100–115. |
| Journal | Pedregosa, F. et al. (2011). Scikit-learn. *JMLR*, 12, 2825–2830. |

Full references are listed in the project report.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{hichkad2025ppv,
  author       = {Priyanka Rajeev Hichkad},
  title        = {An Autonomous Machine Learning Framework for Dynamic Prediction
                  of Blast-Induced Ground Vibration in Surface Mines},
  year         = {2025},
  institution  = {Department of Mining Engineering,
                  Indian Institute of Technology (BHU), Varanasi},
  supervisor   = {Dr. Satyabrata Behera},
  note         = {B.Tech Final Year Project (UG Project), May 2026.
                  Field data from Bhanegaon Opencast Coal Mine,
                  Wardha Valley Coalfield, Maharashtra, India.}
}
```

---

## Contact

**Priyanka Rajeev Hichkad**  
Roll No. 23155082 | B.Tech Part III (Mining Engineering)  
Indian Institute of Technology (BHU), Varanasi — 221 005, India

**Project Supervisor: Dr. Satyabrata Behera**  
Assistant Professor, Department of Mining Engineering  
Indian Institute of Technology (BHU), Varanasi

---

<div align="center">

**Department of Mining Engineering · Indian Institute of Technology (BHU) · Varanasi, India**  
*B.Tech Final Year Project · May 2026*

*This is an original research project. All field data, models, and code are the intellectual work of the author under the supervision of Dr. Satyabrata Behera.*

</div>
