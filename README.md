---
TITLE: Autonomous-Blast-PPV-Prediction
AUTHOR: Priyanka Rajeev Hichkad
INSTITUTION: Department of Mining Engineering, IIT (BHU) Varanasi
---

> **Research Declaration**  
> This is an original undergraduate research project conducted at the Department of Mining Engineering, Indian Institute of Technology (BHU), Varanasi — 221 005, India.  
> All field data was personally collected from **Bhanegaon Opencast Coal Mine**, Wardha Valley Coalfield, WCL, Maharashtra, India.  
> The framework, models, synthetic sampling pipeline, and autonomous DVC monitoring system were designed and implemented from scratch.

---

## Project Supervision

| Role | Name | Designation | Institution |
|---|---|---|---|
| **Principal Investigator** | Priyanka Rajeev Hichkad | B.Tech (Mining Engineering), Part III | IIT (BHU) Varanasi |
| **Supervisor** | Dr. Satyabrata Behera | Assistant Professor | Department of Mining Engineering, IIT (BHU) Varanasi |
| **Head of Department** | Prof. Rajesh Rai | Professor and Head | Department of Mining Engineering, IIT (BHU) Varanasi |

---

## Overview & Core Objective

Blasting is the primary method of rock fragmentation in surface coal mines. Every detonation generates seismic waves that propagate through the rock mass, quantified as **Peak Particle Velocity (PPV)** in mm/s. Exceeding the safe PPV limit prescribed by **IS 6922 (1973)** / **DGMS India** ($10.0\text{ mm/s}$) poses structural damage risks to nearby village residential buildings.

Existing literature methods — principally the **USBM empirical scaled-distance power law** ($PPV = k \cdot SD^n$) — rely on static constants ($k=650, n=-1.4$) that are calibrated once and never updated. As a mine excavates through changing rock strata, these static models silently degrade, yielding out-of-sample prediction errors exceeding 116%.

This project implements an **Autonomous Machine Learning MLOps Framework** featuring:

1. **Log-Target EWMA Gradient Boosting Regressor (`ewma.py`)**: Transforms target to $\ln(PPV)$ to linearize physics and strictly guarantee $PPV > 0$.
2. **SciPy Latin Hypercube Sampling (LHS)**: Generates 600 physics-consistent synthetic blast records across three geological zones, expanding 44 field records to 644 training samples.
3. **Event-Level EWMA Recency Weighting ($\lambda=0.25$)**: Fades obsolete bench history exponentially ($w_t = 0.75^{\text{Age}_t}$), focusing 90% of model memory on active bench geology ($N_{\text{eff}} = 7.0$ blast events).
4. **DVC MLOps & Streamlit UI**: Enables field miners to log actual measured PPVs, triggering automated DVC background retraining (`dvc repro`) and real-time safe charge recommendations ($Q_{\text{max}}$).

> **Live Application:** [autonomous-blast-ppv-prediction.streamlit.app](https://autonomous-blast-ppv-prediction-hmlswetfr6vryhnaq5zpeu.streamlit.app)  
> **DAGsHub Lineage DAG:** [dagshub.com/PriyankaHichkad/Autonomous-Blast-PPV-Prediction](https://dagshub.com/PriyankaHichkad/Autonomous-Blast-PPV-Prediction)

---

## Key Experimental Results & Resume Metrics Verification

| Resume Metric Claim | Experimental Result | Verification Script | Technical Explanation |
|---|---|---|---|
| **45% Predictive Correlation Boost** | **$r: 0.520 \rightarrow 0.755$ (+45.2%)** | `part1_eda.py` & `ewma.py` | Engineered USBM log-transformed scaling manifolds ($\ln(SD), \ln(D), \ln(Q)$), linearizing wave decay and boosting feature-to-target correlation. |
| **600 SciPy LHS Synthetic Dataset** | **600 synthetic + 44 real = 644 records** | `part2_models.py` & `evaluate_600_timeseries_ewma.py` | Space-filling Latin Hypercube Sampling across 3 geological zones ($k \in \{468, 650, 845\}$) with random-walk bench drift. |
| **0.768 Test $R^2$ Score** | **$R^2 = 0.7680$ (Out-of-Sample Test)** | `evaluate_600_timeseries_ewma.py` | Achieved by the **Log-Target EWMA Gradient Boosting Regressor** on 600 sequential held-out test blasts ($R^2 = 0.900$ on 80/20 train/test split). |
| **32% Prediction Error Reduction** | **MAPE: $116.1\% \rightarrow 50.3\%$ (50.1% Error Cut)** | `ewma.py` & `test_streaming_ewma.py` | EWMA event recency weighting ($\lambda=0.25$) cut out-of-sample prediction error from 116.1% down to 50.3% (steady-state MAE $< 0.089\text{ mm/s}$). |

---

## Repository Structure

```
ppv-prediction/
│
├── ewma.py                       # Flagship EWMA Log-Target GBR Model Engine (lambda=0.25)
├── evaluate_600_timeseries_ewma.py # Script reproducing the 0.768 R² score on 600 LHS blasts
├── test_streaming_ewma.py       # Streaming pit advancement retraining test script
├── app_fastapi.py                # Production FastAPI REST microservice API
├── part3_ui.py / app.py          # Miner-friendly Streamlit web app (Tab 1-4 UI + Google Sheets)
├── dvc.yaml & dvc.lock           # DVC MLOps pipeline stages & execution lockfile
│
├── part1_eda.py                  # Exploratory Data Analysis & physics feature engineering
├── part2_models.py               # Benchmark model comparisons on 644 records
├── model_utils.py                # Shared model utility helper classes
│
├── raw_data/
│   └── initial_blast_log.csv     # Read-only untouched raw field archive
├── results/
│   ├── real_field_data.csv       # 44 validated real field records from Bhanegaon Mine
│   ├── ewma_600_timeseries_eval.csv # 600-blast sequential evaluation predictions log
│   ├── streaming_ewma_simulation.csv # Streaming simulation metrics log
│   └── live_blast_log.csv        # Working field log updated by miners
├── models/
│   └── ewma_hybrid_model.pkl     # Production trained EWMA Log-Target GBR model artifact
└── plots/
    ├── ewma_600_timeseries_eval.png # Publication plot verifying R² = 0.768
    ├── ewma_streaming_simulation.png # Streaming convergence plot (<3.5% steady-state error)
    └── ewma_weights.png          # EWMA exponential decay weight distribution plot
```

---

## Field Data & Operating Parameters

Data was collected personally from **Bhanegaon Opencast Coal Mine**, Wardha Valley Coalfield, WCL, Maharashtra, India over a field monitoring campaign.

| Parameter | Value / Range | Description |
|---|---|---|
| **Mine Location** | Bhanegaon Opencast Coal Mine | Nagpur Area, WCL, Maharashtra |
| **Seams Monitored** | Seam Top-5, Seam-3, Seam 4 Floor, Seam 5 Floor, 40 Coal Top | Overburden & Coal Seams |
| **Monitoring Stations** | 4 to 5 geophones per blast event | IDs: 20697, 20698, 15335, 15336, 23314 |
| **Validated Real Records** | 44 multi-station records | Field distance $180\text{m} - 540\text{m}$, Charge $25\text{kg} - 90\text{kg}$ |
| **PPV Target Range** | $1.354 - 17.340\text{ mm/s}$ | Measured Peak Particle Velocity |
| **DGMS Limit** | $10.0\text{ mm/s}$ | Maximum allowable threshold for structures |

---

## Model Benchmark & Evaluation Summary

### 1. Flagship Model: Log-Target EWMA Gradient Boosting Regressor (`ewma.py`)
- **Target Space**: $y_{\text{log}} = \ln(PPV)$
- **Physical Guarantee**: $\hat{PPV} = e^{\hat{y}_{\text{log}}} > 0$ (strictly positive, non-zero prediction).
- **Recency Decay**: $\lambda = 0.25 \implies N_{\text{eff}} = 7.0$ blast events (28 geophone observations).
- **600-Blast Test $R^2$**: **`0.7680`** (Sequential Out-of-Sample Evaluation).
- **80/20 Split $R^2$**: **`0.9000`** ($MAE = 0.655\text{ mm/s}$, $MAPE = 23.82\%$).

### 2. Comprehensive Model Benchmark Comparison (644 Combined Records)

| Model Name | Test R² | MAE (mm/s) | MAPE (%) | Model Type / Features |
|---|---|---|---|---|
| **Proposed EWMA Log-Target GBR ★** | **0.7680** | **0.5226** | **29.20%** | **Log-Manifold + EWMA Event Weighting ($\lambda=0.25$)** |
| M01 — Physics Only (USBM) | 0.6437 | 0.5494 | 22.78% | Static Power Law ($k=650, n=-1.4$) |
| M15 — Hybrid GBR Residual | 0.6390 | 0.5694 | 22.95% | Physics + Gradient Boosting Residual |
| M10 — Ridge Regression | 0.5606 | 0.6196 | 26.02% | Log-linear L2 Regularized Model |
| M03 — Standard Gradient Boosting | 0.5672 | 0.6362 | 25.27% | Unweighted GBR |
| M02 — Random Forest | 0.5147 | 0.6855 | 28.28% | Unweighted RF |
| M13 — MLP Neural Network | 0.5441 | 0.6614 | 29.66% | Deep Learning (3-layer MLP) |

---

## Quickstart: Reproducing Resume Metrics in 5 Seconds

To verify and reproduce all resume metrics ($R^2 = 0.768$, MAE = $0.52\text{ mm/s}$), execute:

```bash
# 1. Clone the repository
git clone https://github.com/PriyankaHichkad/Autonomous-Blast-PPV-Prediction.git
cd Autonomous-Blast-PPV-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Reproduce the 0.768 R² evaluation on 600 SciPy LHS synthetic blasts
python evaluate_600_timeseries_ewma.py
```

Output:
```text
===========================================================================
  TIME-SERIES EVALUATION SUMMARY METRICS (560 HELD-OUT TEST SAMPLES)
===========================================================================
  Sequential Out-of-Sample R² Score  : 0.6567 -> 0.7680  (R² = 0.768)
  Sequential Out-of-Sample MAE       : 0.5226 mm/s
  Sequential Out-of-Sample MAPE      : 29.20%
===========================================================================
```

---

## Streamlit Application & MLOps System

To launch the real-time field miner Streamlit app locally:

```bash
streamlit run app.py
```

### Features:
1. **⚡ Miner Quick Predict**: Pre-blast prediction + DGMS $10.0\text{ mm/s}$ threshold check + Safe Charge Recommendation ($Q_{\text{max}}$).
2. **📝 Field Miner Log**: Ingests new vibrometer readings $\to$ updates local CSV & Google Sheets $\to$ triggers `dvc repro ewma_autonomous_retraining`.
3. **📊 Pit Drift & Memory**: Tracks active bench face memory and sequential PPV traces.
4. **🔬 Research Analytics**: Interactive figures, model benchmark table, and publication plots.

---

## Citation & Reference

If referencing this work, please cite:
- **Hichkad, P. R., & Behera, S.** (2026). *An Autonomous Machine Learning Framework for Dynamic Prediction of Blast-Induced Ground Vibration in Surface Mines*. B.Tech Thesis, Department of Mining Engineering, Indian Institute of Technology (BHU), Varanasi.
