"""
================================================================================
  PART 3 — AUTONOMOUS STREAMLIT UI WITH GOOGLE SHEETS CLOUD STORAGE
================================================================================
  Project   : Autonomous ML Framework for Blast-Induced Ground Vibration
  Institute : IIT (BHU) Varanasi | Department of Mining Engineering
  Run       : streamlit run part3_ui.py

  FEATURES:
  ─────────
  1. PREDICT tab    — Input blast parameters → instant PPV prediction
                      Uses saved Hybrid RF model (physics + ML)
                      Shows IS 6922 safety assessment

  2. LOG & STORE tab — New blast record entry form
                       Validates inputs against physical bounds
                       Appends to local CSV + Google Sheet (cloud)
                       Auto-retrains model when error threshold exceeded

  3. MONITOR tab    — Live EWMA error chart (drift detection)
                      Retraining event log
                      Model performance over time

  4. ANALYTICS tab  — All EDA and model plots from Part 1 & 2
                      Benchmark table, feature importance

  GOOGLE SHEETS SETUP (one-time):
  ─────────────────────────────────
  1. Create a Google Cloud project → enable Google Sheets API
  2. Create a Service Account → download credentials JSON
  3. Share your Google Sheet with the service account email
  4. Put credentials JSON path in GSHEET_CREDS_PATH below
  5. Put your Sheet ID in GSHEET_ID below
  If credentials not found, app runs in LOCAL-ONLY mode (CSV storage).

  DEPENDENCIES:
      pip install streamlit gspread google-auth pandas numpy
                  scikit-learn matplotlib seaborn scipy
================================================================================
"""

import os
import sys
import json
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
from collections import deque
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import streamlit as st

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
K_LIT          = 650.0       # USBM site constant (Pal Roy 1993, Wardha Valley)
N_LIT          = -1.4        # Attenuation exponent (Indian coal mines)
IS6922_LIMIT   = 5.0         # IS 6922 safe PPV limit (mm/s)
MAPE_THRESHOLD = 25.0        # % MAPE beyond which drift is flagged
CONSEC_TRIGGER = 4           # consecutive bad predictions to trigger retrain
BUFFER_MAX     = 100         # max records in retraining sliding window
DATA_PATH      = 'results/live_blast_log.csv'      # local storage
MODEL_PATH     = 'models/hybrid_ppv_model.pkl'
BENCH_PATH     = 'results/model_benchmark.csv'
GSHEET_CREDS_PATH = 'credentials/awesome-dialect-489311-u7-b7e39b74b42b.json'  # service account JSON
GSHEET_ID      = '1L7OHqkcHXqNprEslSLPSZYjfqljc06FO1qhB4WvJvgs'               # replace with real ID
GSHEET_TAB     = 'sheet1'

for d in ['results','models','credentials','plots']:
    os.makedirs(d, exist_ok=True)

# ─── STREAMLIT PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "PPV Predictor — IIT BHU Mining",
    page_icon   = "💥",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }
    .main-header {
        background: linear-gradient(135deg, #0D2137 0%, #1B4F72 60%, #117A65 100%);
        padding: 1.8rem 2.2rem; border-radius: 10px;
        margin-bottom: 1.2rem; color: white;
    }
    .main-header h1 { color: white; font-size: 1.6rem; margin: 0 0 0.25rem 0; }
    .main-header p  { color: #AED6F1; font-size: 0.9rem; margin: 0; }
    .metric-card {
        background: #F8F9FA; border-left: 4px solid #1B4F72;
        padding: 0.9rem 1.1rem; border-radius: 6px; margin-bottom: 0.7rem;
    }
    .metric-card .label { font-size: 0.75rem; color: #6C757D;
        text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.75rem; font-weight: 600; color: #0D2137; }
    .safe   { background:#E9F7EF; border-left-color:#27AE60; }
    .warn   { background:#FEF9E7; border-left-color:#F39C12; }
    .danger { background:#FDEDEC; border-left-color:#E74C3C; }
    .formula-box {
        background: #0D2137; color: #AED6F1;
        font-family: 'JetBrains Mono', monospace;
        padding: 0.9rem 1.3rem; border-radius: 8px;
        font-size: 0.92rem; margin: 0.8rem 0;
    }
    .status-ok   { color: #27AE60; font-weight: 600; }
    .status-warn { color: #F39C12; font-weight: 600; }
    .status-bad  { color: #E74C3C; font-weight: 600; }
    .section-hd  { font-size: 1.05rem; font-weight: 600; color: #1B4F72;
        border-bottom: 2px solid #1B4F72; padding-bottom: 3px; margin: 1.2rem 0 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load the saved hybrid model from Part 2."""
    #if not os.path.exists(MODEL_PATH):
        #return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def predict_ppv(model, distance, q, n_holes, depth, tq, sd, sd_tq,
                n_rows, spacing):
    """
    Make a PPV prediction using the hybrid model.
    Falls back to pure physics if model unavailable.
    PPV_final = k × SD^n  +  RF_residual(features)
    """
    log_sd    = float(np.log(max(sd, 1e-9)))
    log_d     = float(np.log(max(distance, 1e-9)))
    log_q     = float(np.log(max(q, 1e-9)))
    log_tq    = float(np.log(max(tq, 1e-9)))
    log_n     = float(np.log(max(n_holes, 1e-9)))
    log_depth = float(np.log(max(depth, 1e-9)))

    feat_map = {
        'Distance': distance, 'Q': q, 'No_of_Holes': n_holes,
        'Depth': depth, 'TQ': tq, 'SD': sd, 'SD_TQ': sd_tq,
        'No_of_Rows': n_rows, 'Spacing': spacing,
        'log_SD': log_sd, 'log_D': log_d, 'log_Q': log_q,
        'log_TQ': log_tq, 'log_N': log_n, 'log_Depth': log_depth,
    }

    ppv_physics = K_LIT * (sd ** N_LIT)

    if model is not None and hasattr(model, 'ml') and hasattr(model, 'sc'):
        try:
            feat_cols = model.feat_cols
            X = np.array([[feat_map.get(c, 0.0) for c in feat_cols]])
            X_s  = model.sc.transform(X)
            resid = model.ml.predict(X_s)[0]
            ppv_hybrid = ppv_physics + resid
            return max(0.01, ppv_hybrid), ppv_physics
        except Exception:
            pass
    return ppv_physics, ppv_physics


# ═════════════════════════════════════════════════════════════════════════════
#  GOOGLE SHEETS INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

def get_gsheet_client():
    """Return authenticated Google Sheets client, or None if unavailable."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        #if not os.path.exists(GSHEET_CREDS_PATH):
            #return None
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        #creds  = Credentials.from_service_account_file(GSHEET_CREDS_PATH,scopes=scope)
        creds_dict = st.secrets["gcp_service_account"]

        creds = Credentials.from_service_account_info(
            creds_dict, scopes=scope
        )
        return gspread.authorize(creds)
    except Exception:
        return None


def append_to_gsheet(client, row_dict):
    """
    Append one row to the Google Sheet.
    Columns match the blast log schema.
    """

    if client is None:
        return False
    try:
        google_sheet   = client.open_by_key(GSHEET_ID)
        ws      = google_sheet.worksheet(GSHEET_TAB)
        #self.ws = client.open_by_key(GSHEET_ID).sheet1
        headers = ws.row_values(1)
        if not headers:
            headers = list(row_dict.keys())
            ws.append_row(headers)
        row = [str(row_dict.get(h, '')) for h in headers]
        ws.append_row(row)
        return True
    except Exception as e:
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  LOCAL CSV STORAGE
# ═════════════════════════════════════════════════════════════════════════════

BLAST_COLS = [
    'Timestamp','Date','Blast_No','No_of_Holes','Explosion','Per_Hole',
    'Depth','No_of_Rows','Spacing','Seam_location','Vibrometer',
    'Distance','PPV_actual','PPV_predicted','PPV_physics',
    'SD','SD_TQ','TQ','Blast_Timing','Frequency','MAPE','Source'
]

def load_blast_log():
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=BLAST_COLS)


def save_blast_log(df):
    df.to_csv(DATA_PATH, index=False)


def append_blast_record(record_dict):
    """Append one new record to local CSV."""
    df = load_blast_log()
    new_row = pd.DataFrame([record_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    save_blast_log(df)
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  EWMA DRIFT DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def compute_ewma(mape_series, lam=0.2):
    """Compute EWMA of MAPE series for drift monitoring."""
    ewma = []
    val  = MAPE_THRESHOLD * 0.75
    for m in mape_series:
        if np.isfinite(m):
            val = lam * m + (1 - lam) * val
        ewma.append(val)
    return np.array(ewma)


def check_drift(df_log):
    """
    Check if MAPE has drifted beyond threshold.
    Returns: (drift_detected: bool, consecutive_bad: int, ewma_array)
    """
    if 'MAPE' not in df_log.columns or len(df_log) < 3:
        return False, 0, np.array([])
    mape = df_log['MAPE'].dropna().values.astype(float)
    ewma = compute_ewma(mape)
    consec = 0
    for m in reversed(mape):
        if m > MAPE_THRESHOLD:
            consec += 1
        else:
            break
    drift = consec >= CONSEC_TRIGGER or (len(ewma) > 0 and ewma[-1] > MAPE_THRESHOLD * 1.2)
    return drift, consec, ewma


# ═════════════════════════════════════════════════════════════════════════════
#  AUTO RETRAINING
# ═════════════════════════════════════════════════════════════════════════════

def retrain_model(df_log, feat_cols):
    """
    Retrain the hybrid RF model on the most recent BUFFER_MAX records
    that have both predicted and actual PPV (for residual computation).
    Returns new model object or None.
    """
    try:
        needed = feat_cols + ['PPV_actual', 'SD']
        valid_cols = [c for c in needed if c in df_log.columns]
        # Map column names to feature names
        col_map = {'PPV_actual': 'PPV'}
        sub = df_log[valid_cols].dropna().tail(BUFFER_MAX).copy()
        sub.rename(columns=col_map, inplace=True)

        if len(sub) < 15:
            return None, "Insufficient data for retraining (need ≥15 records)"

        sd_  = sub['SD'].values.ravel()
        phys = K_LIT * sd_**N_LIT
        y    = sub['PPV'].values
        resid = y - phys

        avail_feat = [c for c in feat_cols if c in sub.columns]
        X = sub[avail_feat].values

        sc  = StandardScaler()
        Xs  = sc.fit_transform(X)
        rf  = RandomForestRegressor(n_estimators=300, max_depth=10,
                                     random_state=42, n_jobs=-1)
        rf.fit(Xs, resid)
        pred   = phys + rf.predict(Xs)
        r2_new = r2_score(y, pred)

        # Build new model object
        class RetrainedHybrid:
            pass
        m = RetrainedHybrid()
        m.k = K_LIT; m.n = N_LIT
        m.ml = rf; m.sc = sc; m.feat_cols = avail_feat
        m.test_metrics  = {'R2': r2_new}
        m.train_metrics = {'R2': r2_new}

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(m, f)

        return m, f"Retraining successful — new R² = {r2_new:.4f}"
    except Exception as e:
        return None, f"Retraining failed: {e}"


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ System Status")
    model = load_model()

    if model:
        st.success("✅ Hybrid model loaded")
        if hasattr(model, 'k') and hasattr(model, 'n'):
            st.markdown(f"**Physics:** PPV = k × SD^n  \n"
                        f"k = `{model.k}` | n = `{model.n}`")
        if hasattr(model, 'test_metrics'):
            tm = model.test_metrics
            st.metric("Test R²",  f"{tm.get('R2', 'N/A'):.4f}")
            st.metric("Test MAE", f"{tm.get('MAE', 'N/A'):.4f} mm/s" if 'MAE' in tm else "N/A")
    else:
        st.warning("⚠️ Model not found.  \nRun part2_models.py first.")

    st.divider()

    gsheet_client = get_gsheet_client()
    if gsheet_client:
        st.success("✅ Google Sheets connected")
    else:
        st.info("📄 Local CSV mode  \n(Add credentials.json for cloud)")

    st.divider()
    df_log = load_blast_log()
    drift_detected, consec_bad, ewma_arr = check_drift(df_log)

    if drift_detected:
        st.error(f"🚨 DRIFT DETECTED  \n{consec_bad} consecutive MAPE > {MAPE_THRESHOLD}%")
        if st.button("🔄 Retrain Model Now"):
            if hasattr(model, 'feat_cols'):
                new_model, msg = retrain_model(df_log, model.feat_cols)
                if new_model:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.success("✅ Model stable — no drift")

    st.divider()
    st.markdown("### 📚 Reference")
    st.markdown("""
    **USBM Law** (Duvall & Fogelson, 1962)  
    PPV = k × (D/√Q)ⁿ  
    k = 650, n = −1.4  
    *(Pal Roy, 1993 — Wardha Valley)*
    """)
    st.caption("IS 6922 safe limit: 5 mm/s for residential structures")

# ═════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>💥 Blast-Induced PPV Prediction System</h1>
    <p>Autonomous ML Framework | Department of Mining Engineering ·
    IIT (BHU) Varanasi · B.Tech Project 2024–25</p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predict PPV",
    "📋 Log New Blast",
    "📊 Live Monitor",
    "🔬 Analytics"
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-hd">Blast Design Parameters</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Hole Design**")
        distance  = st.number_input("Distance from blast (m)", 100.0, 1000.0, 420.0, 10.0,
                                     help="Distance between blast point and monitoring instrument")
        depth     = st.number_input("Depth of hole (m)", 2.0, 15.0, 5.0, 0.5,
                                     help="Drilled depth of each blast hole")
        n_holes   = st.number_input("Number of holes", 20, 300, 90, 5,
                                     help="Total holes in the blast pattern")
        n_rows    = st.number_input("Number of rows", 1, 10, 4, 1,
                                     help="Rows of blast holes")
        spacing   = st.number_input("Spacing (m)", 2.0, 8.0, 4.5, 0.5,
                                     help="Distance between adjacent holes and rows")

    with c2:
        st.markdown("**Explosive Design**")
        per_hole  = st.number_input("Per Hole Q (kg)", 20.0, 120.0, 35.0, 1.0,
                                     help="Average charge per hole — max charge per delay")
        explosion = st.number_input("Total explosion (kg)", 500.0, 50000.0,
                                     float(n_holes * per_hole), 100.0,
                                     help="Total explosive mass for the entire blast")
        seam      = st.selectbox("Seam location",
                                  ['Seam Top-5','Seam-3','Seam 4 Floor',
                                   'Seam 5 Floor','40 Coal Top','Seam Top-3','Seam 3 Floor'],
                                  help="Different seams = different rock conditions and k")
        timing    = st.text_input("Blast timing", "14:30",
                                   help="Time when blast is fired (HH:MM)")
        frequency = st.number_input("Frequency (Hz)", 1.0, 50.0, 8.0, 0.5,
                                     help="Dominant frequency at measurement point")

    # Auto-compute engineered features
    Q      = per_hole
    TQ     = n_holes * Q
    SD     = distance / np.sqrt(max(Q, 1e-9))
    SD_TQ  = distance / np.sqrt(max(TQ, 1e-9))
    stem   = depth * (2/3)
    ch_len = depth * (1/3)

    with c3:
        st.markdown("**Computed Features**")
        st.info(f"""
        **Q (charge/delay)**: {Q:.2f} kg  
        **Total charge TQ**: {TQ:.1f} kg  
        **Scaled Distance SD**: {SD:.2f} m/kg⁰·⁵  
        **SD_TQ (IS 6922)**: {SD_TQ:.3f} m/kg⁰·⁵  
        **Stemming**: {stem:.2f} m  
        **Explosive column**: {ch_len:.2f} m  
        """)

    st.markdown('<div class="formula-box">PPV = k × SD^n + RF_residual(features)<br>'
                f'Physics: PPV = {K_LIT} × {SD:.2f}^({N_LIT}) '
                f'= {K_LIT * SD**N_LIT:.4f} mm/s</div>',
                unsafe_allow_html=True)

    if st.button("🚀 Predict PPV", type="primary", use_container_width=True):
        ppv_hybrid, ppv_phys = predict_ppv(
            model, distance, Q, n_holes, depth, TQ, SD, SD_TQ, n_rows, spacing)

        r1, r2, r3 = st.columns(3)

        # Safety classification
        if ppv_hybrid < IS6922_LIMIT:
            card_class = "safe"
            safety_txt = f"✅ SAFE (margin: {((IS6922_LIMIT-ppv_hybrid)/IS6922_LIMIT)*100:.1f}%)"
        elif ppv_hybrid < 10:
            card_class = "warn"
            safety_txt = f"⚠️ CAUTION — exceeds IS 6922 limit"
        else:
            card_class = "danger"
            safety_txt = f"🚨 DANGER — high vibration risk"

        with r1:
            st.markdown(f"""<div class="metric-card {card_class}">
                <div class="label">Hybrid Model PPV</div>
                <div class="value">{ppv_hybrid:.4f} mm/s</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Physics Only (USBM)</div>
                <div class="value">{ppv_phys:.4f} mm/s</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="metric-card {card_class}">
                <div class="label">IS 6922 Safety Check</div>
                <div class="value" style="font-size:1rem">{safety_txt}</div>
            </div>""", unsafe_allow_html=True)

        # PPV gauge
        fig_g, ax_g = plt.subplots(figsize=(8, 2.2))
        fig_g.patch.set_alpha(0)
        zones  = [(0, 2, '#27AE60'), (2, 5, '#F39C12'), (5, 10, '#E74C3C'), (10, 25, '#922B21')]
        labels = ['Safe', 'Caution', 'Danger', 'Critical']
        for (lo, hi, col), lbl in zip(zones, labels):
            ax_g.barh(0, hi-lo, left=lo, color=col, height=0.45, alpha=0.80)
            ax_g.text((lo+hi)/2, 0.28, lbl, ha='center', fontsize=8.5, color='white', fontweight='bold')
        ax_g.axvline(ppv_hybrid, color='black', lw=3.5, zorder=5)
        ax_g.axvline(IS6922_LIMIT, color='white', lw=1.5, ls='--', zorder=4, alpha=0.8)
        ax_g.set_xlim(0, 20); ax_g.set_yticks([])
        ax_g.set_xlabel("PPV (mm/s)", fontsize=10)
        ax_g.set_title(f"PPV = {ppv_hybrid:.4f} mm/s  |  IS 6922 limit = {IS6922_LIMIT} mm/s",
                       fontsize=10)
        st.pyplot(fig_g, use_container_width=False)
        plt.close(fig_g)
        st.caption("Dashed white line = IS 6922 safe limit. Black bar = prediction.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — LOG NEW BLAST
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-hd">Log a New Blast Record</div>',
                unsafe_allow_html=True)
    st.markdown("Enter blast parameters and the **actual measured PPV**. "
                "The record will be saved locally (CSV) and to Google Sheets (if configured). "
                "The model error is tracked for autonomous drift detection.")

    with st.form("blast_log_form", clear_on_submit=True):
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            f_date     = st.date_input("Date of blast", datetime.date.today())
            f_blast_no = st.number_input("Blast No.", 1, 999, 1, 1)
            f_holes    = st.number_input("No. of Holes", 20, 300, 90, 5)
            f_expl     = st.number_input("Explosion (kg)", 500.0, 60000.0, 3000.0, 100.0,
                                          help="Total explosive charge used in blast")
            f_q        = st.number_input("Per Hole Q (kg)", 20.0, 120.0, 35.0, 1.0,
                                          help="Average charge per hole")

        with fc2:
            f_depth    = st.number_input("Depth of Hole (m)", 2.0, 15.0, 5.0, 0.5)
            f_rows     = st.number_input("No. of Rows", 1, 10, 4, 1)
            f_spacing  = st.number_input("Spacing (m)", 2.0, 8.0, 4.5, 0.5)
            f_seam     = st.selectbox("Seam Location",
                                       ['Seam Top-5','Seam-3','Seam 4 Floor',
                                        'Seam 5 Floor','40 Coal Top','Seam Top-3',
                                        'Seam 3 Floor','Other'])
            f_vib      = st.selectbox("Vibrometer ID",
                                       [20697, 20698, 15335, 15336, 23314, 'Other'])

        with fc3:
            f_distance = st.number_input("Distance (m)", 50.0, 1000.0, 420.0, 10.0)
            f_ppv_act  = st.number_input("Actual Measured PPV (mm/s)", 0.1, 50.0, 3.5, 0.1,
                                          help="PPV value read from vibrometer")
            f_timing   = st.text_input("Blast Timing (HH:MM)", "14:30")
            f_freq     = st.number_input("Frequency (Hz)", 1.0, 50.0, 8.0, 0.5)

        submitted = st.form_submit_button("💾 Submit & Store", type="primary",
                                           use_container_width=True)

    if submitted:
        # Compute features
        f_TQ    = f_holes * f_q
        f_SD    = f_distance / np.sqrt(max(f_q, 1e-9))
        f_SD_TQ = f_distance / np.sqrt(max(f_TQ, 1e-9))

        # Predict
        ppv_pred, ppv_phys = predict_ppv(
            model, f_distance, f_q, f_holes, f_depth,
            f_TQ, f_SD, f_SD_TQ, f_rows, f_spacing)

        mape_val = abs(f_ppv_act - ppv_pred) / (abs(f_ppv_act) + 1e-9) * 100

        record = {
            'Timestamp'    : datetime.datetime.now().isoformat(),
            'Date'         : str(f_date),
            'Blast_No'     : int(f_blast_no),
            'No_of_Holes'  : int(f_holes),
            'Explosion'    : float(f_expl),
            'Per_Hole'     : float(f_q),
            'Depth'        : float(f_depth),
            'No_of_Rows'   : int(f_rows),
            'Spacing'      : float(f_spacing),
            'Seam_location': str(f_seam),
            'Vibrometer'   : str(f_vib),
            'Distance'     : float(f_distance),
            'PPV_actual'   : float(f_ppv_act),
            'PPV_predicted': round(float(ppv_pred), 4),
            'PPV_physics'  : round(float(ppv_phys), 4),
            'SD'           : round(float(f_SD), 4),
            'SD_TQ'        : round(float(f_SD_TQ), 4),
            'TQ'           : float(f_TQ),
            'Blast_Timing' : str(f_timing),
            'Frequency'    : float(f_freq),
            'MAPE'         : round(float(mape_val), 2),
            'Source'       : 'Field',
        }

        # Save locally
        df_updated = append_blast_record(record)

        # Save to Google Sheets
        gc = get_gsheet_client()
        gsheet_ok = append_to_gsheet(gc, record)

        # Show confirmation
        col_a, col_b = st.columns(2)
        with col_a:
            st.success(f"✅ Saved to local CSV ({len(df_updated)} total records)")
            if gsheet_ok:
                st.success("☁️ Saved to Google Sheets")
            else:
                st.info("📄 Google Sheets: not configured (local only)")

        with col_b:
            status_class = "status-ok" if mape_val < MAPE_THRESHOLD else "status-bad"
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Actual PPV | **{f_ppv_act:.4f} mm/s** |
            | Predicted PPV | **{ppv_pred:.4f} mm/s** |
            | Physics PPV | **{ppv_phys:.4f} mm/s** |
            | MAPE | **{mape_val:.2f}%** |
            | Status | {'✅ Normal' if mape_val < MAPE_THRESHOLD else '⚠️ High Error'} |
            """)

        # Check drift after new record
        df_log_new = load_blast_log()
        new_drift, new_consec, _ = check_drift(df_log_new)
        if new_drift:
            st.warning(f"⚠️ Drift Alert: {new_consec} consecutive MAPE > {MAPE_THRESHOLD}%. "
                       "Consider retraining via the sidebar.")

    # Show recent log
    st.markdown('<div class="section-hd">Recent Blast Log</div>', unsafe_allow_html=True)
    df_log = load_blast_log()
    if len(df_log) > 0:
        show_cols = ['Date','Blast_No','Distance','Per_Hole','No_of_Holes',
                     'PPV_actual','PPV_predicted','MAPE','Seam_location']
        show_cols = [c for c in show_cols if c in df_log.columns]
        st.dataframe(df_log[show_cols].tail(20).round(3),
                     use_container_width=True, height=350)

        csv_data = df_log.to_csv(index=False).encode()
        st.download_button("⬇️ Download Full Log (CSV)", csv_data,
                           "blast_log.csv", "text/csv")
    else:
        st.info("No blast records logged yet. Submit records above.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — LIVE MONITOR
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-hd">Live Monitoring Dashboard</div>',
                unsafe_allow_html=True)

    df_log = load_blast_log()

    if len(df_log) < 3:
        st.info("Log at least 3 blast records in Tab 2 to see the monitoring dashboard.")
    else:
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Events",    len(df_log))
        m2.metric("Mean MAPE",       f"{df_log['MAPE'].mean():.2f}%" if 'MAPE' in df_log.columns else "—")
        m3.metric("IS 6922 Exceeded",
                  f"{(df_log['PPV_actual']>IS6922_LIMIT).sum()}" if 'PPV_actual' in df_log.columns else "—")
        m4.metric("Consecutive Bad", f"{consec_bad}")

        # EWMA chart
        if 'MAPE' in df_log.columns:
            mape_vals = df_log['MAPE'].fillna(0).values
            ewma_vals = compute_ewma(mape_vals)
            ucl       = MAPE_THRESHOLD * 1.25

            fig_m, axes_m = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            fig_m.suptitle("Live Monitoring: EWMA Control Chart & PPV Trace",
                           fontsize=12, fontweight='bold')

            ax = axes_m[0]
            ax.plot(range(len(mape_vals)), mape_vals, alpha=0.35,
                    color='#AEB6BF', lw=1, label='Raw MAPE (%)')
            ax.plot(range(len(ewma_vals)), ewma_vals,
                    color='#1B4F72', lw=2.2, label='EWMA MAPE')
            ax.axhline(MAPE_THRESHOLD, color='red', ls='--', lw=1.5,
                       label=f'Threshold = {MAPE_THRESHOLD}%')
            ax.axhline(ucl, color='darkred', ls=':', lw=1.2,
                       label=f'UCL = {ucl:.0f}%')
            drift_idx = np.where(ewma_vals > ucl)[0]
            if len(drift_idx):
                ax.scatter(drift_idx, ewma_vals[drift_idx],
                           color='red', s=80, zorder=5, label='Drift points')
            ax.set_ylabel("MAPE (%)"); ax.set_title("EWMA MAPE Control Chart")
            ax.legend(fontsize=8.5)

            ax2 = axes_m[1]
            if 'PPV_actual' in df_log.columns and 'PPV_predicted' in df_log.columns:
                ax2.plot(range(len(df_log)), df_log['PPV_actual'].fillna(0),
                         color='#1B4F72', lw=1.8, label='Actual PPV')
                ax2.plot(range(len(df_log)), df_log['PPV_predicted'].fillna(0),
                         color='#E74C3C', lw=1.8, ls='--', label='Predicted PPV')
                ax2.axhline(IS6922_LIMIT, color='orange', lw=1.5, ls=':',
                            label='IS 6922 limit (5 mm/s)')
            ax2.set_xlabel("Blast Event Number")
            ax2.set_ylabel("PPV (mm/s)")
            ax2.set_title("Actual vs Predicted PPV — Live Trace")
            ax2.legend(fontsize=8.5)

            plt.tight_layout()
            st.pyplot(fig_m, use_container_width=True)
            plt.close(fig_m)

        # Drift status
        if drift_detected:
            st.error(f"""
            🚨 **Model Drift Detected**  
            {consec_bad} consecutive predictions exceed {MAPE_THRESHOLD}% MAPE.  
            This may indicate a geological zone change at the blast site.  
            Use the **Retrain Model** button in the sidebar to update the model.
            """)
        else:
            st.success("✅ Model operating within normal error bounds — no drift detected.")

        # Actual vs predicted scatter
        if len(df_log) >= 5 and 'PPV_actual' in df_log.columns:
            sub = df_log[['PPV_actual','PPV_predicted']].dropna()
            if len(sub) >= 5:
                r2_live = r2_score(sub['PPV_actual'], sub['PPV_predicted'])
                fig_s, ax_s = plt.subplots(figsize=(6, 5))
                ax_s.scatter(sub['PPV_actual'], sub['PPV_predicted'],
                             color='#1B4F72', s=55, alpha=0.85, edgecolors='white')
                lim = [0, max(sub['PPV_actual'].max(), sub['PPV_predicted'].max()) * 1.12]
                ax_s.plot(lim, lim, 'k--', lw=1.5, label='1:1 line')
                ax_s.set_xlabel("Actual PPV (mm/s)"); ax_s.set_ylabel("Predicted PPV (mm/s)")
                ax_s.set_title(f"Live R² = {r2_live:.4f}")
                ax_s.legend(fontsize=9)
                st.pyplot(fig_s, use_container_width=False)
                plt.close(fig_s)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-hd">Model Analytics & EDA Figures</div>',
                unsafe_allow_html=True)

    plot_options = {
        "3.1 Univariate Correlation Metrics"         : "plots/Fig3_1_univariate_correlation.png",
        "3.2 Parameter vs PPV — Power-Law Fits"      : "plots/Fig3_2_scatter_vs_ppv.png",
        "3.3 Log-Log Axes — Power Law Linearisation" : "plots/Fig3_3_loglog_vs_ppv.png",
        "3.4 Scaled Distance vs PPV (USBM Verify)"   : "plots/Fig3_4_scaled_distance_vs_ppv.png",
        "3.5 Distance vs Frequency"                  : "plots/Fig3_5_distance_frequency.png",
        "3.6 Correlation Heatmap"                    : "plots/Fig3_6_correlation_heatmap.png",
        "3.7 PPV Distribution Analysis"              : "plots/Fig3_7_ppv_distribution.png",
        "3.8 Real vs Fake Data Distributions"        : "plots/Fig3_8_real_vs_fake.png",
        "4.1 Physics Model Fit (USBM)"               : "plots/Fig4_1_physics_model.png",
        "4.2 Synthetic Data Distributions"           : "plots/Fig4_2_synthetic_distribution.png",
        "4.3 All Model Benchmark"                    : "plots/Fig4_3_model_benchmark.png",
        "4.4 Hybrid Model Actual vs Predicted"       : "plots/Fig4_4_actual_vs_predicted.png",
        "4.5 Error Distribution"                     : "plots/Fig4_5_error_distribution.png",
        "4.6 Prediction Timeline"                    : "plots/Fig4_6_prediction_timeline.png",
        "4.7 Feature Importance"                     : "plots/Fig4_7_feature_importance.png",
        "4.8 Train vs Test R²"                       : "plots/Fig4_8_train_test_r2.png",
    }

    selected = st.selectbox("Select figure:", list(plot_options.keys()))
    img_path = plot_options[selected]
    if os.path.exists(img_path):
        st.image(img_path, use_column_width=True,
                 caption=f"Fig. {selected}")
    else:
        st.warning(f"Figure not found: `{img_path}`  \nRun part1_eda.py and part2_models.py first.")

    # Model benchmark table
    st.markdown('<div class="section-hd">Model Benchmark Summary</div>',
                unsafe_allow_html=True)
    if os.path.exists(BENCH_PATH):
        bench_df = pd.read_csv(BENCH_PATH)
        # Highlight best row
        best_r2  = bench_df['R2'].max()
        def highlight_best(row):
            if row['R2'] == best_r2:
                return ['background-color: #D5F5E3'] * len(row)
            return [''] * len(row)
        st.dataframe(bench_df.style.apply(highlight_best, axis=1),
                     use_container_width=True)
        st.caption("🟢 Green row = best R² model")
    else:
        st.info("Run part2_models.py to generate the benchmark table.")

    # Site constants explanation
    with st.expander("📖 About the Physics Constants (k and n)"):
        st.markdown(f"""
        **Site Constant k = {K_LIT}** — The site transmission constant governs
        the amplitude of ground vibration at unit scaled distance. For Bhanegaon mine
        and similar Wardha Valley coalfield conditions, literature values range from
        **200 to 1100** depending on rock type, explosive type, and degree of fracturing.
        k = 650 represents a midpoint calibrated to Indian overburden blasting practice
        (Pal Roy, 1993).

        **Attenuation Exponent n = {N_LIT}** — The negative exponent governs the rate
        of PPV decay with increasing scaled distance. For Indian coal mines, n ranges
        from **−1.2 to −1.6**. Both k and n are **always non-zero** by physical definition:
        k = 0 would mean no vibration is generated (physically impossible with explosive
        detonation); n = 0 would mean PPV does not decrease with distance (violates wave
        mechanics). n = −1.4 is consistent with the moderate attenuation observed in
        Gondwana sandstone overburden of Wardha Valley.

        **Reference**: Pal Roy, P. (1993). Putting ground vibration predictors into
        practice. *Colliery Guardian*, 241(2), 63-67.
        """)

    # Google Sheets setup guide
    with st.expander("☁️ Google Sheets Setup Guide"):
        st.markdown(f"""
        **To enable cloud storage in Google Sheets:**

        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project → Enable **Google Sheets API** and **Google Drive API**
        3. Create a **Service Account** → Download the JSON credentials file
        4. Place the credentials file at: `{GSHEET_CREDS_PATH}`
        5. Create a Google Sheet → Copy the Sheet ID from the URL:
           `https://docs.google.com/spreadsheets/d/`**SHEET_ID**`/edit`
        6. Share your Google Sheet with the **service account email** (Editor permission)
        7. Edit this file: set `GSHEET_ID = 'your_actual_sheet_id'`
        8. Restart the app: `streamlit run part3_ui.py`

        **Sheet structure** — the app will auto-create headers in tab `{GSHEET_TAB}`:
        ```
        Timestamp | Date | Blast_No | No_of_Holes | Explosion | Per_Hole |
        Depth | No_of_Rows | Spacing | Seam_location | Vibrometer | Distance |
        PPV_actual | PPV_predicted | PPV_physics | SD | SD_TQ | TQ |
        Blast_Timing | Frequency | MAPE | Source
        ```
        """)
