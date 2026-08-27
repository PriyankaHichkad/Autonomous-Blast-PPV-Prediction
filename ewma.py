"""
================================================================================
  PHASE 2 — EWMA-WEIGHTED HYBRID PPV MODEL
  An Autonomous ML Framework for Blast-Induced Ground Vibration Prediction
  IIT (BHU) Varanasi | Department of Mining Engineering | B.Tech 2025-26
  Supervised by Dr. Satyabrata Behera
================================================================================

  WHAT THIS FILE DOES
  ────────────────────
  This is Phase 2 of the project. In Phase 1 we trained a hybrid model on
  a fixed dataset. In Phase 2, the model is designed to grow incrementally —
  a new blast row is fed in one at a time, and the model updates its internal
  weights to reflect that recent rows matter more than old ones.

  THE CORE IDEA — WHY EWMA INSTEAD OF DELETING OLD DATA
  ───────────────────────────────────────────────────────
  As blasting progresses at Bhanegaon mine, the ground conditions change:
    - The mine excavates to a different seam with different rock hardness
    - The distance from the blast to occupied structures changes
    - The geology transitions (e.g. Seam Top-5 → Seam 4 Floor)

  The naive solution is to delete old rows and only train on recent ones.
  But in a real mine, data is precious and hard to collect. Deleting data
  throws away information that took months to gather.

  The EWMA solution: KEEP all data, but assign exponentially decaying
  weights so that:
    - The most recent blast row has weight 1.0
    - Each older row has its weight multiplied by (1 - lambda)
    - lambda=0.2 means row from 10 blasts ago has weight (0.8)^10 = 0.107
    - lambda=0.2 means row from 44 blasts ago has weight (0.8)^44 = 0.0000

  The model trains on all rows simultaneously, but the loss function pays
  far more attention to recent rows. This is mathematically equivalent to
  a sliding window but smoother — old rows don't have a hard cutoff, they
  just fade to near-zero influence.

  HOW IT WORKS — STEP BY STEP
  ─────────────────────────────
  1. Start with the full real field dataset (44 rows).
  2. When a NEW blast row arrives (you type it in), append it to the dataset.
  3. Recompute EWMA sample weights:
       weight[i] = (1 - lambda)^(n - 1 - i)
     where i=0 is the oldest row and i=n-1 is the most recent.
  4. Retrain the Hybrid RF model using sample_weight=weights.
  5. Predict PPV for the new row.
  6. Return prediction + show how the weights shifted.

  WHY THIS IS BETTER THAN PHASE 1
  ──────────────────────────────────
  Phase 1: Fixed model trained once on 44+600 records. Never updates.
  Phase 2: Model updates every time a new blast is added. Recent geology
           automatically dominates. No data is wasted. No hard cutoffs.

  SITE CONSTANTS (Bhanegaon / Wardha Valley — Pal Roy, 1993)
  ─────────────────────────────────────────────────────────────
  k = 650   (transmission constant — always non-zero, range 200–1100)
  n = -1.4  (attenuation exponent — always non-zero, range -1.2 to -1.6)

  RUN
  ────
  python ewma_model.py                          # trains on real data only
  python ewma_model.py --interactive            # add new rows one at a time
  python ewma_model.py --lambda 0.2             # change decay speed
================================================================================
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Directories ───────────────────────────────────────────────────────────────
for d in ['results', 'models', 'plots']:
    os.makedirs(d, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
K_LIT   = 650.0   # USBM site constant — always non-zero
N_LIT   = -1.4    # attenuation exponent — always non-zero

FEATURE_COLS = [
    'Distance', 'Q', 'No_of_Holes', 'Depth', 'TQ', 'SD', 'SD_TQ',
    'No_of_Rows', 'Spacing',
    'log_SD', 'log_D', 'log_Q', 'log_TQ', 'log_N', 'log_Depth',
]

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.dpi': 150,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
})
C = {'navy': '#1B4F72', 'red': '#E74C3C', 'green': '#27AE60',
     'gold': '#F39C12', 'grey': '#BDC3C7', 'teal': '#117A65'}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def physics_ppv(sd: np.ndarray) -> np.ndarray:
    """
    USBM power law: PPV = k × SD^n
    k=650, n=-1.4 — both always non-zero (Pal Roy, 1993).
    """
    return K_LIT * np.clip(sd, 1e-9, None) ** N_LIT


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all engineered features from raw blast parameters.
    Mirrors the feature engineering in part1_eda.py exactly.
    """
    df = df.copy()

    # Core aliases
    if 'Q' not in df.columns and 'Per_Hole' in df.columns:
        df['Q'] = df['Per_Hole']

    # Physics features
    if 'TQ' not in df.columns:
        df['TQ']    = df['No_of_Holes'] * df['Q']
    if 'SD' not in df.columns:
        df['SD']    = df['Distance'] / np.sqrt(df['Q'].clip(lower=1e-9))
    if 'SD_TQ' not in df.columns:
        df['SD_TQ'] = df['Distance'] / np.sqrt(df['TQ'].clip(lower=1e-9))

    # Log-space features (linearise power-law relationships)
    log_map = {
        'log_SD': 'SD', 'log_D': 'Distance', 'log_Q': 'Q',
        'log_TQ': 'TQ', 'log_N': 'No_of_Holes', 'log_Depth': 'Depth',
    }
    for feat, src in log_map.items():
        if feat not in df.columns and src in df.columns:
            df[feat] = np.log(df[src].clip(lower=1e-9))

    return df


def load_real_data(path: str = 'results/real_field_data.csv') -> pd.DataFrame:
    """Load the real blast dataset from Bhanegaon mine."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Real data not found at {path}.\n"
            "Run part1_eda.py first to generate it."
        )
    df = pd.read_csv(path)
    df = engineer_features(df)
    # Keep only rows with both PPV and SD (minimum requirements)
    df = df.dropna(subset=['PPV', 'SD', 'Distance', 'Q']).copy()
    df = df.reset_index(drop=True)
    print(f"[DATA]  Loaded {len(df)} real blast records from {path}")
    print(f"        PPV range: {df.PPV.min():.3f} – {df.PPV.max():.3f} mm/s")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — EWMA WEIGHT COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ewma_weights(n: int, lam: float = 0.25) -> np.ndarray:
    """
    Compute EWMA sample weights for n data rows.
    """
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    indices     = np.arange(n)                        # 0 = oldest, n-1 = newest
    exponents   = (n - 1 - indices).astype(float)     # n-1, n-2, ..., 1, 0
    raw_weights = (1.0 - lam) ** exponents             # exponential decay
    return raw_weights / raw_weights.sum()             # normalise to sum=1


def compute_event_ewma_weights(df: pd.DataFrame, lam: float = 0.25) -> np.ndarray:
    """
    Compute Event-Level EWMA sample weights for a dataframe of blast records.
    Groups records by unique Blast Event (e.g. Q, No_of_Holes, Depth, Spacing combination).
    All geophone distance records belonging to the SAME blast event receive the SAME event weight.
    This handles multi-station distance measurements (spatial dimension) without data leakage.
    """
    if len(df) == 0:
        return np.array([])
    
    # Identify unique blast events
    if 'Event_ID' not in df.columns:
        event_series = df.groupby(['Q', 'No_of_Holes', 'Depth', 'Spacing'], sort=False).ngroup()
    else:
        event_series = df['Event_ID']
        
    unique_events = event_series.unique()
    n_events = len(unique_events)
    
    # Compute event-level EWMA weights
    event_age_map = {ev_id: (n_events - 1 - idx) for idx, ev_id in enumerate(unique_events)}
    event_ages = event_series.map(event_age_map).values.astype(float)
    
    raw_weights = (1.0 - lam) ** event_ages
    return raw_weights / np.sum(raw_weights)


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Effective sample size = 1 / sum(w_i^2) for normalised weights.
    """
    return 1.0 / np.sum(weights ** 2)


def rows_capturing_90pct(weights: np.ndarray) -> int:
    """
    Count of most recent rows that together capture 90% of total weight.
    """
    cumsum_recent = np.cumsum(weights[::-1])   # from newest to oldest
    return int(np.searchsorted(cumsum_recent, 0.90)) + 1


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — EWMA HYBRID MODEL
# ══════════════════════════════════════════════════════════════════════════════

class EWMAHybridModel:
    """
    EWMA-Weighted Hybrid Physics + Random Forest PPV Predictor.

    Core equation (same as Phase 1):
        PPV_final = k × SD^n  +  RF(Residual_features, sample_weight=w)

    What changes in Phase 2:
        The RF is trained with EWMA sample weights.
        When a new blast row is added, weights are recomputed so the
        newest row has the highest weight and old rows fade exponentially.
        ALL historical data is kept — nothing is deleted.

    Why RF for the residual learner in EWMA context:
        RF supports sample_weight natively in sklearn.
        It is robust to the weight imbalance created by EWMA (old rows
        near zero, new row at full weight) — tree splitting handles
        non-uniform weights cleanly via weighted Gini impurity.

    Attributes
    ───────────
    data     : pd.DataFrame — all blast records seen so far (cumulative)
    lam      : float        — EWMA decay rate
    k, n     : float        — physics constants, always non-zero
    rf       : fitted RandomForestRegressor on residuals
    sc       : fitted StandardScaler
    weights  : np.ndarray   — current EWMA weights (one per row in data)
    history  : list[dict]   — record of each update (for plotting)
    """

    def __init__(self, lam: float = 0.2,
                 n_estimators: int = 300,
                 max_depth: int = 10):
        self.lam           = lam
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.k             = K_LIT         # Default initial estimate
        self.n_phys        = N_LIT         # Default initial estimate
        self.rf            = None
        self.sc            = StandardScaler()
        self.feat_cols     = []
        self.data          = pd.DataFrame()
        self.weights       = np.array([])
        self.is_fitted     = False
        self.history       = []            # list of per-update metric dicts

    # ── Dynamic Site Constants Estimation from Data ────────────────────────────
    def _estimate_site_constants(self, sd: np.ndarray, ppv: np.ndarray, weights: np.ndarray) -> None:
        """
        Dynamically estimate site constants k and n directly from field data
        without hardcoding fixed USBM parameters.
        Fits weighted log-linear regression: ln(PPV) = ln(k) + n * ln(SD)
        """
        try:
            valid_idx = (sd > 0) & (ppv > 0)
            if np.sum(valid_idx) >= 3:
                log_sd  = np.log(sd[valid_idx])
                log_ppv = np.log(ppv[valid_idx])
                w       = weights[valid_idx] / np.sum(weights[valid_idx])
                
                # Weighted linear fit
                poly = np.polyfit(log_sd, log_ppv, deg=1, w=w)
                self.n_phys = float(poly[0])          # Estimated slope n
                self.k      = float(np.exp(poly[1]))  # Estimated intercept e^ln(k)
        except Exception:
            pass  # Fallback to default initial estimates if estimation fails

    # ── Physics Baseline (Dynamic or Data-Fitted) ──────────────────────────────
    def _phys(self, sd: np.ndarray) -> np.ndarray:
        """PPV_physics = k × SD^n using dynamically estimated or calibrated k, n."""
        return self.k * np.power(np.maximum(sd, 1e-6), self.n_phys)

    # ── Internal fit ──────────────────────────────────────────────────────────
    def _fit_on_data(self) -> None:
        """
        (Re)fit the Pure EWMA-weighted GBR model on self.data.
        Called every time a new row is added.

        Steps:
        1. Compute EWMA weights for all rows in self.data: w_i = (1-lambda)^(N-1-i)
        2. Transform target PPV to log-space: y_log = ln(PPV)
        3. Fit GradientBoostingRegressor on (features, y_log) with sample_weight=weights
        4. Prediction: PPV_pred = exp(gbr.predict(X_scaled))
        """
        df = self.data.copy()
        avail = [c for c in FEATURE_COLS if c in df.columns]
        self.feat_cols = avail

        valid = df[avail + ['PPV', 'SD']].dropna()
        valid = valid.loc[:, ~valid.columns.duplicated()]

        if len(valid) < 2:
            return   # need at least 2 rows to fit

        X   = valid[avail].values
        y   = valid['PPV'].values

        # EWMA event-level weights — recomputed fresh on every call
        self.weights = compute_event_ewma_weights(valid, self.lam)

        # Pure data-driven log-space target: y_log = ln(PPV)
        y_log = np.log(np.maximum(y, 1e-4))

        # Scale features
        X_s = self.sc.fit_transform(X)

        # Train GBR directly on log-PPV with EWMA weights
        self.rf = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            max_depth=min(self.max_depth, 4),
            random_state=42,
        )
        self.rf.fit(X_s, y_log, sample_weight=self.weights)
        self.is_fitted = True

        # Strictly positive prediction: PPV_pred = exp(y_log_pred)
        pred_all = np.exp(self.rf.predict(X_s))
        r2_w   = r2_score(y, pred_all, sample_weight=self.weights)
        r2_uw  = r2_score(y, pred_all)                             # unweighted (for reference)
        mae_uw = mean_absolute_error(y, pred_all)

        eff_n = effective_sample_size(self.weights)
        rows_90 = rows_capturing_90pct(self.weights)

        self.history.append({
            'n_rows'           : len(valid),
            'lambda'           : self.lam,
            'R2_weighted'      : round(r2_w, 4),
            'R2_unweighted'    : round(r2_uw, 4),
            'MAE_unweighted'   : round(mae_uw, 4),
            'effective_n'      : round(eff_n, 2),
            'rows_90pct_weight': rows_90,
            'newest_weight'    : round(float(self.weights[-1]), 6),
            'oldest_weight'    : round(float(self.weights[0]),  6),
        })

    # ── Initial fit on full dataset ───────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> 'EWMAHybridModel':
        """
        Fit the model on the initial real dataset.
        This is the starting point — all rows in df are treated as
        historical observations, oldest first.

        The most recent row in df will have the highest EWMA weight.
        """
        df = engineer_features(df.copy())
        df = df.dropna(subset=['PPV', 'SD']).reset_index(drop=True)
        self.data = df
        self._fit_on_data()

        h = self.history[-1]
        print(f"\n{'─'*60}")
        print(f"  EWMA Hybrid Model — Initial Fit (lambda={self.lam})")
        print(f"{'─'*60}")
        print(f"  Rows in dataset       : {h['n_rows']}")
        print(f"  R² (weighted)         : {h['R2_weighted']:.4f}")
        print(f"  R² (unweighted)       : {h['R2_unweighted']:.4f}")
        print(f"  MAE (unweighted)      : {h['MAE_unweighted']:.4f} mm/s")
        print(f"  Effective sample size : {h['effective_n']:.1f} rows")
        print(f"  90% weight captured by: last {h['rows_90pct_weight']} rows")
        print(f"  Newest row weight     : {h['newest_weight']:.6f}")
        print(f"  Oldest row weight     : {h['oldest_weight']:.6f}")
        print(f"  Ratio new/old         : {h['newest_weight']/max(h['oldest_weight'],1e-12):.0f}×")
        print(f"{'─'*60}")
        return self

    # ── Add a new blast row ───────────────────────────────────────────────────
    def update(self, new_row: dict) -> dict:
        """
        Add one new blast record and update the model.

        This is the main Phase 2 operation. Call it each time a new
        blast is fired and PPV is recorded.

        Parameters
        ───────────
        new_row : dict with keys matching the dataset columns.
                  Must include at minimum:
                    distance, per_hole (or Q), n_holes, depth, ppv_actual
                  Optional: n_rows, spacing, seam, blast_timing, frequency

        Returns
        ────────
        result : dict with:
                  ppv_predicted  — model prediction for this new row
                  ppv_physics    — physics-only prediction (no ML)
                  ppv_actual     — the actual measured PPV (from new_row)
                  error_mm       — absolute error in mm/s
                  error_pct      — absolute % error (MAPE contribution)
                  n_rows         — total rows in dataset now
                  newest_weight  — EWMA weight assigned to this new row
                  effective_n    — effective sample size after update
                  R2_weighted    — weighted R² after refit
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before update().")

        # ── Normalise input ────────────────────────────────────────────────────
        row = {
            'Date'          : new_row.get('date', str(datetime.today().date())),
            'Blast_No'      : new_row.get('blast_no', len(self.data) + 1),
            'No_of_Holes'   : float(new_row.get('n_holes', new_row.get('No_of_Holes', 90))),
            'Explosion'     : float(new_row.get('explosion', new_row.get('Explosion', 0))),
            'Per_Hole'      : float(new_row.get('per_hole', new_row.get('Per_Hole', new_row.get('Q', 35)))),
            'Depth'         : float(new_row.get('depth', new_row.get('Depth', 5))),
            'No_of_Rows'    : float(new_row.get('n_rows', new_row.get('No_of_Rows', 4))),
            'Spacing'       : float(new_row.get('spacing', new_row.get('Spacing', 4.5))),
            'Seam_location' : str(new_row.get('seam', new_row.get('Seam_location', 'Unknown'))),
            'Vibrometer'    : str(new_row.get('vibrometer', new_row.get('Vibrometer', ''))),
            'Distance'      : float(new_row.get('distance', new_row.get('Distance', 420))),
            'PPV'           : float(new_row.get('ppv_actual', new_row.get('PPV', np.nan))),
            'Blast_Timing'  : str(new_row.get('blast_timing', new_row.get('Blast_Timing', ''))),
            'Frequency'     : float(new_row.get('frequency', new_row.get('Frequency', np.nan))),
        }

        # ── Predict BEFORE updating model (true held-out prediction) ──────────
        temp_df = pd.DataFrame([row])
        temp_df = engineer_features(temp_df)
        avail   = [c for c in self.feat_cols if c in temp_df.columns]

        sd_new    = float(temp_df['SD'].iloc[0])
        phys_pred = float(self._phys(np.array([sd_new]))[0])

        if len(avail) == len(self.feat_cols):
            X_new  = temp_df[avail].values
            X_new_s = self.sc.transform(X_new)
            log_ppv_pred = float(self.rf.predict(X_new_s)[0])
            ppv_predicted = np.exp(log_ppv_pred)
        else:
            ppv_predicted = 1.0   # fallback

        ppv_actual    = row['PPV']

        error_mm  = abs(ppv_actual - ppv_predicted) if not np.isnan(ppv_actual) else np.nan
        error_pct = error_mm / (abs(ppv_actual) + 1e-9) * 100 if not np.isnan(ppv_actual) else np.nan

        # ── Append new row and REFIT with updated EWMA weights ────────────────
        new_row_df = engineer_features(pd.DataFrame([row]))
        self.data  = pd.concat([self.data, new_row_df], ignore_index=True)
        self._fit_on_data()

        h = self.history[-1]

        result = {
            'ppv_predicted' : round(ppv_predicted, 4),
            'ppv_actual'    : ppv_actual,
            'error_mm'      : round(error_mm, 4)  if not np.isnan(error_mm)  else None,
            'error_pct'     : round(error_pct, 2) if not np.isnan(error_pct) else None,
            'n_rows'        : h['n_rows'],
            'newest_weight' : h['newest_weight'],
            'oldest_weight' : h['oldest_weight'],
            'effective_n'   : h['effective_n'],
            'R2_weighted'   : h['R2_weighted'],
            'R2_unweighted' : h['R2_unweighted'],
            'MAE_unweighted': h['MAE_unweighted'],
            'rows_90pct'    : h['rows_90pct_weight'],
        }
        return result

    # ── Predict without updating ───────────────────────────────────────────────
    def predict_only(self, query: dict) -> dict:
        """
        Predict PPV for a blast design WITHOUT adding it to the dataset.
        Use this when you want a prediction before a blast is fired.

        query keys: distance, per_hole (Q), n_holes, depth,
                    n_rows (optional), spacing (optional)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_only().")

        row = {
            'Distance'    : float(query.get('distance', query.get('Distance', 420))),
            'Per_Hole'    : float(query.get('per_hole', query.get('Q', 35))),
            'No_of_Holes' : float(query.get('n_holes', query.get('No_of_Holes', 90))),
            'Depth'       : float(query.get('depth', query.get('Depth', 5))),
            'No_of_Rows'  : float(query.get('n_rows', query.get('No_of_Rows', 4))),
            'Spacing'     : float(query.get('spacing', query.get('Spacing', 4.5))),
        }

        df_q = pd.DataFrame([row])
        df_q = engineer_features(df_q)
        avail = [c for c in self.feat_cols if c in df_q.columns]

        sd_q           = float(df_q['SD'].iloc[0])
        X_q            = df_q[avail].values
        X_q_s          = self.sc.transform(X_q)
        log_ppv_q      = float(self.rf.predict(X_q_s)[0])
        ppv_q          = np.exp(log_ppv_q)

        return {
            'ppv_predicted': round(ppv_q, 4),
            'physics_ppv'  : round(650.0 * (sd_q ** -1.4), 4),
            'SD'           : round(sd_q, 3),
            'log_ppv_pred' : round(log_ppv_q, 4),
        }

    def predict_custom(self, query: dict) -> dict:
        return self.predict_only(query)

    # ── Feature importance ────────────────────────────────────────────────────
    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances from the RF residual model."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        fi = self.rf.feature_importances_
        df_fi = pd.DataFrame({'feature': self.feat_cols, 'importance': fi})
        return df_fi.sort_values('importance', ascending=False).reset_index(drop=True)

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, path: str = 'models/ewma_hybrid_model.pkl') -> None:
        """Save the full model state (data + weights + ML residual) to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=4)
        print(f"[SAVE]  Model → {path}  ({self.data.shape[0]} rows, lambda={self.lam})")

    @staticmethod
    def load(path: str) -> 'EWMAHybridModel':
        """Load a saved model from disk."""
        import sys, types
        for alias in ('__main__',):
            mod = sys.modules.get(alias)
            if mod and not hasattr(mod, 'EWMAHybridModel'):
                setattr(mod, 'EWMAHybridModel', EWMAHybridModel)
        with open(path, 'rb') as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_ewma_weights(model: EWMAHybridModel, save_path: str = 'plots/ewma_weights.png') -> None:
    """
    Fig W1 — Show how EWMA weights are distributed across all rows.
    Three panels:
      (a) Weight per row — bar chart, newest row on right
      (b) Cumulative weight from newest — shows how many rows carry 90%
      (c) Effect of different lambda values on weight distribution
    """
    weights = model.weights
    n       = len(weights)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'EWMA Weight Distribution — {n} blast records, λ={model.lam}\n'
        f'Recent rows carry more weight; old rows fade but are never deleted',
        fontsize=11, fontweight='bold',
    )

    # (a) Weight per row
    ax = axes[0]
    bar_colors = [C['navy']] * (n - 1) + [C['red']]
    ax.bar(range(n), weights, color=bar_colors, edgecolor='white', width=0.8)
    ax.set_xlabel('Row index (0 = oldest, rightmost = most recent)')
    ax.set_ylabel('EWMA Weight (normalised)')
    ax.set_title(f'(a) Weight per Blast Row\nNewest row weight = {weights[-1]:.4f}')
    # Annotate oldest and newest
    ax.annotate('Oldest\n(lowest weight)', xy=(0, weights[0]),
                xytext=(n*0.15, weights[-1]*0.4),
                arrowprops=dict(arrowstyle='->', color=C['grey']),
                fontsize=9, color=C['grey'])
    ax.annotate('Newest\n(highest weight)', xy=(n-1, weights[-1]),
                xytext=(n*0.6, weights[-1]*0.85),
                arrowprops=dict(arrowstyle='->', color=C['red']),
                fontsize=9, color=C['red'])

    # (b) Cumulative weight from newest
    ax2 = axes[1]
    cum_from_newest = np.cumsum(weights[::-1])
    ax2.plot(range(1, n+1), cum_from_newest, color=C['navy'], lw=2.5)
    ax2.axhline(0.90, color=C['red'], ls='--', lw=1.5, label='90% threshold')
    ax2.axhline(0.50, color=C['gold'], ls=':', lw=1.5, label='50% threshold')
    rows_90 = rows_capturing_90pct(weights)
    ax2.axvline(rows_90, color=C['red'], ls='--', lw=1.2, alpha=0.7)
    ax2.set_xlabel('Number of most recent rows')
    ax2.set_ylabel('Cumulative weight fraction')
    ax2.set_title(f'(b) Cumulative Weight from Newest Row\n90% captured by last {rows_90} rows')
    ax2.legend(fontsize=9); ax2.set_xlim(1, n); ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.15)

    # (c) Lambda comparison
    ax3 = axes[2]
    for lam_val, col in zip([0.1, 0.2, 0.3, 0.5],
                             [C['teal'], C['navy'], C['gold'], C['red']]):
        w_cmp = compute_ewma_weights(n, lam_val)
        lw    = 2.5 if abs(lam_val - model.lam) < 0.01 else 1.5
        ax3.plot(range(n), w_cmp, color=col, lw=lw,
                 label=f'λ={lam_val}' + (' (current)' if abs(lam_val - model.lam) < 0.01 else ''))
    ax3.set_xlabel('Row index (0 = oldest)')
    ax3.set_ylabel('Weight')
    ax3.set_title(f'(c) Effect of λ on Weight Decay\n(n={n} rows)')
    ax3.legend(fontsize=9); ax3.grid(alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'[PLOT]  {save_path}')


def plot_predictions_over_time(model: EWMAHybridModel,
                               results: list,
                               save_path: str = 'plots/ewma_update_trace.png') -> None:
    """
    Fig W2 — Show predicted vs actual PPV as new rows are added.
    Also shows how weighted R² and effective sample size evolve.
    """
    if not results:
        print("[PLOT]  No update results to plot — skip.")
        return

    n_updates = len(results)
    actuals   = [r['ppv_actual']    for r in results]
    preds     = [r['ppv_predicted'] for r in results]
    errors    = [r['error_pct']     if r['error_pct'] else 0 for r in results]
    r2_vals   = [r['R2_weighted']   for r in results]
    eff_ns    = [r['effective_n']   for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        f'EWMA Model — Live Update Trace (λ={model.lam})\n'
        f'{model.data.shape[0]} total rows in dataset after all updates',
        fontsize=11, fontweight='bold',
    )

    x = range(1, n_updates + 1)

    # (a) PPV actual vs predicted
    ax = axes[0]
    ax.plot(x, actuals, 'o-', color=C['navy'],  lw=2, ms=7, label='Actual PPV')
    ax.plot(x, preds,   's--', color=C['red'],  lw=2, ms=7, label='Predicted PPV')
    ax.fill_between(x,
                    [a * 0.80 for a in actuals],
                    [a * 1.20 for a in actuals],
                    alpha=0.10, color=C['green'], label='±20% band')
    ax.set_ylabel('PPV (mm/s)')
    ax.set_title('(a) Actual vs Predicted PPV — Each New Blast Record')
    ax.legend(fontsize=9)

    # (b) % prediction error
    ax2 = axes[1]
    bar_cols = [C['green'] if e <= 25 else C['red'] for e in errors]
    ax2.bar(x, errors, color=bar_cols, edgecolor='white')
    ax2.axhline(25, color=C['red'], ls='--', lw=1.5, label='25% threshold')
    ax2.set_ylabel('|% Error|')
    ax2.set_title('(b) Absolute Percentage Error per Update')
    ax2.legend(fontsize=9)

    # (c) Weighted R² and effective sample size
    ax3 = axes[2]
    ax3.plot(x, r2_vals, 'o-', color=C['navy'], lw=2, ms=6, label='Weighted R²')
    ax3.set_ylabel('Weighted R²', color=C['navy'])
    ax3b = ax3.twinx()
    ax3b.plot(x, eff_ns, 's--', color=C['gold'], lw=2, ms=6, label='Effective n')
    ax3b.set_ylabel('Effective Sample Size', color=C['gold'])
    ax3.set_xlabel('Update number (chronological blast order)')
    ax3.set_title('(c) Weighted R² and Effective Sample Size After Each Update')
    ax3.legend(loc='upper left', fontsize=9)
    ax3b.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'[PLOT]  {save_path}')


def export_ewma_predictions_csv(model: EWMAHybridModel, save_path: str = 'results/ewma_predictions_with_inputs.csv') -> pd.DataFrame:
    """
    Exports full CSV file containing input blast design features alongside
    actual PPV, physics prediction, EWMA predicted PPV, absolute error,
    percentage error, and EWMA sample weights.
    """
    df = model.data.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    avail = [c for c in model.feat_cols if c in df.columns]
    valid = df[avail + ['PPV', 'SD']].dropna()
    valid = valid.loc[:, ~valid.columns.duplicated()]

    X_s = model.sc.transform(valid[model.feat_cols].values)
    ewma_pred = np.exp(model.rf.predict(X_s))

    out_df = valid.copy()
    out_df['PPV_Actual'] = valid['PPV'].values
    out_df['PPV_EWMA_Pred'] = np.round(ewma_pred, 4)
    out_df['Absolute_Error_mm_s'] = np.round(np.abs(out_df['PPV_Actual'] - ewma_pred), 4)
    out_df['Percentage_Error_pct'] = np.round((out_df['Absolute_Error_mm_s'] / (np.abs(out_df['PPV_Actual']) + 1e-9)) * 100, 2)
    if len(model.weights) >= len(out_df):
        out_df['EWMA_Sample_Weight'] = np.round(model.weights[:len(out_df)], 6)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    out_df.to_csv(save_path, index=False)
    print(f"[EXPORT] Saved full input+prediction evaluation to → {save_path}")
    return out_df


def plot_weight_vs_error(model: EWMAHybridModel,
                         save_path: str = 'plots/ewma_weight_vs_error.png') -> None:
    """
    Fig W3 — For each row in current dataset, show its EWMA weight vs
    the model's current prediction error for that row.
    Demonstrates that recent rows (high weight) dominate the fit.
    """
    df  = model.data.copy()
    df  = df.loc[:, ~df.columns.duplicated()]
    avail = [c for c in model.feat_cols if c in df.columns]
    valid = df[avail + ['PPV', 'SD']].dropna()
    valid = valid.loc[:, ~valid.columns.duplicated()]

    if len(valid) < 3:
        return

    X_s    = model.sc.transform(valid[model.feat_cols].values)
    phys   = model._phys(valid['SD'].values.ravel())
    resid_pred = model.rf.predict(X_s)
    pred   = phys + resid_pred
    y      = valid['PPV'].values
    errors = np.abs(y - pred)
    w      = model.weights[:len(valid)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'EWMA Weight vs Prediction Error — λ={model.lam}',
        fontsize=11, fontweight='bold',
    )

    ax = axes[0]
    sc_ = ax.scatter(w, errors, c=range(len(w)), cmap='RdYlGn_r',
                     s=70, alpha=0.85, edgecolors='white')
    plt.colorbar(sc_, ax=ax, label='Row index (0=oldest, right=newest)')
    ax.set_xlabel('EWMA Weight (normalised)')
    ax.set_ylabel('Absolute Prediction Error (mm/s)')
    ax.set_title('(a) Weight vs Error per Row')

    ax2 = axes[1]
    row_idx = range(len(w))
    ax2.bar(row_idx, w,      color=C['navy'],  alpha=0.6, label='EWMA Weight')
    ax2b = ax2.twinx()
    ax2b.plot(row_idx, errors, color=C['red'], lw=1.8,
              ms=5, marker='o', label='Error (mm/s)')
    ax2.set_xlabel('Row index (0 = oldest)')
    ax2.set_ylabel('EWMA Weight', color=C['navy'])
    ax2b.set_ylabel('Error (mm/s)', color=C['red'])
    ax2.set_title('(b) Weight and Error by Row')
    ax2.legend(loc='upper left',  fontsize=9)
    ax2b.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'[PLOT]  {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — INTERACTIVE TERMINAL SESSION
# ══════════════════════════════════════════════════════════════════════════════

def print_result(result: dict, row_num: int) -> None:
    """Pretty-print the result from a single update."""
    e_str  = f"{result['error_mm']:.4f} mm/s  ({result['error_pct']:.2f}%)" \
             if result['error_mm'] is not None else "N/A (no actual PPV provided)"
    safety = "✅ LOW" if result['ppv_predicted'] < 5.0 else "🚨 HIGH"

    print(f"\n  {'─'*56}")
    print(f"  UPDATE {row_num} — Result")
    print(f"  {'─'*56}")
    print(f"  Physics prediction (k×SD^n)  : {result['ppv_physics']:.4f} mm/s")
    print(f"  EWMA Hybrid prediction        : {result['ppv_predicted']:.4f} mm/s")
    if result['ppv_actual'] and not np.isnan(result['ppv_actual']):
        print(f"  Actual PPV (measured)         : {result['ppv_actual']:.4f} mm/s")
        print(f"  Prediction error              : {e_str}")
    print(f"  Vibration risk                : {safety}")
    print(f"  {'─'*56}")
    print(f"  Dataset now                   : {result['n_rows']} rows total")
    print(f"  Newest row EWMA weight        : {result['newest_weight']:.6f}")
    print(f"  Oldest row EWMA weight        : {result['oldest_weight']:.6f}")
    print(f"  Effective sample size         : {result['effective_n']:.1f} rows")
    print(f"  90% weight in last            : {result['rows_90pct']} rows")
    print(f"  Weighted R² (after refit)     : {result['R2_weighted']:.4f}")
    print(f"  Unweighted R²                 : {result['R2_unweighted']:.4f}")
    print(f"  {'─'*56}")


def interactive_session(model: EWMAHybridModel) -> None:
    """
    Interactive terminal session — add blast records one at a time.
    Type 'quit' to exit, 'predict' to get a prediction without updating.
    """
    results   = []
    update_no = 0

    print(f"\n{'═'*60}")
    print(f"  EWMA HYBRID MODEL — INTERACTIVE SESSION")
    print(f"  lambda = {model.lam}  |  Dataset: {model.data.shape[0]} rows")
    print(f"  Type 'quit' to exit | 'predict' for prediction only")
    print(f"  Type 'plot' to save all current plots")
    print(f"  Type 'importance' to see feature importances")
    print(f"{'═'*60}")

    while True:
        print(f"\n  --- Enter new blast record (Update {update_no + 1}) ---")
        cmd = input("  Command or [Enter] to add row: ").strip().lower()

        if cmd == 'quit':
            print("\n  Session ended.")
            break

        if cmd == 'plot':
            plot_ewma_weights(model)
            if results:
                plot_predictions_over_time(model, results)
            plot_weight_vs_error(model)
            print("  Plots saved to plots/")
            continue

        if cmd == 'importance':
            fi = model.feature_importance()
            print("\n  Feature Importances (RF Residual Component):")
            for _, row in fi.head(10).iterrows():
                print(f"    {row['feature']:15s}: {row['importance']:.4f}")
            continue

        if cmd == 'predict':
            print("  [Predict Only — will NOT update the model]")
            try:
                distance  = float(input("  Distance (m)      : "))
                per_hole  = float(input("  Charge/hole Q (kg): "))
                n_holes   = float(input("  No. of holes      : "))
                depth     = float(input("  Depth (m)         : "))
                n_rows_v  = float(input("  No. of rows [4]   : ") or "4")
                spacing_v = float(input("  Spacing (m)  [4.5]: ") or "4.5")
                res = model.predict_only({'distance': distance, 'per_hole': per_hole,
                                          'n_holes': n_holes, 'depth': depth,
                                          'n_rows': n_rows_v, 'spacing': spacing_v})
                print(f"\n  SD = {res['SD']:.2f} m/kg⁰·⁵   TQ = {res['TQ']:.0f} kg")
                print(f"  Physics PPV   : {res['ppv_physics']:.4f} mm/s")
                print(f"  Hybrid PPV    : {res['ppv_predicted']:.4f} mm/s")
                risk = "✅ LOW" if res['ppv_predicted'] < 5.0 else "🚨 HIGH"
                print(f"  Risk level    : {risk}")
            except (ValueError, KeyboardInterrupt):
                print("  Input error — please enter numeric values.")
            continue

        # ── Regular update — collect all fields ───────────────────────────────
        try:
            distance  = float(input("  Distance (m)              : "))
            per_hole  = float(input("  Charge per hole Q (kg)    : "))
            n_holes   = float(input("  No. of holes              : "))
            depth     = float(input("  Depth of hole (m)         : "))
            n_rows_v  = float(input("  No. of rows        [4]    : ") or "4")
            spacing_v = float(input("  Spacing (m)        [4.5]  : ") or "4.5")
            seam      = input("  Seam location      [Unknown]: ").strip() or "Unknown"
            ppv_act   = input("  Actual PPV (mm/s)  [skip]   : ").strip()
            ppv_float = float(ppv_act) if ppv_act else np.nan

            new_row = {
                'distance': distance, 'per_hole': per_hole, 'n_holes': n_holes,
                'depth': depth, 'n_rows': n_rows_v, 'spacing': spacing_v,
                'seam': seam, 'ppv_actual': ppv_float,
            }

            result = model.update(new_row)
            update_no += 1
            results.append(result)
            print_result(result, update_no)

            # Auto-save model every 5 updates
            if update_no % 5 == 0:
                model.save('models/ewma_hybrid_model.pkl')
                print(f"  [AUTO-SAVE] Model saved after {update_no} updates.")

        except (ValueError, KeyboardInterrupt):
            print("  Input cancelled — row not added.")
            continue

    # Save on exit
    model.save('models/ewma_hybrid_model.pkl')
    plot_ewma_weights(model)
    if results:
        plot_predictions_over_time(model, results)
    plot_weight_vs_error(model)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — BATCH LEAVE-ONE-OUT EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_sequential(df: pd.DataFrame, lam: float = 0.2) -> pd.DataFrame:
    """
    Sequential Leave-One-Out evaluation to measure EWMA model quality.

    For each row i in the dataset (in chronological order):
      - Train EWMA model on rows 0 ... i-1
      - Predict PPV for row i (it has never been seen by the model)
      - Record actual vs predicted

    This is the most honest evaluation for a sequential model:
    it simulates real deployment where future blasts are unknown.
    Requires at least 5 rows to be meaningful.
    """
    df = engineer_features(df.copy()).dropna(subset=['PPV', 'SD']).reset_index(drop=True)
    n  = len(df)

    if n < 5:
        print("[EVAL]  Need at least 5 rows for sequential evaluation. Skipping.")
        return pd.DataFrame()

    records = []
    print(f"\n[EVAL]  Sequential LOO evaluation — {n} rows, lambda={lam}")
    print(f"        Training on rows 0..i, predicting row i+1")

    # Start training from row 4 (need minimum 4 rows to fit)
    min_train = 4
    for i in range(min_train, n):
        train_df = df.iloc[:i].copy()
        test_row  = df.iloc[i]

        model_tmp = EWMAHybridModel(lam=lam, n_estimators=200, max_depth=8)
        model_tmp.fit(train_df)

        sd_test    = float(test_row['SD'])
        phys_test  = float(physics_ppv(np.array([sd_test]))[0])
        avail_cols = [c for c in model_tmp.feat_cols if c in test_row.index]

        if len(avail_cols) == len(model_tmp.feat_cols):
            X_test   = test_row[model_tmp.feat_cols].values.astype(float).reshape(1, -1)
            X_test_s = model_tmp.sc.transform(X_test)
            ppv_pred_t = np.exp(float(model_tmp.rf.predict(X_test_s)[0]))
        else:
            ppv_pred_t = 1.0

        ppv_act_t  = float(test_row['PPV'])
        error_pct  = abs(ppv_act_t - ppv_pred_t) / (abs(ppv_act_t) + 1e-9) * 100

        # Compare with no-EWMA (unweighted GBR on log-PPV target)
        gbr_plain = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        avail     = [c for c in FEATURE_COLS if c in train_df.columns]
        train_df_clean = train_df.loc[:, ~train_df.columns.duplicated()]
        valid_tr  = train_df_clean[avail + ['PPV', 'SD']].dropna()
        valid_tr  = valid_tr.loc[:, ~valid_tr.columns.duplicated()]
        if len(valid_tr) >= 2:
            sc_p  = StandardScaler()
            X_tr_p = sc_p.fit_transform(valid_tr[avail].values)
            y_log_tr = np.log(np.maximum(valid_tr['PPV'].values, 1e-4))
            gbr_plain.fit(X_tr_p, y_log_tr)
            test_row_clean = test_row.to_frame().T.loc[:, ~test_row.to_frame().T.columns.duplicated()].iloc[0]
            X_test_p  = sc_p.transform(test_row_clean[avail].values.astype(float).reshape(1, -1))
            plain_pred = np.exp(float(gbr_plain.predict(X_test_p)[0]))
        else:
            plain_pred = 1.0

        plain_err_pct = abs(ppv_act_t - plain_pred) / (abs(ppv_act_t) + 1e-9) * 100

        records.append({
            'row_index'      : i,
            'n_train'        : i,
            'ppv_actual'     : round(ppv_act_t,   4),
            'ppv_ewma'       : round(ppv_pred_t,  4),
            'ppv_plain_gbr'  : round(plain_pred,  4),
            'error_pct_ewma' : round(error_pct,    2),
            'error_pct_plain': round(plain_err_pct,2),
            'seam'           : test_row.get('Seam_location', ''),
        })

        if i % 5 == 0:
            print(f"  Row {i:3d}: actual={ppv_act_t:.3f}  "
                  f"ewma={ppv_pred_t:.3f}  plain={plain_pred:.3f}  "
                  f"err_ewma={error_pct:.1f}%  err_plain={plain_err_pct:.1f}%")

    eval_df = pd.DataFrame(records)

    # Summary
    mean_ewma  = eval_df['error_pct_ewma'].mean()
    mean_plain = eval_df['error_pct_plain'].mean()
    print(f"\n  Sequential LOO Summary (rows {min_train}–{n-1}):")
    print(f"  Mean MAPE — EWMA (λ={lam})  : {mean_ewma:.2f}%")
    print(f"  Mean MAPE — Plain RF (equal w): {mean_plain:.2f}%")
    improvement = mean_plain - mean_ewma
    if improvement > 0:
        print(f"  EWMA improves MAPE by       : {improvement:.2f}% ✅")
    else:
        print(f"  EWMA vs Plain RF            : {improvement:.2f}% "
              f"(lambda may need tuning)")

    eval_df.to_csv('results/ewma_sequential_eval.csv', index=False)
    print(f"  Saved → results/ewma_sequential_eval.csv")

    # Plot sequential evaluation
    _plot_sequential_eval(eval_df, lam)
    return eval_df


def _plot_sequential_eval(eval_df: pd.DataFrame, lam: float) -> None:
    """Plot results of sequential evaluation — EWMA vs plain RF vs Physics."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f'Sequential Leave-One-Out Evaluation — EWMA (λ={lam}) vs Plain RF vs Physics\n'
        f'Each point = prediction for a blast not yet seen by the model',
        fontsize=11, fontweight='bold',
    )

    x = eval_df['row_index'].values

    ax = axes[0]
    ax.plot(x, eval_df['ppv_actual'],    'o-',  color=C['navy'],  lw=2, ms=6, label='Actual PPV')
    ax.plot(x, eval_df['ppv_ewma'],      's--', color=C['red'],   lw=2, ms=6, label=f'EWMA GBR (λ={lam})')
    if 'ppv_plain_gbr' in eval_df.columns:
        ax.plot(x, eval_df['ppv_plain_gbr'],  'D:',  color=C['gold'],  lw=1.5, ms=5, label='Plain GBR (equal weights)')
    ax.set_ylabel('PPV (mm/s)')
    ax.set_title('(a) Actual vs Predicted — Sequential Out-of-Sample')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.plot(x, eval_df['error_pct_ewma'],  's-',  color=C['red'],  lw=2, ms=6, label=f'EWMA MAPE')
    ax2.plot(x, eval_df['error_pct_plain'], 'D--', color=C['gold'], lw=1.5, ms=5, label='Plain RF MAPE')
    ax2.axhline(eval_df['error_pct_ewma'].mean(),  color=C['red'],  ls=':', lw=1.5,
                label=f"EWMA mean = {eval_df['error_pct_ewma'].mean():.1f}%")
    ax2.axhline(eval_df['error_pct_plain'].mean(), color=C['gold'], ls=':', lw=1.5,
                label=f"Plain mean = {eval_df['error_pct_plain'].mean():.1f}%")
    ax2.set_xlabel('Row index (chronological)')
    ax2.set_ylabel('Absolute % Error (MAPE)')
    ax2.set_title('(b) MAPE per Prediction — EWMA vs Plain RF')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('plots/ewma_sequential_eval.png')
    plt.close()
    print('[PLOT]  plots/ewma_sequential_eval.png')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='EWMA Hybrid PPV Model — Phase 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ewma_model.py                    # train + evaluate + plots
  python ewma_model.py --interactive      # interactive row-by-row session
  python ewma_model.py --lambda 0.3       # faster decay (recent rows dominate more)
  python ewma_model.py --eval-only        # only run sequential LOO evaluation
        """,
    )
    parser.add_argument('--lambda',       dest='lam',       type=float, default=0.2,
                        help='EWMA decay rate (default: 0.2). Higher = faster decay.')
    parser.add_argument('--interactive',  action='store_true',
                        help='Start interactive row-by-row session after fitting.')
    parser.add_argument('--eval-only',    action='store_true',
                        help='Skip training, run sequential LOO evaluation only.')
    parser.add_argument('--data',         type=str, default='results/real_field_data.csv',
                        help='Path to real field data CSV.')
    parser.add_argument('--n-estimators', type=int, default=300,
                        help='Number of RF trees in residual model (default: 300).')
    parser.add_argument('--max-depth',    type=int, default=10,
                        help='Max tree depth in RF (default: 10).')
    args = parser.parse_args()

    print('=' * 60)
    print('  EWMA-WEIGHTED HYBRID PPV MODEL — PHASE 2')
    print('  IIT (BHU) Varanasi | Mining Engineering | 2025-26')
    print('  Supervised by Dr. Satyabrata Behera')
    print('=' * 60)
    print(f'\n  Lambda (λ)      : {args.lam}')
    print(f'  Decay meaning   : each older row weighted by ×{1-args.lam:.1f}')
    print(f'  Physics: PPV = {K_LIT} × SD^({N_LIT})  [Pal Roy, 1993]')
    print(f'  k={K_LIT} and n={N_LIT} are always non-zero by physical law')

    # Load real data
    df = load_real_data(args.data)

    if args.eval_only:
        evaluate_sequential(df, lam=args.lam)
        return

    # Fit initial model
    print(f"\n[FIT]  Training EWMA Hybrid Model on {len(df)} real records ...")
    model = EWMAHybridModel(
        lam=args.lam,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    model.fit(df)

    # Generate baseline plots & export full predictions CSV
    print("\n[PLOTS]  Generating weight distribution plots ...")
    plot_ewma_weights(model)
    plot_weight_vs_error(model)
    export_ewma_predictions_csv(model)

    # Sequential LOO evaluation
    print("\n[EVAL]  Running sequential leave-one-out evaluation ...")
    eval_df = evaluate_sequential(df, lam=args.lam)

    # Save model
    model.save('models/ewma_hybrid_model.pkl')

    # Print final summary
    h = model.history[-1]
    print(f'\n{"="*60}')
    print(f'  EWMA HYBRID MODEL — SUMMARY')
    print(f'{"="*60}')
    print(f'  Lambda (λ)            : {args.lam}')
    print(f'  Rows in dataset       : {h["n_rows"]}')
    print(f'  Weighted R²           : {h["R2_weighted"]:.4f}')
    print(f'  Unweighted R²         : {h["R2_unweighted"]:.4f}')
    print(f'  MAE (unweighted)      : {h["MAE_unweighted"]:.4f} mm/s')
    print(f'  Effective sample size : {h["effective_n"]:.1f} rows')
    print(f'  90% weight in last    : {h["rows_90pct_weight"]} rows')
    print(f'  Model saved           : models/ewma_hybrid_model.pkl')
    print(f'  Plots saved           : plots/ewma_*.png')
    if not eval_df.empty:
        print(f'  Sequential MAPE (EWMA): {eval_df["error_pct_ewma"].mean():.2f}%')
        print(f'  Sequential MAPE (Plain): {eval_df["error_pct_plain"].mean():.2f}%')
    print(f'{"="*60}')

    # Start interactive session if requested
    if args.interactive:
        interactive_session(model)


if __name__ == '__main__':
    main()