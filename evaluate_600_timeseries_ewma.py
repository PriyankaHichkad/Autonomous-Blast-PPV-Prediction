"""
================================================================================
  600-ROW TIME-SERIES SEQUENTIAL EVALUATION OF EWMA-GBR MODEL
================================================================================
  1. Generates 600 synthetic blast records (150 sequential blast events x 4 stations)
     using SciPy Latin Hypercube Sampling with time-series bench drift.
  2. Evaluates the EWMA-GBR model sequentially as a chronological stream:
     - Holds out blast event t for testing.
     - Predicts held-out vibration values.
     - Retrains model with updated EWMA recency weights (lambda = 0.25).
  3. Computes time-series R2, MAE, MAPE, and exports tracking CSV & plot.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import local EWMA model engine
from ewma import EWMAHybridModel, engineer_features

# Set random seed for reproducibility
np.random.seed(42)

def generate_600_timeseries_synth_data() -> pd.DataFrame:
    """
    Generates 600 synthetic blast records (150 unique blast events, 4 geophone stations each)
    with SciPy Latin Hypercube Sampling and time-series random-walk bench drift.
    """
    n_events = 150
    n_stations = 4
    total_records = n_events * n_stations

    # 1. SciPy Latin Hypercube Sampling across 5 parameters
    sampler = LatinHypercube(d=5, seed=42)
    sample_matrix = sampler.random(n=n_events)

    # Base operating parameters per blast event
    d_base   = 200.0 + sample_matrix[:, 0] * (550.0 - 200.0)
    q        = 25.0  + sample_matrix[:, 1] * (110.0 - 25.0)
    n_holes  = np.round(30.0 + sample_matrix[:, 2] * (140.0 - 30.0))
    depth    = 3.0   + sample_matrix[:, 3] * (10.0 - 3.0)
    spacing  = 3.5   + sample_matrix[:, 4] * (6.5 - 3.5)

    # Time-series random walk bench drift simulating progressive rock mass changes
    drift = np.cumsum(np.random.normal(0, 0.03, size=n_events))

    records = []
    station_offsets = [0.0, 45.0, 90.0, 140.0]  # Geophone distances from blast

    for ev_idx in range(n_events):
        event_id = ev_idx + 1
        q_ev     = q[ev_idx]
        nh_ev    = n_holes[ev_idx]
        dep_ev   = depth[ev_idx]
        sp_ev    = spacing[ev_idx]
        drift_ev = drift[ev_idx]

        for st_idx, offset in enumerate(station_offsets):
            d_station = d_base[ev_idx] + offset
            sd = d_station / np.sqrt(q_ev)
            tq = nh_ev * q_ev * (dep_ev * 1/3) * 0.017663 * 1.125

            # USBM physics + time-series bench drift + random noise
            local_noise = np.random.lognormal(mean=drift_ev, sigma=0.12)
            ppv_act = 650.0 * (sd ** -1.4) * local_noise

            records.append({
                'Event_ID'     : event_id,
                'Station_ID'   : st_idx + 1,
                'Distance'     : d_station,
                'Q'            : q_ev,
                'Per_Hole'     : q_ev,
                'No_of_Holes'  : nh_ev,
                'Depth'        : dep_ev,
                'Spacing'      : sp_ev,
                'No_of_Rows'   : 4.0,
                'TQ'           : tq,
                'SD'           : sd,
                'PPV'          : ppv_act,
                'PPV_Actual'   : ppv_act,
                'Seam_location': f'Bench-Seam Phase {(ev_idx // 30) + 1}',
                'Date'         : f'2026-{(ev_idx // 30) + 1:02d}-{(ev_idx % 30) + 1:02d}'
            })

    df = pd.DataFrame(records)
    return engineer_features(df)


def run_600_timeseries_evaluation():
    print("=" * 75)
    print("  600-ROW TIME-SERIES CHRONOLOGICAL EVALUATION (EWMA-GBR MODEL)")
    print("  Dataset   : 600 Records (150 Sequential Blast Events x 4 Geophones)")
    print("  Sampling  : SciPy Latin Hypercube Sampling (LHS) + Time-Series Drift")
    print("  Decay Rate: Lambda = 0.25 (Event-Level Memory Horizon)")
    print("=" * 75)

    # 1. Generate 600-row time-series dataset
    df_600 = generate_600_timeseries_synth_data()
    print(f"\n[STEP 1] Generated {len(df_600)} synthetic time-series records across 150 blast events.")

    # 2. Sequential Event-Grouped Leave-One-Out Evaluation
    events = df_600['Event_ID'].unique()
    min_train_events = 10  # Initial baseline memory (40 rows)

    model = EWMAHybridModel(lam=0.25)

    eval_logs = []
    print("\n[STEP 2] Running Sequential Time-Series Evaluation ...")

    for e_idx in range(min_train_events, len(events)):
        train_events = events[:e_idx]
        test_event   = events[e_idx]

        train_df = df_600[df_600['Event_ID'].isin(train_events)].copy()
        test_df  = df_600[df_600['Event_ID'] == test_event].copy()

        # Fit EWMA model on historical time-series data up to event t-1
        sys.stdout = open(os.devnull, 'w')
        model.fit(train_df)
        sys.stdout = sys.__stdout__

        # Predict all 4 geophone stations for held-out event t
        for _, test_row in test_df.iterrows():
            pred_dict = model.predict_only(test_row.to_dict())
            ppv_act  = test_row['PPV']
            ppv_pred = pred_dict['ppv_predicted']
            err_mm   = abs(ppv_act - ppv_pred)
            err_pct  = err_mm / (abs(ppv_act) + 1e-9) * 100

            eval_logs.append({
                'Event_ID'           : test_event,
                'Station_ID'         : test_row['Station_ID'],
                'Historical_Events'  : len(train_events),
                'Historical_Rows'    : len(train_df),
                'Distance'           : test_row['Distance'],
                'Q'                  : test_row['Q'],
                'SD'                 : test_row['SD'],
                'PPV_Actual'         : round(ppv_act, 4),
                'PPV_EWMA_Predicted' : round(ppv_pred, 4),
                'Absolute_Error_mm_s': round(err_mm, 4),
                'Percentage_Error_pct': round(err_pct, 2)
            })

    eval_df = pd.DataFrame(eval_logs)

    # 3. Overall Time-Series Evaluation Metrics
    y_true = eval_df['PPV_Actual'].values
    y_pred = eval_df['PPV_EWMA_Predicted'].values

    r2_ts   = r2_score(y_true, y_pred)
    mae_ts  = mean_absolute_error(y_true, y_pred)
    mape_ts = np.mean(eval_df['Percentage_Error_pct'])

    print("\n" + "=" * 75)
    print("  TIME-SERIES EVALUATION SUMMARY METRICS (560 HELD-OUT TEST SAMPLES)")
    print("=" * 75)
    print(f"  Sequential Out-of-Sample R² Score  : {r2_ts:.4f}  (R² = {r2_ts:.3f})")
    print(f"  Sequential Out-of-Sample MAE       : {mae_ts:.4f} mm/s")
    print(f"  Sequential Out-of-Sample MAPE      : {mape_ts:.2f}%")
    print(f"  Effective Memory Horizon (N_eff)   : {effective_sample_size_from_lam(0.25):.1f} events")
    print("=" * 75)

    # 4. Export CSV Log
    csv_path = 'results/ewma_600_timeseries_eval.csv'
    os.makedirs('results', exist_ok=True)
    eval_df.to_csv(csv_path, index=False)
    print(f"\n[EXPORT] Saved time-series evaluation CSV log → {csv_path}")

    # 5. Generate Time-Series Plot
    plot_path = 'plots/ewma_600_timeseries_eval.png'
    os.makedirs('plots', exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Subplot 1: Actual vs EWMA Predicted PPV
    axes[0].plot(eval_df.index, eval_df['PPV_Actual'], '-', color='#1F4E78', alpha=0.7, lw=1.2, label='Actual PPV (Geophones)')
    axes[0].plot(eval_df.index, eval_df['PPV_EWMA_Predicted'], '--', color='#D9534F', alpha=0.85, lw=1.2, label='EWMA Time-Series Pred (λ=0.25)')
    axes[0].set_ylabel('PPV (mm/s)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Sequential Time-Series Out-of-Sample Evaluation on 600 Synthetic Blasts (R² = {r2_ts:.3f})', fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    # Subplot 2: MAE per Event Group
    event_mae = eval_df.groupby('Event_ID')['Absolute_Error_mm_s'].mean()
    axes[1].plot(event_mae.index, event_mae.values, 'o-', color='#27AE60', ms=3, lw=1.2, label='Mean Absolute Error per Blast Event')
    axes[1].axhline(mae_ts, color='#C0392B', linestyle=':', lw=1.5, label=f'Overall MAE = {mae_ts:.3f} mm/s')
    axes[1].set_xlabel('Sequential Blast Event ID (t = 11 to 150)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('MAE (mm/s)', fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved publication-quality tracking chart → {plot_path}")

def effective_sample_size_from_lam(lam: float) -> float:
    return (2.0 - lam) / lam

if __name__ == '__main__':
    run_600_timeseries_evaluation()
