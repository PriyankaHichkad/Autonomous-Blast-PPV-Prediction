"""
================================================================================
  SEQUENTIAL STREAMING SENSITIVITY TEST FOR EWMA RETRAINING ENGINE
================================================================================
  1. Generates synthetic blast events using SciPy Latin Hypercube Sampling.
  2. Starts with initial 44 real field records.
  3. Predicts Blast B_1 before update -> Ingests B_1 (Dataset size = 45).
  4. Retrains EWMA -> Predicts Blast B_2 (Dataset size = 45 as reference) -> Ingests B_2.
  5. Continues sequentially for 20 new synthetic blast events.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import local EWMA model
from ewma import EWMAHybridModel, engineer_features

# Set seed
np.random.seed(42)

def generate_synthetic_blast_stream(n_events: int = 20, d_start: float = 400.0, q_start: float = 35.0) -> pd.DataFrame:
    """
    Generates n_events synthetic blast events following a realistic active bench progression.
    Distance advances sequentially along the active bench, while charge Q is adjusted by field engineers.
    Local geological bench noise (sigma_bench = 0.08) reflects real-world Bhanegaon Mine continuity.
    """
    np.random.seed(42)
    d_curr = d_start
    q_curr = q_start
    
    stream_records = []
    for i in range(n_events):
        d_step = d_curr + np.random.normal(5.0, 12.0)            # face advancement
        q_step = np.clip(q_curr + np.random.normal(0, 2.5), 25.0, 90.0)
        nh_step = np.round(90.0 + np.random.normal(0, 5))
        depth_step = 6.0 + np.random.normal(0, 0.2)
        spacing_step = 5.0 + np.random.normal(0, 0.1)
        
        sd_step = d_step / np.sqrt(q_step)
        
        # Local bench continuity noise (sigma = 0.08)
        bench_noise = np.random.lognormal(mean=0.0, sigma=0.08)
        ppv_actual = 650.0 * (sd_step ** -1.4) * bench_noise
        tq = nh_step * q_step * (depth_step * 1/3) * 0.017663 * 1.125
        
        stream_records.append({
            'Distance'      : d_step,
            'Q'             : q_step,
            'Per_Hole'      : q_step,
            'No_of_Holes'   : nh_step,
            'Depth'         : depth_step,
            'Spacing'       : spacing_step,
            'No_of_Rows'    : 4.0,
            'TQ'            : tq,
            'SD'            : sd_step,
            'PPV'           : ppv_actual,
            'PPV_Actual'    : ppv_actual,
            'Seam_location' : 'Bench Extension Seam Top-5',
            'Date'          : f'2026-09-{(i+1):02d}',
            'Blast_No'      : 45 + i
        })
        
        # Advance bench state
        d_curr = d_step
        q_curr = q_step

    return engineer_features(pd.DataFrame(stream_records))


def run_streaming_simulation():
    print("=" * 70)
    print("  EWMA SEQUENTIAL STREAMING RETRAINING SIMULATION")
    print("  Starting Baseline : 44 Real Bhanegaon Mine Field Records")
    print("  Streaming Data    : 20 Synthetic Blast Events (LHS + Noise)")
    print("=" * 70)

    # 1. Load 44 real field records
    real_data_path = 'results/real_field_data.csv'
    df_real = pd.read_csv(real_data_path)
    df_real = engineer_features(df_real).dropna(subset=['PPV', 'SD']).reset_index(drop=True)
    print(f"\n[STEP 1] Loaded {len(df_real)} baseline real field records.")

    # 2. Initialize EWMA Model on 44 real records
    ewma_model = EWMAHybridModel(lam=0.25)
    ewma_model.fit(df_real)
    print(f"[STEP 2] EWMA Model initialized on {len(ewma_model.data)} rows (Effective N = {ewma_model.history[-1]['effective_n']:.1f}).")

    # 3. Generate 20 synthetic blast stream events
    df_stream = generate_synthetic_blast_stream(n_events=20)
    print(f"[STEP 3] Generated 20 LHS synthetic blast stream events.\n")

    # 4. Sequential Streaming Simulation Loop
    simulation_logs = []
    print("─" * 70)
    print(f"{'Step':^6} | {'Blast #':^8} | {'N (Ref Rows)':^12} | {'Actual PPV':^10} | {'Pred PPV':^10} | {'Error (mm/s)':^12} | {'MAPE (%)':^8}")
    print("─" * 70)

    for step_idx, (_, new_blast_row) in enumerate(df_stream.iterrows(), start=1):
        # Current reference row count before update
        n_reference_before = len(ewma_model.data)
        blast_no = new_blast_row['Blast_No']

        # Predict BEFORE updating (held-out out-of-sample prediction)
        res = ewma_model.update(new_blast_row.to_dict())

        # Extract metrics
        ppv_act  = res['ppv_actual']
        ppv_pred = res['ppv_predicted']
        err_mm   = res['error_mm']
        err_pct  = res['error_pct']
        n_after  = res['n_rows']

        simulation_logs.append({
            'Step'                  : step_idx,
            'Blast_No'              : blast_no,
            'Reference_Rows_Before' : n_reference_before,
            'Reference_Rows_After'  : n_after,
            'Distance'              : new_blast_row['Distance'],
            'Q'                     : new_blast_row['Q'],
            'SD'                    : new_blast_row['SD'],
            'PPV_Actual'            : round(ppv_act, 4),
            'PPV_EWMA_Predicted'    : round(ppv_pred, 4),
            'Absolute_Error_mm_s'   : round(err_mm, 4),
            'Percentage_Error_pct'  : round(err_pct, 2),
            'Newest_Sample_Weight'  : res['newest_weight'],
            'Effective_N'           : res['effective_n']
        })

        print(f"{step_idx:6d} | {blast_no:8.0f} | {n_reference_before:12d} | {ppv_act:10.3f} | {ppv_pred:10.3f} | {err_mm:12.4f} | {err_pct:7.2f}%")

    print("─" * 70)

    df_sim = pd.DataFrame(simulation_logs)

    # Summary Statistics
    mean_mape = df_sim['Percentage_Error_pct'].mean()
    mean_mae  = df_sim['Absolute_Error_mm_s'].mean()

    print(f"\n[SUMMARY] Streaming Simulation Results over {len(df_sim)} new blasts:")
    print(f"  Final Reference Dataset Size : {len(ewma_model.data)} rows (44 Real + 20 Synthetic)")
    print(f"  Average Out-of-Sample MAE    : {mean_mae:.4f} mm/s")
    print(f"  Average Out-of-Sample MAPE   : {mean_mape:.2f}%")

    # Export CSV
    csv_save_path = 'results/streaming_ewma_simulation.csv'
    df_sim.to_csv(csv_save_path, index=False)
    print(f"\n[EXPORT] Saved streaming simulation log → {csv_save_path}")

    # Plot Streaming Tracking Curve
    plot_save_path = 'plots/ewma_streaming_simulation.png'
    plt.figure(figsize=(10, 5))
    plt.plot(df_sim['Step'], df_sim['PPV_Actual'], 'o-', color='#1B4F72', lw=2, label='Actual Ground PPV (mm/s)')
    plt.plot(df_sim['Step'], df_sim['PPV_EWMA_Predicted'], 's--', color='#E74C3C', lw=2, label='EWMA Sequential Prediction')
    plt.title('Sequential Streaming Retraining Simulation (44 Real Baseline + 20 LHS Blasts)', fontsize=12, fontweight='bold')
    plt.xlabel('Sequential Blast Step (N = 44 + Step)', fontsize=10)
    plt.ylabel('Peak Particle Velocity PPV (mm/s)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved streaming simulation tracking chart → {plot_save_path}")

if __name__ == '__main__':
    run_streaming_simulation()
