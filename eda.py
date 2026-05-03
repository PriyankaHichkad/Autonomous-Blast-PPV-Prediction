"""
================================================================================
  PART 1 — DATA GENERATION, FEATURE ENGINEERING & EXPLORATORY DATA ANALYSIS
================================================================================
  Project   : Autonomous ML Framework for Blast-Induced Ground Vibration
  Institute : IIT (BHU) Varanasi | Department of Mining Engineering
  Run       : python part1_eda.py

  What this file does:
  ─────────────────────
  1. Creates the full real dataset from your field values (44 records)
  2. Generates realistic fake/precautionary data to expand to ~150 rows
     using the USBM physics law with site-specific noise
  3. Applies complete feature engineering:
       Q = Per_Hole (charge per delay)
       TQ = No_of_Holes × Q
       SD = Distance / sqrt(Q)        [USBM scaled distance]
       SD_TQ = Distance / sqrt(TQ)    [IS 6922 scaled distance]
       Stemming = (2/3) × Depth
       Ch_length = (1/3) × Depth
       + all log-space features
  4. Full EDA with power-law, linear, log-log correlations
  5. Distance vs Frequency, PPV distribution, heatmap, all scatter plots
  6. Saves clean combined dataset for Part 2
================================================================================
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

for d in ['plots', 'results', 'models', 'data']:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    'font.family'       : 'DejaVu Serif',
    'font.size'         : 11,
    'axes.titlesize'    : 12,
    'axes.labelsize'    : 11,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 200,
    'savefig.bbox'      : 'tight',
})
C = ['#1B4F72', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD', '#2E86C1', '#117A65']

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — BUILD THE EXACT FIELD DATASET FROM YOUR COLLECTED DATA
# ══════════════════════════════════════════════════════════════════════════════

def build_real_dataset():
    """
    Reconstruct the exact 44-record field dataset from Bhanegaon mine
    using the values you collected and shared.

    Column meanings:
    ─────────────────
    Date          : Day the blast occurred
    Blast_No      : Sequential reference number
    No_of_Holes   : Number of holes where explosives were placed
    Explosion     : Maximum overall charge used in the blast (kg)
    Per_Hole      : Average charge per hole (kg) — this is Q in USBM formula
    Depth         : Depth of each blast hole (m)
    No_of_Rows    : Number of rows of blast holes
    Spacing       : Distance between holes and between rows (m)
    Seam_location : Rock/seam type — different seams = different rock conditions
    Vibrometer    : Instrument ID (each blast has 5 vibrometers at diff distances)
    Distance      : Distance from blast to vibrometer (m)
    PPV           : Peak Particle Velocity — output variable (mm/s)
    Blast_Timing  : When the blast was fired
    Frequency     : Dominant frequency at measurement point (Hz)
    """
    rows = [
        # Date, Blast_No, Holes, Explosion, Per_Hole, Depth, Rows, Spacing, Seam, Vib, Dist, PPV, Timing, Freq
        ('31.10.25',1,100,350,35,6,4,5,'Seam Top-5',20697,400,4.826,'14:30',6.75),
        ('31.10.25',1,100,350,35,6,4,5,'Seam Top-5',20698,430,5.280,'14:30',7.875),
        ('31.10.25',1,100,350,35,6,4,5,'Seam Top-5',15335,470,np.nan,'14:30',np.nan),
        ('31.10.25',1,100,350,35,6,4,5,'Seam Top-5',15336,480,13.28,'14:30',9.125),
        ('31.10.25',1,100,350,35,6,4,5,'Seam Top-5',23314,np.nan,10.29,'14:30',9.0),
        ('01.11.25',2,45,65,35,3,3,5,'Seam-3',20697,310,8.818,'14:30',27.0),
        ('01.11.25',2,45,65,35,5,3,5,'Seam-3',20698,320,17.34,'14:30',10.63),
        ('01.11.25',2,45,65,35,5,3,5,'Seam-3',15335,np.nan,16.34,'14:30',10.75),
        ('01.11.25',2,45,65,35,5,3,5,'Seam-3',15336,450,3.543,'14:30',6.25),
        ('01.11.25',2,45,65,35,5,3,5,'Seam-3',23314,180,7.240,'14:30',24.63),
        ('01.11.25',2,70,65,35,5,3,5,'Seam-3',20697,np.nan,2.096,'14:30',33.0),
        ('01.11.25',2,70,65,35,5,3,5,'Seam-3',20698,np.nan,1.323,'14:30',15.75),
        ('01.11.25',2,70,65,35,5,3,5,'Seam-3',15335,np.nan,2.654,'14:30',14.88),
        ('01.11.25',2,70,65,35,5,3,5,'Seam-3',15336,np.nan,1.921,'14:30',6.25),
        ('01.11.25',2,70,65,35,5,3,5,'Seam-3',23314,np.nan,1.983,'14:30',8.875),
        ('02.11.25',1,70,32,90,6,5,5,'Seam Top-5',20697,250,2.009,'02:36',28.88),
        ('02.11.25',1,70,32,90,6,5,5,'Seam Top-5',20698,280,4.326,'02:36',14.25),
        ('02.11.25',1,70,32,90,6,5,5,'Seam Top-5',15335,300,2.958,'02:36',8.125),
        ('02.11.25',1,70,32,90,6,5,5,'Seam Top-5',15336,430,3.175,'02:36',28.75),
        ('02.11.25',1,70,32,90,6,5,5,'Seam Top-5',23314,np.nan,2.471,'02:36',7.125),
        ('03.11.25',1,100,250,25,6,4,4.5,'Seam 4 Floor',20697,450,1.637,'14:08',6.5),
        ('03.11.25',1,100,250,25,6,4,4.5,'Seam 4 Floor',20698,430,3.791,'14:08',6.0),
        ('03.11.25',1,100,250,25,6,4,4.5,'Seam 4 Floor',15335,450,3.753,'14:08',6.25),
        ('03.11.25',1,100,250,25,6,4,4.5,'Seam 4 Floor',15336,380,np.nan,'14:08',np.nan),
        ('03.11.25',1,100,250,25,6,4,4.5,'Seam 4 Floor',23314,400,3.331,'14:08',10.63),
        ('04.11.25',1,90,250,33,4.5,3,4,'Seam 5 Floor',20697,450,1.829,'15:14',5.75),
        ('04.11.25',1,90,250,33,4.5,3,4,'Seam 5 Floor',20698,470,1.576,'15:14',19.88),
        ('04.11.25',1,90,250,33,4.5,3,4,'Seam 5 Floor',15335,420,1.824,'15:14',7.625),
        ('04.11.25',1,90,250,33,4.5,3,4,'Seam 5 Floor',15336,440,np.nan,'15:14',np.nan),
        ('04.11.25',1,90,250,33,4.5,3,4,'Seam 5 Floor',23314,430,1.964,'15:14',5.75),
        ('05.11.25',1,100,250,35,4.5,4,4.5,'40 Coal Top',20697,480,np.nan,'15:22',np.nan),
        ('05.11.25',1,100,250,35,4.5,4,4.5,'40 Coal Top',20698,470,np.nan,'15:22',np.nan),
        ('05.11.25',1,100,250,35,4.5,4,4.5,'40 Coal Top',15335,430,np.nan,'15:22',np.nan),
        ('05.11.25',1,100,250,35,4.5,4,4.5,'40 Coal Top',15336,410,np.nan,'15:22',np.nan),
        ('05.11.25',1,100,250,35,4.5,4,4.5,'40 Coal Top',23314,380,np.nan,'15:22',np.nan),
        ('05.11.25',2,80,3600,43,5,3,5,'Seam Top-3',20697,510,1.979,'15:25',5.75),
        ('05.11.25',2,80,3600,43,5,3,5,'Seam Top-3',20698,480,2.310,'15:25',5.375),
        ('05.11.25',2,80,3600,43,5,3,5,'Seam Top-3',15335,460,2.994,'15:25',3.0),
        ('05.11.25',2,80,3600,43,5,3,5,'Seam Top-3',15336,430,5.061,'15:25',2.75),
        ('05.11.25',2,80,3600,43,5,3,5,'Seam Top-3',23314,420,7.377,'15:25',2.75),
        ('06.11.25',1,104,3600,50,4,7,4,'Seam 3 Floor',20697,460,1.693,'14:16',5.375),
        ('06.11.25',1,104,3600,50,4,7,4,'Seam 3 Floor',20698,480,1.354,'14:16',3.375),
        ('06.11.25',1,104,3600,50,4,7,4,'Seam 3 Floor',15335,480,1.405,'14:16',8.375),
        ('06.11.25',1,104,3600,50,4,7,4,'Seam 3 Floor',15336,470,1.690,'14:16',4.75),
        ('06.11.25',1,104,3600,50,4,7,4,'Seam 3 Floor',23314,450,1.796,'14:16',5.125),
        ('06.11.25',2,60,3600,45,6,4,4,'Seam 5 Floor',20697,480,3.364,'14:36',4.125),
        ('06.11.25',2,60,3600,45,6,4,4,'Seam 5 Floor',20698,500,3.713,'14:36',4.875),
        ('06.11.25',2,60,3600,45,6,4,4,'Seam 5 Floor',15335,510,3.078,'14:36',5.0),
        ('06.11.25',2,60,3600,45,6,4,4,'Seam 5 Floor',15336,490,3.812,'14:36',7.125),
        ('06.11.25',2,60,3600,45,6,4,4,'Seam 5 Floor',23314,470,4.378,'14:36',5.0),
        ('07.11.25',1,120,3600,25,3.5,5,4,'Seam 4 Floor',20697,380,5.747,'14:16',5.625),
        ('07.11.25',1,120,3600,25,3.5,5,4,'Seam 4 Floor',20698,420,4.582,'14:16',7.75),
        ('07.11.25',1,120,3600,25,3.5,5,4,'Seam 4 Floor',15335,440,7.081,'14:16',7.625),
        ('07.11.25',1,120,3600,25,3.5,5,4,'Seam 4 Floor',15336,470,5.120,'14:16',9.875),
        ('07.11.25',1,120,3600,25,3.5,5,4,'Seam 4 Floor',23314,400,3.305,'14:16',8.0),
        ('08.11.25',1,160,3600,35,6,5,4,'Seam 4 Floor',20697,480,1.786,'14:39',5.875),
        ('08.11.25',1,160,3600,35,6,5,4,'Seam 4 Floor',20698,490,1.503,'14:39',5.5),
        ('08.11.25',1,160,3600,35,6,5,4,'Seam 4 Floor',15335,520,1.759,'14:39',8.5),
        ('08.11.25',1,160,3600,35,6,5,4,'Seam 4 Floor',15336,540,1.769,'14:39',6.25),
        ('08.11.25',1,160,3600,35,6,5,4,'Seam 4 Floor',23314,500,2.577,'14:39',5.375),
    ]
    cols = ['Date','Blast_No','No_of_Holes','Explosion','Per_Hole','Depth',
            'No_of_Rows','Spacing','Seam_location','Vibrometer',
            'Distance','PPV','Blast_Timing','Frequency']
    #df = pd.DataFrame(rows, columns=cols)
    #df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%y', errors='coerce')
    
    df = pd.read_excel("raw_data/Blasting Data BTP.xlsx")
    df.columns = df.columns.str.strip()
    rename_map = {
        'Distance (meters)' : 'Distance',
        'No. of Hole'       : 'No_of_Holes',
        'Per Hole'          : 'Per_Hole',     
        'Depth of Hole'     : 'Depth',
        'No. of Row'        : 'No_of_Rows',
        'Spacing'           : 'Spacing',
        'Frequency'         : 'Frequency',
        'PPV'               : 'PPV',
        'Blast No.'         : 'Blast_No',
        'Blast Timing'      : 'Blast_Timing',
        'Seam location'     : 'Seam_location',
        'Vibromater'        : 'Vibrometer',
        'Explosion'         : 'Explosion',
    }
    df.rename(columns={k: v for k, v in rename_map.items()
                       if k in df.columns}, inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — GENERATE REALISTIC FAKE / PRECAUTIONARY DATA
# ══════════════════════════════════════════════════════════════════════════════

def generate_fake_data(df_real, n_fake=120):
    """
    Generate realistic fake blast records to expand the dataset from 44 to ~164 rows.

    WHY fake data at Part 1 stage?
    ────────────────────────────────
    With only 44 real records across 10 unique blast events, EDA visualisations
    are limited and model patterns are harder to observe. Precautionary fake data:
    - Uses the same column structure and physical constraints as real data
    - PPV generated using literature USBM law (k=650, n=-1.4) with 18% noise
    - All parameters sampled from the real data distributions
    - Clearly labelled (Source='Fake') so real vs fake is always distinguishable
    - Does NOT replace or distort real data — appended separately

    Column physical constraints used:
    ────────────────────────────────────
    Per_Hole (Q)    : 25–90 kg    (DGMS India limits for opencast blasting)
    No_of_Holes     : 40–180      (typical for Bhanegaon scale operations)
    Distance        : 150–700 m   (vibrometer placement range)
    Depth           : 3–7 m       (overburden hole depth range)
    Spacing         : 3–6 m       (burden × spacing design)
    No_of_Rows      : 2–6
    Frequency       : 2–35 Hz     (observed range in field data)
    Seam_location   : sampled from real unique values
    """
    k_lit = 650.0    # literature site constant for Bhanegaon / Wardha Valley
    n_lit = -1.4     # literature attenuation exponent

    seams     = df_real['Seam_location'].dropna().unique().tolist()
    vibs      = [20697, 20698, 15335, 15336, 23314]
    timings   = ['14:00','14:15','14:30','14:45','15:00','15:15','15:30']

    fake_rows = []
    for i in range(n_fake):
        Q       = np.clip(np.random.normal(40, 15), 25, 90)
        N       = int(np.clip(np.random.normal(90, 35), 40, 180))
        Depth   = np.clip(np.random.normal(5, 1), 3, 7)
        Rows    = int(np.clip(np.random.normal(4, 1.2), 2, 6))
        Spacing = np.clip(np.random.normal(4.5, 0.8), 3, 6)
        Expl    = round(N * Q * np.random.uniform(0.9, 1.1))
        Dist    = np.clip(np.random.normal(430, 90), 150, 700)
        Seam    = np.random.choice(seams)
        Vib     = np.random.choice(vibs)
        Timing  = np.random.choice(timings)
        Freq    = np.clip(np.random.exponential(8) + 2.5, 2, 35)
        Date    = pd.Timestamp('2025-11-01') + pd.Timedelta(days=np.random.randint(0, 30))

        SD      = Dist / np.sqrt(Q)
        noise   = np.random.normal(1.0, 0.18)
        PPV_raw = k_lit * SD**n_lit * noise
        PPV     = round(float(np.clip(PPV_raw, 0.5, 20)), 3)

        fake_rows.append({
            'Date': Date, 'Blast_No': i+100,
            'No_of_Holes': N, 'Explosion': Expl, 'Per_Hole': round(Q, 1),
            'Depth': round(Depth, 1), 'No_of_Rows': Rows, 'Spacing': round(Spacing, 1),
            'Seam_location': Seam, 'Vibrometer': Vib, 'Distance': round(Dist),
            'PPV': PPV, 'Blast_Timing': Timing, 'Frequency': round(Freq, 2),
            'Source': 'Fake'
        })

    return pd.DataFrame(fake_rows)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """
    Apply all domain-specific feature engineering.

    Per_Hole (Q) is the maximum charge per delay — directly used in USBM formula.
    Explosion is the total charge used in the entire blast event.

    Features derived:
    ─────────────────
    Q         = Per_Hole                  (alias for USBM notation)
    TQ        = No_of_Holes × Q           (total charge per blast)
    SD        = Distance / sqrt(Q)        (USBM scaled distance, Duvall 1962)
    SD_TQ     = Distance / sqrt(TQ)       (IS 6922 scaled distance)
    Stemming  = (2/3) × Depth             (standard blasting practice)
    Ch_length = (1/3) × Depth             (explosive column length)
    log_*     = natural log of each quantity (linearises power-law relationships)
    """
    df = df.copy()

    # Core physics features
    df['Q']        = df['Per_Hole']
    df['TQ']       = df['No_of_Holes'] * df['Q']
    df['SD']       = df['Distance'] / np.sqrt(df['Q'].clip(lower=1e-9))
    df['SD_TQ']    = df['Distance'] / np.sqrt(df['TQ'].clip(lower=1e-9))
    df['Stemming'] = df['Depth'] * (2/3)
    df['Ch_length']= df['Depth'] * (1/3)

    # Log-space features (needed because PPV follows power law)
    for feat, src in [
        ('log_PPV', 'PPV'), ('log_SD', 'SD'), ('log_SD_TQ', 'SD_TQ'),
        ('log_D', 'Distance'), ('log_Q', 'Q'), ('log_TQ', 'TQ'),
        ('log_N', 'No_of_Holes'), ('log_Depth', 'Depth'),
    ]:
        if src in df.columns:
            df[feat] = np.log(df[src].clip(lower=1e-9))

    return df


def print_dataset_summary(df, label=""):
    ppv = df['PPV'].dropna()
    print(f"\n{'═'*60}")
    print(f"  DATASET SUMMARY  {label}")
    print(f"{'═'*60}")
    print(f"  Total records   : {len(df)}")
    print(f"  Valid PPV rows  : {len(ppv)}")
    print(f"  PPV range       : {ppv.min():.3f} – {ppv.max():.3f} mm/s")
    print(f"  PPV mean ± std  : {ppv.mean():.3f} ± {ppv.std():.3f} mm/s")
    print(f"  PPV > 5 mm/s    : {(ppv>5).sum()} ({(ppv>5).mean()*100:.1f}%)")
    print(f"  Distance range  : {df['Distance'].min():.0f}–{df['Distance'].max():.0f} m")
    print(f"  Q range         : {df['Per_Hole'].min():.0f}–{df['Per_Hole'].max():.0f} kg")
    print(f"  SD range        : {df['SD'].min():.1f}–{df['SD'].max():.1f} m/kg⁰·⁵")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — POWER-LAW R² UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def power_r2(x, y):
    """Fit y = a * x^b via log-log OLS. Returns (a, b, R²)."""
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 5:
        return None, None, np.nan
    lx, ly = np.log(x[ok]), np.log(y[ok])
    b, loga = np.polyfit(lx, ly, 1)
    a  = np.exp(loga)
    yp = a * x[ok]**b
    ss_res = np.sum((y[ok] - yp)**2)
    ss_tot = np.sum((y[ok] - y[ok].mean())**2)
    r2 = max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return a, b, r2


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — EDA PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_univariate_correlation(df):
    """
    Fig 3.1 — Bar chart of Power-Law R², Pearson R², Spearman ρ for each
    feature vs PPV. Power-law R² is the correct metric because PPV follows
    a power-law decay with distance and charge (USBM empirical law).
    """
    features = {
        'Distance'    : 'Distance (m)',
        'Q'           : 'Charge per Hole Q (kg)',
        'No_of_Holes' : 'No. of Holes',
        'Depth'       : 'Depth of Hole (m)',
        'TQ'          : 'Total Charge TQ (kg)',
        'SD'          : 'Scaled Distance SD (m/kg⁰·⁵)',
        'SD_TQ'       : 'Scaled Distance SD_TQ',
        'Spacing'     : 'Spacing (m)',
        'Frequency'   : 'Frequency (Hz)',
    }
    features = {k:v for k,v in features.items() if k in df.columns}

    rows = []
    for feat, label in features.items():
        x  = df[feat].values.astype(float)
        y  = df['PPV'].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[ok], y[ok]
        if len(xv) < 5:
            continue
        rp, _ = stats.pearsonr(xv, yv)
        rs, _ = stats.spearmanr(xv, yv)
        a, b, r2p = power_r2(xv, yv)
        rows.append({'Feature': label, 'N': ok.sum(),
                     'Pearson_R2': round(rp**2, 4),
                     'PowerLaw_R2': round(r2p, 4) if not np.isnan(r2p) else 0,
                     'Spearman_rho': round(rs, 4),
                     'a': round(a, 4) if a else np.nan,
                     'b': round(b, 4) if b else np.nan})

    res = pd.DataFrame(rows).sort_values('PowerLaw_R2', ascending=False)
    res.to_csv('results/univariate_correlation.csv', index=False)
    print("\n  UNIVARIATE CORRELATION TABLE")
    print(res[['Feature','N','Pearson_R2','PowerLaw_R2','Spearman_rho','a','b']].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle('Fig. 3.1: Univariate Correlation — Blast Parameters vs PPV',
                 fontsize=12, fontweight='bold')
    for ax, col, title in zip(axes,
        ['Pearson_R2', 'PowerLaw_R2', 'Spearman_rho'],
        ['Pearson R²', 'Power-Law R²', 'Spearman ρ']):
        vals  = np.abs(res[col].values)
        short = [f.split(' (')[0].split('Dist')[0][:18] for f in res['Feature'].values]
        cols_ = [C[0] if v >= 0.3 else C[2] if v >= 0.1 else '#BDC3C7' for v in vals]
        bars  = ax.barh(short, vals, color=cols_, edgecolor='white', height=0.62)
        ax.set_xlabel(title); ax.set_title(title, fontweight='bold')
        ax.axvline(0.3, color='red', ls='--', lw=1.2, alpha=0.7)
        for bar, v in zip(bars, res[col].values):
            ax.text(abs(v) + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{v:+.3f}', va='center', fontsize=8.5)
        ax.set_xlim(0, 1.15)
    plt.tight_layout()
    plt.savefig('plots/Fig3_1_univariate_correlation.png')
    plt.close()
    print('[PLOT] Fig3_1_univariate_correlation.png')
    return res, features


def plot_scatter_vs_ppv(df, features, real_only=False):
    """
    Fig 3.2 — Scatter of each parameter vs PPV with power-law fit (linear axes).
    Fig 3.3 — Same on log-log axes (linearises the power-law relationship).
    """
    items  = list(features.items())
    ncols  = 3
    nrows  = (len(items) + ncols - 1) // ncols
    suffix = ' (Real)' if real_only else ''

    for fignum, use_log, fname, title_base in [
        ('3.2', False, 'plots/Fig3_2_scatter_vs_ppv.png',
         'Fig. 3.2: Parameter vs PPV — Power-Law Fits'),
        ('3.3', True,  'plots/Fig3_3_loglog_vs_ppv.png',
         'Fig. 3.3: Log-Log Axes — Power-Law Linearisation'),
    ]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
        fig.suptitle(title_base + suffix, fontsize=12, fontweight='bold', y=1.01)
        flat = axes.flatten()

        for i, (feat, label) in enumerate(items):
            ax = flat[i]
            x  = df[feat].values.astype(float)
            y  = df['PPV'].values.astype(float)
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            xv, yv = x[ok], y[ok]

            color = C[0] if not real_only else C[2]
            ax.scatter(xv, yv, color=color, s=40, alpha=0.70,
                       edgecolors='white', zorder=3)
            a, b, r2 = power_r2(xv, yv)
            if a is not None and not np.isnan(r2):
                xl = np.linspace(xv.min(), xv.max(), 250)
                ax.plot(xl, a * xl**b, color=C[1], lw=2.2,
                        label=f'a={a:.3f}, b={b:.3f}\nR²={r2:.3f}')
                ax.legend(fontsize=8)
            if use_log:
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.grid(True, which='both', alpha=0.15)
            ax.set_xlabel(label, fontsize=8.5)
            ax.set_ylabel('PPV (mm/s)', fontsize=8.5)
            ax.set_title(feat, fontweight='bold', fontsize=10)
            ax.tick_params(labelsize=8)

        for j in range(i + 1, len(flat)):
            flat[j].set_visible(False)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f'[PLOT] {fname.split("/")[1]}')


def plot_scaled_distance(df):
    """Fig 3.4 — SD and SD_TQ vs PPV on log-log, verifying the USBM law."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Fig. 3.4: Scaled Distance vs PPV — USBM Power Law Verification',
                 fontsize=12, fontweight='bold')
    for ax, feat, label, c in [
        (axes[0], 'SD',    'SD = D/√Q  (m/kg⁰·⁵)',    C[0]),
        (axes[1], 'SD_TQ', 'SD_TQ = D/√TQ  (m/kg⁰·⁵)', C[2]),
    ]:
        x  = df[feat].values.astype(float)
        y  = df['PPV'].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        xv, yv = x[ok], y[ok]
        ax.scatter(xv, yv, color=c, s=55, alpha=0.8, edgecolors='white', zorder=3)
        a, b, r2 = power_r2(xv, yv)
        if a is not None:
            xl = np.linspace(xv.min(), xv.max(), 300)
            ax.plot(xl, a * xl**b, color=C[1], lw=2.5,
                    label=f'PPV = {a:.4f}·{feat}^{b:.4f}\nR² = {r2:.4f}')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(label); ax.set_ylabel('PPV (mm/s)')
        ax.set_title(feat, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.18)
    plt.tight_layout()
    plt.savefig('plots/Fig3_4_scaled_distance_vs_ppv.png')
    plt.close()
    print('[PLOT] Fig3_4_scaled_distance_vs_ppv.png')


def plot_distance_frequency(df):
    """
    Fig 3.5 — Distance vs Frequency coloured by PPV.
    Frequency decreases with distance (seismic wave attenuation).
    """
    sub = df[['Distance', 'Frequency', 'PPV']].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Fig. 3.5: Distance vs Frequency — Vibration Characteristics',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    sc = ax.scatter(sub['Distance'], sub['Frequency'],
                    c=sub['PPV'], cmap='RdYlGn_r', s=65, alpha=0.85,
                    edgecolors='white', zorder=3)
    plt.colorbar(sc, ax=ax, label='PPV (mm/s)')
    xv = sub['Distance'].values; yv = sub['Frequency'].values
    ok = (xv > 0) & (yv > 0)
    a, b, r2 = power_r2(xv[ok], yv[ok])
    if a is not None:
        xl = np.linspace(xv[ok].min(), xv[ok].max(), 200)
        ax.plot(xl, a * xl**b, color=C[1], lw=2,
                label=f'f = {a:.3f}·D^{b:.3f}  R²={r2:.3f}')
        ax.legend(fontsize=9)
    ax.set_xlabel('Distance (m)'); ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Distance vs Frequency (coloured by PPV)')

    ax2 = axes[1]
    ax2.scatter(sub['Frequency'], sub['PPV'],
                color=C[3], s=65, alpha=0.85, edgecolors='white')
    xv2 = sub['Frequency'].values; yv2 = sub['PPV'].values
    ok2 = (xv2 > 0) & (yv2 > 0)
    a2, b2, r2b = power_r2(xv2[ok2], yv2[ok2])
    if a2 is not None:
        xl2 = np.linspace(xv2[ok2].min(), xv2[ok2].max(), 200)
        ax2.plot(xl2, a2 * xl2**b2, color=C[0], lw=2,
                 label=f'R²={r2b:.3f}')
        ax2.legend(fontsize=9)
    ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('PPV (mm/s)')
    ax2.set_title('Frequency vs PPV (power-law fit)')

    plt.tight_layout()
    plt.savefig('plots/Fig3_5_distance_frequency.png')
    plt.close()
    print('[PLOT] Fig3_5_distance_frequency.png')


def plot_correlation_heatmap(df):
    """Fig 3.6 — Full Pearson correlation matrix of all numeric features."""
    cols = ['Distance', 'Q', 'No_of_Holes', 'Depth', 'TQ', 'SD', 'SD_TQ',
            'Spacing', 'No_of_Rows', 'Frequency', 'PPV']
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Pearson Correlation Coefficient'})
    ax.set_title('Fig. 3.6: Pearson Correlation Matrix — All Blast Parameters',
                 fontsize=12, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig('plots/Fig3_6_correlation_heatmap.png')
    plt.close()
    print('[PLOT] Fig3_6_correlation_heatmap.png')


def plot_ppv_distribution(df):
    """
    Fig 3.7 — PPV histogram, box plot, Q-Q plot, and seam-wise PPV boxplots.
    """
    ppv = df['PPV'].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Fig. 3.7: PPV Distribution Analysis', fontsize=12, fontweight='bold')

    # Histogram
    ax = axes[0, 0]
    ax.hist(ppv, bins=18, color=C[0], edgecolor='white', alpha=0.85)
    ax.axvline(ppv.mean(),   color=C[1], lw=2, ls='--',
               label=f'Mean = {ppv.mean():.2f} mm/s')
    ax.axvline(ppv.median(), color=C[2], lw=2, ls='-.',
               label=f'Median = {ppv.median():.2f} mm/s')
    ax.axvline(5.0, color='red', lw=1.8, ls=':',
               label='IS 6922 limit = 5 mm/s')
    ax.set_xlabel('PPV (mm/s)'); ax.set_ylabel('Count')
    ax.set_title('PPV Histogram'); ax.legend(fontsize=9)

    # Box plot by seam
    ax2 = axes[0, 1]
    if 'Seam_location' in df.columns:
        seam_data = [df[df['Seam_location'] == s]['PPV'].dropna().values
                     for s in df['Seam_location'].dropna().unique()]
        seam_labs = df['Seam_location'].dropna().unique().tolist()
        bp = ax2.boxplot(seam_data, patch_artist=True, labels=seam_labs)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(C[i % len(C)]); box.set_alpha(0.75)
        ax2.set_xlabel('Seam Location'); ax2.set_ylabel('PPV (mm/s)')
        ax2.set_title('PPV by Seam Location')
        ax2.set_xticklabels(seam_labs, rotation=25, ha='right', fontsize=8)
    else:
        bp = ax2.boxplot(ppv, patch_artist=True)
        bp['boxes'][0].set_facecolor(C[0])

    # Q-Q plot
    ax3 = axes[1, 0]
    (osm, osr), (slope, intercept, r) = stats.probplot(ppv, dist='norm')
    ax3.scatter(osm, osr, color=C[0], s=45, alpha=0.8, edgecolors='white')
    xl = np.array([min(osm), max(osm)])
    ax3.plot(xl, slope * xl + intercept, color=C[1], lw=2,
             label=f'R²={r**2:.3f}')
    ax3.set_xlabel('Theoretical Quantiles'); ax3.set_ylabel('Sample Quantiles')
    ax3.set_title('Normal Q-Q Plot'); ax3.legend(fontsize=9)

    # Distance vs PPV scatter
    ax4 = axes[1, 1]
    sub = df[['Distance', 'PPV']].dropna()
    ax4.scatter(sub['Distance'], sub['PPV'],
                color=C[4], s=45, alpha=0.7, edgecolors='white')
    xv = sub['Distance'].values; yv = sub['PPV'].values
    ok = (xv > 0) & (yv > 0)
    a, b, r2 = power_r2(xv[ok], yv[ok])
    if a:
        xl = np.linspace(xv[ok].min(), xv[ok].max(), 200)
        ax4.plot(xl, a * xl**b, color=C[1], lw=2.2,
                 label=f'PPV={a:.3f}·D^{b:.3f}  R²={r2:.3f}')
        ax4.legend(fontsize=9)
    ax4.set_xlabel('Distance (m)'); ax4.set_ylabel('PPV (mm/s)')
    ax4.set_title('Distance vs PPV')

    plt.tight_layout()
    plt.savefig('plots/Fig3_7_ppv_distribution.png')
    plt.close()
    print('[PLOT] Fig3_7_ppv_distribution.png')

    # Print stats
    print(f"\n  PPV DESCRIPTIVE STATISTICS  (n={len(ppv)})")
    for stat, val in [('Mean',ppv.mean()),('Median',ppv.median()),
                      ('Std Dev',ppv.std()),('CV %',ppv.std()/ppv.mean()*100),
                      ('Min',ppv.min()),('Max',ppv.max()),
                      ('Skewness',ppv.skew()),('Kurtosis',ppv.kurtosis())]:
        print(f"  {stat:12s} : {val:.4f}")
    print(f"  IS 6922 >5   : {(ppv>5).sum()} records ({(ppv>5).mean()*100:.1f}%)")


def plot_real_vs_fake(df_combined):
    """Fig 3.8 — Compare real vs fake data distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Fig. 3.8: Real vs Fake Data — Parameter Distribution Comparison',
                 fontsize=12, fontweight='bold')
    feats = ['PPV', 'Distance', 'Q', 'No_of_Holes', 'TQ', 'SD']
    for ax, feat in zip(axes.flatten(), feats):
        if feat not in df_combined.columns:
            ax.set_visible(False); continue
        real_ = df_combined[df_combined['Source'] == 'Real'][feat].dropna()
        fake_ = df_combined[df_combined['Source'] == 'Fake'][feat].dropna()
        ax.hist(real_, bins=12, alpha=0.75, color=C[0], density=True,
                edgecolor='white', label=f'Real (n={len(real_)})')
        ax.hist(fake_, bins=25, alpha=0.50, color=C[3], density=True,
                edgecolor='white', label=f'Fake (n={len(fake_)})')
        ax.set_xlabel(feat); ax.set_ylabel('Density')
        ax.set_title(feat, fontweight='bold'); ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig('plots/Fig3_8_real_vs_fake.png')
    plt.close()
    print('[PLOT] Fig3_8_real_vs_fake.png')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 60)
    print('  PART 1 — EDA & FEATURE ENGINEERING')
    print('  IIT (BHU) Varanasi | Mining Engineering')
    print('=' * 60)

    # Build real dataset
    df_raw = build_real_dataset()
    print(f'\n[DATA]  Raw records built: {len(df_raw)}')

    # Apply feature engineering to real data
    df_real_all = engineer_features(df_raw)

    # Valid records only (have both Distance and PPV)
    df_real = df_real_all.dropna(subset=['Distance', 'PPV']).copy()
    df_real['Source'] = 'Real'
    print(f'[DATA]  Valid records (Distance+PPV): {len(df_real)}')

    print_dataset_summary(df_real, "(Real Field Data)")

    # Generate fake data
    df_fake_raw = generate_fake_data(df_real, n_fake=120)
    df_fake = engineer_features(df_fake_raw)
    print(f'[DATA]  Fake records generated: {len(df_fake)}')

    # Combine
    df_combined = pd.concat([df_real, df_fake], ignore_index=True)
    print_dataset_summary(df_combined, "(Real + Fake Combined)")

    # EDA plots on real data
    print('\n[EDA]  Running univariate analysis...')
    corr_df, features = plot_univariate_correlation(df_real)

    print('[EDA]  Plotting individual scatter plots...')
    plot_scatter_vs_ppv(df_real, features)
    plot_scaled_distance(df_real)
    plot_distance_frequency(df_real)
    plot_correlation_heatmap(df_real)
    plot_ppv_distribution(df_real)
    plot_real_vs_fake(df_combined)

    # Save outputs
    df_real.to_csv('results/real_field_data.csv', index=False)
    df_combined.to_csv('results/combined_real_fake.csv', index=False)
    df_real.to_excel('data/clean_field_data.xlsx', index=False)
    df_combined.to_excel('data/combined_real_fake.xlsx', index=False)

    print('\n[SAVE]  results/real_field_data.csv')
    print('[SAVE]  results/combined_real_fake.csv')
    print('\n  PART 1 COMPLETE — All figures saved to plots/')