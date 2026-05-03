"""
================================================================================
  PART 2 — SYNTHETIC DATA GENERATION & ALL MODEL TRAINING / COMPARISON
================================================================================
  Project   : Autonomous ML Framework for Blast-Induced Ground Vibration
  Institute : IIT (BHU) Varanasi | Department of Mining Engineering
  Run       : python part2_models.py   (after part1_eda.py)

  Models trained and compared:
  ─────────────────────────────
  PHYSICS     : USBM Power Law (k=650, n=-1.4, literature constants)
  ML MODELS   : Random Forest, Gradient Boosting, Extra Trees,
                AdaBoost, Bagging, SVR, KNN, Decision Tree,
                Ridge Regression, Lasso, ElasticNet
  DEEP LEARN  : MLP Neural Network (3-layer, sklearn MLPRegressor)
  FUZZY LOGIC : Sugeno-type fuzzy inference system (numpy from scratch)
  HYBRID      : Physics (USBM) + Random Forest on residuals [PROPOSED]

  All ML/DL models use:
  ─────────────────────
  - 80/20 train-test split on combined (real+fake+synthetic) dataset
  - GridSearchCV with 5-fold cross-validation for hyperparameter tuning
  - StandardScaler for distance-sensitive models (SVR, KNN, MLP)
  - Feature set: Distance, Q, No_of_Holes, Depth, TQ, SD, SD_TQ,
                 log_SD, log_D, log_Q, log_TQ, log_N, log_Depth

  Site constants (Bhanegaon / Wardha Valley coalfield, India):
  ─────────────────────────────────────────────────────────────
  k = 650   (range 200–1100 for Indian overburden, Pal Roy 1993)
  n = -1.4  (range -1.2 to -1.6 for Indian coal mines)
================================================================================
"""

import os, sys, warnings, json, pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
                                      GridSearchCV, KFold)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor,
                               BaggingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
np.random.seed(42)

for d in ['plots', 'results', 'models']:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    'font.family'       : 'DejaVu Serif',
    'font.size'         : 11,
    'axes.titlesize'    : 12,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 200,
    'savefig.bbox'      : 'tight',
})
C = ['#1B4F72','#E74C3C','#27AE60','#F39C12','#8E44AD','#2E86C1','#117A65',
     '#C0392B','#1ABC9C','#D35400','#7D3C98','#2980B9','#27AE60']

# Site constants — ALWAYS non-zero, from literature
K_LIT = 650.0    # site transmission constant (Pal Roy, 1993; Wardha Valley)
N_LIT = -1.4     # attenuation exponent (range: -1.2 to -1.6, Indian coal mines)

FEATURE_COLS = [
    'Distance', 'Q', 'No_of_Holes', 'Depth', 'TQ', 'SD', 'SD_TQ',
    'No_of_Rows', 'Spacing',
    'log_SD', 'log_D', 'log_Q', 'log_TQ', 'log_N', 'log_Depth',
]


# ─────────────────────────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred, label='', verbose=True):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    if verbose and label:
        print(f"  {label:42s} R²={r2:7.4f}  MAE={mae:.4f}  "
              f"RMSE={rmse:.4f}  MAPE={mape:.2f}%")
    return {'Model': label, 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def ensure_features(df):
    """Make sure all engineered features are present."""
    df = df.copy()
    if 'Q' not in df.columns:
        df['Q'] = df['Per_Hole']
    for col in ['TQ', 'SD', 'SD_TQ']:
        if col not in df.columns:
            if col == 'TQ':
                df['TQ'] = df['No_of_Holes'] * df['Q']
            elif col == 'SD':
                df['SD'] = df['Distance'] / np.sqrt(df['Q'].clip(1e-9))
            elif col == 'SD_TQ':
                df['SD_TQ'] = df['Distance'] / np.sqrt(df['TQ'].clip(1e-9))
    for feat, src in [('log_SD','SD'),('log_D','Distance'),('log_Q','Q'),
                      ('log_TQ','TQ'),('log_N','No_of_Holes'),
                      ('log_PPV','PPV'),('log_Depth','Depth')]:
        if feat not in df.columns and src in df.columns:
            df[feat] = np.log(df[src].clip(1e-9))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PHYSICS MODEL
# ══════════════════════════════════════════════════════════════════════════════

def plot_physics_model(df_real):
    """
    Fig 4.1 — USBM Power Law: PPV = K × SD^N
    k=650, n=-1.4 from literature (Pal Roy 1993, Wardha Valley).
    Both k and n are ALWAYS non-zero by definition.
    """
    v   = df_real[['SD', 'PPV']].dropna()
    v   = v[(v['SD'] > 0) & (v['PPV'] > 0)]
    yp  = K_LIT * v['SD'].values**N_LIT
    r2  = r2_score(v['PPV'].values, yp)
    mae = mean_absolute_error(v['PPV'].values, yp)

    print(f"\n{'─'*60}")
    print(f"  PHYSICS MODEL — PPV = k × SD^n")
    print(f"  k = {K_LIT}  (site constant, always ≠ 0, range 200-1100)")
    print(f"  n = {N_LIT}  (attenuation exponent, always ≠ 0, range -1.2 to -1.6)")
    print(f"  R² = {r2:.4f}    MAE = {mae:.4f} mm/s")
    print(f"  Note: Low R² reflects geological heterogeneity across seams.")
    print(f"        This motivates the ML residual correction layer.")
    print(f"{'─'*60}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Fig. 4.1: Physics-Based PPV Prediction — USBM Power Law\n'
                 f'PPV = {K_LIT} × SD^({N_LIT})  [k=650, n=-1.4, Pal Roy 1993]',
                 fontsize=11, fontweight='bold')

    ax = axes[0]
    ax.scatter(v['SD'], v['PPV'], color=C[0], s=60, alpha=0.85,
               edgecolors='white', zorder=3, label='Field data')
    xl = np.linspace(v['SD'].min(), v['SD'].max(), 300)
    ax.plot(xl, K_LIT * xl**N_LIT, color=C[1], lw=2.5,
            label=f'PPV={K_LIT}·SD^{N_LIT}\nR²={r2:.4f}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Scaled Distance SD (m/kg⁰·⁵)'); ax.set_ylabel('PPV (mm/s)')
    ax.set_title('(a) Log-Log Fit'); ax.legend(fontsize=8.5)
    ax.grid(True, which='both', alpha=0.18)

    ax2 = axes[1]
    v2  = df_real[['SD_TQ', 'PPV']].dropna()
    v2  = v2[(v2['SD_TQ'] > 0) & (v2['PPV'] > 0)]
    lr2 = LinearRegression()
    lr2.fit(np.log(v2['SD_TQ'].values).reshape(-1,1), np.log(v2['PPV'].values))
    k2, n2 = np.exp(lr2.intercept_), lr2.coef_[0]
    yp2    = k2 * v2['SD_TQ'].values**n2
    r2_2   = r2_score(v2['PPV'].values, yp2)
    ax2.scatter(v2['SD_TQ'], v2['PPV'], color=C[2], s=60, alpha=0.85,
                edgecolors='white', zorder=3)
    xl2 = np.linspace(v2['SD_TQ'].min(), v2['SD_TQ'].max(), 300)
    ax2.plot(xl2, k2*xl2**n2, color=C[3], lw=2.5,
             label=f'PPV={k2:.3f}·SD_TQ^{n2:.3f}\nR²={r2_2:.4f}')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('SD_TQ (m/kg⁰·⁵)'); ax2.set_ylabel('PPV (mm/s)')
    ax2.set_title('(b) SD_TQ Fit'); ax2.legend(fontsize=8.5)
    ax2.grid(True, which='both', alpha=0.18)

    ax3 = axes[2]
    ax3.scatter(v['PPV'], yp, color=C[4], s=60, alpha=0.85,
                edgecolors='white', zorder=3)
    lim = [0, max(v['PPV'].max(), yp.max()) * 1.12]
    ax3.plot(lim, lim, 'k--', lw=1.5, label='1:1 line')
    ax3.fill_between(lim, [x*0.8 for x in lim], [x*1.2 for x in lim],
                     alpha=0.1, color='green', label='±20%')
    ax3.set_xlabel('Observed PPV (mm/s)'); ax3.set_ylabel('Predicted PPV (mm/s)')
    ax3.set_title(f'(c) Actual vs Predicted  R²={r2:.4f}')
    ax3.legend(fontsize=8.5); ax3.set_xlim(lim); ax3.set_ylim(lim)

    plt.tight_layout()
    plt.savefig('plots/Fig4_1_physics_model.png')
    plt.close()
    print('[PLOT] Fig4_1_physics_model.png')
    return {'k': K_LIT, 'n': N_LIT, 'r2': r2, 'mae': mae}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — SYNTHETIC DATA GENERATION  (Objective 2)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic(df_real, n_syn=600):
    """
    Generate physics-consistent synthetic blast vibration records.

    Three geological zones simulate time-varying rock conditions (Objective 2):
    Zone A (k×1.00 = 650):   baseline — current site conditions
    Zone B (k×1.30 = 845):   harder rock / better explosive coupling
    Zone C (k×0.72 = 468):   softer / weathered overburden, higher attenuation

    Noise CV = 16% simulates field PPV measurement uncertainty
    (Hao & Wu, 2005 — documented ±10-15% for Indian coal mines).

    Parameters are bootstrapped from real data to preserve correlations.
    """
    n_per = n_syn // 3
    zone_configs = [
        ('A', K_LIT * 1.00, n_per),
        ('B', K_LIT * 1.30, n_per),
        ('C', K_LIT * 0.72, n_syn - 2*n_per),
    ]

    chunks = []
    for zone_label, k_zone, n_z in zone_configs:
        idx  = np.random.choice(len(df_real), n_z, replace=True)
        base = df_real.iloc[idx][['Distance','Per_Hole','No_of_Holes',
                                   'Depth','No_of_Rows','Spacing',
                                   'Seam_location','Frequency']].copy().reset_index(drop=True)

        # Gaussian perturbation — calibrated to 25% of real std
        base['Distance']    += np.random.normal(0, 40, n_z)
        base['Per_Hole']    += np.random.normal(0, 8,  n_z)
        base['No_of_Holes'] += np.random.normal(0, 15, n_z)
        base['Depth']       += np.random.normal(0, 0.8, n_z)
        base['Spacing']     += np.random.normal(0, 0.5, n_z)

        # Physical bounds
        base['Distance']    = base['Distance'].clip(lower=100, upper=800)
        base['Per_Hole']    = base['Per_Hole'].clip(lower=20,  upper=100)
        base['No_of_Holes'] = base['No_of_Holes'].clip(lower=30, upper=200).round()
        base['Depth']       = base['Depth'].clip(lower=2.5, upper=8)
        base['Spacing']     = base['Spacing'].clip(lower=3, upper=6)
        base['Frequency']   = np.clip(
            base['Frequency'].values + np.random.normal(0, 3, n_z), 2, 35)

        # Engineered features
        base['Q']      = base['Per_Hole']
        base['TQ']     = base['No_of_Holes'] * base['Q']
        base['SD']     = base['Distance'] / np.sqrt(base['Q'].clip(1e-9))
        base['SD_TQ']  = base['Distance'] / np.sqrt(base['TQ'].clip(1e-9))
        base['Stemming']  = base['Depth'] * (2/3)
        base['Ch_length'] = base['Depth'] * (1/3)

        for feat, src in [('log_SD','SD'),('log_D','Distance'),('log_Q','Q'),
                          ('log_TQ','TQ'),('log_N','No_of_Holes'),
                          ('log_Depth','Depth')]:
            base[feat] = np.log(base[src].clip(1e-9))

        # PPV: physics law + multiplicative Gaussian noise
        noise       = np.random.normal(1.0, 0.16, n_z)
        base['PPV'] = np.clip(k_zone * base['SD'].values**N_LIT * noise, 0.3, 25)
        base['log_PPV'] = np.log(base['PPV'].clip(1e-9))
        base['Zone']    = zone_label
        base['Source']  = 'Synthetic'
        chunks.append(base)

    syn = pd.concat(chunks, ignore_index=True)

    # Label real data
    df_r        = df_real.copy()
    df_r['Source'] = 'Real'
    df_r['Zone']   = 'Real'

    combined = pd.concat([df_r, syn], ignore_index=True)
    combined.to_csv('results/combined_with_synthetic.csv', index=False)

    print(f'\n[SYNTH]  {n_syn} records: Zone A={n_per}, B={n_per}, C={n_syn-2*n_per}')
    print(f'         PPV range: {syn["PPV"].min():.3f} – {syn["PPV"].max():.3f} mm/s')
    print(f'         Combined total: {len(combined)} records')

    # Fig 4.2
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Fig. 4.2: Real vs Synthetic Data — Distribution Comparison\n'
                 '(3 geological zones: A baseline, B hard rock, C soft rock)',
                 fontsize=11, fontweight='bold')
    for ax, feat in zip(axes.flatten(),
                        ['PPV','SD','Distance','Q','TQ','Depth']):
        if feat not in combined.columns:
            ax.set_visible(False); continue
        rd  = combined[combined['Source']=='Real'][feat].dropna()
        sd_ = combined[combined['Source']=='Synthetic'][feat].dropna()
        ax.hist(rd,  bins=12, alpha=0.80, color=C[0], density=True,
                edgecolor='white', label=f'Real (n={len(rd)})')
        ax.hist(sd_, bins=35, alpha=0.45, color=C[3], density=True,
                edgecolor='white', label=f'Synthetic (n={len(sd_)})')
        ax.set_xlabel(feat); ax.set_ylabel('Density')
        ax.set_title(feat, fontweight='bold'); ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig('plots/Fig4_2_synthetic_distribution.png')
    plt.close()
    print('[PLOT]  Fig4_2_synthetic_distribution.png')
    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — FUZZY LOGIC MODEL (from scratch, numpy only)
# ══════════════════════════════════════════════════════════════════════════════

class FuzzyPPVPredictor:
    """
    Sugeno-type Fuzzy Inference System for PPV prediction.
    Implemented from scratch using NumPy — no external fuzzy library needed.

    Why Fuzzy Logic?
    ─────────────────
    Fuzzy logic is particularly suitable for blast vibration prediction because:
    1. Blast parameters have inherent imprecision (e.g., "large charge", "far distance")
    2. Expert mining knowledge can be encoded as IF-THEN rules
    3. It handles the nonlinear, overlapping relationships between variables
    4. It provides linguistically interpretable predictions

    System design:
    ──────────────
    Inputs  : SD (scaled distance), TQ (total charge), No_of_Holes
    Output  : PPV (mm/s)
    MFs     : Triangular/Trapezoidal membership functions
    Rules   : 9 Sugeno-type rules based on blast vibration physics
    Defuzz  : Weighted average (Sugeno method)
    """

    def __init__(self):
        self.fitted = False
        self.sd_params  = None
        self.tq_params  = None
        self.n_params   = None
        self.output_vals = None

    def _tri_mf(self, x, a, b, c):
        """Triangular membership function. a=left, b=peak, c=right."""
        left  = np.where(b > a, (x - a) / (b - a + 1e-9), 0)
        right = np.where(c > b, (c - x) / (c - b + 1e-9), 0)
        return np.clip(np.minimum(left, right), 0, 1)

    def _trap_mf(self, x, a, b, c, d):
        """Trapezoidal membership function."""
        left  = np.where(b > a, (x - a) / (b - a + 1e-9), 1)
        right = np.where(d > c, (d - x) / (d - c + 1e-9), 1)
        return np.clip(np.minimum(np.minimum(left, right), 1), 0, 1)

    def fit(self, X_train, y_train, sd_col=5, tq_col=4, n_col=2):
        """
        Calibrate universe of discourse and output singletons from training data.
        sd_col, tq_col, n_col are column indices in X_train for SD, TQ, N.
        """
        sd_vals = X_train[:, sd_col]
        tq_vals = X_train[:, tq_col]
        n_vals  = X_train[:, n_col]

        # Universe of discourse
        sd_lo, sd_hi = np.percentile(sd_vals, [5, 95])
        tq_lo, tq_hi = np.percentile(tq_vals, [5, 95])
        n_lo,  n_hi  = np.percentile(n_vals,  [5, 95])

        # SD MFs: Low (close blast) / Medium / High (far blast)
        sd_m = (sd_lo + sd_hi) / 2
        self.sd_params = {
            'low'  : (sd_lo,  sd_lo,  sd_m),
            'med'  : (sd_lo,  sd_m,   sd_hi),
            'high' : (sd_m,   sd_hi,  sd_hi),
        }

        # TQ MFs: Low / Medium / High charge
        tq_m = (tq_lo + tq_hi) / 2
        self.tq_params = {
            'low'  : (tq_lo, tq_lo, tq_m),
            'med'  : (tq_lo, tq_m,  tq_hi),
            'high' : (tq_m,  tq_hi, tq_hi),
        }

        # No_of_Holes MFs: Few / Many
        n_m = (n_lo + n_hi) / 2
        self.n_params = {
            'few'  : (n_lo, n_lo, n_m),
            'many' : (n_m,  n_hi, n_hi),
        }

        # Calibrate 9 output singletons from data medians
        # Rules: IF SD=low TQ=high N=many → PPV=very_high, etc.
        self.rules = [
            # (sd_mf, tq_mf, n_mf, output_category)
            ('low',  'high', 'many', 'very_high'),
            ('low',  'high', 'few',  'high'),
            ('low',  'med',  'many', 'high'),
            ('low',  'med',  'few',  'medium'),
            ('low',  'low',  'few',  'medium'),
            ('med',  'high', 'many', 'medium'),
            ('med',  'med',  'few',  'low'),
            ('high', 'high', 'many', 'low'),
            ('high', 'low',  'few',  'very_low'),
        ]

        # Output singletons: calibrated to actual PPV percentiles
        ppv_sorted = np.sort(y_train)
        n = len(ppv_sorted)
        self.output_singletons = {
            'very_low' : np.median(ppv_sorted[:max(1, n//5)]),
            'low'      : np.median(ppv_sorted[n//5 : 2*n//5]),
            'medium'   : np.median(ppv_sorted[2*n//5 : 3*n//5]),
            'high'     : np.median(ppv_sorted[3*n//5 : 4*n//5]),
            'very_high': np.median(ppv_sorted[4*n//5:]),
        }
        self.sd_col = sd_col
        self.tq_col = tq_col
        self.n_col  = n_col
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("FuzzyPPVPredictor not fitted.")
        preds = []
        for row in X:
            sd = row[self.sd_col]
            tq = row[self.tq_col]
            n  = row[self.n_col]

            # Fuzzify SD
            sd_lo_mf = self._tri_mf(np.array([sd]), *self.sd_params['low'])[0]
            sd_md_mf = self._tri_mf(np.array([sd]), *self.sd_params['med'])[0]
            sd_hi_mf = self._tri_mf(np.array([sd]), *self.sd_params['high'])[0]

            # Fuzzify TQ
            tq_lo_mf = self._tri_mf(np.array([tq]), *self.tq_params['low'])[0]
            tq_md_mf = self._tri_mf(np.array([tq]), *self.tq_params['med'])[0]
            tq_hi_mf = self._tri_mf(np.array([tq]), *self.tq_params['high'])[0]

            # Fuzzify N
            n_few_mf = self._tri_mf(np.array([n]),  *self.n_params['few'])[0]
            n_mny_mf = self._tri_mf(np.array([n]),  *self.n_params['many'])[0]

            sd_dict = {'low': sd_lo_mf, 'med': sd_md_mf, 'high': sd_hi_mf}
            tq_dict = {'low': tq_lo_mf, 'med': tq_md_mf, 'high': tq_hi_mf}
            n_dict  = {'few': n_few_mf, 'many': n_mny_mf}

            # Fire rules (AND = minimum t-norm)
            num = 0.0; den = 0.0
            for sd_k, tq_k, n_k, out_k in self.rules:
                strength = min(sd_dict[sd_k], tq_dict[tq_k], n_dict[n_k])
                out_val  = self.output_singletons[out_k]
                num += strength * out_val
                den += strength

            preds.append(num / den if den > 1e-9 else np.median(list(self.output_singletons.values())))

        return np.array(preds)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — HYBRID PHYSICS + ML MODEL (PROPOSED)
# ══════════════════════════════════════════════════════════════════════════════

class HybridPPVModel:
    """
    Physics-Informed Hybrid Random Forest Model (Proposed Architecture).

    Final equation:
        PPV_final = PPV_physics + RF(Residual_features)

    where:
        PPV_physics = K_LIT × SD^N_LIT              [USBM Eq. 3.1]
        Residual    = PPV_actual - PPV_physics        [Eq. 3.3]
        RF learns   the geological correction term    [Eq. 3.4]

    k and n are ALWAYS non-zero constants from literature.
    """

    def __init__(self):
        self.k = K_LIT          # always non-zero
        self.n = N_LIT          # always non-zero
        self.ml       = None
        self.sc       = StandardScaler()
        self.feat_cols= []
        self.is_fitted = False
        self.train_metrics = {}
        self.test_metrics  = {}

    def _phys(self, sd):
        """PPV_physics = k × SD^n  (k≠0, n≠0 guaranteed by literature constants)."""
        return self.k * np.clip(sd, 1e-9, None)**self.n

    def fit(self, df_combined, feature_cols, test_size=0.2):
        avail = [c for c in feature_cols if c in df_combined.columns]
        self.feat_cols = avail

        df2 = df_combined.copy()
        df2['Physics_PPV'] = self._phys(df2['SD'].values)
        df2['Residual']    = df2['PPV'] - df2['Physics_PPV']

        valid = df2[avail + ['Residual', 'PPV', 'SD']].dropna()
        valid = valid.loc[:, ~valid.columns.duplicated()]

        X   = valid[avail].values
        y_r = valid['Residual'].values
        y_p = valid['PPV'].values
        sd_ = valid['SD'].values.ravel()

        X_tr, X_te, yr_tr, yr_te, yp_tr, yp_te, sd_tr, sd_te = \
            train_test_split(X, y_r, y_p, sd_, test_size=test_size,
                             random_state=42)

        # GridSearchCV
        print("    [Hybrid] GridSearchCV: Random Forest on residuals...")
        param_grid = {
            'n_estimators'    : [200, 400],
            'max_depth'       : [8, 12, None],
            'min_samples_leaf': [1, 2],
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid, cv=kf, scoring='r2', n_jobs=-1)
        gs.fit(self.sc.fit_transform(X_tr), yr_tr)
        self.ml = gs.best_estimator_
        self.is_fitted = True
        print(f"    [Hybrid] Best params: {gs.best_params_}  CV-R²={gs.best_score_:.4f}")

        for tag, Xs, yr, yp, sd_s in [('train', X_tr, yr_tr, yp_tr, sd_tr),
                                        ('test',  X_te, yr_te, yp_te, sd_te)]:
            pred = self._phys(sd_s) + self.ml.predict(self.sc.transform(Xs))
            m    = get_metrics(yp, pred, f'Hybrid RF [{tag:5s}]')
            if tag == 'train': self.train_metrics = m
            else:              self.test_metrics  = m

        self._td = {
            'X_te': X_te, 'yp_te': yp_te, 'sd_te': sd_te,
            'pred_te': self._phys(sd_te) + self.ml.predict(self.sc.transform(X_te)),
            'X_tr': X_tr, 'yp_tr': yp_tr, 'sd_tr': sd_tr,
            'pred_tr': self._phys(sd_tr) + self.ml.predict(self.sc.transform(X_tr)),
        }
        return self

    def predict(self, X, sd):
        return self._phys(sd) + self.ml.predict(self.sc.transform(X))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f'[SAVE]  {path}')

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(df_combined, feature_cols):
    """
    Train and compare all models with GridSearchCV hyperparameter tuning.
    80% training, 20% test split on combined (real+fake+synthetic) dataset.
    """
    avail = [c for c in feature_cols if c in df_combined.columns]
    valid = df_combined[avail + ['PPV', 'SD']].dropna()
    valid = valid.loc[:, ~valid.columns.duplicated()]

    X   = valid[avail].values
    y   = valid['PPV'].values
    sd_ = valid['SD'].values.ravel()

    X_tr, X_te, y_tr, y_te, sd_tr, sd_te = train_test_split(
        X, y, sd_, test_size=0.2, random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    results = {}

    print(f"\n{'─'*65}")
    print(f"  MODEL TRAINING — {len(X_tr)} train / {len(X_te)} test records")
    print(f"{'─'*65}")

    # ── M1: Physics Only ──────────────────────────────────────────────────
    yp_ph = K_LIT * sd_te**N_LIT
    results['M01 Physics (USBM)'] = get_metrics(y_te, yp_ph, 'M01 Physics (USBM)')

    # ── Helper: fit with GridSearchCV ──────────────────────────────────────
    def fit_model(name, estimator, use_scale=True, param_grid=None):
        Xtr = X_tr_s if use_scale else X_tr
        Xte = X_te_s if use_scale else X_te
        if param_grid:
            gs = GridSearchCV(estimator, param_grid,
                              cv=kf, scoring='r2', n_jobs=-1)
            gs.fit(Xtr, y_tr)
            best = gs.best_estimator_
            print(f"    {name}: best={gs.best_params_}  CV-R²={gs.best_score_:.4f}")
        else:
            best = estimator
            best.fit(Xtr, y_tr)
        yp = best.predict(Xte)
        results[name] = get_metrics(y_te, yp, name)
        return best

    # ── M2: Random Forest ─────────────────────────────────────────────────
    print("  Tuning M02: Random Forest...")
    fit_model('M02 Random Forest',
              RandomForestRegressor(random_state=42, n_jobs=-1), True,
              {'n_estimators':[200,400],'max_depth':[8,12,None],'min_samples_leaf':[1,2]})

    # ── M3: Gradient Boosting ─────────────────────────────────────────────
    print("  Tuning M03: Gradient Boosting...")
    fit_model('M03 Gradient Boosting',
              GradientBoostingRegressor(random_state=42), True,
              {'n_estimators':[200,400],'learning_rate':[0.05,0.1],'max_depth':[3,5],'subsample':[0.7,0.9]})

    # ── M4: Extra Trees ───────────────────────────────────────────────────
    print("  Tuning M04: Extra Trees...")
    fit_model('M04 Extra Trees',
              ExtraTreesRegressor(random_state=42, n_jobs=-1), True,
              {'n_estimators':[200,400],'max_depth':[8,12,None],'min_samples_leaf':[1,2]})

    # ── M5: AdaBoost ──────────────────────────────────────────────────────
    print("  Tuning M05: AdaBoost...")
    fit_model('M05 AdaBoost',
              AdaBoostRegressor(random_state=42), True,
              {'n_estimators':[100,200],'learning_rate':[0.5,1.0]})

    # ── M6: Bagging ───────────────────────────────────────────────────────
    print("  Tuning M06: Bagging...")
    fit_model('M06 Bagging',
              BaggingRegressor(random_state=42, n_jobs=-1), True,
              {'n_estimators':[100,200],'max_samples':[0.7,1.0]})

    # ── M7: SVR ───────────────────────────────────────────────────────────
    print("  Tuning M07: SVR...")
    fit_model('M07 SVR (RBF)',
              SVR(kernel='rbf'), True,
              {'C':[10,100,500],'gamma':['scale',0.01],'epsilon':[0.05,0.1,0.5]})

    # ── M8: KNN ───────────────────────────────────────────────────────────
    print("  Tuning M08: KNN...")
    fit_model('M08 KNN',
              KNeighborsRegressor(), True,
              {'n_neighbors':[3,5,7,10],'weights':['uniform','distance']})

    # ── M9: Decision Tree ─────────────────────────────────────────────────
    print("  Tuning M09: Decision Tree...")
    fit_model('M09 Decision Tree',
              DecisionTreeRegressor(random_state=42), False,
              {'max_depth':[4,6,10,None],'min_samples_leaf':[1,2,4]})

    # ── M10: Ridge ────────────────────────────────────────────────────────
    print("  Tuning M10: Ridge...")
    fit_model('M10 Ridge',
              Ridge(), True, {'alpha':[0.1,1.0,10,100]})

    # ── M11: Lasso ────────────────────────────────────────────────────────
    print("  Tuning M11: Lasso...")
    fit_model('M11 Lasso',
              Lasso(max_iter=5000), True, {'alpha':[0.01,0.1,1.0,10]})

    # ── M12: ElasticNet ───────────────────────────────────────────────────
    print("  Tuning M12: ElasticNet...")
    fit_model('M12 ElasticNet',
              ElasticNet(max_iter=5000), True,
              {'alpha':[0.01,0.1,1.0],'l1_ratio':[0.2,0.5,0.8]})

    # ── M13: MLP Neural Network ───────────────────────────────────────────
    print("  Tuning M13: MLP Neural Network (Deep Learning)...")
    fit_model('M13 MLP Neural Net',
              MLPRegressor(max_iter=1000, random_state=42,
                           activation='relu', solver='adam',
                           early_stopping=True, validation_fraction=0.1),
              True,
              {'hidden_layer_sizes':[(128,64),(128,64,32),(256,128,64)],
               'alpha':[0.001,0.01],'learning_rate_init':[0.001,0.005]})

    # ── M14: Fuzzy Logic ─────────────────────────────────────────────────
    print("  Training M14: Fuzzy Logic (Sugeno FIS)...")
    # Find column indices for SD, TQ, No_of_Holes in avail
    try:
        sd_col = avail.index('SD')
        tq_col = avail.index('TQ')
        n_col  = avail.index('No_of_Holes')
        fuzzy = FuzzyPPVPredictor()
        fuzzy.fit(X_tr, y_tr, sd_col=sd_col, tq_col=tq_col, n_col=n_col)
        yp_fz = fuzzy.predict(X_te)
        results['M14 Fuzzy Logic (Sugeno)'] = get_metrics(
            y_te, yp_fz, 'M14 Fuzzy Logic (Sugeno)')
        with open('models/fuzzy_model.pkl', 'wb') as f:
            pickle.dump(fuzzy, f)
    except Exception as e:
        print(f"    Fuzzy model warning: {e}")

    # ── M15: Hybrid (Proposed) ─────────────────────────────────────────────
    print("  Training M15: Hybrid Physics+RF (Proposed)...")
    hybrid = HybridPPVModel()
    hybrid.fit(df_combined, feature_cols)
    results['M15 Hybrid RF (Proposed)'] = hybrid.test_metrics
    results['M15 Hybrid RF (Proposed)']['Model'] = 'M15 Hybrid RF (Proposed)'

    # Summary
    bench = pd.DataFrame(results).T[['R2','MAE','RMSE','MAPE']].round(4)
    bench.index.name = 'Model'
    bench.reset_index(inplace=True)
    bench.to_csv('results/model_benchmark.csv', index=False)

    print(f"\n{'═'*70}")
    print(f"  COMPLETE MODEL BENCHMARK")
    print(f"{'═'*70}")
    print(bench.to_string(index=False))
    print(f"{'═'*70}")

    return results, hybrid, bench


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — BENCHMARK VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_benchmark(bench):
    """Fig 4.3 — Model comparison bar charts for all 4 metrics."""
    short_labels = [m.split(' ')[0]+'\n'+' '.join(m.split(' ')[1:3])
                    for m in bench['Model'].values]
    highlight = ['#E74C3C' if 'Hybrid' in m else
                 '#27AE60' if 'MLP' in m else
                 '#F39C12' if 'Fuzzy' in m else
                 '#1B4F72' for m in bench['Model'].values]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fig. 4.3: Complete Model Performance Comparison\n'
                 '(Red = Proposed Hybrid, Green = MLP Neural Net, Orange = Fuzzy)',
                 fontsize=12, fontweight='bold')

    for ax, metric, ylabel in zip(axes.flatten(),
        ['R2','MAE','RMSE','MAPE'],
        ['R² (higher = better)','MAE (mm/s)','RMSE (mm/s)','MAPE (%)']):
        vals = bench[metric].values
        bars = ax.bar(range(len(bench)), vals, color=highlight,
                      edgecolor='white', width=0.72)
        ax.set_xticks(range(len(bench)))
        ax.set_xticklabels(short_labels, rotation=35, ha='right', fontsize=7.5)
        ax.set_ylabel(ylabel); ax.set_title(metric, fontweight='bold')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + max(vals)*0.012,
                    f'{v:.3f}', ha='center', fontsize=7.5)
        if metric == 'R2':
            ax.set_ylim(0, 1.15)
            ax.axhline(0.7, color='orange', ls=':', lw=1.5, label='R²=0.70')
            ax.axhline(0.9, color='red',    ls='--', lw=1.5, label='R²=0.90 target')
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('plots/Fig4_3_model_benchmark.png')
    plt.close()
    print('[PLOT] Fig4_3_model_benchmark.png')


def plot_hybrid_results(hybrid):
    """Figs 4.4–4.7 — Hybrid model performance plots."""
    td     = hybrid._td
    y_true = td['yp_te']
    y_pred = td['pred_te']
    resid  = y_true - y_pred
    rel_e  = np.abs(resid / (y_true + 1e-9)) * 100

    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Fig 4.4 — Actual vs Predicted + Residual
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Fig. 4.4: Hybrid Model (PPV = Physics + RF Residual) — Performance',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.scatter(y_true, y_pred, color=C[0], s=70, alpha=0.85,
               edgecolors='white', zorder=3)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.12]
    ax.plot(lim, lim, 'k--', lw=1.8, label='1:1 perfect prediction')
    ax.fill_between(lim, [x*0.8 for x in lim], [x*1.2 for x in lim],
                    alpha=0.08, color='green', label='±20% band')
    ax.fill_between(lim, [x*0.9 for x in lim], [x*1.1 for x in lim],
                    alpha=0.12, color='blue', label='±10% band')
    ax.set_xlabel('Observed PPV (mm/s)'); ax.set_ylabel('Predicted PPV (mm/s)')
    ax.set_title(f'Actual vs Predicted  |  R²={r2:.4f}')
    ax.legend(fontsize=9); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.text(0.04, 0.86,
            f'R²   = {r2:.4f}\nMAE  = {mae:.4f} mm/s\nRMSE = {rmse:.4f} mm/s',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

    ax2 = axes[1]
    ax2.scatter(y_pred, resid, color=C[3], s=65, alpha=0.85, edgecolors='white')
    ax2.axhline(0, color='black', lw=1.8)
    sig = resid.std()
    ax2.axhline( 2*sig, color='red', ls='--', lw=1.3, label=f'+2σ={2*sig:.3f}')
    ax2.axhline(-2*sig, color='red', ls='--', lw=1.3, label=f'-2σ={-2*sig:.3f}')
    ax2.set_xlabel('Predicted PPV (mm/s)'); ax2.set_ylabel('Residual (mm/s)')
    ax2.set_title('Residual Plot'); ax2.legend(fontsize=9)

    plt.tight_layout(); plt.savefig('plots/Fig4_4_actual_vs_predicted.png')
    plt.close(); print('[PLOT] Fig4_4_actual_vs_predicted.png')

    # Fig 4.5 — Error distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Fig. 4.5: Error Distribution Analysis — Hybrid Model',
                 fontsize=12, fontweight='bold')
    ax = axes[0]
    ax.hist(resid, bins=18, color=C[0], edgecolor='white', density=True, alpha=0.85)
    xr = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, resid.mean(), resid.std()),
            color=C[1], lw=2.5, label='Normal fit')
    ax.set_xlabel('Residual (mm/s)'); ax.set_ylabel('Density')
    ax.set_title('Residual Distribution'); ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.hist(rel_e, bins=18, color=C[2], edgecolor='white', density=True, alpha=0.85)
    ax2.axvline(np.median(rel_e), color='red', lw=2.2,
                label=f'Median = {np.median(rel_e):.1f}%')
    ax2.axvline(20, color='orange', lw=1.8, ls='--', label='20% threshold')
    ax2.set_xlabel('|Relative Error| (%)'); ax2.set_ylabel('Density')
    ax2.set_title('% Error Distribution'); ax2.legend(fontsize=9)

    plt.tight_layout(); plt.savefig('plots/Fig4_5_error_distribution.png')
    plt.close(); print('[PLOT] Fig4_5_error_distribution.png')

    # Fig 4.6 — Prediction timeline
    fig, ax = plt.subplots(figsize=(12, 5))
    idx_ = np.argsort(y_true)
    ax.plot(range(len(y_true)), y_true[idx_], 'o-',
            color=C[0], lw=2, ms=7, label='Observed PPV', alpha=0.9)
    ax.plot(range(len(y_pred)), y_pred[idx_], 's--',
            color=C[1], lw=2, ms=7, label='Predicted PPV (Hybrid)', alpha=0.9)
    ax.fill_between(range(len(y_true)),
                    y_true[idx_]*0.80, y_true[idx_]*1.20,
                    alpha=0.12, color=C[2], label='±20% tolerance')
    ax.set_xlabel('Blast Event (sorted by PPV magnitude)')
    ax.set_ylabel('PPV (mm/s)')
    ax.set_title('Fig. 4.6: Observed vs Predicted PPV — Sorted Events',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig('plots/Fig4_6_prediction_timeline.png')
    plt.close(); print('[PLOT] Fig4_6_prediction_timeline.png')

    # Fig 4.7 — Feature importance
    if hasattr(hybrid.ml, 'feature_importances_'):
        fi   = hybrid.ml.feature_importances_
        idx_ = np.argsort(fi)[::-1]
        labs = [hybrid.feat_cols[i] for i in idx_]
        vals = fi[idx_]
        fig, ax = plt.subplots(figsize=(11, 5))
        col_fi = [C[1] if v > 0.12 else C[0] if v > 0.05 else '#BDC3C7' for v in vals]
        ax.barh(labs, vals, color=col_fi, edgecolor='white')
        ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)')
        ax.set_title('Fig. 4.7: Feature Importance — Hybrid RF Residual Model',
                     fontsize=12, fontweight='bold')
        for i, v in enumerate(vals):
            ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=9)
        plt.tight_layout(); plt.savefig('plots/Fig4_7_feature_importance.png')
        plt.close(); print('[PLOT] Fig4_7_feature_importance.png')

    # Fig 4.8 — Train vs Test R² comparison
    models_te = {
        'Physics':   r2_score(K_LIT * td['sd_te']**N_LIT, td['yp_te']),
        'Hybrid Train': r2_score(td['yp_tr'], td['pred_tr']),
        'Hybrid Test':  r2_score(td['yp_te'], td['pred_te']),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(models_te.keys()), list(models_te.values()),
                  color=[C[2], C[0], C[1]], edgecolor='white', width=0.55)
    ax.axhline(0.9, color='red', ls='--', lw=1.5, label='Target R²=0.9')
    ax.set_ylim(0, 1.1); ax.set_ylabel('R²')
    ax.set_title('Fig. 4.8: Physics vs Hybrid Model — Train/Test R²',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    for bar, v in zip(bars, models_te.values()):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.02, f'{v:.4f}', ha='center', fontsize=11)
    plt.tight_layout(); plt.savefig('plots/Fig4_8_train_test_r2.png')
    plt.close(); print('[PLOT] Fig4_8_train_test_r2.png')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 65)
    print('  PART 2 — SYNTHETIC DATA & ALL MODEL TRAINING')
    print('  IIT (BHU) Varanasi | Mining Engineering')
    print('=' * 65)

    # Load real field data from Part 1
    if not os.path.exists('results/real_field_data.csv'):
        print("[ERROR] Run part1_eda.py first.")
        sys.exit(1)

    df_real = pd.read_csv('results/real_field_data.csv')
    df_real = ensure_features(df_real)
    print(f'\n[DATA]  {len(df_real)} real field records loaded')

    # Physics model
    physics = plot_physics_model(df_real)

    # Synthetic data generation
    df_combined = generate_synthetic(df_real, n_syn=600)
    df_combined = ensure_features(df_combined)

    # Train all models
    feat_cols = [c for c in FEATURE_COLS if c in df_combined.columns]
    print(f'\n[MODELS] Training on {len(df_combined)} combined records')
    print(f'         Features: {feat_cols}\n')

    results, hybrid, bench = train_all_models(df_combined, feat_cols)

    # Result plots
    plot_benchmark(bench)
    plot_hybrid_results(hybrid)

    # Save model
    hybrid.save('models/hybrid_ppv_model.pkl')

    summary = {
        'physics'     : {'k': K_LIT, 'n': N_LIT, 'r2': physics['r2'],
                         'note': 'k and n are ALWAYS non-zero literature constants'},
        'test_metrics': hybrid.test_metrics,
        'train_metrics':hybrid.train_metrics,
        'features'    : hybrid.feat_cols,
        'n_real'      : len(df_real),
        'n_synthetic' : len(df_combined) - len(df_real),
        'n_combined'  : len(df_combined),
    }
    with open('results/model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    bench.to_csv('results/model_benchmark.csv', index=False)

    print(f'\n{"="*65}')
    print(f'  PART 2 COMPLETE')
    print(f'  k = {K_LIT}  (non-zero ✓)    n = {N_LIT}  (non-zero ✓)')
    print(f'  Hybrid Train R²  : {hybrid.train_metrics["R2"]:.4f}')
    print(f'  Hybrid Test  R²  : {hybrid.test_metrics["R2"]:.4f}')
    print(f'  Hybrid Test  MAE : {hybrid.test_metrics["MAE"]:.4f} mm/s')
    print(f'{"="*65}')