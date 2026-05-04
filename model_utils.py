"""
model_utils.py
==============
Shared model class imported by BOTH part2_models.py and part3_ui.py.

WHY THIS FILE EXISTS
─────────────────────
Python's pickle serialises an object by recording:
    (<module_name>, <class_name>)

When part2_models.py runs as the main script (python part2_models.py),
Python sets __name__ = '__main__', so any class defined there is stored
in the pickle as ('__main__', 'HybridPPVModel').

When part3_ui.py later calls pickle.load(), Python looks for
'__main__.HybridPPVModel' in part3's __main__ — which is part3_ui.py —
and the class is not there.  Result:

    AttributeError: Can't get attribute 'HybridPPVModel'
    on <module '__main__' from '.../part3_ui.py'>

The solution: define HybridPPVModel here, in model_utils.py.
Both part2 and part3 import it from this file.
Pickle always records ('model_utils', 'HybridPPVModel').
Loading works from any script because model_utils.py is always importable.

DO NOT move HybridPPVModel back into part2_models.py or part3_ui.py.
"""

import os
import sys
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Site constants — always non-zero, from literature ────────────────────────
K_LIT = 650.0   # site transmission constant (Pal Roy 1993, Wardha Valley)
N_LIT = -1.4    # attenuation exponent (Indian coal mines: −1.2 to −1.6)


# ══════════════════════════════════════════════════════════════════════════════
#  HYBRID PPV MODEL
# ══════════════════════════════════════════════════════════════════════════════

class HybridPPVModel:
    """
    Physics-Informed Hybrid Random Forest Model.

    Architecture
    ─────────────
    PPV_physics = k × SD^n                 [USBM power law, Eq. 3.1]
    Residual    = PPV_actual − PPV_physics  [geological correction, Eq. 3.2]
    PPV_final   = PPV_physics + RF(X)      [hybrid prediction, Eq. 3.3]

    k = 650, n = −1.4  (Pal Roy 1993 — Wardha Valley coalfield, India)
    Both k and n are ALWAYS non-zero by physical law.

    This class is defined here (model_utils.py) so that pickle always
    records the module as 'model_utils', making load() work from any script.
    """

    def __init__(self):
        self.k             = K_LIT    # site constant, always non-zero
        self.n             = N_LIT    # attenuation exponent, always non-zero
        self.ml            = None     # RandomForestRegressor (residual learner)
        self.sc            = StandardScaler()
        self.feat_cols     = []
        self.is_fitted     = False
        self.train_metrics = {}
        self.test_metrics  = {}
        self._td           = {}       # test-data dict, used for result plots

    # ── Physics component ─────────────────────────────────────────────────────
    def _phys(self, sd):
        """PPV_physics = k × SD^n.  k ≠ 0, n ≠ 0 guaranteed."""
        assert self.k != 0 and self.n != 0, "Physics constants must be non-zero"
        return self.k * np.clip(sd, 1e-9, None) ** self.n

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, df_combined, feature_cols, test_size=0.2):
        """
        Train the RF residual model.
        Prints train/test metrics.
        """
        avail = [c for c in feature_cols if c in df_combined.columns]
        self.feat_cols = avail

        df2 = df_combined.copy()
        df2['Physics_PPV'] = self._phys(df2['SD'].values)
        df2['Residual']    = df2['PPV'] - df2['Physics_PPV']

        valid = df2[avail + ['Residual', 'PPV', 'SD']].dropna()
        valid = valid.loc[:, ~valid.columns.duplicated()]

        X    = valid[avail].values
        y_r  = valid['Residual'].values
        y_p  = valid['PPV'].values
        sd_v = valid['SD'].values.ravel()

        X_tr, X_te, yr_tr, yr_te, yp_tr, yp_te, sd_tr, sd_te = \
            train_test_split(X, y_r, y_p, sd_v,
                             test_size=test_size, random_state=42)

        # GridSearchCV on RF residual learner
        print("    GridSearchCV: tuning Hybrid RF residual model …")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {'n_estimators'    : [200, 400],
             'max_depth'       : [8, 12, None],
             'min_samples_leaf': [1, 2]},
            cv=kf, scoring='r2', n_jobs=-1, refit=True,
        )
        gs.fit(self.sc.fit_transform(X_tr), yr_tr)
        self.ml = gs.best_estimator_
        self.is_fitted = True
        print(f"    Best params : {gs.best_params_}")
        print(f"    CV R²       : {gs.best_score_:.4f}")

        # Metrics
        def _met(Xs, yr, yp, sd, tag):
            pred = self._phys(sd) + self.ml.predict(self.sc.transform(Xs))
            r2   = r2_score(yp, pred)
            mae  = mean_absolute_error(yp, pred)
            rmse = np.sqrt(mean_squared_error(yp, pred))
            mape = np.mean(np.abs((yp - pred) / (np.abs(yp) + 1e-9))) * 100
            print(f"    Hybrid [{tag:5s}]  R²={r2:.4f}  MAE={mae:.4f}  "
                  f"RMSE={rmse:.4f}  MAPE={mape:.2f}%")
            return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

        self.train_metrics = _met(X_tr, yr_tr, yp_tr, sd_tr, 'train')
        self.test_metrics  = _met(X_te, yr_te, yp_te, sd_te, 'test ')

        # Store for result plots
        pred_te = self._phys(sd_te) + self.ml.predict(self.sc.transform(X_te))
        pred_tr = self._phys(sd_tr) + self.ml.predict(self.sc.transform(X_tr))
        self._td = {
            'X_te': X_te, 'yp_te': yp_te, 'sd_te': sd_te, 'pred_te': pred_te,
            'X_tr': X_tr, 'yp_tr': yp_tr, 'sd_tr': sd_tr, 'pred_tr': pred_tr,
        }
        return self

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, X, sd):
        """Predict PPV for feature matrix X and scaled distance array sd."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._phys(sd) + self.ml.predict(self.sc.transform(X))

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        """
        Save the model to a pickle file.

        IMPORTANT: before dumping, assert the class module is 'model_utils'.
        If somehow __module__ has been overridden (e.g. by running as __main__),
        this check will catch it and print a warning.
        """
        module = type(self).__module__
        if module == '__main__':
            # Force the correct module so pickle records 'model_utils'
            type(self).__module__ = 'model_utils'

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=4)

        # Restore __module__ (good practice, though not strictly needed)
        type(self).__module__ = 'model_utils'
        print(f"[SAVE]  {path}  (module=model_utils.HybridPPVModel)")

    @staticmethod
    def load(path: str) -> 'HybridPPVModel':
        """
        Load a model from a pickle file.

        Registers 'model_utils.HybridPPVModel' under both '__main__' and
        'part2_models' aliases before loading, so files saved under any of
        those names will deserialise correctly.
        """
        import sys

        # Register aliases — covers all past save locations
        for alias in ('__main__', 'part2_models'):
            mod = sys.modules.get(alias)
            if mod is not None and not hasattr(mod, 'HybridPPVModel'):
                setattr(mod, 'HybridPPVModel', HybridPPVModel)
            elif mod is None:
                import types
                fake = types.ModuleType(alias)
                fake.HybridPPVModel = HybridPPVModel
                sys.modules[alias] = fake

        with open(path, 'rb') as f:
            obj = pickle.load(f)

        # Guarantee the loaded object is a proper HybridPPVModel instance
        if not isinstance(obj, HybridPPVModel):
            raise TypeError(
                f"Loaded object is {type(obj)}, not HybridPPVModel. "
                "Re-run part2_models.py to regenerate the model file."
            )
        return obj

    # ── Validation ────────────────────────────────────────────────────────────
    def validate(self) -> tuple:
        """
        Check the loaded model is complete and ready for inference.
        Returns (ok: bool, message: str).
        """
        required = ['k', 'n', 'ml', 'sc', 'feat_cols', 'is_fitted']
        missing  = [a for a in required if not hasattr(self, a)]
        if missing:
            return False, f"Corrupt model — missing attributes: {missing}"
        if not self.is_fitted:
            return False, "Model is_fitted=False — re-run part2_models.py"
        if self.k == 0 or self.n == 0:
            return False, f"Invalid constants: k={self.k}, n={self.n} (must be non-zero)"
        if not self.feat_cols:
            return False, "feat_cols is empty — model was not trained correctly"
        return True, (
            f"Model OK — k={self.k}, n={self.n}, "
            f"features={len(self.feat_cols)}, "
            f"test_R²={self.test_metrics.get('R2', 'N/A')}"
        )