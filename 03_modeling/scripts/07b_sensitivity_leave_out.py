"""
07b_sensitivity_leave_out.py
----------------------------
Companion sensitivity analyses for the FINAL primary model
(07_final_primary_log.py), addressing two reviewer-style concerns:

  (1) Leverage of extreme warm years (2016 and 2024) on the slope estimate.
      -> Refit under leave-2024-out (n=7) and leave-{2016, 2024}-out (n=6)
         and report the chill slope, its SE, its p-value and the in-sample
         R^2 for each reduced sample.

  (2) Sensitivity to the offset c in the variance-stabilising transform
      log(y + c).
      -> Refit the two-predictor OLS under c in {0.1, 0.25, 0.5, 0.75, 1.0}
         on the FULL 8-year sample and report the chill slope, SE, p-value
         and back-transformed in-sample R^2.

No new data are fabricated — everything is refit from the same
feature_matrix_chill_complete.csv used by script 07.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

BRANCH = Path(r"D:/olive_yield_RS_chill")
FEAT = BRANCH / "01_preprocessing" / "outputs" / "feature_matrix_chill_complete.csv"
OUT_DIR = BRANCH / "03_modeling" / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS_DEFAULT = 0.5


def back_transform(log_pred, eps):
    return np.clip(np.exp(log_pred) - eps, 0.0, None)


def metrics_raw(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return r2, rmse, mae


# --------------------------------------------------------------------------
# Load and aggregate to year level (same as 07_final_primary_log.py)
df = pd.read_csv(FEAT)
year_df = df.groupby("yield_year").agg(
    yield_tn_ha=("yield_tn_ha", "mean"),
    Tmean_chill_lag1=("Tmean_chill_lag1", "first"),
    Tmean_mean_CREC=("Tmean_mean_CREC", "first"),
).reset_index()
print("[Data] years:", year_df["yield_year"].tolist())


def fit_two_pred(subset, eps):
    y = subset["yield_tn_ha"].values
    log_y = np.log(y + eps)
    X = subset[["Tmean_chill_lag1", "Tmean_mean_CREC"]].values
    X_ols = sm.add_constant(X)
    ols = sm.OLS(log_y, X_ols).fit()
    yhat_raw = back_transform(ols.fittedvalues, eps)
    r2_raw, rmse_raw, mae_raw = metrics_raw(y, yhat_raw)
    return dict(
        n=len(y),
        intercept=float(ols.params[0]),
        beta_chill=float(ols.params[1]),
        se_chill=float(ols.bse[1]),
        p_chill=float(ols.pvalues[1]),
        beta_creg=float(ols.params[2]),
        se_creg=float(ols.bse[2]),
        p_creg=float(ols.pvalues[2]),
        R2_log=float(ols.rsquared),
        R2_raw=r2_raw,
        RMSE_raw=rmse_raw,
        MAE_raw=mae_raw,
    )


# --------------------------------------------------------------------------
# (1) Leave-out sensitivity (c = 0.5, primary spec)
print("\n" + "=" * 72)
print("LEAVE-OUT SENSITIVITY  (c = 0.5)")
print("=" * 72)

leave_out_specs = [
    ("full (n=8)", []),
    ("leave_2024 (n=7)", [2024]),
    ("leave_2016_2024 (n=6)", [2016, 2024]),
]

rows_lo = []
for label, exclude in leave_out_specs:
    sub = year_df[~year_df["yield_year"].isin(exclude)].copy()
    r = fit_two_pred(sub, EPS_DEFAULT)
    r["spec"] = label
    r["excluded"] = ",".join(str(e) for e in exclude) if exclude else "-"
    rows_lo.append(r)
    print(f"  {label:<22}  n={r['n']}  "
          f"beta_chill={r['beta_chill']:+.4f} (SE={r['se_chill']:.4f}, p={r['p_chill']:.4f})  "
          f"R2_log={r['R2_log']:.3f}  R2_raw={r['R2_raw']:.3f}")

df_lo = pd.DataFrame(rows_lo)[[
    "spec", "excluded", "n",
    "beta_chill", "se_chill", "p_chill",
    "beta_creg", "se_creg", "p_creg",
    "R2_log", "R2_raw", "RMSE_raw", "MAE_raw",
]]
df_lo.to_csv(OUT_DIR / "sensitivity_leave_out.csv", index=False)
print(f"\n  -> {OUT_DIR/'sensitivity_leave_out.csv'}")

# --------------------------------------------------------------------------
# (2) Offset-c sensitivity (full n=8 sample)
print("\n" + "=" * 72)
print("OFFSET-c SENSITIVITY  (full n=8 sample)")
print("=" * 72)

c_grid = [0.1, 0.25, 0.5, 0.75, 1.0]
rows_c = []
for c in c_grid:
    r = fit_two_pred(year_df, c)
    r["c_offset"] = c
    rows_c.append(r)
    print(f"  c={c:<4}  beta_chill={r['beta_chill']:+.4f} (SE={r['se_chill']:.4f}, "
          f"p={r['p_chill']:.4f})  R2_log={r['R2_log']:.3f}  R2_raw={r['R2_raw']:.3f}")

df_c = pd.DataFrame(rows_c)[[
    "c_offset", "n",
    "beta_chill", "se_chill", "p_chill",
    "beta_creg", "se_creg", "p_creg",
    "R2_log", "R2_raw", "RMSE_raw", "MAE_raw",
]]
df_c.to_csv(OUT_DIR / "sensitivity_offset.csv", index=False)
print(f"\n  -> {OUT_DIR/'sensitivity_offset.csv'}")

print("\n[Done]")
