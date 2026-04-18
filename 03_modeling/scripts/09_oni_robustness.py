"""
09_oni_robustness.py
ONI robustness variant: replace Tmean_chill_lag1 with NOAA ONI aggregated
over May-Aug of Y-1, then also fit a joint model ONI + T_chill to test
whether T_chill captures variance beyond ENSO.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YEAR_DATA = ROOT / "03_modeling" / "outputs" / "tables" / "year_level_dataset.csv"
ONI_FILE = ROOT / "00_raw_data" / "oni.data"
OUT_DIR = ROOT / "03_modeling" / "outputs" / "tables"

# --- Parse NOAA PSL ONI file ---
oni_rows = []
with open(ONI_FILE) as f:
    for line in f:
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            yr = int(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if yr < 1900 or yr > 2100:
            continue
        oni_rows.append([yr] + vals)

oni_df = pd.DataFrame(oni_rows, columns=["year"] + [f"m{i}" for i in range(1, 13)])
# Replace missing sentinel (-99.9 etc)
oni_df = oni_df.replace({-99.9: np.nan, -999.9: np.nan})
# Chill window = May-Aug (months 5..8) of Y-1
oni_df["ONI_chill"] = oni_df[["m5", "m6", "m7", "m8"]].mean(axis=1)
oni_chill = oni_df[["year", "ONI_chill"]].copy()
oni_chill["year_target"] = oni_chill["year"] + 1  # assign to yield year Y
oni_chill = oni_chill[["year_target", "ONI_chill"]].rename(columns={"year_target": "yield_year"})

# --- Merge ---
year = pd.read_csv(YEAR_DATA)
ext = year.merge(oni_chill, on="yield_year", how="left")
ext.to_csv(OUT_DIR / "year_level_dataset_extended.csv", index=False)
print(ext[["yield_year", "Tmean_chill_lag1", "ONI_chill"]])

ext["log_y"] = np.log(ext["yield_tn_ha"] + 0.5)

def fit(name, cols):
    X = sm.add_constant(ext[cols])
    m = sm.OLS(ext["log_y"], X).fit()
    return m

m_oni = fit("ONI + T_CREC", ["ONI_chill", "Tmean_mean_CREC"])
m_onionly = fit("ONI only", ["ONI_chill"])
m_joint = fit("ONI + T_chill + T_CREC", ["ONI_chill", "Tmean_chill_lag1", "Tmean_mean_CREC"])

# --- LOYO for ONI + T_CREC ---
def loyo_r2(ext, cols):
    preds_log = np.zeros(len(ext))
    for i in range(len(ext)):
        train = ext.drop(ext.index[i])
        test = ext.iloc[[i]]
        Xtr = sm.add_constant(train[cols])
        Xte = sm.add_constant(test[cols], has_constant="add")
        m = sm.OLS(train["log_y"], Xtr).fit()
        preds_log[i] = m.predict(Xte).iloc[0]
    preds_y = np.exp(preds_log) - 0.5
    preds_y = np.clip(preds_y, 0, None)
    obs_y = ext["yield_tn_ha"].values
    obs_log = ext["log_y"].values
    r2_log = 1 - np.sum((obs_log - preds_log) ** 2) / np.sum((obs_log - obs_log.mean()) ** 2)
    r2_raw = 1 - np.sum((obs_y - preds_y) ** 2) / np.sum((obs_y - obs_y.mean()) ** 2)
    rmse_raw = float(np.sqrt(np.mean((obs_y - preds_y) ** 2)))
    mae_raw = float(np.mean(np.abs(obs_y - preds_y)))
    return r2_log, r2_raw, rmse_raw, mae_raw

loyo_oni = loyo_r2(ext, ["ONI_chill", "Tmean_mean_CREC"])

# In-sample back-transformed metrics for ONI+T_CREC
pred_log = m_oni.predict(sm.add_constant(ext[["ONI_chill", "Tmean_mean_CREC"]]))
pred_y = np.clip(np.exp(pred_log) - 0.5, 0, None)
obs_y = ext["yield_tn_ha"].values
r2_is = 1 - np.sum((obs_y - pred_y) ** 2) / np.sum((obs_y - obs_y.mean()) ** 2)
rmse_is = float(np.sqrt(np.mean((obs_y - pred_y) ** 2)))
mae_is = float(np.mean(np.abs(obs_y - pred_y)))

with open(OUT_DIR / "oni_robustness.txt", "w") as f:
    f.write("ONI ROBUSTNESS\n==============\n\n")
    f.write("ONI_chill = mean ONI over May-Aug of Y-1\n\n")
    f.write("(a) ONI only  log(y+0.5) ~ ONI_chill\n")
    f.write(f"   R2 (log) = {m_onionly.rsquared:.4f}\n")
    f.write(f"   beta_ONI = {m_onionly.params['ONI_chill']:+.4f} (p={m_onionly.pvalues['ONI_chill']:.4f})\n\n")
    f.write("(b) ONI + T_CREC\n")
    f.write(m_oni.summary().as_text() + "\n\n")
    f.write(f"   in-sample back-trans R2={r2_is:.4f} RMSE={rmse_is:.3f} MAE={mae_is:.3f}\n")
    f.write(f"   LOYO log R2={loyo_oni[0]:.4f} raw R2={loyo_oni[1]:.4f} "
            f"RMSE={loyo_oni[2]:.3f} MAE={loyo_oni[3]:.3f}\n\n")
    f.write("(c) Joint ONI + T_chill + T_CREC\n")
    f.write(m_joint.summary().as_text() + "\n\n")
    f.write(f"   beta_T_chill p = {m_joint.pvalues['Tmean_chill_lag1']:.4f}\n")
    f.write(f"   beta_ONI     p = {m_joint.pvalues['ONI_chill']:.4f}\n")

print("Wrote:", OUT_DIR / "oni_robustness.txt")
print(f"ONI-only R2={m_onionly.rsquared:.3f}")
print(f"ONI+T_CREC R2={m_oni.rsquared:.3f} in-sample back-trans={r2_is:.3f}")
print(f"Joint: beta_T_chill p={m_joint.pvalues['Tmean_chill_lag1']:.4f}")
print(f"LOYO ONI+T_CREC: log R2={loyo_oni[0]:.3f} raw R2={loyo_oni[1]:.3f} RMSE={loyo_oni[2]:.2f} MAE={loyo_oni[3]:.2f}")
