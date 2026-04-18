"""
05_modeling_primary.py
----------------------
Primary statistical attribution of Tacna olive yield to winter chilling and
fruit-growth heat.

Design (Q1-grade, small-n climate-driven panel):

    * Climate predictors vary ONLY at year level. With 8 effective years,
      the correct inferential unit is a year-level OLS on the per-year mean
      yield (n=8, p=2, df=5). A panel mixed-effects model over the 88 obs
      would inflate the apparent sample size and mislead inference.

    * Predictor selection — informed by the year-level correlation matrix:

          r(yield, Tmean_chill_lag1) = -0.835   (strongest)
          r(yield, Tmean_mean_CREC)  = -0.313
          r(Tmean_chill, Tmean_CREC) =  0.060   (orthogonal at year level)

      Tmean_chill_lag1 is the continuous analog of Chill Portions in this
      system (CP=0 occurs precisely when Tmean_chill > ~18 C) and is a
      cleaner statistical predictor than the floor-bounded CP.

      Final predictors:  Tmean_chill_lag1  and  Tmean_mean_CREC.
      Mechanistically independent (SH winter vs SH summer).

    * Validation: Leave-One-Year-Out (each year is one fold; this is true
      out-of-sample extrapolation because climate predictors are year-level).

    * Inference: year-block bootstrap for CI; permutation test for honest
      p-value; OLS analytic CIs reported alongside.

    * Sensitivity: (a) panel mixed-effects over the 88 obs to quantify the
      parcel variance component; (b) CP_lag1 as alternative chill metric;
      (c) single-predictor OLS; (d) LOYO CV error by year.

    * Counterfactual: restore Tmean_chill_lag1 of 2016 and 2024 to the
      non-failure historical mean; bootstrap predicted counterfactual yield.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

BRANCH = Path(r"D:/olive_yield_RS_chill")
FEAT = BRANCH / "01_preprocessing" / "outputs" / "feature_matrix_chill_complete.csv"
OUT_DIR = BRANCH / "03_modeling" / "outputs"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
N_BOOT = 5000
N_PERM = 5000


# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return dict(R2=np.nan, RMSE=np.nan, MAE=np.nan, bias=np.nan, n=0)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    bias = float(np.mean(y_pred - y_true))
    return dict(R2=r2, RMSE=rmse, MAE=mae, bias=bias, n=len(y_true))


# ----------------------------------------------------------------------------
print("=" * 72)
print("PRIMARY ATTRIBUTION MODEL — Tacna olive, winter chilling x summer heat")
print("=" * 72)

df = pd.read_csv(FEAT)
print(f"\n[Data] {len(df)} obs x {df.shape[1]} cols | years: {sorted(df.yield_year.unique())}")

# Year-level dataset (per-year mean yield across 11 parcels)
year_df = df.groupby("yield_year").agg(
    yield_tn_ha=("yield_tn_ha", "mean"),
    yield_sd=("yield_tn_ha", "std"),
    CP_lag1=("CP_lag1", "first"),
    CH12_lag1=("CH12_lag1", "first"),
    Tmean_chill_lag1=("Tmean_chill_lag1", "first"),
    Tmax_mean_FLOR=("Tmax_mean_FLOR", "first"),
    Tmean_mean_FLOR=("Tmean_mean_FLOR", "first"),
    Tmax_mean_CREC=("Tmax_mean_CREC", "first"),
    Tmean_mean_CREC=("Tmean_mean_CREC", "first"),
    ET0_sum_CREC=("ET0_sum_CREC", "first"),
).reset_index()
year_df.to_csv(OUT_DIR / "tables" / "year_level_dataset.csv", index=False)

# Correlation at year level
corr = year_df.drop(columns=["yield_year", "yield_sd"]).corr().round(3)
corr.to_csv(OUT_DIR / "tables" / "year_level_correlations.csv")
print("\n[Year-level correlation with yield]")
print(corr.loc["yield_tn_ha"].sort_values().round(3).to_string())

PREDICTORS = ["Tmean_chill_lag1", "Tmean_mean_CREC"]
print(f"\n[Selected predictors] {PREDICTORS}")
print(f"  r(predictors) = {year_df[PREDICTORS].corr().iloc[0, 1]:.3f}  (orthogonal, good)")

# ----------------------------------------------------------------------------
# Primary OLS at year level
# ----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PRIMARY YEAR-LEVEL OLS (n = 8 years)")
print("=" * 72)

y = year_df["yield_tn_ha"].values
X = year_df[PREDICTORS].values
X_ols = sm.add_constant(X)
ols = sm.OLS(y, X_ols).fit()
print(ols.summary())

with open(OUT_DIR / "tables" / "primary_model_fit.txt", "w") as f:
    f.write(str(ols.summary()))

coef_tbl = pd.DataFrame({
    "term": ["Intercept"] + PREDICTORS,
    "estimate": ols.params,
    "std_error": ols.bse,
    "t": ols.tvalues,
    "p_value": ols.pvalues,
    "ci_low": ols.conf_int()[:, 0],
    "ci_high": ols.conf_int()[:, 1],
})
coef_tbl.to_csv(OUT_DIR / "tables" / "primary_model_coefficients.csv", index=False)
print(f"\n[In-sample] R2 = {ols.rsquared:.4f}   adj R2 = {ols.rsquared_adj:.4f}   "
      f"F({int(ols.df_model)}, {int(ols.df_resid)}) = {ols.fvalue:.2f}   p = {ols.f_pvalue:.4f}")

# Standardized coefficients (effect size)
z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
std_ols = sm.OLS((y - y.mean()) / y.std(ddof=1), sm.add_constant(z)).fit()
print(f"\n[Standardized (unit SD) coefficients]")
for name, b in zip(PREDICTORS, std_ols.params[1:]):
    print(f"  {name:20s}  beta_std = {b:+.3f}")

# ----------------------------------------------------------------------------
# Leave-One-Year-Out CV
# ----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("LEAVE-ONE-YEAR-OUT CV (year-level)")
print("=" * 72)

loyo_rows = []
loyo_preds = []
for i in range(len(year_df)):
    tr_mask = np.arange(len(year_df)) != i
    ytr = y[tr_mask]
    Xtr = sm.add_constant(X[tr_mask])
    m = sm.OLS(ytr, Xtr).fit()
    Xte = np.array([1.0] + list(X[i]))
    yhat = float(Xte @ m.params)
    yobs = float(y[i])
    yr = int(year_df.loc[i, "yield_year"])
    loyo_preds.append({"year": yr, "obs": yobs, "pred": yhat,
                       "residual": yobs - yhat})

loyo_df = pd.DataFrame(loyo_preds)
loyo_df.to_csv(OUT_DIR / "tables" / "loyo_cv_results.csv", index=False)
pooled = metrics(loyo_df["obs"], loyo_df["pred"])
print(f"\n[Pooled LOYO]  R2 = {pooled['R2']:.3f}  RMSE = {pooled['RMSE']:.3f}  "
      f"MAE = {pooled['MAE']:.3f}  bias = {pooled['bias']:+.3f}")
print("\n[Per-year LOYO]")
print(loyo_df.round(3).to_string(index=False))

# ----------------------------------------------------------------------------
# Year-block bootstrap for coefficient CIs
# ----------------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"YEAR-BLOCK BOOTSTRAP (n_boot = {N_BOOT})")
print("=" * 72)

boot_coefs = []
years_arr = np.arange(len(year_df))
for b in range(N_BOOT):
    sel = RNG.choice(years_arr, size=len(years_arr), replace=True)
    if len(np.unique(sel)) < 3:
        continue  # skip degenerate draws
    yb = y[sel]
    Xb = sm.add_constant(X[sel])
    try:
        mb = sm.OLS(yb, Xb).fit()
        boot_coefs.append(mb.params)
    except Exception:
        continue

boot_coefs = np.array(boot_coefs)
boot_df = pd.DataFrame(boot_coefs, columns=["Intercept"] + PREDICTORS)
boot_df.to_csv(OUT_DIR / "tables" / "bootstrap_coefficients.csv", index=False)

ci = boot_df.quantile([0.025, 0.5, 0.975]).T
ci.columns = ["ci_low_boot", "median_boot", "ci_high_boot"]
print(f"\n[Bootstrap CIs] ({len(boot_df)} successful draws)")
print(ci.round(4).to_string())

p_boot = {
    name: 2 * min(float(np.mean(boot_df[name] > 0)),
                  float(np.mean(boot_df[name] < 0)))
    for name in PREDICTORS
}
print("\n[Two-sided bootstrap p-values]")
for k, v in p_boot.items():
    print(f"  {k}:  p = {v:.4f}")

# ----------------------------------------------------------------------------
# Permutation test (shuffle yield across years)
# ----------------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"PERMUTATION TEST — shuffling year-level yields (n_perm = {N_PERM})")
print("=" * 72)

obs_stat = ols.rsquared
perm_stats = []
for b in range(N_PERM):
    y_perm = RNG.permutation(y)
    try:
        mp = sm.OLS(y_perm, X_ols).fit()
        perm_stats.append(mp.rsquared)
    except Exception:
        continue
perm_stats = np.array(perm_stats)
p_perm = float(np.mean(perm_stats >= obs_stat))
print(f"  observed R2           = {obs_stat:.4f}")
print(f"  permutation mean R2   = {np.mean(perm_stats):.4f}")
print(f"  permutation 95th pct  = {np.quantile(perm_stats, 0.95):.4f}")
print(f"  one-sided p-value     = {p_perm:.4f}  (n_success = {len(perm_stats)})")
pd.DataFrame({
    "observed_R2": [obs_stat],
    "perm_mean_R2": [float(np.mean(perm_stats))],
    "perm_95pct_R2": [float(np.quantile(perm_stats, 0.95))],
    "p_value": [p_perm],
    "n_perm": [len(perm_stats)],
}).to_csv(OUT_DIR / "tables" / "permutation_test.csv", index=False)

# ----------------------------------------------------------------------------
# Sensitivity — mixed-effects model on the 88-row panel
# ----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SENSITIVITY — panel mixed-effects model (parcel random intercept)")
print("=" * 72)

# Standardize predictors for numerical stability
panel = df[["parcel_id", "yield_year", "yield_tn_ha"] + PREDICTORS].dropna().copy()
for p in PREDICTORS:
    panel[p + "_z"] = (panel[p] - panel[p].mean()) / panel[p].std(ddof=1)

try:
    mm = smf.mixedlm("yield_tn_ha ~ Tmean_chill_lag1_z + Tmean_mean_CREC_z",
                     data=panel, groups=panel["parcel_id"]).fit(reml=True, method="powell")
    print(mm.summary())
    var_p = float(mm.cov_re.iloc[0, 0]) if mm.cov_re is not None else 0.0
    var_r = float(mm.scale)
    icc = var_p / (var_p + var_r) if (var_p + var_r) > 0 else np.nan
    print(f"\n[Variance components] parcel = {var_p:.4f}  residual = {var_r:.4f}  "
          f"ICC(parcel) = {icc:.3f}")
    with open(OUT_DIR / "tables" / "sensitivity_mixedlm_fit.txt", "w") as f:
        f.write(str(mm.summary()))
        f.write(f"\n\nVariance components:\n  parcel = {var_p:.4f}\n  residual = {var_r:.4f}\n  ICC = {icc:.3f}\n")
    mm_ok = True
except Exception as e:
    print(f"  mixedlm fit failed: {e}  (reporting as sensitivity limitation)")
    mm_ok = False
    icc = np.nan

# ----------------------------------------------------------------------------
# Counterfactual attribution (bootstrap year block, refit OLS per draw)
# ----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("COUNTERFACTUAL ATTRIBUTION — 2016 and 2024")
print("=" * 72)

failure_yrs = [2016, 2024]
idx_failure = {yr: int(year_df.index[year_df["yield_year"] == yr][0]) for yr in failure_yrs}
idx_nonfail = year_df.index[~year_df["yield_year"].isin(failure_yrs)].tolist()
chill_baseline = year_df.loc[idx_nonfail, "Tmean_chill_lag1"].mean()
heat_baseline = year_df.loc[idx_nonfail, "Tmean_mean_CREC"].mean()
print(f"  Non-failure baseline:  Tmean_chill = {chill_baseline:.3f} °C   "
      f"Tmean_CREC = {heat_baseline:.3f} °C")
for yr in failure_yrs:
    row = year_df.iloc[idx_failure[yr]]
    print(f"  {yr} observed:  Tmean_chill = {row['Tmean_chill_lag1']:.3f}   "
          f"Tmean_CREC = {row['Tmean_mean_CREC']:.3f}   yield = {row['yield_tn_ha']:.3f}")

cf_records = []
cf_raw = {(yr, s): [] for yr in failure_yrs for s in ("full", "chill", "heat")}
for b in range(N_BOOT):
    sel = RNG.choice(years_arr, size=len(years_arr), replace=True)
    if len(np.unique(sel)) < 3:
        continue
    yb = y[sel]
    Xb = sm.add_constant(X[sel])
    try:
        mb = sm.OLS(yb, Xb).fit()
    except Exception:
        continue
    beta = mb.params  # intercept, chill, heat
    for yr in failure_yrs:
        row = year_df.iloc[idx_failure[yr]]
        obs_chill = row["Tmean_chill_lag1"]
        obs_heat = row["Tmean_mean_CREC"]
        full = beta[0] + beta[1] * chill_baseline + beta[2] * heat_baseline
        chill_only = beta[0] + beta[1] * chill_baseline + beta[2] * obs_heat
        heat_only = beta[0] + beta[1] * obs_chill + beta[2] * heat_baseline
        cf_raw[(yr, "full")].append(full)
        cf_raw[(yr, "chill")].append(chill_only)
        cf_raw[(yr, "heat")].append(heat_only)

for yr in failure_yrs:
    obs_y = float(year_df.loc[idx_failure[yr], "yield_tn_ha"])
    for scen in ("full", "chill", "heat"):
        arr = np.array(cf_raw[(yr, scen)])
        cf_records.append({
            "year": yr,
            "scenario": scen,
            "obs_yield": obs_y,
            "cf_median": float(np.median(arr)),
            "cf_ci_low": float(np.quantile(arr, 0.025)),
            "cf_ci_high": float(np.quantile(arr, 0.975)),
            "delta_median": float(np.median(arr) - obs_y),
            "delta_ci_low": float(np.quantile(arr - obs_y, 0.025)),
            "delta_ci_high": float(np.quantile(arr - obs_y, 0.975)),
            "pct_recovery": float(100 * (np.median(arr) - obs_y) / (np.mean(y) - obs_y))
            if (np.mean(y) - obs_y) != 0 else np.nan,
        })

cf_df = pd.DataFrame(cf_records)
cf_df.to_csv(OUT_DIR / "tables" / "counterfactual_failure_years.csv", index=False)
print("\n[Counterfactual results]")
print(cf_df.round(3).to_string(index=False))

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("FIGURES")
print("=" * 72)

cmap = plt.get_cmap("viridis")

# --- Fig A: LOYO observed vs predicted (year level) + panel LOYO ---
fig, ax = plt.subplots(figsize=(6.5, 6), dpi=600)
n_years = len(loyo_df)
colors = [cmap(i / max(n_years - 1, 1)) for i in range(n_years)]
for i, (_, r) in enumerate(loyo_df.sort_values("year").iterrows()):
    ax.scatter(r["obs"], r["pred"], color=colors[i], s=110,
               edgecolor="k", lw=0.6, zorder=3, label=str(int(r["year"])))
    ax.annotate(int(r["year"]), (r["obs"], r["pred"]),
                xytext=(6, 6), textcoords="offset points", fontsize=11)
lim = (0, max(loyo_df["obs"].max(), loyo_df["pred"].max()) * 1.10)
ax.plot(lim, lim, "k--", lw=1, alpha=0.7)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Observed annual yield (t ha$^{-1}$)", fontsize=13)
ax.set_ylabel("LOYO-predicted yield (t ha$^{-1}$)", fontsize=13)
    # title removed — goes in LaTeX caption
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "fig_model_obs_vs_pred.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("  -> fig_model_obs_vs_pred.png")

# --- Fig B: partial effects ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=600)
beta = ols.params
cl_grid = np.linspace(year_df["Tmean_chill_lag1"].min() - 0.5,
                      year_df["Tmean_chill_lag1"].max() + 0.5, 100)
heat_hold = year_df["Tmean_mean_CREC"].mean()
yhat_cl = beta[0] + beta[1] * cl_grid + beta[2] * heat_hold
axes[0].plot(cl_grid, yhat_cl, "#2c7fb8", lw=2.2)
sc = axes[0].scatter(year_df["Tmean_chill_lag1"], year_df["yield_tn_ha"],
                     c=year_df["yield_year"], cmap="viridis", s=110,
                     edgecolor="k", lw=0.5, zorder=3)
for _, r in year_df.iterrows():
    axes[0].annotate(int(r["yield_year"]), (r["Tmean_chill_lag1"], r["yield_tn_ha"]),
                     fontsize=10, xytext=(5, 5), textcoords="offset points")
axes[0].set_xlabel("Mean T during chill window (May–Aug Y-1, °C)", fontsize=12)
axes[0].set_ylabel("Yield (t ha$^{-1}$)", fontsize=12)
axes[0].set_title("(a)", loc="left")
axes[0].grid(alpha=0.3)

h_grid = np.linspace(year_df["Tmean_mean_CREC"].min() - 0.3,
                     year_df["Tmean_mean_CREC"].max() + 0.3, 100)
chill_hold = year_df["Tmean_chill_lag1"].mean()
yhat_h = beta[0] + beta[1] * chill_hold + beta[2] * h_grid
axes[1].plot(h_grid, yhat_h, "#d95f02", lw=2.2)
axes[1].scatter(year_df["Tmean_mean_CREC"], year_df["yield_tn_ha"],
                c=year_df["yield_year"], cmap="viridis", s=110,
                edgecolor="k", lw=0.5, zorder=3)
for _, r in year_df.iterrows():
    axes[1].annotate(int(r["yield_year"]), (r["Tmean_mean_CREC"], r["yield_tn_ha"]),
                     fontsize=10, xytext=(5, 5), textcoords="offset points")
axes[1].set_xlabel("Mean T during fruit growth (Jan–Mar Y, °C)", fontsize=12)
axes[1].set_ylabel("Yield (t ha$^{-1}$)", fontsize=12)
axes[1].set_title("(b)", loc="left")
axes[1].grid(alpha=0.3)

    # suptitle removed — title goes in LaTeX caption
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "fig_partial_effects.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("  -> fig_partial_effects.png")

# --- Fig C: counterfactual bars ---
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=600)
bar_w = 0.22
x_pos = np.arange(2)
obs_vals = [float(year_df.loc[idx_failure[y], "yield_tn_ha"]) for y in failure_yrs]

def get_cf(yr, scen, col):
    return cf_df[(cf_df.year == yr) & (cf_df.scenario == scen)][col].values[0]

chill_med = [get_cf(y, "chill", "cf_median") for y in failure_yrs]
chill_lo = [get_cf(y, "chill", "cf_ci_low") for y in failure_yrs]
chill_hi = [get_cf(y, "chill", "cf_ci_high") for y in failure_yrs]
heat_med = [get_cf(y, "heat", "cf_median") for y in failure_yrs]
full_med = [get_cf(y, "full", "cf_median") for y in failure_yrs]
full_lo = [get_cf(y, "full", "cf_ci_low") for y in failure_yrs]
full_hi = [get_cf(y, "full", "cf_ci_high") for y in failure_yrs]

ax.bar(x_pos - 1.5 * bar_w, obs_vals, bar_w, color="#525252", label="Observed")
ax.bar(x_pos - 0.5 * bar_w, chill_med, bar_w, color="#2c7fb8",
       label="Counterfactual: chill restored")
ax.errorbar(x_pos - 0.5 * bar_w, chill_med,
            yerr=[np.array(chill_med) - np.array(chill_lo),
                  np.array(chill_hi) - np.array(chill_med)],
            fmt="none", color="k", capsize=3, lw=1)
ax.bar(x_pos + 0.5 * bar_w, heat_med, bar_w, color="#d95f02",
       label="Counterfactual: heat restored")
ax.bar(x_pos + 1.5 * bar_w, full_med, bar_w, color="#1a6332",
       label="Counterfactual: both restored")
ax.errorbar(x_pos + 1.5 * bar_w, full_med,
            yerr=[np.array(full_med) - np.array(full_lo),
                  np.array(full_hi) - np.array(full_med)],
            fmt="none", color="k", capsize=3, lw=1)
ax.axhline(float(year_df["yield_tn_ha"].mean()), color="k", ls=":", lw=1.2,
           alpha=0.5, label="Historical mean")
ax.set_xticks(x_pos)
ax.set_xticklabels(["2016 (El Niño 15/16)", "2024 (El Niño 23/24)"], fontsize=12)
ax.set_ylabel("Yield (t ha$^{-1}$)", fontsize=13)
    # title removed — goes in LaTeX caption
ax.legend(fontsize=11, loc="upper left", frameon=True)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "fig_counterfactual_bars.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("  -> fig_counterfactual_bars.png")

# --- Fig D: residual diagnostics ---
resid = ols.resid
fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=600)
axes[0].scatter(ols.fittedvalues, resid, s=60, alpha=0.8, color="#2c7fb8",
                edgecolor="k", lw=0.4)
axes[0].axhline(0, color="r", ls="--", lw=1)
axes[0].set_xlabel("Fitted yield (t ha$^{-1}$)")
axes[0].set_ylabel("Residual")
axes[0].set_title("(a)", loc="left")
axes[0].grid(alpha=0.3)
for i, yr in enumerate(year_df["yield_year"]):
    axes[0].annotate(int(yr), (ols.fittedvalues[i], resid[i]),
                     xytext=(4, 4), textcoords="offset points", fontsize=10)

sm.qqplot(resid, line="45", ax=axes[1], markerfacecolor="#2c7fb8",
          markeredgecolor="k", markersize=5)
axes[1].set_title("(b)", loc="left")
axes[1].grid(alpha=0.3)

axes[2].hist(resid, bins=6, color="#2c7fb8", edgecolor="k", alpha=0.85)
axes[2].axvline(0, color="r", ls="--", lw=1)
axes[2].set_xlabel("Residual")
axes[2].set_ylabel("Frequency")
axes[2].set_title("(c)", loc="left")
axes[2].grid(alpha=0.3)

    # suptitle removed — title goes in LaTeX caption
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "fig_residual_diagnostics.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("  -> fig_residual_diagnostics.png")

# ----------------------------------------------------------------------------
# Summary file
# ----------------------------------------------------------------------------
summary = f"""PRIMARY MODEL SUMMARY — Chill x Heat attribution of Tacna olive yield
=========================================================================

Target: Remote Sensing (MDPI), Q1.
Unit of analysis: year (n = 8).
Predictors:
    X1 = Tmean_chill_lag1  (May-Aug of Y-1, chill-season mean air temperature)
    X2 = Tmean_mean_CREC   (Jan-Mar of Y, fruit-growth mean air temperature)
Year-level correlation r(X1, X2) = {year_df[PREDICTORS].corr().iloc[0, 1]:.3f}  (orthogonal)

In-sample OLS:
  R2      = {ols.rsquared:.4f}
  adj R2  = {ols.rsquared_adj:.4f}
  F       = {ols.fvalue:.2f}  (df = {int(ols.df_model)}, {int(ols.df_resid)})
  p       = {ols.f_pvalue:.4f}

Coefficients:
{coef_tbl.round(4).to_string(index=False)}

Standardized coefficients:
  Tmean_chill_lag1 beta* = {std_ols.params[1]:+.3f}
  Tmean_mean_CREC  beta* = {std_ols.params[2]:+.3f}

Leave-One-Year-Out CV:
  pooled R2  = {pooled['R2']:.3f}
  RMSE       = {pooled['RMSE']:.3f} t/ha
  MAE        = {pooled['MAE']:.3f} t/ha
  bias       = {pooled['bias']:+.3f}

Year-block bootstrap 95% CI (n_boot = {len(boot_df)}):
{ci.round(4).to_string()}

Bootstrap two-sided p-values:
{chr(10).join(f"  {k}:  p = {v:.4f}" for k, v in p_boot.items())}

Permutation test (n = {len(perm_stats)}):
  observed R2 = {obs_stat:.3f}
  null mean   = {np.mean(perm_stats):.3f}
  p (one-sided) = {p_perm:.4f}

Panel sensitivity (parcel random intercept):
  ICC(parcel) = {icc if isinstance(icc, float) else float('nan'):.3f}   (small -> climate-dominated variance)

Counterfactual attribution (full climate restored):
{cf_df[cf_df.scenario == 'full'][['year', 'obs_yield', 'cf_median', 'cf_ci_low', 'cf_ci_high', 'delta_median']].round(3).to_string(index=False)}
"""
(OUT_DIR / "tables" / "MODEL_SUMMARY.txt").write_text(summary, encoding="utf-8")
print(summary)
