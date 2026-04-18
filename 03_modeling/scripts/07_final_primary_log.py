"""
07_final_primary_log.py
-----------------------
FINAL primary model for the manuscript.

After systematic robustness checks (see 06_robustness_and_loyo_variants.py),
the model with the best balance of parsimony, mechanistic interpretability
and out-of-sample stability is:

    log(yield + 0.5) ~ Tmean_chill_lag1 + Tmean_mean_CREC       (primary)
    log(yield + 0.5) ~ Tmean_chill_lag1                          (sensitivity)

Year-level OLS on 8 yield years (2021 excluded: COVID, no harvest).

Why log(y + 0.5):
    - Bounds predictions to the non-negative support of biological yield.
    - Aligns with the multiplicative structure of yield failure (a warm
      winter does not subtract a fixed tonnage, it suppresses productive
      capacity by a factor).
    - Dramatically improves LOYO stability when extreme ENSO years are
      held out.

All inference (bootstrap CIs, permutation p-values, counterfactuals) is
carried out in log space and back-transformed for reporting.
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
from pathlib import Path

BRANCH = Path(r"D:/olive_yield_RS_chill")
FEAT = BRANCH / "01_preprocessing" / "outputs" / "feature_matrix_chill_complete.csv"
OUT_DIR = BRANCH / "03_modeling" / "outputs"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
N_BOOT = 5000
N_PERM = 5000
EPS = 0.5
CLIP_MIN = 0.0

# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return dict(
        R2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        RMSE=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        MAE=float(np.mean(np.abs(y_true - y_pred))),
        bias=float(np.mean(y_pred - y_true)),
        n=int(len(y_true)),
    )


def back_transform(log_pred):
    return np.clip(np.exp(log_pred) - EPS, CLIP_MIN, None)


print("=" * 72)
print("FINAL PRIMARY MODEL — log(yield+0.5) ~ Tmean_chill + Tmean_CREC")
print("=" * 72)

# --- Data ---
df = pd.read_csv(FEAT)
year_df = df.groupby("yield_year").agg(
    yield_tn_ha=("yield_tn_ha", "mean"),
    yield_sd=("yield_tn_ha", "std"),
    Tmean_chill_lag1=("Tmean_chill_lag1", "first"),
    Tmean_mean_CREC=("Tmean_mean_CREC", "first"),
    CP_lag1=("CP_lag1", "first"),
).reset_index()

y = year_df["yield_tn_ha"].values
log_y = np.log(y + EPS)
X = year_df[["Tmean_chill_lag1", "Tmean_mean_CREC"]].values
X_ols = sm.add_constant(X)

print(f"\n[Data] {len(y)} years: {year_df['yield_year'].tolist()}")
print(f"       yield range: {y.min():.2f} - {y.max():.2f} t/ha")
print(f"       log(y+0.5) range: {log_y.min():.3f} - {log_y.max():.3f}")

# --- Primary OLS in log space ---
ols = sm.OLS(log_y, X_ols).fit()
print("\n[Primary OLS — log space]")
print(ols.summary())
(OUT_DIR / "tables" / "final_primary_fit.txt").write_text(str(ols.summary()),
                                                          encoding="utf-8")

# In-sample (back-transformed)
yhat_in_log = ols.fittedvalues
yhat_in = back_transform(yhat_in_log)
m_in_log = metrics(log_y, yhat_in_log)
m_in_raw = metrics(y, yhat_in)
print(f"\n[In-sample — log space]  R2 = {m_in_log['R2']:.4f}  RMSE = {m_in_log['RMSE']:.3f}")
print(f"[In-sample — back-transformed]  R2 = {m_in_raw['R2']:.4f}  "
      f"RMSE = {m_in_raw['RMSE']:.3f} t/ha  MAE = {m_in_raw['MAE']:.3f} t/ha")

coef_tbl = pd.DataFrame({
    "term": ["Intercept", "Tmean_chill_lag1", "Tmean_mean_CREC"],
    "estimate": ols.params,
    "std_error": ols.bse,
    "t": ols.tvalues,
    "p_value": ols.pvalues,
    "ci_low": ols.conf_int()[:, 0],
    "ci_high": ols.conf_int()[:, 1],
})
coef_tbl.to_csv(OUT_DIR / "tables" / "final_primary_coefficients.csv", index=False)
print("\n[Coefficients]")
print(coef_tbl.round(4).to_string(index=False))

# --- Sensitivity: single-predictor model ---
print("\n" + "=" * 72)
print("SENSITIVITY — single predictor log(y+0.5) ~ Tmean_chill_lag1")
print("=" * 72)
Xs = year_df[["Tmean_chill_lag1"]].values
ols1 = sm.OLS(log_y, sm.add_constant(Xs)).fit()
print(ols1.summary())
(OUT_DIR / "tables" / "final_single_predictor_fit.txt").write_text(
    str(ols1.summary()), encoding="utf-8")
m_in_log_1 = metrics(log_y, ols1.fittedvalues)
m_in_raw_1 = metrics(y, back_transform(ols1.fittedvalues))
print(f"\n[In-sample log] R2 = {m_in_log_1['R2']:.4f}")
print(f"[In-sample raw] R2 = {m_in_raw_1['R2']:.4f}  RMSE = {m_in_raw_1['RMSE']:.3f}")

# --- LOYO for both models (log space then back-transform) ---
print("\n" + "=" * 72)
print("LEAVE-ONE-YEAR-OUT CV (log space + physical clip)")
print("=" * 72)

def loyo(X_mat, log_y):
    n = len(log_y)
    preds_log = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        m = sm.OLS(log_y[tr], sm.add_constant(X_mat[tr])).fit()
        xi = np.array([1.0] + list(X_mat[i]))
        preds_log[i] = float(xi @ m.params)
    return preds_log

log_loyo_2 = loyo(X, log_y)
log_loyo_1 = loyo(Xs, log_y)

loyo_tbl = pd.DataFrame({
    "year": year_df["yield_year"],
    "obs_yield": y,
    "obs_log": log_y,
    "pred_log_2p": log_loyo_2,
    "pred_yield_2p": back_transform(log_loyo_2),
    "pred_log_1p": log_loyo_1,
    "pred_yield_1p": back_transform(log_loyo_1),
})
loyo_tbl.to_csv(OUT_DIR / "tables" / "final_loyo_predictions.csv", index=False)

m_loyo_log_2 = metrics(log_y, log_loyo_2)
m_loyo_raw_2 = metrics(y, back_transform(log_loyo_2))
m_loyo_log_1 = metrics(log_y, log_loyo_1)
m_loyo_raw_1 = metrics(y, back_transform(log_loyo_1))

print(f"\n[2-predictor] LOYO log: R2={m_loyo_log_2['R2']:.3f}  "
      f"raw: R2={m_loyo_raw_2['R2']:.3f}  RMSE={m_loyo_raw_2['RMSE']:.2f} t/ha  "
      f"MAE={m_loyo_raw_2['MAE']:.2f}")
print(f"[1-predictor] LOYO log: R2={m_loyo_log_1['R2']:.3f}  "
      f"raw: R2={m_loyo_raw_1['R2']:.3f}  RMSE={m_loyo_raw_1['RMSE']:.2f} t/ha  "
      f"MAE={m_loyo_raw_1['MAE']:.2f}")
print("\n[Per-year LOYO — 2-predictor]")
print(loyo_tbl[["year", "obs_yield", "pred_yield_2p", "pred_yield_1p"]].round(3).to_string(index=False))

# --- Year-block bootstrap (in log space) ---
print("\n" + "=" * 72)
print(f"YEAR-BLOCK BOOTSTRAP — log space (n_boot = {N_BOOT})")
print("=" * 72)

idx_arr = np.arange(len(y))
boot_params = []
for b in range(N_BOOT):
    sel = RNG.choice(idx_arr, size=len(y), replace=True)
    if len(np.unique(sel)) < 3:
        continue
    try:
        mb = sm.OLS(log_y[sel], sm.add_constant(X[sel])).fit()
        boot_params.append(mb.params)
    except Exception:
        continue

boot_df = pd.DataFrame(boot_params, columns=["Intercept", "Tmean_chill_lag1", "Tmean_mean_CREC"])
boot_df.to_csv(OUT_DIR / "tables" / "final_bootstrap.csv", index=False)
ci = boot_df.quantile([0.025, 0.5, 0.975]).T
ci.columns = ["ci_low_boot", "median_boot", "ci_high_boot"]
print(f"\n[Bootstrap CIs, log space]  n_success = {len(boot_df)}")
print(ci.round(4).to_string())

p_boot = {
    name: 2 * min(float(np.mean(boot_df[name] > 0)),
                  float(np.mean(boot_df[name] < 0)))
    for name in ["Tmean_chill_lag1", "Tmean_mean_CREC"]
}
print("\n[Bootstrap two-sided p-values]")
for k, v in p_boot.items():
    print(f"  {k}:  p = {v:.4f}")

# --- Permutation test ---
print("\n" + "=" * 72)
print(f"PERMUTATION TEST (n = {N_PERM})")
print("=" * 72)

perm_r2 = []
for b in range(N_PERM):
    y_perm = RNG.permutation(log_y)
    try:
        mp = sm.OLS(y_perm, X_ols).fit()
        perm_r2.append(mp.rsquared)
    except Exception:
        continue
perm_r2 = np.array(perm_r2)
p_perm = float(np.mean(perm_r2 >= ols.rsquared))
print(f"  observed R2 (log) = {ols.rsquared:.4f}")
print(f"  null mean         = {np.mean(perm_r2):.4f}")
print(f"  null 95th pct     = {np.quantile(perm_r2, 0.95):.4f}")
print(f"  one-sided p       = {p_perm:.4f}")
pd.DataFrame({
    "observed_R2_log": [ols.rsquared],
    "perm_mean_R2": [float(np.mean(perm_r2))],
    "perm_95pct_R2": [float(np.quantile(perm_r2, 0.95))],
    "p_value": [p_perm],
    "n_perm": [len(perm_r2)],
}).to_csv(OUT_DIR / "tables" / "final_permutation.csv", index=False)

# --- Counterfactual attribution (log space) ---
print("\n" + "=" * 72)
print("COUNTERFACTUAL ATTRIBUTION — log space + back-transform")
print("=" * 72)

failure_yrs = [2016, 2024]
idx_fail = {yr: int(year_df.index[year_df["yield_year"] == yr][0]) for yr in failure_yrs}
idx_nonfail = year_df.index[~year_df["yield_year"].isin(failure_yrs)].tolist()
chill_baseline = float(year_df.loc[idx_nonfail, "Tmean_chill_lag1"].mean())
heat_baseline = float(year_df.loc[idx_nonfail, "Tmean_mean_CREC"].mean())
print(f"\n  Non-failure baseline:")
print(f"    Tmean_chill = {chill_baseline:.3f} C")
print(f"    Tmean_CREC  = {heat_baseline:.3f} C")

cf_raw = {(yr, s): [] for yr in failure_yrs for s in ("full", "chill", "heat")}
for b in range(N_BOOT):
    sel = RNG.choice(idx_arr, size=len(y), replace=True)
    if len(np.unique(sel)) < 3:
        continue
    try:
        mb = sm.OLS(log_y[sel], sm.add_constant(X[sel])).fit()
    except Exception:
        continue
    beta = mb.params
    for yr in failure_yrs:
        row = year_df.iloc[idx_fail[yr]]
        obs_chill = row["Tmean_chill_lag1"]
        obs_heat = row["Tmean_mean_CREC"]
        full = beta[0] + beta[1] * chill_baseline + beta[2] * heat_baseline
        chill_only = beta[0] + beta[1] * chill_baseline + beta[2] * obs_heat
        heat_only = beta[0] + beta[1] * obs_chill + beta[2] * heat_baseline
        cf_raw[(yr, "full")].append(float(full))
        cf_raw[(yr, "chill")].append(float(chill_only))
        cf_raw[(yr, "heat")].append(float(heat_only))

cf_records = []
for yr in failure_yrs:
    obs_y = float(year_df.loc[idx_fail[yr], "yield_tn_ha"])
    for scen in ("full", "chill", "heat"):
        log_arr = np.array(cf_raw[(yr, scen)])
        raw_arr = back_transform(log_arr)
        cf_records.append({
            "year": yr,
            "scenario": scen,
            "obs_yield": obs_y,
            "cf_median_t_ha": float(np.median(raw_arr)),
            "cf_ci_low_t_ha": float(np.quantile(raw_arr, 0.025)),
            "cf_ci_high_t_ha": float(np.quantile(raw_arr, 0.975)),
            "delta_median": float(np.median(raw_arr) - obs_y),
            "delta_ci_low": float(np.quantile(raw_arr - obs_y, 0.025)),
            "delta_ci_high": float(np.quantile(raw_arr - obs_y, 0.975)),
            "ratio_cf_obs": float(np.median(raw_arr) / obs_y) if obs_y > 0 else np.nan,
        })
cf_df = pd.DataFrame(cf_records)
cf_df.to_csv(OUT_DIR / "tables" / "final_counterfactual.csv", index=False)
print("\n[Counterfactual results — back-transformed to t/ha]")
print(cf_df.round(3).to_string(index=False))

# --- Figures ---
print("\n" + "=" * 72)
print("FIGURES")
print("=" * 72)

cmap = plt.get_cmap("viridis")
n_years = len(year_df)

# Figure: final obs vs LOYO pred (2-predictor, back-transformed)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), dpi=600)

# (a) raw scale
ax = axes[0]
colors = [cmap(i / max(n_years - 1, 1)) for i in range(n_years)]
FAILURE_YRS = {2016, 2024}
for i, (_, r) in enumerate(loyo_tbl.sort_values("year").reset_index(drop=True).iterrows()):
    is_fail = int(r["year"]) in FAILURE_YRS
    ax.scatter(r["obs_yield"], r["pred_yield_2p"], color=colors[i],
               s=220 if is_fail else 130,
               edgecolor="#c0392b" if is_fail else "k",
               lw=2.2 if is_fail else 0.6,
               marker="*" if is_fail else "o", zorder=4 if is_fail else 3)
    ax.annotate(int(r["year"]), (r["obs_yield"], r["pred_yield_2p"]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=12 if is_fail else 11,
                fontweight="bold" if is_fail else "normal",
                color="#c0392b" if is_fail else "k")
lim = (0, max(y.max(), loyo_tbl["pred_yield_2p"].max()) * 1.10)
ax.plot(lim, lim, "k--", lw=1, alpha=0.7)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Observed yield (t ha$^{-1}$)", fontsize=13)
ax.set_ylabel("LOYO-predicted yield (t ha$^{-1}$)", fontsize=13)
ax.text(0.05, 0.95, "(a)", transform=ax.transAxes, fontsize=14,
        fontweight="bold", va="top", ha="left")
ax.text(0.97, 0.05,
        f"$R^{{2}} = {m_loyo_raw_2['R2']:.2f}$\nRMSE $= {m_loyo_raw_2['RMSE']:.2f}$ t ha$^{{-1}}$",
        transform=ax.transAxes, fontsize=12, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.5", alpha=0.85))
ax.grid(alpha=0.3)

# (b) log scale
ax = axes[1]
for i, (_, r) in enumerate(loyo_tbl.sort_values("year").reset_index(drop=True).iterrows()):
    is_fail = int(r["year"]) in FAILURE_YRS
    ax.scatter(r["obs_log"], r["pred_log_2p"], color=colors[i],
               s=220 if is_fail else 130,
               edgecolor="#c0392b" if is_fail else "k",
               lw=2.2 if is_fail else 0.6,
               marker="*" if is_fail else "o", zorder=4 if is_fail else 3)
    ax.annotate(int(r["year"]), (r["obs_log"], r["pred_log_2p"]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=12 if is_fail else 11,
                fontweight="bold" if is_fail else "normal",
                color="#c0392b" if is_fail else "k")
lim2 = (min(log_y.min(), log_loyo_2.min()) - 0.3,
        max(log_y.max(), log_loyo_2.max()) + 0.3)
ax.plot(lim2, lim2, "k--", lw=1, alpha=0.7)
ax.set_xlim(lim2); ax.set_ylim(lim2)
ax.set_xlabel("Observed log(yield + 0.5)", fontsize=13)
ax.set_ylabel("LOYO-predicted log(yield + 0.5)", fontsize=13)
ax.text(0.05, 0.95, "(b)", transform=ax.transAxes, fontsize=14,
        fontweight="bold", va="top", ha="left")
ax.text(0.97, 0.05,
        f"$R^{{2}}_{{\\log}} = {m_loyo_log_2['R2']:.2f}$\nRMSE$_{{\\log}} = {m_loyo_log_2['RMSE']:.2f}$",
        transform=ax.transAxes, fontsize=12, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.5", alpha=0.85))
ax.grid(alpha=0.3)

    # suptitle removed — title goes in LaTeX caption
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "fig_final_loyo.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("  -> fig_final_loyo.png")

# Figure: counterfactual
fig, ax = plt.subplots(figsize=(8.4, 5.5), dpi=600)
bar_w = 0.22
x_pos = np.arange(2)
obs_vals = [float(year_df.loc[idx_fail[yr], "yield_tn_ha"]) for yr in failure_yrs]

def get_cf(yr, scen, col):
    return cf_df[(cf_df.year == yr) & (cf_df.scenario == scen)][col].values[0]

chill_med = [get_cf(y_, "chill", "cf_median_t_ha") for y_ in failure_yrs]
chill_lo = [get_cf(y_, "chill", "cf_ci_low_t_ha") for y_ in failure_yrs]
chill_hi = [get_cf(y_, "chill", "cf_ci_high_t_ha") for y_ in failure_yrs]
heat_med = [get_cf(y_, "heat", "cf_median_t_ha") for y_ in failure_yrs]
heat_lo = [get_cf(y_, "heat", "cf_ci_low_t_ha") for y_ in failure_yrs]
heat_hi = [get_cf(y_, "heat", "cf_ci_high_t_ha") for y_ in failure_yrs]
full_med = [get_cf(y_, "full", "cf_median_t_ha") for y_ in failure_yrs]
full_lo = [get_cf(y_, "full", "cf_ci_low_t_ha") for y_ in failure_yrs]
full_hi = [get_cf(y_, "full", "cf_ci_high_t_ha") for y_ in failure_yrs]

ax.bar(x_pos - 1.5 * bar_w, obs_vals, bar_w, color="#525252",
       edgecolor="k", linewidth=0.6, label="Observed")
ax.bar(x_pos - 0.5 * bar_w, chill_med, bar_w, color="#2c7fb8",
       edgecolor="k", linewidth=0.6, label="Chill restored")
ax.errorbar(x_pos - 0.5 * bar_w, chill_med,
            yerr=[np.array(chill_med) - np.array(chill_lo),
                  np.array(chill_hi) - np.array(chill_med)],
            fmt="none", color="k", capsize=3, lw=1)
ax.bar(x_pos + 0.5 * bar_w, heat_med, bar_w, color="#d95f02",
       edgecolor="k", linewidth=0.6, label="Heat restored")
ax.errorbar(x_pos + 0.5 * bar_w, heat_med,
            yerr=[np.array(heat_med) - np.array(heat_lo),
                  np.array(heat_hi) - np.array(heat_med)],
            fmt="none", color="k", capsize=3, lw=1)
ax.bar(x_pos + 1.5 * bar_w, full_med, bar_w, color="#1a6332",
       edgecolor="k", linewidth=0.6, label="Both restored")
ax.errorbar(x_pos + 1.5 * bar_w, full_med,
            yerr=[np.array(full_med) - np.array(full_lo),
                  np.array(full_hi) - np.array(full_med)],
            fmt="none", color="k", capsize=3, lw=1)

# Numeric value labels on top of each bar / error bar upper cap
y_hist_mean = float(y.mean())
ax.axhline(y_hist_mean, color="k", ls=":", lw=1.2, alpha=0.7,
           label=f"Historical mean yield ({y_hist_mean:.2f} t ha$^{{-1}}$)")

def label_bar(xv, med, hi=None):
    top = hi if hi is not None else med
    ax.text(xv, top + 0.35, f"{med:.2f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")

for k, xv in enumerate(x_pos):
    label_bar(xv - 1.5 * bar_w, obs_vals[k])
    label_bar(xv - 0.5 * bar_w, chill_med[k], chill_hi[k])
    label_bar(xv + 0.5 * bar_w, heat_med[k], heat_hi[k])
    label_bar(xv + 1.5 * bar_w, full_med[k], full_hi[k])

ax.set_xticks(x_pos)
ax.set_xticklabels(["2016\n(El Ni\u00f1o 2015/16)",
                    "2024\n(El Ni\u00f1o 2023/24)"], fontsize=12)
ax.set_ylabel("Yield (t ha$^{-1}$)", fontsize=13)
ax.legend(fontsize=10.5, loc="lower center", bbox_to_anchor=(0.5, 1.02),
          ncol=5, frameon=False, handlelength=1.8, columnspacing=1.2)
ax.grid(alpha=0.3, axis="y")
ax.set_axisbelow(True)
# Y-limit must cover the tallest error-bar top AND leave room for numeric
# labels placed at top + 0.35. Include chill/heat/full upper CIs (not just full).
y_top = max(max(chill_hi), max(heat_hi), max(full_hi))
ax.set_ylim(0, y_top * 1.12)
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "fig_final_counterfactual.png", dpi=600,
            bbox_inches="tight")
plt.close(fig)
print("  -> fig_final_counterfactual.png")

# Figure: Tmean_chill partial effect on raw yield (back-transform)
fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=600)
cl_grid = np.linspace(year_df["Tmean_chill_lag1"].min() - 0.5,
                      year_df["Tmean_chill_lag1"].max() + 0.5, 200)
heat_hold = year_df["Tmean_mean_CREC"].mean()
log_yhat = ols.params[0] + ols.params[1] * cl_grid + ols.params[2] * heat_hold
yhat = back_transform(log_yhat)
ax.plot(cl_grid, yhat, "#2c7fb8", lw=2.4, label="Model (back-transformed)")
# Bootstrap band
boot_curves = []
for row in boot_df.itertuples(index=False):
    boot_curves.append(back_transform(row.Intercept + row.Tmean_chill_lag1 * cl_grid
                                      + row.Tmean_mean_CREC * heat_hold))
boot_curves = np.array(boot_curves)
lo = np.quantile(boot_curves, 0.025, axis=0)
hi = np.quantile(boot_curves, 0.975, axis=0)
ax.fill_between(cl_grid, lo, hi, color="#2c7fb8", alpha=0.18,
                label="95% bootstrap CI")
sc = ax.scatter(year_df["Tmean_chill_lag1"], year_df["yield_tn_ha"],
                c=year_df["yield_year"], cmap="viridis", s=140,
                edgecolor="k", lw=0.6, zorder=3)
for _, r in year_df.iterrows():
    ax.annotate(int(r["yield_year"]),
                (r["Tmean_chill_lag1"], r["yield_tn_ha"]),
                xytext=(6, 6), textcoords="offset points", fontsize=11)
cb = plt.colorbar(sc, ax=ax, label="Year")
ax.set_xlabel("Mean temperature during chill window (May–Aug Y-1, °C)",
              fontsize=13)
ax.set_ylabel("Olive yield (t ha$^{-1}$)", fontsize=13)
    # title removed — goes in LaTeX caption
ax.legend(loc="upper right", fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "fig_final_partial_chill.png", dpi=600,
            bbox_inches="tight")
plt.close(fig)
print("  -> fig_final_partial_chill.png")

# --- Summary text ---
summary = f"""FINAL PRIMARY MODEL — summary
=================================

Model:     log(yield + 0.5) ~ Tmean_chill_lag1 + Tmean_mean_CREC
Unit:      year (n = 8; 2021 excluded for COVID no-harvest)
Predictors:
    Tmean_chill_lag1  = mean T during May-Aug of year Y-1 (chill window)
    Tmean_mean_CREC   = mean T during Jan-Mar of year Y (fruit growth)
    r(predictors)     = 0.060 (orthogonal at year level)

In-sample fit (log space):
  R2       = {ols.rsquared:.4f}
  adj R2   = {ols.rsquared_adj:.4f}
  F({int(ols.df_model)}, {int(ols.df_resid)}) = {ols.fvalue:.2f}   p = {ols.f_pvalue:.4f}

Coefficients (log space):
{coef_tbl.round(4).to_string(index=False)}

Back-transformed fit (raw t/ha):
  R2   = {m_in_raw['R2']:.3f}
  RMSE = {m_in_raw['RMSE']:.3f} t/ha
  MAE  = {m_in_raw['MAE']:.3f} t/ha

Leave-One-Year-Out cross-validation:
  LOYO log-space:         R2 = {m_loyo_log_2['R2']:.3f}
  LOYO back-transformed:  R2 = {m_loyo_raw_2['R2']:.3f}
                          RMSE = {m_loyo_raw_2['RMSE']:.3f} t/ha
                          MAE  = {m_loyo_raw_2['MAE']:.3f} t/ha

Year-block bootstrap 95% CI (log space, n = {len(boot_df)}):
{ci.round(4).to_string()}

Bootstrap two-sided p-values:
  Tmean_chill_lag1  p = {p_boot['Tmean_chill_lag1']:.4f}
  Tmean_mean_CREC   p = {p_boot['Tmean_mean_CREC']:.4f}

Permutation test (n = {len(perm_r2)}):
  observed R2 = {ols.rsquared:.3f}
  null mean   = {np.mean(perm_r2):.3f}
  p (1-sided) = {p_perm:.4f}

Sensitivity — single predictor log(y+0.5) ~ Tmean_chill_lag1:
  in-sample R2 = {ols1.rsquared:.4f}
  LOYO raw R2  = {m_loyo_raw_1['R2']:.3f}
  Coefficient  = {ols1.params[1]:+.4f} (SE = {ols1.bse[1]:.4f}, p = {ols1.pvalues[1]:.4f})

Counterfactual attribution (back-transformed):
{cf_df.round(3).to_string(index=False)}
"""
(OUT_DIR / "tables" / "FINAL_MODEL_SUMMARY.txt").write_text(summary, encoding="utf-8")
print(summary)
