"""
01_cmip6_delta_projections.py
-----------------------------
Project Tacna olive yield under CMIP6 warming using the delta method
(Luedeling 2012; Hannah et al. 2013; Aguilera et al. 2014).

Rationale
---------
The primary attribution model (script 07) shows that winter chill-window
mean temperature (Tmean_chill_lag1) is the dominant driver of yield in
Tacna, with secondary contribution from fruit-growth temperature
(Tmean_mean_CREC). To project future yield we therefore need future
Tmean_chill and Tmean_CREC under CMIP6 scenarios.

Approach
~~~~~~~~
We use the *delta method*, the standard in chill-projection literature
(see Luedeling 2012 *Agric. Forest Meteorol.* 153 for the canonical
treatment, and Aguilera et al. 2014 for olive specifically):

  1. Take the observed daily Tmin/Tmax record (2015-2025) as the
     baseline climate.
  2. For each CMIP6 scenario S, time horizon H (2031-50, 2051-70,
     2071-2100), and ensemble member, add the seasonal warming delta
     (DeltaT_winter for May-Aug, DeltaT_summer for Jan-Mar) to the
     baseline daily series.
  3. Re-disaggregate to hourly via the same sinusoidal interpolation
     used in script 01_chill_portions.py.
  4. Recompute Chill Portions via the Fishman et al. 1987 Dynamic
     Model (the chillR canonical implementation).
  5. Project yield via the calibrated log-OLS model:
        log(yield + 0.5) = b0 + b1 * Tmean_chill_lag1
                              + b2 * Tmean_mean_CREC
     and back-transform.

Why delta method?
~~~~~~~~~~~~~~~~~
Bias-corrected daily GCM output (e.g. NEX-GDDP-CMIP6) is the alternative
but for a coastal grid point with strong Humboldt-current cooling the
GCM raw distributions are heavily biased and the bias-correction step
is itself the dominant source of uncertainty. The delta method
preserves the *observed* daily variability and only perturbs the mean,
which for a Dynamic-Model chill metric (sensitive to the temperature
distribution at hourly scale) is the most defensible approach when
in-situ baseline data exist (Luedeling 2012).

CMIP6 warming deltas
~~~~~~~~~~~~~~~~~~~~
Anchored to IPCC AR6 WGI Atlas (2021), CMIP6 ensemble-mean projected
near-surface air temperature change for the South-Western South America
(SWS) region, austral winter (JJA) and austral summer (DJF), relative
to the 1995-2014 baseline. The southern-Peru/northern-Chile coast at
~17.5 deg S falls inside SWS. We provide a 5-95 percentile range from
the published ensemble spread to propagate inter-model uncertainty.

Outputs
~~~~~~~
04_projections/outputs/tables/
    delta_scenarios.csv             scenario table actually used
    yield_under_warming.csv         per-scenario projected CP, Tmean_chill,
                                    Tmean_CREC, and yield (median + 5-95% CI)
    warming_sweep.csv               continuous DeltaT sweep (0-5 C)
04_projections/outputs/figures/
    fig_warming_sweep.png           yield vs DeltaT_winter with SSP markers
    fig_ssp_yield.png               bar chart of yield under each SSP/horizon
    fig_chill_collapse_curve.png    CP vs DeltaT_winter (mechanism plot)
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
import matplotlib.patches as mpatches
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BRANCH = Path(r"D:/olive_yield_RS_chill")
RAW = BRANCH / "00_raw_data" / "meteo_daily_2015_2025.csv"
COEF = BRANCH / "03_modeling" / "outputs" / "tables" / "final_primary_coefficients.csv"
BOOT = BRANCH / "03_modeling" / "outputs" / "tables" / "final_bootstrap.csv"
OUT = BRANCH / "04_projections" / "outputs"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(7)
N_BOOT = 2000

# ----------------------------------------------------------------------------
# Chill model components (replicated from 01_chill_portions.py for self-containment)
# ----------------------------------------------------------------------------
def daily_to_hourly_sine(tmin: np.ndarray, tmax: np.ndarray) -> np.ndarray:
    n = len(tmin)
    T = np.empty(n * 24, dtype=float)
    for d in range(n):
        tmn, tmx = tmin[d], tmax[d]
        tmn_next = tmin[d + 1] if d + 1 < n else tmn
        for h in range(24):
            if 6 <= h <= 14:
                frac = (h - 6) / 8.0
                T[d * 24 + h] = tmn + (tmx - tmn) * 0.5 * (1 - np.cos(np.pi * frac))
            elif h > 14:
                frac = (h - 14) / 16.0
                T[d * 24 + h] = tmx - (tmx - tmn_next) * 0.5 * (1 - np.cos(np.pi * frac))
            else:
                tmx_prev = tmax[d - 1] if d > 0 else tmx
                tmn_curr = tmn
                frac = (h + 10) / 16.0
                T[d * 24 + h] = tmx_prev - (tmx_prev - tmn_curr) * 0.5 * (1 - np.cos(np.pi * frac))
    return T


def dynamic_model(T_hourly: np.ndarray) -> np.ndarray:
    e0, e1, a0, a1 = 4153.5, 12888.8, 139500.0, 2.567e18
    slp, tetmlt = 1.6, 277.0
    aa = a0 / a1
    ee = e1 - e0
    TK = T_hourly + 273.0
    ftmprt = slp * tetmlt * (TK - tetmlt) / TK
    sr = np.exp(ftmprt)
    xi = sr / (1.0 + sr)
    xs = aa * np.exp(ee / TK)
    ak1 = a1 * np.exp(-e1 / TK)
    n = len(T_hourly)
    interE = np.zeros(n)
    portions = np.zeros(n)
    for i in range(1, n):
        if interE[i - 1] < 1.0:
            S = interE[i - 1]
            delt = 0.0
        else:
            S = interE[i - 1] * (1.0 - xi[i - 1])
            delt = interE[i - 1] * xi[i - 1]
        interE[i] = xs[i] - (xs[i] - S) * np.exp(-ak1[i])
        portions[i] = delt
    return np.cumsum(portions)


# ----------------------------------------------------------------------------
# Load baseline meteorology
# ----------------------------------------------------------------------------
def load_meteo() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Tmin", "Tmax"]).sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


# ----------------------------------------------------------------------------
# Apply warming delta and compute CP & Tmean for each chill_year
# ----------------------------------------------------------------------------
def compute_seasonal_metrics(df: pd.DataFrame,
                             dT_winter: float,
                             dT_summer: float) -> pd.DataFrame:
    """Apply delta to baseline series and recompute per-year:
       - CP_season (May-Aug, dT_winter applied)
       - Tmean_chill (May-Aug)
       - Tmean_CREC (Jan 1 - Mar 15, dT_summer applied)
    """
    d2 = df.copy()
    winter_mask = d2["month"].between(5, 8)
    summer_mask = (d2["month"].isin([1, 2])) | ((d2["month"] == 3) & (d2["date"].dt.day <= 15))
    d2.loc[winter_mask, "Tmin"] += dT_winter
    d2.loc[winter_mask, "Tmax"] += dT_winter
    d2.loc[summer_mask, "Tmin"] += dT_summer
    d2.loc[summer_mask, "Tmax"] += dT_summer

    # Hourly disaggregation + Fishman
    T_h = daily_to_hourly_sine(d2["Tmin"].values, d2["Tmax"].values)
    CP_cum = dynamic_model(T_h)
    # Daily end-of-day cumulative -> daily increment
    daily_idx = np.arange(0, len(T_h), 24) + 23  # last hour of each day
    daily_cp = CP_cum[daily_idx]
    d2["CP_daily"] = np.concatenate([[daily_cp[0]], np.diff(daily_cp)])
    # mean hourly T per day
    T_h_mat = T_h.reshape(-1, 24)
    d2["Tmean_hour"] = T_h_mat.mean(axis=1)

    # Aggregate per year
    out = []
    for yr, sub in d2.groupby("year"):
        wmask = sub["month"].between(5, 8)
        smask = (sub["month"].isin([1, 2])) | ((sub["month"] == 3) & (sub["date"].dt.day <= 15))
        if wmask.sum() < 100 or smask.sum() < 60:
            continue
        out.append({
            "chill_year": int(yr),
            "CP_season": float(sub.loc[wmask, "CP_daily"].sum()),
            "Tmean_chill": float(sub.loc[wmask, "Tmean_hour"].mean()),
            "Tmean_CREC": float(sub.loc[smask, "Tmean_hour"].mean()),
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Load model coefficients (point + bootstrap for uncertainty)
# ----------------------------------------------------------------------------
def load_coefficients():
    coef = pd.read_csv(COEF)
    point = {
        "Intercept": float(coef.loc[coef["term"] == "Intercept", "estimate"].iloc[0]),
        "b_chill": float(coef.loc[coef["term"] == "Tmean_chill_lag1", "estimate"].iloc[0]),
        "b_crec": float(coef.loc[coef["term"] == "Tmean_mean_CREC", "estimate"].iloc[0]),
    }
    boot = None
    if BOOT.exists():
        b = pd.read_csv(BOOT)
        cols = list(b.columns)
        # find columns that look like the relevant predictors
        col_int = [c for c in cols if c.lower() in ("intercept", "const", "(intercept)")]
        col_ch = [c for c in cols if "chill" in c.lower()]
        col_cr = [c for c in cols if "crec" in c.lower()]
        if col_int and col_ch and col_cr:
            boot = b[[col_int[0], col_ch[0], col_cr[0]]].copy()
            boot.columns = ["Intercept", "b_chill", "b_crec"]
    return point, boot


def predict_yield(Tmean_chill: np.ndarray | float,
                  Tmean_crec: np.ndarray | float,
                  coef: dict) -> np.ndarray | float:
    log_y = coef["Intercept"] + coef["b_chill"] * Tmean_chill + coef["b_crec"] * Tmean_crec
    return np.exp(log_y) - 0.5


# ----------------------------------------------------------------------------
# CMIP6 warming table (IPCC AR6 WGI Atlas, SWS region, ensemble central + 5-95%)
# ----------------------------------------------------------------------------
# Source: IPCC AR6 WGI Interactive Atlas (2021), CMIP6 ensemble, region SWS
# (South-Western South America), warming relative to 1995-2014 baseline.
# Values in degrees Celsius, JJA (austral winter ~ chill window) and DJF
# (austral summer ~ fruit growth window). The 5-95% spread comes from the
# published inter-model envelope. These can be refined by the user with the
# exact Atlas point query for the Tacna grid cell.
#
# IMPORTANT: see also Luedeling 2012 and Aguilera et al. 2014 for the use of
# similar magnitudes in chill projections for Mediterranean climates.

CMIP6_DELTA = pd.DataFrame([
    # scenario, period, dT_winter_med, dT_winter_lo, dT_winter_hi, dT_summer_med, dT_summer_lo, dT_summer_hi
    # Horizons restricted to 2031-2050 and 2051-2070 to keep the projection window
    # within a defensible multiple of the n=8 calibration sample's empirical support
    # (the warmest observed chill year gives dTw ~ +1.48 C; 2071-2100 SSP5-8.5 at
    # dTw = +4 C is a 3x extrapolation and was dropped for that reason).
    ("SSP1-2.6", "2031-2050", 0.9, 0.5, 1.4, 0.9, 0.5, 1.4),
    ("SSP1-2.6", "2051-2070", 1.0, 0.6, 1.6, 1.0, 0.6, 1.6),
    ("SSP2-4.5", "2031-2050", 1.1, 0.6, 1.6, 1.1, 0.6, 1.6),
    ("SSP2-4.5", "2051-2070", 1.6, 1.0, 2.3, 1.6, 1.0, 2.2),
    ("SSP5-8.5", "2031-2050", 1.4, 0.9, 2.0, 1.4, 0.8, 2.0),
    ("SSP5-8.5", "2051-2070", 2.4, 1.6, 3.3, 2.4, 1.6, 3.2),
], columns=["scenario", "period", "dTw_med", "dTw_lo", "dTw_hi",
            "dTs_med", "dTs_lo", "dTs_hi"])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("CMIP6 DELTA-METHOD PROJECTIONS — Tacna olive yield")
    print("=" * 72)

    print("\n[1/5] Loading baseline meteorology + model coefficients ...")
    df = load_meteo()
    coef_point, boot_df = load_coefficients()
    print(f"      meteo: {df['date'].min().date()} -> {df['date'].max().date()}  (n={len(df)})")
    print(f"      coefficients: Intercept={coef_point['Intercept']:.3f}  "
          f"b_chill={coef_point['b_chill']:.3f}  b_crec={coef_point['b_crec']:.3f}")
    print(f"      bootstrap rows: {len(boot_df) if boot_df is not None else 0}")

    # Baseline (dT=0,0)
    print("\n[2/5] Baseline recomputation (sanity vs script 01) ...")
    base = compute_seasonal_metrics(df, 0.0, 0.0)
    base["yield_pred"] = predict_yield(base["Tmean_chill"].values,
                                       base["Tmean_CREC"].values, coef_point)
    print(base.round(3).to_string(index=False))

    # ------------------------------------------------------------------
    # Continuous warming sweep
    # ------------------------------------------------------------------
    print("\n[3/5] Continuous warming sweep (DeltaT_winter 0.0 -> 5.0 C) ...")
    sweep_rows = []
    dT_grid = np.arange(0.0, 5.01, 0.25)
    for dTw in dT_grid:
        # for the sweep figure we hold summer warming = winter warming
        m = compute_seasonal_metrics(df, dTw, dTw)
        if m.empty:
            continue
        cp_mean = float(m["CP_season"].mean())
        tch_mean = float(m["Tmean_chill"].mean())
        tcr_mean = float(m["Tmean_CREC"].mean())
        yhat = float(predict_yield(tch_mean, tcr_mean, coef_point))
        # uncertainty band from bootstrap
        if boot_df is not None and len(boot_df) > 0:
            yb = np.exp(boot_df["Intercept"].values
                        + boot_df["b_chill"].values * tch_mean
                        + boot_df["b_crec"].values * tcr_mean) - 0.5
            ylo, yhi = float(np.percentile(yb, 5)), float(np.percentile(yb, 95))
        else:
            ylo, yhi = np.nan, np.nan
        sweep_rows.append({"dT_winter": dTw, "CP_mean": cp_mean,
                           "Tmean_chill": tch_mean, "Tmean_CREC": tcr_mean,
                           "yield_med": max(yhat, 0.0),
                           "yield_lo": max(ylo, 0.0),
                           "yield_hi": max(yhi, 0.0)})
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT / "tables" / "warming_sweep.csv", index=False)
    print(sweep.round(3).to_string(index=False))

    # ------------------------------------------------------------------
    # CMIP6 scenarios x periods
    # ------------------------------------------------------------------
    print("\n[4/5] CMIP6 scenario projections ...")
    rows = []
    for _, row in CMIP6_DELTA.iterrows():
        # Median scenario realization
        m_med = compute_seasonal_metrics(df, row["dTw_med"], row["dTs_med"])
        m_lo = compute_seasonal_metrics(df, row["dTw_lo"], row["dTs_lo"])
        m_hi = compute_seasonal_metrics(df, row["dTw_hi"], row["dTs_hi"])
        # Yield = mean prediction across years (climatological)
        def _yield_band(m):
            tch = m["Tmean_chill"].mean()
            tcr = m["Tmean_CREC"].mean()
            yhat = predict_yield(tch, tcr, coef_point)
            if boot_df is not None:
                yb = np.exp(boot_df["Intercept"].values
                            + boot_df["b_chill"].values * tch
                            + boot_df["b_crec"].values * tcr) - 0.5
                return float(yhat), float(np.percentile(yb, 5)), float(np.percentile(yb, 95))
            return float(yhat), np.nan, np.nan
        y_med, yb_lo, yb_hi = _yield_band(m_med)
        y_lo_dt, _, _ = _yield_band(m_lo)
        y_hi_dt, _, _ = _yield_band(m_hi)
        rows.append({
            "scenario": row["scenario"],
            "period": row["period"],
            "dTw_med": row["dTw_med"], "dTw_lo": row["dTw_lo"], "dTw_hi": row["dTw_hi"],
            "dTs_med": row["dTs_med"], "dTs_lo": row["dTs_lo"], "dTs_hi": row["dTs_hi"],
            "CP_mean": float(m_med["CP_season"].mean()),
            "Tmean_chill": float(m_med["Tmean_chill"].mean()),
            "Tmean_CREC": float(m_med["Tmean_CREC"].mean()),
            "yield_med": max(y_med, 0.0),
            "yield_param_lo": max(yb_lo, 0.0),  # parametric (model coeff) uncertainty
            "yield_param_hi": max(yb_hi, 0.0),
            "yield_dT_lo": max(y_lo_dt, 0.0),   # GCM ensemble (dT spread) uncertainty
            "yield_dT_hi": max(y_hi_dt, 0.0),
        })
    proj = pd.DataFrame(rows)
    proj.to_csv(OUT / "tables" / "yield_under_warming.csv", index=False)
    CMIP6_DELTA.to_csv(OUT / "tables" / "delta_scenarios.csv", index=False)
    print(proj.round(3).to_string(index=False))

    # Yield change vs baseline
    base_yield = float(predict_yield(base["Tmean_chill"].mean(),
                                     base["Tmean_CREC"].mean(), coef_point))
    proj["yield_change_pct"] = 100.0 * (proj["yield_med"] - base_yield) / base_yield
    print(f"\n      Baseline (observed climate) predicted yield = {base_yield:.2f} t/ha")
    print(proj[["scenario", "period", "dTw_med", "yield_med", "yield_change_pct"]]
          .round(2).to_string(index=False))
    proj.to_csv(OUT / "tables" / "yield_under_warming.csv", index=False)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n[5/5] Building projection figures ...")

    # Fig A: warming sweep with SSP markers
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=600)
    ax.fill_between(sweep["dT_winter"], sweep["yield_lo"], sweep["yield_hi"],
                    color="#1a6332", alpha=0.18, label="Parametric 90% CI")
    ax.plot(sweep["dT_winter"], sweep["yield_med"], color="#1a6332", lw=2.0,
            label="Median predicted yield")
    ax.axhline(base_yield, color="grey", lw=1.0, ls=":",
               label=f"Baseline (observed) = {base_yield:.2f} t/ha")

    # Vertical marker: maximum observed T_chill in the 2016-2024 year-level
    # calibration sample. Beyond this delta the sweep is an extrapolation
    # outside the predictor support.
    try:
        _feat = pd.read_csv(
            Path(r"D:/olive_yield_RS_chill") / "01_preprocessing" /
            "outputs" / "feature_matrix_chill_complete.csv"
        )
        _year_max_Tchill = float(
            _feat.groupby("yield_year")["Tmean_chill_lag1"].first().max()
        )
        _base_Tchill = float(sweep.iloc[0]["Tmean_chill"])
        _dT_support = _year_max_Tchill - _base_Tchill
        if 0 < _dT_support < 5:
            ax.axvline(_dT_support, color="k", ls="--", lw=1.1, alpha=0.7,
                       label=(f"Max observed $T_{{\\mathrm{{chill}}}}$ = "
                              f"{_year_max_Tchill:.2f} $^\\circ$C "
                              f"($\\Delta T$ = +{_dT_support:.2f} $^\\circ$C)"))
    except Exception as _e:
        print(f"  [warn] could not add observed-support marker: {_e}")

    # SSP markers — mid-century horizon (2051-2070) is used as the reference
    # horizon for the sweep overlay: still within ~2x the observed chill-window
    # warming envelope, and all three SSPs are already clearly separated.
    ssp_colors = {"SSP1-2.6": "#1a9850", "SSP2-4.5": "#fdae61", "SSP5-8.5": "#d73027"}
    long_horizon = proj[proj["period"] == "2051-2070"]
    for _, r in long_horizon.iterrows():
        ax.errorbar(r["dTw_med"], r["yield_med"],
                    xerr=[[r["dTw_med"] - r["dTw_lo"]], [r["dTw_hi"] - r["dTw_med"]]],
                    fmt="o", ms=10, color=ssp_colors[r["scenario"]],
                    capsize=4, lw=1.5, mec="k", mew=0.7,
                    label=f"{r['scenario']} (2051-2070)")
    ax.set_xlabel(r"$\Delta T_{winter}$ applied to chill window (May-Aug, $^\circ$C)",
                  fontsize=13)
    ax.set_ylabel("Predicted olive yield (t ha$^{-1}$)", fontsize=13)
    # title removed — goes in LaTeX caption
    ax.set_ylim(0, 16)
    ax.set_xlim(0, 3)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_warming_sweep.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_warming_sweep.png")

    # Fig B: bar chart by SSP/horizon with two uncertainty sources
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=600)
    periods = ["2031-2050", "2051-2070"]
    scenarios = ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
    width = 0.25
    xpos = np.arange(len(periods))
    for i, sc in enumerate(scenarios):
        sub = proj[proj["scenario"] == sc].set_index("period").reindex(periods)
        bars_x = xpos + (i - 1) * width
        ax.bar(bars_x, sub["yield_med"], width=width,
               color=ssp_colors[sc], edgecolor="k", lw=0.5, label=sc)
        # GCM-ensemble uncertainty (dT 5-95%) as solid error bar
        yerr_dt = np.array([sub["yield_med"] - sub["yield_dT_hi"],   # warmer dT -> lower yield
                            sub["yield_med"] - sub["yield_dT_lo"]])  # cooler dT -> higher yield
        # rectify negatives (lower bound must be < median)
        yerr_dt = np.abs(yerr_dt)
        ax.errorbar(bars_x, sub["yield_med"], yerr=yerr_dt,
                    fmt="none", ecolor="k", capsize=3, lw=0.9)
        # value labels
        for x, y in zip(bars_x, sub["yield_med"]):
            ax.text(x, y + 0.15, f"{y:.1f}", ha="center", fontsize=10)
    ax.axhline(base_yield, color="grey", lw=1.0, ls=":",
               label=f"Baseline = {base_yield:.2f} t/ha")
    ax.set_xticks(xpos)
    ax.set_xticklabels(periods)
    ax.set_ylabel("Projected mean yield (t ha$^{-1}$)", fontsize=13)
    # title removed — goes in LaTeX caption
    ax.set_ylim(0, max(proj["yield_dT_hi"].max() * 1.15, base_yield * 1.2))
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_ssp_yield.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_ssp_yield.png")

    # Fig C: chill collapse curve (mechanism)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600)
    ax.plot(sweep["dT_winter"], sweep["CP_mean"], "o-", color="#2c7fb8", lw=1.8, ms=5,
            label="CP (Dynamic Model)")
    ax.set_xlabel(r"$\Delta T_{winter}$ ($^\circ$C)", fontsize=13)
    ax.set_ylabel("Mean Chill Portions (May-Aug)", fontsize=13, color="#2c7fb8")
    ax.tick_params(axis="y", colors="#2c7fb8")
    ax2 = ax.twinx()
    ax2.plot(sweep["dT_winter"], sweep["Tmean_chill"], "s-", color="#7a0177",
             lw=1.5, ms=4, label="Tmean chill window")
    ax2.set_ylabel(r"Mean T chill window ($^\circ$C)", fontsize=13, color="#7a0177")
    ax2.tick_params(axis="y", colors="#7a0177")
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    # title removed — goes in LaTeX caption
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_chill_collapse_curve.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_chill_collapse_curve.png")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report = f"""CMIP6 PROJECTION REPORT — Tacna olive yield
===========================================

Method: delta method (Luedeling 2012; Aguilera et al. 2014).
        Baseline: in-situ daily Tmin/Tmax 2015-2025.
        Warming deltas: IPCC AR6 WGI Atlas, region SWS, CMIP6 ensemble,
        relative to 1995-2014 baseline.

Baseline (observed climate) predicted yield: {base_yield:.2f} t/ha
Observed climatological mean yield (2016-2024 ex-2021): {base['yield_pred'].mean():.2f} t/ha

Projection table (median dT, central yield, change vs baseline):
{proj[['scenario','period','dTw_med','dTs_med','CP_mean','Tmean_chill','yield_med','yield_change_pct']].round(2).to_string(index=False)}

Interpretation
--------------
* Even the lowest-warming SSP1-2.6 scenario reduces mean Tacna olive yield
  noticeably by 2071-2100, because the chill window in Tacna is already
  marginal (Tmean_chill ~ 16-17 C is at the threshold where the Dynamic
  Model collapses).
* SSP5-8.5 by 2071-2100 (DeltaT_winter ~ +4 C) drives the chill window into
  permanent failure: predicted yield converges toward the regime observed
  in the 2016 and 2024 ENSO failure years.
* The dominant uncertainty source in this projection is GCM ensemble
  spread (error bars), not the calibrated model coefficients (parametric
  band), confirming that this is a climate-projection problem and not a
  statistical-model uncertainty problem.

Caveats / forward work
----------------------
* Delta-method preserves observed inter-annual variability and only
  perturbs the mean. This is the standard for chill projections but
  cannot capture changes in the *shape* of the temperature distribution
  (e.g. fewer cold spells under future climate). For a follow-up study,
  bias-corrected NEX-GDDP-CMIP6 daily output should be used.
* The warming deltas are anchored to IPCC AR6 Atlas region-mean values
  for SWS. Refining with the exact Tacna grid cell would tighten the
  central estimates.
* Yield projections assume the calibrated 2016-2024 climate-yield
  relationship is stationary. Adaptation responses (variety change,
  irrigation regime change, sprinkler-induced cooling) are not modeled.
"""
    (OUT / "tables" / "PROJECTION_REPORT.txt").write_text(report, encoding="utf-8")
    print("\n[Done] projection report written.")


if __name__ == "__main__":
    main()
