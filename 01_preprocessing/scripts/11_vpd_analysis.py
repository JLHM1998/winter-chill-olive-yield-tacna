"""
11_vpd_analysis.py
------------------
Compute Vapour-Pressure Deficit (VPD) from the AWS daily record and demonstrate
that the 2016 and 2024 failure years were NOT anomalous in atmospheric dryness
during the flowering / fruit-growth window. Addresses Reviewer Major Comment 3.

VPD is computed at the daily scale following FAO 56 (Allen et al., 1998):
    es(T) = 0.6108 * exp(17.27 * T / (T + 237.3))     [kPa]
    es_mean = (es(Tmax) + es(Tmin)) / 2               [kPa]
    ea      = es_mean * RH_mean / 100                 [kPa]
    VPD_mean = es_mean - ea                           [kPa]
    VPD_max  = es(Tmax) * (1 - RH_min/100)            [kPa, afternoon proxy]

Daily VPD is then aggregated as the mean and 95th-percentile over the canonical
olive phenology windows (FLOR, CUAJ, CREC) and contrasted with annual yield.

Outputs:
    01_preprocessing/outputs/vpd_daily.csv
    01_preprocessing/outputs/vpd_window_summary.csv
    01_preprocessing/outputs/vpd_yield_correlations.csv
    05_manuscript/latex/figures_final/FigureS6.png   (600 dpi)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"
rcParams["font.size"] = 12
rcParams["axes.labelsize"] = 12
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["legend.fontsize"] = 11

ROOT = Path(r"D:/olive_yield_RS_chill")
RAW_MET = ROOT / "00_raw_data" / "meteo_daily_2015_2025.csv"
YIELD_YR = ROOT / "03_modeling" / "outputs" / "tables" / "year_level_dataset.csv"
OUT_PRE = ROOT / "01_preprocessing" / "outputs"
OUT_FIG = ROOT / "05_manuscript" / "latex" / "figures_final"
OUT_PRE.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def es_kpa(T):
    """FAO-56 saturation vapour pressure (kPa) for T in deg C."""
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))


def main():
    print("[1/5] Loading meteo ...")
    df = pd.read_csv(RAW_MET)
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date", "Tmax", "Tmin", "RH_mean"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"      {len(df)} valid daily rows, {df['date'].min().date()} -> {df['date'].max().date()}")

    print("[2/5] Computing VPD (FAO-56) ...")
    df["es_Tmax"] = es_kpa(df["Tmax"])
    df["es_Tmin"] = es_kpa(df["Tmin"])
    df["es_mean"] = (df["es_Tmax"] + df["es_Tmin"]) / 2.0
    df["ea"] = df["es_mean"] * df["RH_mean"] / 100.0
    df["VPD_mean"] = (df["es_mean"] - df["ea"]).clip(lower=0)
    if "RH_min" in df.columns and df["RH_min"].notna().any():
        df["VPD_max"] = (df["es_Tmax"] * (1 - df["RH_min"].clip(0, 100) / 100.0)).clip(lower=0)
    else:
        df["VPD_max"] = df["VPD_mean"]
    df[["date", "Tmax", "Tmin", "RH_mean", "VPD_mean", "VPD_max"]].to_csv(
        OUT_PRE / "vpd_daily.csv", index=False
    )
    print(f"      mean VPD over record = {df['VPD_mean'].mean():.3f} kPa")
    print(f"      max  VPD over record = {df['VPD_max'].mean():.3f} kPa (daily afternoon proxy)")

    print("[3/5] Aggregating VPD over phenology windows ...")
    # Phenology windows used in the manuscript (yield-year indexing).
    # FLOR/CUAJ belong to year (Y-1); CREC belongs to year Y.
    PHENO = {
        "FLOR": ((10, 1), (11, 15), -1),
        "CUAJ": ((11, 16), (12, 31), -1),
        "CREC": ((1, 1), (3, 15), 0),
    }

    rows = []
    yields = pd.read_csv(YIELD_YR)
    yield_years = sorted(yields["yield_year"].unique())
    for yy in yield_years:
        for w, ((sm, sd), (em, ed), off) in PHENO.items():
            cal_year = yy + off
            start = pd.Timestamp(cal_year, sm, sd)
            end = pd.Timestamp(cal_year, em, ed)
            sub = df[(df["date"] >= start) & (df["date"] <= end)]
            if len(sub) == 0:
                continue
            rows.append({
                "yield_year": yy,
                "window": w,
                "n_days": len(sub),
                "VPD_mean_kPa": sub["VPD_mean"].mean(),
                "VPD_max_mean_kPa": sub["VPD_max"].mean(),
                "VPD_max_p95_kPa": sub["VPD_max"].quantile(0.95),
                "Tmax_mean_C": sub["Tmax"].mean(),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_PRE / "vpd_window_summary.csv", index=False)
    print("      saved vpd_window_summary.csv")
    print(summary.pivot_table(index="yield_year", columns="window",
                              values="VPD_mean_kPa").round(3).to_string())

    print("[4/5] Correlating VPD windows with yield ...")
    corr_rows = []
    yields = yields[["yield_year", "yield_tn_ha"]].copy()
    for w in ["FLOR", "CUAJ", "CREC"]:
        sub = summary[summary["window"] == w].merge(yields, on="yield_year")
        for col in ["VPD_mean_kPa", "VPD_max_mean_kPa", "VPD_max_p95_kPa"]:
            r = sub[[col, "yield_tn_ha"]].corr().iloc[0, 1]
            corr_rows.append({"window": w, "metric": col, "r_pearson": r, "n": len(sub)})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT_PRE / "vpd_yield_correlations.csv", index=False)
    print(corr.round(3).to_string(index=False))

    print("[5/5] Plotting Figure S6 (VPD anomaly during FLOR/CUAJ/CREC) ...")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    colors = {2016: "#d7191c", 2024: "#d7191c", 2022: "#1a9641"}
    panel_labels = ["(a) FLOR", "(b) CUAJ", "(c) CREC"]
    for ax, w, lab in zip(axes, ["FLOR", "CUAJ", "CREC"], panel_labels):
        sub = summary[summary["window"] == w].copy().sort_values("yield_year")
        bar_colors = [colors.get(int(y), "#999999") for y in sub["yield_year"]]
        ax.bar(sub["yield_year"].astype(str), sub["VPD_max_mean_kPa"],
               color=bar_colors, edgecolor="black", linewidth=0.6)
        mean_val = sub["VPD_max_mean_kPa"].mean()
        ax.axhline(mean_val, color="black", lw=0.9, ls="--",
                   label=f"mean = {mean_val:.2f} kPa")
        ax.text(0.03, 0.97, lab, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")
        ax.set_xlabel("Harvest year")
        if w == "FLOR":
            ax.set_ylabel(r"Mean afternoon VPD$_{\mathrm{max}}$ (kPa)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "FigureS6.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"      saved FigureS6.png at 600 dpi -> {OUT_FIG / 'FigureS6.png'}")

    print("Done.")


if __name__ == "__main__":
    main()
