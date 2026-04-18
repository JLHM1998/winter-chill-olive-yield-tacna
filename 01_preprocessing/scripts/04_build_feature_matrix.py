"""
04_build_feature_matrix.py
--------------------------
Build the final parcel-year feature matrix for the Chill Portions attribution
analysis.

Schema (one row per parcel_id x yield_year):
    parcel_id, group, variety, yield_year, yield_tn_ha
    CP_lag1         : chill portions, May-Aug of (yield_year - 1)   [Dynamic]
    CH12_lag1       : chill hours <12 C, same window
    CU_lag1         : Utah CU, same window                          [robustness]
    Tmean_chill_lag1: mean T during chill window
    Tmax_FLOR       : mean daily Tmax during flowering (Oct-Nov Y-1)
    Tmax_CREC       : mean daily Tmax during fruit growth (Jan-Mar Y)
    Tmean_CREC      : mean daily Tmean during fruit growth
    ET0_CREC        : cumulative ET0 during fruit growth
    NDRE_repro_lag1 : mean NDRE Flor+Cuaj of year (Y-1)  [primary canopy proxy]
    CIre_repro_lag1 : mean CIre Flor+Cuaj of year (Y-1)
    NDVI_CREC       : mean NDVI during Crec (contemporaneous canopy)
    NDRE_Flor_lag1  : NDRE at flowering only (descriptive)

Exclusions:
    - 2021: no harvest (COVID). Documented, excluded.

Design notes:
    - The Reposo window (May-Aug, SH winter) is excluded as a VI proxy window
      because of sparse S2 coverage (persistent coastal stratus). Instead we
      use NDRE/CIre averaged over Flor+Cuaj of the previous cycle as the
      canopy-state anchor (319 S2 observations, uniform across years).
    - Heat stress in Tacna does not manifest as extreme daily events
      (KDD35 = 0 across the entire record). The relevant metric is mean
      Tmax/Tmean per phenology window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

BRANCH = Path(r"D:/olive_yield_RS_chill")
IN_DIR = BRANCH / "01_preprocessing" / "outputs"
RAW_YIELD = BRANCH / "00_raw_data" / "yield_parcels_2016_2024.csv"
S2_RAW = BRANCH / "00_raw_data" / "s2_indices_per_parcel.csv"
OUT_DIR = IN_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Year-exclusion rationale: COVID pandemic, no harvest campaign
EXCLUDED_YEARS = [2021]


def build_repro_canopy_proxy(s2_raw_path: Path) -> pd.DataFrame:
    """
    Compute mean NDRE and CIre per (parcel_id, yield_year) aggregated over the
    Flor + Cuaj windows of the PRIOR calendar year (Y-1). Returns a DataFrame
    indexed for merging with yield_year.
    """
    df = pd.read_csv(s2_raw_path, parse_dates=["date"])
    # Flor (Y-1) = Oct 1 - Nov 15 of year (yield_year - 1)
    # Cuaj (Y-1) = Nov 16 - Dec 31 of year (yield_year - 1)
    # So for any date in Oct-Dec of calendar year C, yield_year = C + 1.
    mask_reproductive = (df["date"].dt.month >= 10) & (df["date"].dt.month <= 12)
    rep = df.loc[mask_reproductive].copy()
    rep["yield_year"] = rep["date"].dt.year + 1
    out = rep.groupby(["parcel_id", "yield_year"]).agg(
        NDRE_repro_lag1=("NDRE", "mean"),
        CIre_repro_lag1=("CIre", "mean"),
        NDVI_repro_lag1=("NDVI", "mean"),
        n_repro_imgs=("NDRE", "count"),
    ).reset_index()
    return out


def build_contemporaneous_canopy(s2_raw_path: Path) -> pd.DataFrame:
    """Mean NDVI/NDRE during Crec (Jan 1 - Mar 15 of yield_year)."""
    df = pd.read_csv(s2_raw_path, parse_dates=["date"])
    mask = (df["date"].dt.month >= 1) & (df["date"].dt.month <= 3) & (df["date"].dt.day <= 31)
    mask &= ~((df["date"].dt.month == 3) & (df["date"].dt.day > 15))
    crec = df.loc[mask].copy()
    crec["yield_year"] = crec["date"].dt.year
    out = crec.groupby(["parcel_id", "yield_year"]).agg(
        NDVI_CREC=("NDVI", "mean"),
        NDRE_CREC=("NDRE", "mean"),
        n_crec_imgs=("NDVI", "count"),
    ).reset_index()
    return out


def main():
    print("[1/5] Loading inputs ...")
    yld = pd.read_csv(RAW_YIELD)
    chill = pd.read_csv(IN_DIR / "chill_season_totals.csv")
    heat = pd.read_csv(IN_DIR / "heat_indices_wide.csv")
    print(f"      yield rows: {len(yld)}")
    print(f"      chill rows: {len(chill)}")
    print(f"      heat  rows: {len(heat)}")

    print("[2/5] Excluding 2021 (COVID no-harvest campaign) ...")
    yld = yld[~yld["year"].isin(EXCLUDED_YEARS)].copy()
    yld = yld.dropna(subset=["yield_tn_ha"])
    yld = yld.rename(columns={"year": "yield_year"})
    print(f"      yield after exclusion: {len(yld)} rows | years: {sorted(yld['yield_year'].unique())}")

    print("[3/5] Merging chill variables (CP lag1) ...")
    chill_sel = chill[["yield_year", "CP_season", "CH12_season", "CU_season", "Tmean_season"]].rename(
        columns={
            "CP_season": "CP_lag1",
            "CH12_season": "CH12_lag1",
            "CU_season": "CU_lag1",
            "Tmean_season": "Tmean_chill_lag1",
        }
    )
    mat = yld.merge(chill_sel, on="yield_year", how="left")

    print("[4/5] Merging heat-stress window aggregates ...")
    heat_keep = ["yield_year",
                 "Tmax_mean_FLOR", "Tmean_mean_FLOR",
                 "Tmax_mean_CREC", "Tmean_mean_CREC",
                 "ET0_sum_CREC", "Tmax_mean_ENVE", "Tmean_mean_ENVE",
                 "KDD35_ANN", "CDD30_ANN", "Tmean_mean_ANN"]
    have = [c for c in heat_keep if c in heat.columns]
    mat = mat.merge(heat[have], on="yield_year", how="left")

    print("[5/5] Merging Sentinel-2 canopy proxies ...")
    repro = build_repro_canopy_proxy(S2_RAW)
    crec = build_contemporaneous_canopy(S2_RAW)
    mat = mat.merge(repro, on=["parcel_id", "yield_year"], how="left")
    mat = mat.merge(crec, on=["parcel_id", "yield_year"], how="left")

    # Final cleanup
    mat = mat.sort_values(["parcel_id", "yield_year"]).reset_index(drop=True)
    # Retain rows with complete primary predictors: yield, CP_lag1
    mat_complete = mat.dropna(subset=["yield_tn_ha", "CP_lag1"]).copy()

    print(f"\nFeature matrix: {mat.shape}  |  complete rows: {mat_complete.shape[0]}")
    mat.to_csv(OUT_DIR / "feature_matrix_chill.csv", index=False)
    mat_complete.to_csv(OUT_DIR / "feature_matrix_chill_complete.csv", index=False)
    print(f"      -> feature_matrix_chill.csv")
    print(f"      -> feature_matrix_chill_complete.csv")

    # Summary of key predictors vs yield per year
    print("\nSummary by yield_year (mean across parcels):")
    summary_cols = ["yield_tn_ha", "CP_lag1", "CH12_lag1", "Tmean_chill_lag1",
                    "Tmax_mean_FLOR", "Tmean_mean_CREC", "NDRE_repro_lag1", "NDVI_CREC"]
    have_s = [c for c in summary_cols if c in mat.columns]
    print(mat.groupby("yield_year")[have_s].mean(numeric_only=True).round(3).to_string())

    # Missingness report
    print("\nMissingness in complete set (primary predictors):")
    for c in ["CP_lag1", "CH12_lag1", "Tmax_mean_FLOR", "Tmean_mean_CREC",
              "NDRE_repro_lag1", "NDVI_CREC"]:
        if c in mat_complete.columns:
            miss = mat_complete[c].isna().sum()
            print(f"  {c:20s}  missing {miss} / {len(mat_complete)}")


if __name__ == "__main__":
    main()
