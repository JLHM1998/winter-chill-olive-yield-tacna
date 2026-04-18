"""
03_s2_phenology_aggregation.py
------------------------------
Aggregate Sentinel-2 vegetation indices per parcel and olive phenology phase,
producing one row per (parcel_id, yield_year) with the mean/median of each VI
in each phenology window.

The raw file `s2_indices_per_parcel.csv` already ships with a `phenology` label
per image provided by the project's GEE pipeline. We re-compute phase from the
image date for consistency and add an overlap-aware assignment to the yield-year
calendar.

Indices available: NDVI, EVI, NDRE, NDWI, SAVI, CIre, GNDVI, MTCI.

Phenology -> yield_year linkage:
    Reposo   : May  1 (Y-1) - Aug 14 (Y-1)    (dormancy, overlaps chill season)
    Brot     : Aug 15 (Y-1) - Sep 30 (Y-1)
    Flor     : Oct  1 (Y-1) - Nov 15 (Y-1)
    Cuaj     : Nov 16 (Y-1) - Dec 31 (Y-1)
    Crec     : Jan  1 (Y)   - Mar 15 (Y)
    Enve     : Mar 16 (Y)   - Apr 30 (Y)     (harvest begins)

For the CP attribution story, the most critical windows are Reposo and Brot:
they represent canopy condition going into floral induction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

BRANCH = Path(r"D:/olive_yield_RS_chill")
RAW = BRANCH / "00_raw_data" / "s2_indices_per_parcel.csv"
OUT_DIR = BRANCH / "01_preprocessing" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VI_COLS = ["NDVI", "EVI", "NDRE", "NDWI", "SAVI", "CIre", "GNDVI", "MTCI"]

# (start_month, start_day, end_month, end_day, year_offset_from_yield_year)
PHENO_WINDOWS = {
    "Reposo": (5, 1, 8, 14, -1),
    "Brot":   (8, 15, 9, 30, -1),
    "Flor":   (10, 1, 11, 15, -1),
    "Cuaj":   (11, 16, 12, 31, -1),
    "Crec":   (1, 1, 3, 15, 0),
    "Enve":   (3, 16, 4, 30, 0),
}


def assign_yield_year_and_window(dates: pd.Series) -> pd.DataFrame:
    """
    For each date, return (yield_year, window) it belongs to.
    A date can belong to at most one window under the definitions above.
    """
    out = pd.DataFrame({"date": dates.values})
    out["yield_year"] = np.nan
    out["window"] = None
    for i, d in enumerate(dates):
        for wname, (sm, sd, em, ed, off) in PHENO_WINDOWS.items():
            # The window belongs to yield_year Y; the date's calendar year
            # is (Y + off). So Y = date.year - off.
            yy_candidate = d.year - off
            start = pd.Timestamp(yy_candidate + off, sm, sd)
            end = pd.Timestamp(yy_candidate + off, em, ed)
            if start <= d <= end:
                out.at[i, "yield_year"] = yy_candidate
                out.at[i, "window"] = wname
                break
    return out


def main():
    print("[1/4] Loading Sentinel-2 indices ...")
    df = pd.read_csv(RAW)
    df["date"] = pd.to_datetime(df["date"])
    print(f"      rows: {len(df)} | parcels: {df['parcel_id'].nunique()} | "
          f"dates: {df['date'].nunique()} | range: {df['date'].min().date()} -> {df['date'].max().date()}")

    # Filter to dates with cloud_roi_pct acceptable (already ~0 for most)
    if "cloud_roi_pct" in df.columns:
        before = len(df)
        df = df[df["cloud_roi_pct"] < 20].copy()
        print(f"      cloud filter <20%: {len(df)}/{before} rows retained")

    print("[2/4] Assigning (yield_year, phenology window) per image ...")
    unique_dates = pd.Series(sorted(df["date"].unique()))
    mapping = assign_yield_year_and_window(unique_dates)
    mapping["date"] = pd.to_datetime(mapping["date"])
    df = df.merge(mapping[["date", "yield_year", "window"]], on="date", how="left")
    df = df.dropna(subset=["yield_year", "window"]).copy()
    df["yield_year"] = df["yield_year"].astype(int)
    print(f"      assigned: {len(df)} rows | windows: {df['window'].value_counts().to_dict()}")

    print("[3/4] Aggregating VIs per (parcel_id, yield_year, window) ...")
    agg = df.groupby(["parcel_id", "yield_year", "window"])[VI_COLS].agg(["mean", "median"])
    agg.columns = [f"{v}_{stat}" for v, stat in agg.columns]
    agg = agg.reset_index()
    n_obs = df.groupby(["parcel_id", "yield_year", "window"]).size().rename("n_img").reset_index()
    agg = agg.merge(n_obs, on=["parcel_id", "yield_year", "window"])
    agg.to_csv(OUT_DIR / "s2_by_phenology_long.csv", index=False)
    print(f"      -> s2_by_phenology_long.csv ({len(agg)} rows)")

    print("[4/4] Pivoting to wide format (parcel_id, yield_year) ...")
    value_cols = [c for c in agg.columns if c not in ("parcel_id", "yield_year", "window", "n_img")]
    wide = agg.pivot_table(index=["parcel_id", "yield_year"], columns="window", values=value_cols)
    wide.columns = [f"{v}_{w}" for v, w in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(OUT_DIR / "s2_by_phenology_wide.csv", index=False)
    print(f"      -> s2_by_phenology_wide.csv shape={wide.shape}")

    # Preview: NDRE and CIre at Reposo (pre-floral induction proxy)
    preview_cols = ["parcel_id", "yield_year", "NDRE_mean_Reposo", "CIre_mean_Reposo",
                    "NDVI_mean_Reposo", "NDRE_mean_Brot"]
    have = [c for c in preview_cols if c in wide.columns]
    if have:
        print("\nPreview — canopy state at Reposo/Brot (pre-flowering):")
        print(wide[have].groupby("yield_year").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
