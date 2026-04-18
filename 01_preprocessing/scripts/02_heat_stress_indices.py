"""
02_heat_stress_indices.py
-------------------------
Compute heat-stress and agroclimatic indices from daily station meteorology.

Indices (annual, per calendar year + per olive phenology window):
    KDD35   : killing degree days above 35 C
    CDD30   : cooling degree days above 30 C (Tmax based)
    TROPN   : number of tropical nights (Tmin > 20 C)
    HI_max  : mean of daily maximum Heat Index
    HOT_DAYS: count of days with Tmax > 32 C
    ET0_sum : cumulative reference ET (Penman-Monteith preferred, Hargreaves fallback)
    GDD10   : growing degree days base 10 C

Phenology windows (olive, Tacna, Southern Hemisphere):
    BROT : Brotacion        Aug 15 - Sep 30
    FLOR : Floracion        Oct 1  - Nov 15
    CUAJ : Cuajado          Nov 16 - Dec 31
    CREC : Crec. Fruto      Jan 1  - Mar 15
    ENVE : Envero           Mar 16 - Apr 30

Aggregation is by yield_year (calendar year of harvest). The harvest in year Y
corresponds to bud break in Aug-Sep of year (Y-1) and fruit growth Jan-Apr of
year Y. Accordingly:
    BROT(Y)  := Aug 15 (Y-1) - Sep 30 (Y-1)
    FLOR(Y)  := Oct 1  (Y-1) - Nov 15 (Y-1)
    CUAJ(Y)  := Nov 16 (Y-1) - Dec 31 (Y-1)
    CREC(Y)  := Jan 1  (Y)   - Mar 15 (Y)
    ENVE(Y)  := Mar 16 (Y)   - Apr 30 (Y)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

BRANCH = Path(r"D:/olive_yield_RS_chill")
RAW = BRANCH / "00_raw_data" / "meteo_daily_2015_2025.csv"
OUT_DIR = BRANCH / "01_preprocessing" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phenology window definitions — (start_month, start_day, end_month, end_day, offset_from_yield_year)
# offset = -1 means "previous calendar year"; 0 means "current calendar year"
PHENO_WINDOWS = {
    "BROT": (8, 15, 9, 30, -1),
    "FLOR": (10, 1, 11, 15, -1),
    "CUAJ": (11, 16, 12, 31, -1),
    "CREC": (1, 1, 3, 15, 0),
    "ENVE": (3, 16, 4, 30, 0),
}


def window_mask(dates: pd.Series, yield_year: int, sm, sd, em, ed, offset) -> pd.Series:
    start = pd.Timestamp(yield_year + offset, sm, sd)
    end = pd.Timestamp(yield_year + offset, em, ed)
    return (dates >= start) & (dates <= end)


def compute_indices(sub: pd.DataFrame) -> dict:
    """Compute heat/agroclimatic indices for a subset of daily rows."""
    if len(sub) == 0:
        return {k: np.nan for k in [
            "KDD35", "CDD30", "TROPN", "HI_max", "HOT_DAYS", "ET0_sum",
            "GDD10", "Tmax_mean", "Tmin_mean", "Tmean_mean", "RH_mean",
        ]}
    kdd35 = np.clip(sub["Tmax"] - 35.0, 0, None).sum()
    cdd30 = np.clip(sub["Tmax"] - 30.0, 0, None).sum()
    tropn = int((sub["Tmin"] > 20.0).sum())
    hi_max = sub["Heat_index_max"].mean() if "Heat_index_max" in sub else np.nan
    hot_days = int((sub["Tmax"] > 32.0).sum())
    # Prefer Penman-Monteith ET0 if present
    if "ET0_PM" in sub and sub["ET0_PM"].notna().any():
        et0 = sub["ET0_PM"].sum()
    elif "ET0" in sub:
        et0 = sub["ET0"].sum()
    else:
        et0 = np.nan
    gdd10 = sub["GDD_10"].sum() if "GDD_10" in sub else np.nan
    return {
        "KDD35": float(kdd35),
        "CDD30": float(cdd30),
        "TROPN": tropn,
        "HI_max": float(hi_max) if pd.notna(hi_max) else np.nan,
        "HOT_DAYS": hot_days,
        "ET0_sum": float(et0) if pd.notna(et0) else np.nan,
        "GDD10": float(gdd10) if pd.notna(gdd10) else np.nan,
        "Tmax_mean": float(sub["Tmax"].mean()),
        "Tmin_mean": float(sub["Tmin"].mean()),
        "Tmean_mean": float(sub["Tmean"].mean()) if "Tmean" in sub else np.nan,
        "RH_mean": float(sub["RH_mean"].mean()) if "RH_mean" in sub else np.nan,
    }


def main():
    print("[1/3] Loading meteorology ...")
    df = pd.read_csv(RAW)
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    if df["date"].isna().any():
        mask = df["date"].isna()
        df.loc[mask, "date"] = pd.to_datetime(df.loc[mask, "date"], dayfirst=True, errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"      rows: {len(df)} | {df['date'].min().date()} -> {df['date'].max().date()}")

    print("[2/3] Computing indices per yield_year x phenology window ...")
    yield_years = list(range(2016, 2025))  # 2016-2024 (yield calendar years available)
    records = []
    for yy in yield_years:
        for wname, (sm, sd, em, ed, off) in PHENO_WINDOWS.items():
            mask = window_mask(df["date"], yy, sm, sd, em, ed, off)
            sub = df.loc[mask]
            idx = compute_indices(sub)
            idx.update({"yield_year": yy, "window": wname, "n_days": len(sub)})
            records.append(idx)
    long_df = pd.DataFrame(records)
    long_df.to_csv(OUT_DIR / "heat_indices_long.csv", index=False)
    print(f"      -> heat_indices_long.csv ({len(long_df)} rows)")

    print("[3/3] Pivoting to wide format (one row per yield_year) ...")
    value_cols = ["KDD35", "CDD30", "TROPN", "HI_max", "HOT_DAYS",
                  "ET0_sum", "GDD10", "Tmax_mean", "Tmin_mean", "Tmean_mean", "RH_mean"]
    wide = long_df.pivot_table(
        index="yield_year", columns="window", values=value_cols
    )
    wide.columns = [f"{v}_{w}" for v, w in wide.columns]
    wide = wide.reset_index()

    # Add annual (full-year) aggregates
    for yy in yield_years:
        sub = df[df["date"].dt.year == yy]
        row = compute_indices(sub)
        for k, v in row.items():
            wide.loc[wide["yield_year"] == yy, f"{k}_ANN"] = v

    wide.to_csv(OUT_DIR / "heat_indices_wide.csv", index=False)
    print(f"      -> heat_indices_wide.csv ({wide.shape})")

    # Preview critical signal: KDD35 and HOT_DAYS per yield_year at FLOR and CREC
    preview_cols = ["yield_year", "KDD35_FLOR", "HOT_DAYS_FLOR", "KDD35_CREC",
                    "HOT_DAYS_CREC", "Tmax_mean_FLOR", "Tmean_mean_CREC"]
    have = [c for c in preview_cols if c in wide.columns]
    print("\nCritical heat-stress preview (per yield_year):")
    print(wide[have].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
