"""
12_s2_ndvi_mask.py
------------------
Mixed-pixel mitigation for Sentinel-2 vegetation indices over the La Yarada
olive orchard. Addresses Reviewer Major Comment 1.

Background
----------
The raw `s2_indices_per_parcel.csv` file already ships with PARCEL-LEVEL means
of each VI per Sentinel-2 scene (computed in GEE). Because individual olive
canopies are smaller than the 10-20 m S2 pixel footprint, parcel-mean reflectance
is contaminated by a large bare-soil fraction. We cannot do per-pixel
unmixing inside this repository, but we CAN apply two complementary corrections
at the parcel-scene level:

  (a) NDVI-threshold filtering: discard parcel-scene observations with
      NDVI < tau_NDVI (the parcel-mean is so soil-dominated that it carries
      no canopy signal). We use tau_NDVI = 0.25 (slightly above bare-soil
      NDVI of 0.10-0.20 for the hyper-arid Tacna desert).

  (b) NDVI-conditional regression correction: regress NDRE on NDVI across all
      retained parcel-scene observations and use the residual NDRE
      (NDRE - f(NDVI)) as a soil-corrected canopy proxy. This removes the
      first-order linear effect of bare-soil dilution.

After (a)+(b) we recompute the parcel-year mean NDRE and CIre over the
Flor + Cuaj window of (Y-1) and re-correlate with parcel-level annual yield.

Outputs
-------
    01_preprocessing/outputs/s2_masked_per_parcel_year.csv
    01_preprocessing/outputs/s2_masked_correlations.csv
    05_manuscript/latex/figures_final/FigureS7.png   (600 dpi)
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
RAW_S2 = ROOT / "00_raw_data" / "s2_indices_per_parcel.csv"
RAW_Y = ROOT / "00_raw_data" / "yield_parcels_2016_2024.csv"
OUT_PRE = ROOT / "01_preprocessing" / "outputs"
OUT_FIG = ROOT / "05_manuscript" / "latex" / "figures_final"

NDVI_THRESH = 0.25  # bare-soil rejection threshold for parcel-scene means

PHENO = {
    "Flor": ((10, 1), (11, 15), -1),
    "Cuaj": ((11, 16), (12, 31), -1),
}


def assign_window(date: pd.Timestamp):
    for w, ((sm, sd), (em, ed), off) in PHENO.items():
        for cand_yy in (date.year, date.year + 1):
            cal = cand_yy + off
            start = pd.Timestamp(cal, sm, sd)
            end = pd.Timestamp(cal, em, ed)
            if start <= date <= end:
                return cand_yy, w
    return None, None


def main():
    print("[1/6] Loading raw Sentinel-2 parcel-scene table ...")
    s2 = pd.read_csv(RAW_S2)
    s2["date"] = pd.to_datetime(s2["date"])
    n0 = len(s2)
    print(f"      {n0} parcel-scene rows | {s2['parcel_id'].nunique()} parcels | "
          f"{s2['date'].nunique()} dates")
    print(f"      raw NDVI distribution:  median = {s2['NDVI'].median():.3f}, "
          f"p25 = {s2['NDVI'].quantile(0.25):.3f}, "
          f"p75 = {s2['NDVI'].quantile(0.75):.3f}")
    print(f"      raw NDRE distribution:  median = {s2['NDRE'].median():.3f}, "
          f"p25 = {s2['NDRE'].quantile(0.25):.3f}, "
          f"p75 = {s2['NDRE'].quantile(0.75):.3f}")

    print(f"[2/6] Applying NDVI > {NDVI_THRESH} bare-soil filter ...")
    s2_m = s2[s2["NDVI"] > NDVI_THRESH].copy()
    print(f"      retained {len(s2_m)}/{n0} ({100*len(s2_m)/n0:.1f}%) parcel-scene observations")

    print("[3/6] NDVI-conditional regression correction (soil-detrending NDRE / CIre) ...")
    # First-order linear detrending of NDRE on NDVI to remove the bare-soil
    # mixing contribution. NDRE_corr = NDRE - (a + b * NDVI) + b * NDVI_pure
    # where NDVI_pure is the 95th percentile NDVI in the dataset (closest to
    # full canopy cover). This pegs the corrected index to the canopy endmember.
    ndvi_pure = s2_m["NDVI"].quantile(0.95)
    print(f"      NDVI_pure (canopy endmember, p95) = {ndvi_pure:.3f}")
    for vi in ("NDRE", "CIre"):
        x = s2_m["NDVI"].values
        y = s2_m[vi].values
        b, a = np.polyfit(x, y, 1)
        s2_m[f"{vi}_soilcorr"] = y - (a + b * x) + (a + b * ndvi_pure)
        print(f"      {vi}: slope on NDVI = {b:.3f}, intercept = {a:.3f} -> "
              f"corrected mean = {s2_m[f'{vi}_soilcorr'].mean():.3f}")

    print("[4/6] Assigning Flor + Cuaj window of (Y-1) ...")
    win = s2_m["date"].apply(assign_window)
    s2_m["yield_year"] = [w[0] for w in win]
    s2_m["window"] = [w[1] for w in win]
    s2_m = s2_m.dropna(subset=["yield_year", "window"]).copy()
    s2_m["yield_year"] = s2_m["yield_year"].astype(int)
    s2_m = s2_m[s2_m["yield_year"].between(2016, 2024)]
    s2_m = s2_m[s2_m["yield_year"] != 2021]  # COVID
    print(f"      {len(s2_m)} parcel-scene rows in Flor+Cuaj of (Y-1)")

    print("[5/6] Aggregating to parcel-year (Flor+Cuaj mean) ...")
    grp = s2_m.groupby(["parcel_id", "yield_year"]).agg(
        NDRE_raw_repro=("NDRE", "mean"),
        NDRE_corr_repro=("NDRE_soilcorr", "mean"),
        CIre_raw_repro=("CIre", "mean"),
        CIre_corr_repro=("CIre_soilcorr", "mean"),
        NDVI_repro=("NDVI", "mean"),
        n_scenes=("NDRE", "count"),
    ).reset_index()
    grp.to_csv(OUT_PRE / "s2_masked_per_parcel_year.csv", index=False)
    print(f"      {len(grp)} parcel-year rows  (mean n_scenes = "
          f"{grp['n_scenes'].mean():.1f})")

    yields = pd.read_csv(RAW_Y)[["parcel_id", "year", "yield_tn_ha"]]
    yields = yields.rename(columns={"year": "yield_year"})
    yields = yields.dropna(subset=["yield_tn_ha"])
    merged = grp.merge(yields, on=["parcel_id", "yield_year"], how="inner")
    print(f"      merged with yield: {len(merged)} parcel-year rows")

    corr_rows = []
    for col in ["NDRE_raw_repro", "NDRE_corr_repro", "CIre_raw_repro", "CIre_corr_repro"]:
        r = merged[[col, "yield_tn_ha"]].corr().iloc[0, 1]
        corr_rows.append({"index": col, "r_pearson": r, "n": len(merged)})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_PRE / "s2_masked_correlations.csv", index=False)
    print("\nParcel-level correlations (across all parcel-years):")
    print(corr_df.round(3).to_string(index=False))

    # Parcel-level WITHIN-YEAR correlations (controlling for the year effect)
    within_year_r = []
    for yy, sub in merged.groupby("yield_year"):
        if len(sub) < 4:
            continue
        for col in ["NDRE_raw_repro", "NDRE_corr_repro"]:
            r = sub[[col, "yield_tn_ha"]].corr().iloc[0, 1]
            within_year_r.append({"year": yy, "index": col, "r": r, "n": len(sub)})
    wy = pd.DataFrame(within_year_r)
    print("\nWithin-year parcel-level correlations:")
    print(wy.pivot_table(index="year", columns="index", values="r").round(3).to_string())

    print("[6/6] Plotting Figure S7 (raw vs soil-corrected NDRE) ...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    ax.scatter(merged["NDRE_raw_repro"], merged["yield_tn_ha"],
               s=22, alpha=0.7, c="#888888", edgecolors="black", linewidth=0.4,
               label=f"raw NDRE (r = {merged[['NDRE_raw_repro','yield_tn_ha']].corr().iloc[0,1]:+.2f})")
    ax.set_xlabel(r"Mean NDRE over Flor+Cuaj of $Y-1$ (raw)")
    ax.set_ylabel(r"Parcel yield (t ha$^{-1}$)")
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(merged["NDRE_corr_repro"], merged["yield_tn_ha"],
               s=22, alpha=0.7, c="#1f78b4", edgecolors="black", linewidth=0.4,
               label=f"NDVI-masked + soil-detrended\n(r = {merged[['NDRE_corr_repro','yield_tn_ha']].corr().iloc[0,1]:+.2f})")
    ax.set_xlabel(r"Mean NDRE over Flor+Cuaj of $Y-1$ (corrected)")
    ax.text(0.03, 0.97, "(b)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_FIG / "FigureS7.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"      saved FigureS7.png at 600 dpi")

    print("Done.")


if __name__ == "__main__":
    main()
