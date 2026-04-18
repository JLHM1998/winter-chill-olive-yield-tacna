"""
01_eda_enso_overlay.py
----------------------
Four-panel EDA figure aligning the ENSO ONI index (NOAA PSL) with:
    (a) Annual mean yield across all 11 parcels (+/- SD)
    (b) Chill Portions (Dynamic Model) for the May-Aug chill window
    (c) Mean air temperature during chill window
    (d) ENSO ONI monthly, with El Nino / La Nina thresholds shaded

All annual panels (a-c) are aligned to the harvest year so that the
chill accumulation of winter Y-1 and the yield of harvest Y share the
same x-position. Composite tick labels ("W'15 / H'16") make the
temporal link explicit.

Output: 02_eda/outputs/fig_eda_enso_chill_yield.png (600 DPI)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
from pathlib import Path

BRANCH = Path(r"D:/olive_yield_RS_chill")
ONI_FILE = BRANCH / "00_raw_data" / "oni.data"
FEATURE = BRANCH / "01_preprocessing" / "outputs" / "feature_matrix_chill_complete.csv"
CHILL = BRANCH / "01_preprocessing" / "outputs" / "chill_season_totals.csv"
OUT_DIR = BRANCH / "02_eda" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_oni() -> pd.DataFrame:
    """Parse NOAA PSL ONI fixed-width text file."""
    with open(ONI_FILE) as f:
        first = f.readline().split()
        y0, y1 = int(first[0]), int(first[1])
        rows = []
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                break
            try:
                year = int(parts[0])
            except ValueError:
                break
            if year < y0 or year > y1:
                break
            vals = [float(v) for v in parts[1:13]]
            rows.append([year] + vals)
    df = pd.DataFrame(rows, columns=["year"] + [f"m{i}" for i in range(1, 13)])
    long = df.melt(id_vars="year", var_name="month", value_name="ONI")
    long["month"] = long["month"].str.replace("m", "").astype(int)
    long["date"] = pd.to_datetime(dict(year=long["year"], month=long["month"], day=15))
    long = long.sort_values("date").reset_index(drop=True)
    long.loc[long["ONI"] < -90, "ONI"] = np.nan
    return long


def main():
    print("[1/3] Loading data ...")
    oni = load_oni()
    feat = pd.read_csv(FEATURE)
    chill = pd.read_csv(CHILL)
    print(f"      ONI rows: {len(oni)} | {oni['date'].min().date()} -> {oni['date'].max().date()}")

    # Restrict to the analysis window: chill years 2015-2023 -> harvest 2016-2024
    # (the calibration sample). Without this filter, any extra rows in the CSVs
    # would render bars/lines past the 2024-12-31 xlim.
    chill = chill[chill["yield_year"].between(2016, 2024)].reset_index(drop=True)
    feat = feat[feat["yield_year"].between(2016, 2024)].reset_index(drop=True)

    # Yield: annual mean and SD across 11 parcels
    yld_annual = feat.groupby("yield_year")["yield_tn_ha"].agg(["mean", "std", "count"]).reset_index()
    print("      yield annual:", yld_annual.to_dict("records"))

    # ---- Align chill data to HARVEST year (chill_year + 1) ----
    # This ensures panels (a)-(c) share the same x-position per event
    harvest_years = chill["yield_year"].values  # already = chill_year + 1
    # x-position: April 15 of the harvest year (same reference for all panels)
    x_harvest = pd.to_datetime([f"{y}-04-15" for y in harvest_years])
    x_yield = pd.to_datetime(yld_annual["yield_year"].astype(str) + "-04-15")

    # ENSO window for plotting: Jan 2014 - Dec 2024 (end of analysis period)
    oni_plot = oni[(oni["date"] >= "2014-01-01") & (oni["date"] <= "2024-12-31")].copy()

    print("[2/3] Building figure ...")
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), dpi=600, sharex=False)

    # ---- Shared x-limits for all panels ----
    xlim = (pd.Timestamp("2015-06-01"), pd.Timestamp("2024-12-31"))

    # === (a) Yield — plotted at harvest year ===
    ax = axes[0]
    ax.errorbar(x_yield, yld_annual["mean"], yerr=yld_annual["std"],
                fmt="o-", color="#1a6332", ecolor="#1a6332", capsize=3,
                lw=1.4, ms=6, label="Mean annual yield (11 parcels)")
    ax.axvline(pd.Timestamp("2021-04-15"), color="grey", ls="--", alpha=0.5)
    ax.set_ylabel("Yield (t ha$^{-1}$)", fontsize=12)
    ax.text(0.985, 0.96, "(a)", transform=ax.transAxes,
            ha="right", va="top", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(xlim)
    ax.set_ylim(0, yld_annual["mean"].max() + yld_annual["std"].max() + 2.5)
    ax.annotate("2021: no harvest\n(COVID)", xy=(pd.Timestamp("2021-04-15"), 0.5),
                xytext=(pd.Timestamp("2021-10-15"), 3.5),
                fontsize=9, color="grey", va="top", ha="left",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.8, alpha=0.6))

    # === (b) Chill Portions — NOW aligned to harvest year ===
    ax = axes[1]
    cp_vals = chill["CP_season"].values
    ax.bar(x_harvest, cp_vals, width=80, color="#2c7fb8", alpha=0.9,
           label="Chill Portions (May\u2013Aug of $Y-1$)")
    # Annotate CP values on bars for unambiguous alignment
    for xp, cpv in zip(x_harvest, cp_vals):
        ax.text(xp, cpv + 0.08, f"{cpv:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#2c7fb8")
    ax2 = ax.twinx()
    ax2.plot(x_harvest, chill["CH12_season"], "s-", color="#d95f02", ms=5, lw=1.2,
             label="Chill Hours < 12 \u00b0C (h)")
    ax.set_ylabel("Chill Portions", fontsize=12, color="#2c7fb8")
    ax2.set_ylabel("CH$_{12}$ (h)", fontsize=12, color="#d95f02")
    ax.tick_params(axis="y", colors="#2c7fb8")
    ax2.tick_params(axis="y", colors="#d95f02")
    ax.text(0.985, 0.96, "(b)", transform=ax.transAxes,
            ha="right", va="top", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.set_ylim(top=max(cp_vals) * 1.35)
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.set_xlim(xlim)
    ax2.set_xlim(xlim)

    # === (c) Mean chill-window temperature — aligned to harvest year ===
    ax = axes[2]
    ax.plot(x_harvest, chill["Tmean_season"], "o-", color="#7a0177", lw=1.4, ms=6,
            label="Mean $T$ during chill window (May\u2013Aug of $Y-1$)")
    ax.axhline(18.0, color="red", ls=":", alpha=0.6, label="Mechanistic CP-collapse threshold (~18 \u00b0C)")
    ax.set_ylabel("Temperature (\u00b0C)", fontsize=12)
    ax.text(0.985, 0.96, "(c)", transform=ax.transAxes,
            ha="right", va="top", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    # Expand y-axis top so the upper-left legend does not overlap the 2016/2024 peaks
    ax.set_ylim(bottom=min(chill["Tmean_season"].min() - 0.5, 15.5), top=20.0)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(xlim)

    # === (d) ENSO ONI — monthly bars at calendar date ===
    ax = axes[3]
    colors = np.where(oni_plot["ONI"] > 0, "#d73027", "#4575b4")
    ax.bar(oni_plot["date"], oni_plot["ONI"], width=30, color=colors, alpha=0.8)
    ax.axhline(0.5, color="#d73027", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(-0.5, color="#4575b4", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.fill_between(oni_plot["date"], 0.5, 3, color="#d73027", alpha=0.08)
    ax.fill_between(oni_plot["date"], -3, -0.5, color="#4575b4", alpha=0.08)
    ax.set_ylim(-3.0, 4.0)
    ax.set_ylabel("ONI (\u00b0C)", fontsize=12)
    ax.text(0.985, 0.96, "(d)", transform=ax.transAxes,
            ha="right", va="top", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_xlim(xlim)

    # Annotate key events on ONI panel
    for yr, label, yoff in [
        (2016, "El Ni\u00f1o\n2015\u20132016", 2.7),
        (2021, "Triple La Ni\u00f1a\n2020\u20132022", -2.2),
        (2024, "El Ni\u00f1o\n2023\u20132024", 2.7),
    ]:
        ax.annotate(label, xy=(pd.Timestamp(f"{yr}-04-15"), yoff),
                    fontsize=9.5, ha="center", color="k")

    # ---- Composite x-tick labels: "W'15 / H'16" ----
    tick_positions = [pd.Timestamp(f"{y}-04-15") for y in range(2016, 2025)]
    tick_labels = [f"W\u2019{y-1-2000:02d} \u2192 H\u2019{y-2000:02d}" for y in range(2016, 2025)]

    for a in axes:
        a.set_xticks(tick_positions)
        a.set_xticklabels(tick_labels, fontsize=9.5, rotation=30, ha="right")

    axes[3].set_xlabel("Chill winter ($Y-1$) \u2192 Harvest year ($Y$)", fontsize=12)

    # ---- Shade failure years across ALL panels ----
    for failure_year in [2016, 2024]:
        for a in axes:
            a.axvspan(pd.Timestamp(f"{failure_year}-01-01"),
                      pd.Timestamp(f"{failure_year}-07-31"),
                      color="red", alpha=0.08, zorder=-1)

    # ---- Connecting arrows: winter -> harvest for failure years ----
    for failure_year, chill_yr in [(2016, 2015), (2024, 2023)]:
        # Add arrow annotation in ONI panel pointing from El Nino peak to harvest
        ax_oni = axes[3]
        ax_oni.annotate("",
                        xy=(pd.Timestamp(f"{failure_year}-04-15"), 0),
                        xytext=(pd.Timestamp(f"{chill_yr}-10-15"), 0),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5, alpha=0.5))

    # suptitle removed — title goes in LaTeX caption
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    out = OUT_DIR / "fig_eda_enso_chill_yield.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[3/3] -> {out}")


if __name__ == "__main__":
    main()
