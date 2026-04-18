"""
03_workflow_diagram.py
----------------------
Build the methodological workflow diagram for the manuscript (Figure 2).

A clean matplotlib-only schematic showing:
    Inputs (raw data) -> Preprocessing blocks -> Feature matrix
        -> Statistical inference -> Validation -> Attribution
            -> CMIP6 projections.

Output: 02_eda/outputs/fig02_workflow.png (600 DPI)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(r"D:/olive_yield_RS_chill/02_eda/outputs")

C_DATA = "#cfe2f3"
C_PRE  = "#d9ead3"
C_MOD  = "#fff2cc"
C_VAL  = "#ead1dc"
C_PROJ = "#fce5cd"
EC = "#333333"


def box(ax, x, y, w, h, text, color, fontsize=8.5):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.15",
                       fc=color, ec=EC, lw=0.9)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            wrap=True)


def arrow(ax, x0, y0, x1, y1, color="#333", style="-|>", lw=1.0):
    a = FancyArrowPatch((x0, y0), (x1, y1),
                        arrowstyle=style, mutation_scale=12,
                        color=color, lw=lw,
                        connectionstyle="arc3,rad=0.0",
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=600)
    ax.set_xlim(-1.6, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # ----- Row 1: raw data -----
    y1 = 7.0
    box(ax, 1.6, y1, 2.6, 0.9,
        "In-situ daily met\nTmin / Tmax 2015-2025\n(n = 3388)", C_DATA)
    box(ax, 4.5, y1, 2.6, 0.9,
        "Sentinel-2 SR\n11 parcels, 319 obs\nNDVI/NDRE/CIre/...", C_DATA)
    box(ax, 7.4, y1, 2.6, 0.9,
        "Yield records\n11 parcels x 8 years\n2016-2024 (no 2021)", C_DATA)
    box(ax, 10.3, y1, 2.6, 0.9,
        "NOAA PSL ONI\nENSO 1950-2025", C_DATA)

    # ----- Row 2: preprocessing -----
    y2 = 5.4
    box(ax, 1.6, y2, 2.6, 0.95,
        "Sinusoidal hourly\nFishman Dynamic Model\n-> CP / CH12 / CU", C_PRE)
    box(ax, 4.5, y2, 2.6, 0.95,
        "Phenology aggregation\nFlor / Cuaj / Crec\n+ heat indices", C_PRE)
    box(ax, 7.4, y2, 2.6, 0.95,
        "Parcel-year long\nQA + COVID exclusion\n(2021 dropped)", C_PRE)
    box(ax, 10.3, y2, 2.6, 0.95,
        "ENSO overlay\nEl Nino 15/16 + 23/24\nLa Nina 20-22", C_PRE)

    # arrows row1 -> row2
    for x in (1.6, 4.5, 7.4, 10.3):
        arrow(ax, x, y1 - 0.5, x, y2 + 0.5)

    # ----- Row 3: feature matrix -----
    y3 = 3.8
    box(ax, 6.0, y3, 6.0, 0.85,
        "Year-level feature matrix (n = 8 years)\n"
        "Tmean_chill_lag1 | Tmean_CREC | CP | NDRE_repro | yield",
        "#ffe599", fontsize=9)
    for x in (1.6, 4.5, 7.4, 10.3):
        arrow(ax, x, y2 - 0.55, 6.0, y3 + 0.45)

    # ----- Row 4: inference -----
    y4 = 2.4
    box(ax, 2.5, y4, 3.4, 0.95,
        "Primary log-OLS\nlog(y+0.5) ~ Tmean_chill\n         + Tmean_CREC", C_MOD)
    box(ax, 6.0, y4, 3.0, 0.95,
        "Validation\nLOYO + log + interior\nbootstrap + permutation", C_VAL)
    box(ax, 9.4, y4, 3.2, 0.95,
        "Sensitivity\nRF / mixed-effects /\nalt chill metrics", C_VAL)
    arrow(ax, 6.0, y3 - 0.45, 2.5, y4 + 0.5)
    arrow(ax, 6.0, y3 - 0.45, 6.0, y4 + 0.5)
    arrow(ax, 6.0, y3 - 0.45, 9.4, y4 + 0.5)

    # ----- Row 5: outputs -----
    y5 = 0.8
    box(ax, 2.5, y5, 3.4, 0.95,
        "Attribution\nCounterfactual 2016/2024\nchill vs heat decomposition", C_PROJ)
    box(ax, 6.0, y5, 3.0, 0.95,
        "CMIP6 delta projections\nSSP1-2.6 / 2-4.5 / 5-8.5\n2031-2100", C_PROJ)
    box(ax, 9.4, y5, 3.2, 0.95,
        "Risk surface\nDeltaT sweep 0-5 C\ncollapse threshold ~+1.25 C", C_PROJ)
    arrow(ax, 2.5, y4 - 0.5, 2.5, y5 + 0.5)
    arrow(ax, 6.0, y4 - 0.5, 6.0, y5 + 0.5)
    arrow(ax, 9.4, y4 - 0.5, 9.4, y5 + 0.5)

    # Stage labels (left margin)
    for y, label in [
        (y1, "RAW DATA"),
        (y2, "PREPROCESS"),
        (y3, "FEATURES"),
        (y4, "INFERENCE"),
        (y5, "ATTRIBUTION\n+ PROJECTION"),
    ]:
        ax.text(-1.5, y, label, fontsize=8, fontweight="bold",
                color="#555", ha="left", va="center")

    fig.suptitle("Figure 2 — Analytical workflow",
                 fontsize=12, y=0.98)
    fig.tight_layout()
    out = OUT / "fig02_workflow.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
