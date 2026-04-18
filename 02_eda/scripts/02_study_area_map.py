"""
02_study_area_map.py
--------------------
Build the publication-quality Study Area figure (Figure 1 of the manuscript).

Two panels:
    (a) Regional context: South-American Pacific coast, Peru-Chile-Bolivia
        triple-frontier area, with a red star at the Tacna olive farm.
    (b) Farm-level parcel layout: 11 olive parcels coloured by mean
        observed yield (2016-2024 ex 2021), Sentinel-2 RGB-style basemap.

Output: 02_eda/outputs/fig01_study_area.png (600 DPI)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import contextily as cx
from pathlib import Path

BRANCH = Path(r"D:/olive_yield_RS_chill")
PARCELS = BRANCH / "00_raw_data" / "Parcelas.shp"
YIELD = BRANCH / "00_raw_data" / "yield_parcels_2016_2024.csv"
OUT = BRANCH / "02_eda" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("[1/3] Loading parcels and yield ...")
    parcels = gpd.read_file(PARCELS).to_crs(epsg=3857)  # Web Mercator
    yld = pd.read_csv(YIELD)
    yld_mean = yld[yld["year"] != 2021].groupby("parcel_id")["yield_tn_ha"].mean().reset_index()
    yld_mean.columns = ["Parcela", "yield_mean"]
    parcels = parcels.merge(yld_mean, on="Parcela", how="left")
    print(parcels[["Parcela", "Area", "yield_mean"]].round(2))

    # Centroid (single farm point) for the regional inset
    centroid_3857 = parcels.geometry.unary_union.centroid
    cent_geo = (
        gpd.GeoSeries([centroid_3857], crs=3857).to_crs(4326).iloc[0]
    )
    lon0, lat0 = cent_geo.x, cent_geo.y
    print(f"      farm centroid: lon={lon0:.4f}, lat={lat0:.4f}")

    print("[2/3] Building figure ...")
    fig = plt.figure(figsize=(12, 6.5), dpi=600)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---- Panel (a) regional context ---------------------------------------
    # Build a 6 deg x 5 deg box centred on the farm, in Web Mercator
    deg_buf_lon, deg_buf_lat = 5.0, 4.0
    bbox_geo = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [lon0 - deg_buf_lon, lon0 + deg_buf_lon],
            [lat0 - deg_buf_lat, lat0 + deg_buf_lat],
        ),
        crs=4326,
    ).to_crs(3857)
    xmin, ymin = bbox_geo.geometry.iloc[0].x, bbox_geo.geometry.iloc[0].y
    xmax, ymax = bbox_geo.geometry.iloc[1].x, bbox_geo.geometry.iloc[1].y
    ax_a.set_xlim(xmin, xmax)
    ax_a.set_ylim(ymin, ymax)

    # Try a topographic basemap; fall back to plain
    try:
        cx.add_basemap(ax_a, source=cx.providers.Esri.WorldShadedRelief,
                       zoom=7, attribution_size=6)
    except Exception as e:
        print(f"      [warn] Esri WorldShadedRelief failed: {e}")
        try:
            cx.add_basemap(ax_a, source=cx.providers.OpenTopoMap, zoom=7,
                           attribution_size=6)
        except Exception as e2:
            print(f"      [warn] OpenTopoMap also failed: {e2}")

    # Star marker at farm
    ax_a.plot(centroid_3857.x, centroid_3857.y, marker="*", ms=22,
              color="#d73027", mec="k", mew=1.0, zorder=5,
              label=f"Tacna olive farm ({abs(lat0):.2f}°S, {abs(lon0):.2f}°W)")

    # Annotate countries
    txt_kw = dict(fontsize=10, fontweight="bold", color="#222",
                  ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.25",
                            fc="white", ec="none", alpha=0.7))
    for lon_t, lat_t, label in [
        (-75.0, -14.0, "PERU"),
        (-70.0, -22.5, "CHILE"),
        (-67.5, -16.5, "BOLIVIA"),
        (-72.0, -16.0, "Pacific\nOcean"),
    ]:
        pt = gpd.GeoSeries(gpd.points_from_xy([lon_t], [lat_t]),
                           crs=4326).to_crs(3857).iloc[0]
        if xmin <= pt.x <= xmax and ymin <= pt.y <= ymax:
            ax_a.text(pt.x, pt.y, label, **txt_kw)

    ax_a.set_title("(a)", loc="left", fontsize=11)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_edgecolor("k")
        spine.set_linewidth(0.8)
    ax_a.legend(loc="upper right", fontsize=8, framealpha=0.95)

    # Lat/lon graticule labels (manual, since axes are Mercator)
    lat_ticks = [-13, -15, -17, -19, -21]
    lon_ticks = [-74, -72, -70, -68]
    grat = []
    for la in lat_ticks:
        pt = gpd.GeoSeries(gpd.points_from_xy([lon0 - deg_buf_lon * 0.97], [la]),
                           crs=4326).to_crs(3857).iloc[0]
        if ymin <= pt.y <= ymax:
            ax_a.text(xmin + (xmax - xmin) * 0.015, pt.y, f"{abs(la)}°S",
                      fontsize=7, ha="left", va="center",
                      bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    for lo in lon_ticks:
        pt = gpd.GeoSeries(gpd.points_from_xy([lo], [lat0 - deg_buf_lat * 0.97]),
                           crs=4326).to_crs(3857).iloc[0]
        if xmin <= pt.x <= xmax:
            ax_a.text(pt.x, ymin + (ymax - ymin) * 0.015, f"{abs(lo)}°W",
                      fontsize=7, ha="center", va="bottom",
                      bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    # ---- Panel (b) farm-level layout --------------------------------------
    minx, miny, maxx, maxy = parcels.total_bounds
    pad = 0.2 * max(maxx - minx, maxy - miny)
    ax_b.set_xlim(minx - pad, maxx + pad)
    ax_b.set_ylim(miny - pad, maxy + pad)

    try:
        cx.add_basemap(ax_b, source=cx.providers.Esri.WorldImagery,
                       zoom=17, attribution_size=6)
    except Exception as e:
        print(f"      [warn] Esri WorldImagery failed: {e}")

    # Choropleth on yield_mean
    cmap = plt.cm.RdYlGn
    vmin = float(parcels["yield_mean"].min())
    vmax = float(parcels["yield_mean"].max())
    parcels.plot(
        column="yield_mean", ax=ax_b, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolor="k", lw=1.0, alpha=0.78,
    )
    # Parcel labels at centroid
    for _, row in parcels.iterrows():
        c = row.geometry.centroid
        ax_b.text(c.x, c.y, row["Parcela"], fontsize=7, ha="center",
                  va="center", color="white", fontweight="bold",
                  path_effects=[])
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax_b, fraction=0.038, pad=0.02)
    cb.set_label("Mean observed yield 2016-2024\n(t ha$^{-1}$, ex 2021)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax_b.set_title("(b)", loc="left", fontsize=11)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for spine in ax_b.spines.values():
        spine.set_edgecolor("k")
        spine.set_linewidth(0.8)

    # Scale bar (approx) for panel b — 100 m at this latitude in m
    scale_m = 100.0
    sb_x0 = minx + (maxx - minx) * 0.05
    sb_y0 = miny - pad * 0.25
    ax_b.plot([sb_x0, sb_x0 + scale_m], [sb_y0, sb_y0], color="white", lw=4,
              solid_capstyle="butt")
    ax_b.plot([sb_x0, sb_x0 + scale_m], [sb_y0, sb_y0], color="k", lw=2,
              solid_capstyle="butt")
    ax_b.text(sb_x0 + scale_m / 2, sb_y0 + pad * 0.07, "100 m",
              fontsize=8, ha="center", color="k",
              bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))

    # North arrow
    na_x = maxx + pad * 0.55
    na_y0 = miny + (maxy - miny) * 0.7
    na_y1 = miny + (maxy - miny) * 0.95
    ax_b.annotate("", xy=(na_x, na_y1), xytext=(na_x, na_y0),
                  arrowprops=dict(arrowstyle="->", lw=1.6, color="k"))
    ax_b.text(na_x, na_y1 + pad * 0.05, "N", fontsize=10, ha="center",
              fontweight="bold")

    # suptitle removed — title goes in LaTeX caption
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "fig01_study_area.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[3/3] -> {out}")


if __name__ == "__main__":
    main()
