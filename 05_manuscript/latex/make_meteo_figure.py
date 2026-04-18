"""
Generate Figure 2 — AWS meteorological overview for La Yarada-Los Palos, 2015-2025.
Panels:
 (a) Daily Tmin, Tmax (light) with 30-day centered rolling Tmean (dark)
 (b) Annual chill-window (May-Aug, Y-1 convention) mean temperature vs chill year
 (c) ONI overlay for context (monthly).
Saves to figures_final/Figure2.png.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 11
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

BASE = Path(r"D:/olive_yield_RS_chill")
CSV = BASE / "00_raw_data/meteo_daily_2015_2025.csv"
OUT = BASE / "05_manuscript/latex/figures_final/Figure2.png"

df = pd.read_csv(CSV)
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df = df.sort_values('date').reset_index(drop=True)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Chill-window mask (1 May - 31 Aug)
df['chill_window'] = df['month'].between(5, 8)

# 30-day rolling mean of Tmean
df['Tmean_roll'] = df['Tmean'].rolling(30, center=True, min_periods=15).mean()

# Chill-window annual means: read the authoritative Tmean_season values from
# the preprocessing pipeline (hourly sinusoidal reconstruction of min/max, 1 May
# to 31 Aug) so the figure numbers match the year-level dataset used for model
# calibration (Tmean_chill_lag1 in year_level_dataset.csv).
CHILL_CSV = BASE / "01_preprocessing/outputs/chill_season_totals.csv"
chill = pd.read_csv(CHILL_CSV).rename(columns={
    "chill_year": "year",
    "Tmean_season": "Tchill",
    "n_days": "n",
})
chill = chill[["year", "Tchill", "n", "yield_year"]].copy()
chill["harvest_year"] = chill["yield_year"]
# Restrict to the calibration-relevant chill years (2015-2023 -> harvest
# 2016-2024) so the bar chart aligns with the manuscript's eight-year sample.
chill = chill[chill["year"].between(2015, 2023)].reset_index(drop=True)

# Limit the daily-series panel to the analysis window (up to 31 Dec 2024).
df = df[df['date'] <= pd.Timestamp("2024-12-31")].reset_index(drop=True)
# Tag El Nino chill windows
el_nino_chill_years = {2015, 2023}
la_nina_chill_years = {2020, 2021, 2022}
def color_for(y):
    if y in el_nino_chill_years:
        return '#c0392b'  # red
    if y in la_nina_chill_years:
        return '#2c7fb8'  # blue
    return '#7f7f7f'      # grey
chill['color'] = chill['year'].apply(color_for)

# Threshold line (chill-collapse visual reference ~18 C)
THR = 18.0

# --- FIGURE ---------------------------------------------------------------
fig = plt.figure(figsize=(11, 7.5))
gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.18)

# ---- panel (a): daily series + rolling mean, with chill windows shaded ---
ax1 = fig.add_subplot(gs[0])
ax1.fill_between(df['date'], df['Tmin'], df['Tmax'],
                 color='#d0d7de', alpha=0.6, linewidth=0, label=r'$T_{\min}$ – $T_{\max}$ range')
ax1.plot(df['date'], df['Tmean_roll'], color='#1f77b4', lw=1.3,
         label=r'$T_{\rm mean}$ (30-d rolling)')

# Shade chill windows (May-Aug of each year)
for y in range(2015, 2025):
    start = pd.Timestamp(y, 5, 1)
    end = pd.Timestamp(y, 8, 31)
    col = color_for(y)
    ax1.axvspan(start, end, color=col, alpha=0.10, linewidth=0)

ax1.axhline(THR, color='k', ls='--', lw=0.8, alpha=0.6)
# Place the threshold label in a clear region (upper part of the panel) and
# draw an arrow down to the dashed reference line so the text never overlaps
# the daily Tmin-Tmax band or the rolling-mean trace.
ax1.annotate(r'chill-collapse reference $\approx 18\,^{\circ}$C',
             xy=(pd.Timestamp("2020-07-15"), THR),
             xytext=(pd.Timestamp("2022-06-01"), THR + 6.0),
             fontsize=11, color='k',
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       edgecolor='none', alpha=0.85),
             arrowprops=dict(arrowstyle='->', color='k', lw=0.8,
                             alpha=0.7, connectionstyle='arc3,rad=-0.15'))

ax1.set_ylabel(r'Air temperature ($^{\circ}$C)')
ax1.set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2024-12-31"))
# Show only 2016-2024 tick labels on the x-axis (drop the 2015 label at the left edge)
ax1.xaxis.set_major_locator(mdates.YearLocator(month=1, day=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
year_ticks = [pd.Timestamp(y, 1, 1) for y in range(2016, 2025)]
ax1.set_xticks(year_ticks)
ax1.legend(loc='upper left', fontsize=11, ncol=2, frameon=False)
ax1.text(0.985, 0.96, r'$\mathbf{(a)}$', transform=ax1.transAxes,
         ha='right', va='top', fontsize=14)
ax1.grid(True, alpha=0.3)

# ---- panel (b): chill-window annual mean --------------------------------
ax2 = fig.add_subplot(gs[1])
bars = ax2.bar(chill['year'], chill['Tchill'], color=chill['color'],
               edgecolor='k', linewidth=0.6, width=0.75)
ax2.axhline(THR, color='k', ls='--', lw=0.8, alpha=0.6)
# Label bars
for _, row in chill.iterrows():
    ax2.text(row['year'], row['Tchill'] + 0.15, f"{row['Tchill']:.1f}",
             ha='center', va='bottom', fontsize=11)
ax2.set_xticks(chill['year'])
ax2.set_xticklabels([f"{y}\n(H{y+1})" for y in chill['year']], fontsize=11)
ax2.set_xlabel(r'Chill year ($Y-1$, with harvest year $Y$ in parentheses)')
ax2.set_ylabel(r'$T_{\mathrm{chill}}$ (May–Aug, $^{\circ}$C)')
ax2.set_ylim(14.5, max(chill['Tchill'].max() + 1.5, 22))
ax2.text(0.985, 0.96, r'$\mathbf{(b)}$', transform=ax2.transAxes,
         ha='right', va='top', fontsize=14)
ax2.grid(True, axis='y', alpha=0.3)

# Legend for panel (b)
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor='#c0392b', edgecolor='k', label='El Niño chill window'),
    Patch(facecolor='#2c7fb8', edgecolor='k', label='La Niña chill window'),
    Patch(facecolor='#7f7f7f', edgecolor='k', label='Neutral'),
]
ax2.legend(handles=legend_elems, loc='upper left', fontsize=11, frameon=False, ncol=3)

plt.rcParams['text.usetex'] = False
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print("Saved:", OUT)
print("Chill window summary:")
print(chill[['year', 'harvest_year', 'Tchill', 'n']].to_string(index=False))
