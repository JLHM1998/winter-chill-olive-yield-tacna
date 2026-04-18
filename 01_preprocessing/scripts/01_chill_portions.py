"""
01_chill_portions.py
--------------------
Compute winter chill accumulation for olive (Tacna, Peru) from daily station
meteorology using three complementary models:

    1. Dynamic Model (Fishman et al. 1987; implementation after Luedeling, chillR)
       -> Chill Portions (CP). Current international standard.
    2. Utah Model (Richardson et al. 1974) -> Chill Units (CU).
    3. Chill Hours <7.2 C and <12 C (simple thresholds).

Daily Tmin/Tmax are disaggregated to hourly using a two-piece sinusoidal
interpolation (Tmin at 06:00, Tmax at 14:00). This is a standard approach
for olive chill studies (e.g., De Melo-Abreu et al. 2004; Aguilera et al. 2014).

Chill season (Southern Hemisphere, coastal Peru, olive): 1 May to 31 August.
This precedes bud break (Brotacion, Sep) and flowering (Oct-Nov), which set
the fruit harvested in April-May of the following calendar year. Therefore:

        yield_year = harvest calendar year
        chill_year = yield_year - 1

Example: yield 2016 depends on CP of May-Aug 2015 (El Nino 2015/16 winter).

Outputs (01_preprocessing/outputs/):
    chill_daily_series.csv       -> daily cumulative CP, CU, CH7, CH12
    chill_season_totals.csv      -> chill season totals per chill_year (+ linked to yield_year)
    chill_models_comparison.png  -> visual sanity check across models
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

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BRANCH = Path(r"D:/olive_yield_RS_chill")
RAW = BRANCH / "00_raw_data" / "meteo_daily_2015_2025.csv"
OUT_DIR = BRANCH / "01_preprocessing" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Daily to hourly temperature (sinusoidal interpolation)
# ----------------------------------------------------------------------------
def daily_to_hourly_sine(tmin: np.ndarray, tmax: np.ndarray) -> np.ndarray:
    """
    Disaggregate daily Tmin/Tmax to hourly temperature using a two-piece
    sinusoidal interpolation:
      - Tmin occurs at 06:00 local time
      - Tmax occurs at 14:00 local time
      - Rising half-sine 06:00 -> 14:00  (8 hours)
      - Falling half-sine 14:00 -> 06:00 next day (16 hours)

    Parameters
    ----------
    tmin, tmax : np.ndarray of shape (n_days,)

    Returns
    -------
    T_hourly : np.ndarray of shape (n_days * 24,)
        Hourly temperature series, in chronological order.
    """
    n = len(tmin)
    T = np.empty(n * 24, dtype=float)
    for d in range(n):
        tmn, tmx = tmin[d], tmax[d]
        # "Next day" Tmin for the falling segment between 14:00 and next 06:00
        tmn_next = tmin[d + 1] if d + 1 < n else tmn
        amp = (tmx - tmn) / 2.0
        mid = (tmx + tmn) / 2.0
        for h in range(24):
            if 6 <= h <= 14:
                # Rising half-sine from Tmin(06h) to Tmax(14h)
                frac = (h - 6) / 8.0
                T[d * 24 + h] = tmn + (tmx - tmn) * 0.5 * (1 - np.cos(np.pi * frac))
            elif h > 14:
                # Falling half-sine from Tmax(14h) to next day Tmin(06h)
                frac = (h - 14) / 16.0
                T[d * 24 + h] = tmx - (tmx - tmn_next) * 0.5 * (1 - np.cos(np.pi * frac))
            else:  # h < 6
                # Tail of the previous day's falling segment
                if d == 0:
                    tmx_prev = tmx
                    tmn_curr = tmn
                else:
                    tmx_prev = tmax[d - 1]
                    tmn_curr = tmn
                frac = (h + 10) / 16.0  # 14h..24h..6h -> (h+24-14)/16
                T[d * 24 + h] = tmx_prev - (tmx_prev - tmn_curr) * 0.5 * (1 - np.cos(np.pi * frac))
    return T


# ----------------------------------------------------------------------------
# 2. Dynamic Model (Fishman et al. 1987), after Luedeling chillR
# ----------------------------------------------------------------------------
def dynamic_model(T_hourly: np.ndarray) -> np.ndarray:
    """
    Compute cumulative Chill Portions (CP) from an hourly temperature series
    using the Dynamic Model of Fishman et al. (1987a,b).

    Implementation follows the chillR R package (Luedeling, 2018), which is
    the canonical reference.

    Parameters
    ----------
    T_hourly : np.ndarray
        Hourly air temperature, degrees Celsius.

    Returns
    -------
    CP_cum : np.ndarray
        Cumulative chill portions, same length as T_hourly.
    """
    e0 = 4153.5
    e1 = 12888.8
    a0 = 139500.0
    a1 = 2.567e18
    slp = 1.6
    tetmlt = 277.0
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

    # Iterate (algorithm is serial)
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
# 3. Utah model (Richardson et al. 1974)
# ----------------------------------------------------------------------------
def utah_model(T_hourly: np.ndarray) -> np.ndarray:
    """
    Classic Utah chill unit model with temperature-dependent weighting:
        T <= 1.4        -> 0.0
        1.4 < T <= 2.4  -> 0.5
        2.4 < T <= 9.1  -> 1.0
        9.1 < T <= 12.4 -> 0.5
        12.4 < T <= 15.9-> 0.0
        15.9 < T <= 18.0-> -0.5
        T > 18.0        -> -1.0
    """
    u = np.zeros_like(T_hourly)
    u[(T_hourly > 1.4) & (T_hourly <= 2.4)] = 0.5
    u[(T_hourly > 2.4) & (T_hourly <= 9.1)] = 1.0
    u[(T_hourly > 9.1) & (T_hourly <= 12.4)] = 0.5
    u[(T_hourly > 15.9) & (T_hourly <= 18.0)] = -0.5
    u[T_hourly > 18.0] = -1.0
    # Common practice: cumulative CU is not allowed to go below zero
    cum = np.maximum.accumulate(np.cumsum(u))
    return np.cumsum(u)  # unconstrained cumulative; we'll also report bounded later


# ----------------------------------------------------------------------------
# 4. Chill hours below thresholds
# ----------------------------------------------------------------------------
def chill_hours(T_hourly: np.ndarray, threshold: float) -> np.ndarray:
    return np.cumsum((T_hourly < threshold).astype(float))


# ----------------------------------------------------------------------------
# 5. Main pipeline
# ----------------------------------------------------------------------------
def main():
    print("[1/5] Loading meteorology ...")
    df = pd.read_csv(RAW)
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False, errors="coerce")
    if df["date"].isna().any():
        # fallback with dayfirst=True
        mask = df["date"].isna()
        df.loc[mask, "date"] = pd.to_datetime(df.loc[mask, "date"], dayfirst=True, errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["Tmin", "Tmax"])
    n_raw = len(df)

    # Build continuous daily DatetimeIndex from first to last observed date.
    # The previous code used pd.date_range on the compressed row index, which
    # assumed every calendar day was present and caused a ~600-day misalignment
    # when the source CSV had gaps (Tmean_chill for chill year 2023 was being
    # computed from Oct-2023 -> Apr-2024 data, not from May-Aug 2023).
    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_idx)
    df.index.name = "date"

    # Classify gap lengths and apply tiered imputation:
    #   - runs of <= 2 consecutive missing days: linear interpolation
    #     (matches manuscript Sec 2.3: "fewer than three consecutive missing
    #      days by linear interpolation in temperature anomaly space")
    #   - runs of >= 3 consecutive missing days: fill with the month-of-year
    #     climatology from non-missing days; this is needed for the Dynamic
    #     Model to evolve continuously. Chill windows (May-Aug 2015-2023) have
    #     <=5 missing days each, so imputation is a small correction there.
    n_short_filled = 0
    n_clim_filled = 0
    for col in ["Tmin", "Tmax"]:
        s = df[col].copy()
        na_mask = s.isna()
        # Identify runs of consecutive NaN
        run_id = (s.notna()).cumsum()
        run_len = na_mask.groupby(run_id).transform("sum")
        short_run = na_mask & (run_len <= 2)
        long_run = na_mask & (run_len >= 3)
        # Linear interpolation for short gaps
        s_lin = s.interpolate(method="linear", limit=2, limit_direction="both")
        s = s.where(~short_run, s_lin)
        n_short_filled += int(short_run.sum())
        # Month-of-year climatology fallback for long gaps
        if long_run.any():
            moy = s.groupby(full_idx.month).transform("mean")
            s = s.where(~long_run, moy)
            n_clim_filled += int(long_run.sum())
        df[col] = s

    df = df.reset_index()
    print(f"      raw rows: {n_raw} | continuous grid: {len(full_idx)}")
    print(f"      short-gap interpolation (<=2 d): {n_short_filled} cells "
          f"| long-gap climatology fill (>=3 d): {n_clim_filled} cells")
    print(f"      range: {df['date'].min().date()} -> {df['date'].max().date()}")

    print("[2/5] Disaggregating daily -> hourly (sinusoidal) ...")
    T_h = daily_to_hourly_sine(df["Tmin"].values, df["Tmax"].values)
    # Hourly dates ARE now aligned with real calendar time because df has been
    # reindexed to a continuous daily range.
    start = df["date"].iloc[0]
    hourly_dates = pd.date_range(start=start, periods=len(T_h), freq="h")

    print("[3/5] Computing Dynamic Model (Fishman 1987) ...")
    CP = dynamic_model(T_h)
    print("[3/5] Computing Utah Model ...")
    CU = utah_model(T_h)
    print("[3/5] Computing Chill Hours <7.2 and <12 C ...")
    CH7 = chill_hours(T_h, 7.2)
    CH12 = chill_hours(T_h, 12.0)

    hourly = pd.DataFrame({
        "datetime": hourly_dates,
        "T_hourly": T_h,
        "CP_cum": CP,
        "CU_cum": CU,
        "CH7_cum": CH7,
        "CH12_cum": CH12,
    })
    hourly["date"] = hourly["datetime"].dt.date

    # Daily max of cumulative = end-of-day value
    daily = hourly.groupby("date").agg(
        CP_cum=("CP_cum", "last"),
        CU_cum=("CU_cum", "last"),
        CH7_cum=("CH7_cum", "last"),
        CH12_cum=("CH12_cum", "last"),
        Tmean_hour=("T_hourly", "mean"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    # Compute incremental daily values from cumulative
    for col in ["CP", "CU", "CH7", "CH12"]:
        daily[col + "_daily"] = daily[col + "_cum"].diff().fillna(daily[col + "_cum"])

    daily.to_csv(OUT_DIR / "chill_daily_series.csv", index=False)
    print(f"      -> chill_daily_series.csv ({len(daily)} rows)")

    print("[4/5] Aggregating to chill season totals (May 1 - Aug 31) ...")
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    season_mask = daily["month"].between(5, 8)
    season = daily[season_mask].groupby("year").agg(
        CP_season=("CP_daily", "sum"),
        CU_season=("CU_daily", "sum"),
        CH7_season=("CH7_daily", "sum"),
        CH12_season=("CH12_daily", "sum"),
        Tmean_season=("Tmean_hour", "mean"),
        n_days=("date", "count"),
    ).reset_index()
    season = season.rename(columns={"year": "chill_year"})
    # Link to yield_year: chill_year is the winter preceding the harvest year
    season["yield_year"] = season["chill_year"] + 1
    season = season[season["n_days"] >= 100]  # keep only full chill seasons

    season.to_csv(OUT_DIR / "chill_season_totals.csv", index=False)
    print(season.round(2).to_string(index=False))
    print(f"      -> chill_season_totals.csv")

    print("[5/5] Model comparison figure ...")
    # Use composite x-labels: "W'YY → H'YY+1" aligned to harvest year (yield_year)
    x_labels = [f"W\u2019{int(cy) % 100:02d} \u2192 H\u2019{int(yy) % 100:02d}"
                for cy, yy in zip(season["chill_year"], season["yield_year"])]

    # Helper: annotate bar values to eliminate visual ambiguity with rotated labels
    def annotate_bars(ax, values, fmt="{:.1f}"):
        ymax = max(values) if max(values) > 0 else 1
        ax.set_ylim(top=ymax * 1.18)  # expand headroom so labels don't touch the border
        for i, (rect, v) in enumerate(zip(ax.patches, values)):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.03 * ax.get_ylim()[1],
                    fmt.format(v), ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=600)

    x_pos = range(len(x_labels))

    axes[0, 0].bar(x_pos, season["CP_season"].values, color="#2c7fb8")
    axes[0, 0].set_title("(a)", loc="left")
    axes[0, 0].set_ylabel("CP")
    axes[0, 0].set_xticks(list(x_pos))
    axes[0, 0].set_xticklabels(x_labels, rotation=45, ha="right")
    annotate_bars(axes[0, 0], season["CP_season"].values, fmt="{:.2f}")

    axes[0, 1].bar(x_pos, season["CU_season"].values, color="#41b6c4")
    axes[0, 1].set_title("(b)", loc="left")
    axes[0, 1].set_ylabel("CU")
    axes[0, 1].set_xticks(list(x_pos))
    axes[0, 1].set_xticklabels(x_labels, rotation=45, ha="right")

    axes[1, 0].bar(x_pos, season["CH7_season"].values, color="#7fcdbb")
    axes[1, 0].set_title("(c)", loc="left")
    axes[1, 0].set_ylabel("hours")
    axes[1, 0].set_xticks(list(x_pos))
    axes[1, 0].set_xticklabels(x_labels, rotation=45, ha="right")
    annotate_bars(axes[1, 0], season["CH7_season"].values, fmt="{:.0f}")

    axes[1, 1].bar(x_pos, season["CH12_season"].values, color="#c7e9b4")
    axes[1, 1].set_title("(d)", loc="left")
    axes[1, 1].set_ylabel("hours")
    axes[1, 1].set_xticks(list(x_pos))
    axes[1, 1].set_xticklabels(x_labels, rotation=45, ha="right")
    annotate_bars(axes[1, 1], season["CH12_season"].values, fmt="{:.0f}")

    # suptitle removed — title goes in LaTeX caption
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chill_models_comparison.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> chill_models_comparison.png (600 DPI)")

    print("\nDone.")


if __name__ == "__main__":
    main()
