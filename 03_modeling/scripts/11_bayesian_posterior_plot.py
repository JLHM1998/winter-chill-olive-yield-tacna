"""
11_bayesian_posterior_plot.py
-----------------------------
Plot the closed-form Gaussian posterior of the chill slope from the weak-prior
Bayesian sensitivity (script 10_bayesian_sensitivity.py). Addresses Reviewer
Major Comment 2.

Outputs:
    05_manuscript/latex/figures_final/FigureS8.png  (600 dpi)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
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
YEAR = ROOT / "03_modeling" / "outputs" / "tables" / "year_level_dataset.csv"
OUT_FIG = ROOT / "05_manuscript" / "latex" / "figures_final"

df = pd.read_csv(YEAR)
df["log_y"] = np.log(df["yield_tn_ha"] + 0.5)

cols = ["Tmean_chill_lag1", "Tmean_mean_CREC"]
sds = df[cols].std(ddof=0)
Xs = (df[cols] - df[cols].mean()) / sds
Xs_mat = np.column_stack([np.ones(len(df)), Xs.values])
y = df["log_y"].values

ols = sm.OLS(y, Xs_mat).fit()
sigma2 = float(np.sum(ols.resid ** 2) / (len(y) - Xs_mat.shape[1]))

tau = 1.0
prior_prec = np.diag([0.0, 1 / tau ** 2, 1 / tau ** 2])
A = Xs_mat.T @ Xs_mat / sigma2 + prior_prec
Sigma_post = np.linalg.inv(A)
mu_post = Sigma_post @ (Xs_mat.T @ y) / sigma2
sd_post = np.sqrt(np.diag(Sigma_post))

mu_chill_std = mu_post[1]
sd_chill_std = sd_post[1]
mu_chill_nat = mu_chill_std / sds.iloc[0]
sd_chill_nat = sd_chill_std / sds.iloc[0]
ci_lo_nat = mu_chill_nat - 1.96 * sd_chill_nat
ci_hi_nat = mu_chill_nat + 1.96 * sd_chill_nat

print(f"Posterior beta_T_chill (natural scale): mean = {mu_chill_nat:+.4f}, "
      f"sd = {sd_chill_nat:.4f}, 95% CrI = [{ci_lo_nat:+.4f}, {ci_hi_nat:+.4f}]")

# Build a fine grid for the density
grid = np.linspace(mu_chill_nat - 4 * sd_chill_nat,
                   mu_chill_nat + 4 * sd_chill_nat, 800)
dens = (1 / (sd_chill_nat * np.sqrt(2 * np.pi))) * \
        np.exp(-0.5 * ((grid - mu_chill_nat) / sd_chill_nat) ** 2)

# OLS point estimate from the manuscript fit (natural scale, Table 2)
ols_beta = -0.816
ols_se = 0.272

fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.fill_between(grid, dens, color="#9ecae1", alpha=0.7,
                label=r"Posterior density (Normal$(0,1)$ prior)")
ax.plot(grid, dens, color="#08519c", lw=1.4)

ax.axvline(mu_chill_nat, color="#08519c", lw=1.5, ls="--",
           label=f"Posterior mean = {mu_chill_nat:+.3f}")
ax.axvline(ci_lo_nat, color="#08519c", lw=1.0, ls=":",
           label=f"95% CrI = [{ci_lo_nat:+.3f}, {ci_hi_nat:+.3f}]")
ax.axvline(ci_hi_nat, color="#08519c", lw=1.0, ls=":")
ax.axvline(0, color="black", lw=1.0)
ax.axvline(ols_beta, color="#d7191c", lw=1.6, ls="-",
           label=f"OLS point estimate = {ols_beta:+.3f}")

ax.set_xlabel(r"$\beta_{T_{\mathrm{chill}}}$  (log-yield change per $^{\circ}$C of chill-window warming)")
ax.set_ylabel("Posterior density")
ax.legend(loc="upper left", fontsize=9, frameon=True)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_FIG / "FigureS8.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT_FIG / 'FigureS8.png'} at 600 dpi")
