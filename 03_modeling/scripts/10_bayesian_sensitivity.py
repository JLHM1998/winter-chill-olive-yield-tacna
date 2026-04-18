"""
10_bayesian_sensitivity.py
Weak-prior Bayesian sensitivity for the primary chill attribution model.

Because the full posterior of a Gaussian linear model with a Normal prior
on coefficients has a closed form, we compute it exactly without MCMC:

  prior:       beta ~ N(0, tau^2 I)   with tau = 1.0 (weakly informative
                                      given standardised predictors)
  likelihood:  y    ~ N(X beta, sigma^2 I)
  posterior:   beta | y ~ N(mu_post, Sigma_post)
    Sigma_post = sigma^2 (X'X + (sigma^2/tau^2) I)^{-1}
    mu_post    = Sigma_post X'y / sigma^2
               = (X'X + lambda I)^{-1} X'y   with lambda = sigma^2/tau^2

We standardise predictors, use sigma^2 = residual variance of the OLS fit
(plug-in), and report posterior mean, SD, and 95% credible interval.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YEAR_DATA = ROOT / "03_modeling" / "outputs" / "tables" / "year_level_dataset.csv"
OUT_DIR = ROOT / "03_modeling" / "outputs" / "tables"

df = pd.read_csv(YEAR_DATA)
df["log_y"] = np.log(df["yield_tn_ha"] + 0.5)

cols = ["Tmean_chill_lag1", "Tmean_mean_CREC"]
# Standardise predictors so the Normal(0,1) prior is interpretable
means = df[cols].mean()
sds = df[cols].std(ddof=0)
Xs = (df[cols] - means) / sds
Xs = np.column_stack([np.ones(len(df)), Xs.values])
y = df["log_y"].values

# OLS plug-in for sigma^2
ols = sm.OLS(y, Xs).fit()
sigma2 = float(np.sum(ols.resid ** 2) / (len(y) - Xs.shape[1]))

tau = 1.0  # Normal(0, 1) prior on standardised slopes (flat on intercept)
prior_prec = np.diag([0.0, 1 / tau ** 2, 1 / tau ** 2])  # flat intercept

XtX = Xs.T @ Xs
A = XtX / sigma2 + prior_prec
Sigma_post = np.linalg.inv(A)
mu_post = Sigma_post @ (Xs.T @ y) / sigma2
sd_post = np.sqrt(np.diag(Sigma_post))

# Back-transform slopes to the natural (unstandardised) scale
nat_mu = mu_post.copy()
nat_sd = sd_post.copy()
nat_mu[1] /= sds.iloc[0]
nat_mu[2] /= sds.iloc[1]
nat_sd[1] /= sds.iloc[0]
nat_sd[2] /= sds.iloc[1]

names = ["Intercept_std", "T_chill_std", "T_CREC_std"]
nat_names = ["Intercept(nat,approx)", "T_chill(nat)", "T_CREC(nat)"]

lines = []
lines.append("BAYESIAN SENSITIVITY — weak Normal(0, 1) prior on standardised slopes")
lines.append("=" * 70)
lines.append(f"n = {len(df)}; sigma^2 (plug-in) = {sigma2:.4f}; tau = {tau}\n")
lines.append("Posterior on STANDARDISED scale:")
for i, nm in enumerate(names):
    lo, hi = mu_post[i] - 1.96 * sd_post[i], mu_post[i] + 1.96 * sd_post[i]
    lines.append(f"  {nm:<22s} mean={mu_post[i]:+.4f}  sd={sd_post[i]:.4f}  "
                 f"95% CrI=[{lo:+.4f}, {hi:+.4f}]")
lines.append("\nPosterior on NATURAL scale (slopes only, back-rescaled):")
for i, nm in enumerate(nat_names):
    lo, hi = nat_mu[i] - 1.96 * nat_sd[i], nat_mu[i] + 1.96 * nat_sd[i]
    lines.append(f"  {nm:<22s} mean={nat_mu[i]:+.4f}  sd={nat_sd[i]:.4f}  "
                 f"95% CrI=[{lo:+.4f}, {hi:+.4f}]")

# Does beta_chill exclude zero?
lo_c = mu_post[1] - 1.96 * sd_post[1]
hi_c = mu_post[1] + 1.96 * sd_post[1]
excludes_zero = (lo_c * hi_c) > 0
lines.append("")
lines.append(f"beta_T_chill posterior 95% CrI excludes zero: {excludes_zero}")
try:
    coef_df = pd.read_csv(ROOT / "03_modeling" / "outputs" / "tables" / "final_primary_coefficients.csv")
    ols_nat = float(coef_df.loc[coef_df["term"] == "Tmean_chill_lag1", "estimate"].iloc[0])
except Exception:
    ols_nat = float("nan")
lines.append(f"OLS comparison (natural): beta_chill = {ols_nat:+.4f}")

out = "\n".join(lines)
print(out)
with open(OUT_DIR / "bayesian_sensitivity.txt", "w") as f:
    f.write(out + "\n")
print("Wrote:", OUT_DIR / "bayesian_sensitivity.txt")
