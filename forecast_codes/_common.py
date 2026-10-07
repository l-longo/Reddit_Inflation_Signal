"""
_common.py
==========

Shared helpers for the forecasting scripts.

Nothing here changes the empirics: it only replaces the notebook cells that
loaded the raw Reddit files (post titles / comment bodies, which cannot be
redistributed) with a loader for the already-aggregated daily signal series,
and collects the test statistics and the combination scheme that the three
scripts have in common.
"""

import os
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

# pandas >= 2.2 renamed the month-end resample alias
MEND = "ME" if pd.__version__ >= "2.2" else "M"

TARGET_SUFFIX = {"CPIAUCSL": "cpi", "PCEPILFE": "pce"}


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_daily_signals(path):
    """Daily aggregated Reddit signals — the `df_reg` of the original notebooks.

    Duplicated column names (the sentiment block reuses
    'sentiment_lm_economics_count' for all three subreddits) are restored
    after the csv round-trip so that column names and ordering match the
    notebooks exactly.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [re.sub(r"\.\d+$", "", c) for c in df.columns]
    return df


def load_macro(xlsx_path, target):
    """Monthly macro block: infl, inflation, expect, swap (in this order).

    Taken from df_jae_reddit.xlsx, which stores the FRED series exactly as the
    notebooks built them:
        infl       CPIAUCSL / PCEPILFE
        inflation  12-month log change of infl
        expect     MICH, lagged one month
        swap       EXPINF1YR
    Sample 2002-01 .. 2025-08, month-end index. CPIAUCSL is the ALFRED
    vintage of 2025-09-11 (column CPIAUCSL_20250911 of CPIAUCSL_2.xlsx), the
    one behind the published tables.
    """
    suffix = TARGET_SUFFIX[target]
    dfx = pd.read_excel(xlsx_path, index_col=0)
    dfx.index = pd.to_datetime(dfx.index)
    df_monthly = dfx[[f"infl_{suffix}", f"inflation_{suffix}", "expect", "swap"]].copy()
    df_monthly.columns = ["infl", "inflation", "expect", "swap"]
    return df_monthly


def load_macro_fred(api_key, target, obs_start="2002-01-01", obs_end="2025-08-31"):
    """Same block pulled live from FRED (identical to the notebook cell)."""
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    df_monthly = {}
    df_monthly["infl"] = fred.get_series(
        target, observation_start=obs_start, observation_end=obs_end
    )
    df_monthly = pd.DataFrame(df_monthly)
    df_monthly["inflation"] = (np.log(df_monthly.infl)).diff(12)
    df_monthly["expect"] = fred.get_series(
        "MICH", observation_start=obs_start, observation_end=obs_end
    )
    df_monthly["swap"] = fred.get_series(
        "EXPINF1YR", observation_start=obs_start, observation_end=obs_end
    )
    df_monthly["expect"] = df_monthly.expect.shift(1)
    return df_monthly.resample(MEND).mean()


def monthly_merge(df_monthly, df_reg, ma_window=None):
    """Reproduces

        df_daily  = pd.concat([df_monthly, df_reg.rolling(w).mean()], axis=1)
        df_merged = df_daily.resample('M').mean()

    of the notebooks. df_monthly is already month-end here, and averaging a
    monthly series over its own month is the identity, so the two are the same.
    """
    sig = df_reg if ma_window is None else df_reg.rolling(ma_window).mean()
    sig = sig.resample(MEND).mean()
    return pd.concat([df_monthly, sig], axis=1)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def dm_test(loss_model, loss_bench, h=1):
    """
    Diebold-Mariano test for equal forecast accuracy with squared loss.
    H0: E[d]=0 vs H1: E[d]<0, where d = loss_model - loss_bench.
    - Aligns on common index
    - Newey-West (Bartlett) HAC variance with lag = max(h-1, 0)
    - Harvey-Leybourne-Newbold small-sample correction for h>1
    """
    m, b = loss_model.astype(float).align(loss_bench.astype(float), join="inner")
    d = (m - b).to_numpy()
    d = d[~np.isnan(d)]
    T = d.size
    if T < 3:
        return np.nan, np.nan

    lag = max(int(h) - 1, 0)
    d_mean = d.mean()
    d_center = d - d_mean

    gamma0 = np.dot(d_center, d_center) / T
    var = gamma0
    max_k = min(lag, T - 1)
    for k in range(1, max_k + 1):
        cov = np.dot(d_center[:-k], d_center[k:]) / T
        w = 1.0 - k / (lag + 1.0) if lag > 0 else 0.0
        var += 2 * w * cov

    if var <= 0:
        return np.nan, np.nan

    dm = d_mean / np.sqrt(var / T)

    if h > 1:
        dm /= np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)

    pval = norm.cdf(dm)
    return dm, pval


def fluctuation_test(y1, y2, actual, m, P):
    """Giacomini-Rossi fluctuation test statistic on squared-loss differentials."""
    L1 = (actual - y1) ** 2
    L2 = (actual - y2) ** 2
    dL = L2 - L1
    dvar = np.sqrt(np.sum((dL[:] - np.mean(dL)) ** 2) / P)

    F_stat = np.zeros(P - int(m / 2) * 2 + 1)
    for j in range(int(m / 2), P - int(m / 2) + 1):
        F_stat[j - int(m / 2)] = (1 / (dvar * np.sqrt(m))) * np.sum(
            dL[j - int(m / 2) : j + int(m / 2) - 1]
        )

    return F_stat, dL


def star(p, better):
    if not better:
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# --------------------------------------------------------------------------
# forecast combination
# --------------------------------------------------------------------------
def agg_with_discount(
    df_preds,
    df_loss,
    cols,
    delta=0.95,
    eps=1e-8,
    colname="aggregated_pred_dmsfe",
    normalize_avg=True,
    burn_in=0,
    return_weights=False,
):
    nT = len(df_preds)
    agg = []
    weight_history = []

    S = np.zeros(len(cols), dtype=float)
    W = 0.0

    L = df_loss[cols].to_numpy()
    P = df_preds[cols].to_numpy()

    for t in range(nT):
        if t == 0 or t <= burn_in:
            w = np.ones(len(cols)) / len(cols)
            agg.append(np.nanmean(P[t]))
        else:
            prev_loss = np.nan_to_num(L[t - 1], nan=0.0, posinf=1e6, neginf=1e6)
            S = delta * S + prev_loss
            W = delta * W + 1.0
            dmsfe = S / (W + eps) if normalize_avg else S

            w = 1.0 / (dmsfe + eps)
            w_sum = np.sum(w)
            if w_sum <= 0 or not np.isfinite(w_sum):
                w = np.ones(len(cols)) / len(cols)
                agg.append(np.nanmean(P[t]))
            else:
                w /= w_sum
                agg.append(np.nansum(P[t] * w))

        weight_history.append(w)

    df_preds[colname] = agg

    if return_weights:
        weight_df = pd.DataFrame(weight_history, index=df_preds.index, columns=cols)
        return df_preds, weight_df
    return df_preds


# --------------------------------------------------------------------------
def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
