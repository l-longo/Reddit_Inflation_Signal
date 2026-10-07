"""
nowcast_charts.py  --  nowcast figures of the paper (comparison with the Fed)
============================================================================

Terminal version of JAE_revision_fed_nowcast_charts.ipynb.

Reads the Cleveland Fed inflation nowcasts in data_fed_nowcast/ and the output
of new_codes_real_time.py / new_codes_real_time_llama70B.py (written to the
same folder), aligns the Fed nowcast to each cutoff, and draws, at the cutoff
used in the paper (CPI +14 days, core PCE +22 days):

  fluctuation_test_nowcasts_{target}_{cutoff}_{m}_new.pdf   m = 10 (mu = 0.1), 25 (mu = 0.2)
  cumloss_nowcasts_{target}_{cutoff}_new.pdf

Paper: Figure 9 (fig:f_stat_nowcast) and Figure 10 (fig:cumloss_nowcast).
It also prints, for every cutoff, the RMSE and MAE ratios against AR(1) of the
Fed, Aggr-RIS and Aggr-LLaMA70B nowcasts, with Diebold-Mariano stars.

The Fed MAE is computed on the same Fed column as the RMSE (headline CPI for
CPI, core PCE for PCE).

This script uses no Reddit text.

Usage
-----
    python nowcast_charts.py                  # save the figures, show nothing
    python nowcast_charts.py --show           # also open them
"""

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

import matplotlib

if __name__ == "__main__" and "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "forecast_codes"))

from _common import MEND, dm_test, fluctuation_test, star  # noqa: E402

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
PATH = ROOT / "data_fed_nowcast"
OUT_DIR = ROOT / "figures_paper"
LLM_SUFFIX = "_new"
LLAMA_SUFFIX = "_llama_new"
EVAL_START = "2013-08-31"
EVAL_END = "2025-05-31"
FED_CUTOFFS = [5, 10, 14, 22]

# cutoff and fluctuation windows used in the paper
NOWCAST_SPECS = {
    "CPIAUCSL": {"cutoff": 14, "m_list": [10, 25]},
    "PCEPILFE": {"cutoff": 22, "m_list": [10, 25]},
}
# cutoffs reported in the nowcast table (CPI is released mid-month)
TABLE_CUTOFFS = {"CPIAUCSL": [5, 10, 14], "PCEPILFE": [5, 10, 14, 22]}
CRITICAL = {10: 3.393, 25: 3.179}
MU_LABEL = {10: r"$\mathbf{\mu = 0.1}$", 25: r"$\mathbf{\mu = 0.2}$"}
TARGET_SHORT = {"CPIAUCSL": "CPI", "PCEPILFE": "PCE"}

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "verm": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
}
SERIES = [
    # column, label, colour
    ("fed", "FED Nowcast", OKABE_ITO["sky"]),
    ("aggregated_pred", "Aggr-RIS nowcast", OKABE_ITO["green"]),
    ("aggregated_pred_llama", "Aggr-LLaMA70B nowcast", OKABE_ITO["orange"]),
]
SHOW = False


def _maybe_show():
    if SHOW:
        plt.show()


# ============================================================
# DATA
# ============================================================
def load_fed_nowcast_daily(path=PATH):
    """Append the monthly Cleveland Fed files into one daily frame."""
    files = glob.glob(os.path.join(path, "Year-Over-YearPercentChange-*.csv"))
    if not files:
        raise FileNotFoundError(f"no Year-Over-YearPercentChange-*.csv in {path}")
    dfs = []
    for f in files:
        parts = os.path.basename(f).replace(".csv", "").split("-")
        year, month = parts[-2], parts[-1].zfill(2)
        d = pd.read_csv(f)
        d = d[d["Label"].str.startswith(month + "/")].copy()
        d["date"] = pd.to_datetime(year + "/" + d["Label"], format="%Y/%m/%d")
        dfs.append(d.set_index("date"))
    return pd.concat(dfs).sort_index().drop(columns=["Label"], errors="ignore")


def resample_with_front_extension(df, days, freq=MEND, agg="last"):
    """The first `days` days of each month count towards the previous month."""
    s = df.copy()
    s.index = s.index - pd.to_timedelta(days, unit="D")
    return getattr(s.resample(freq), agg)()


def fed_column(target):
    return "Core PCE Inflation" if target == "PCEPILFE" else "CPI Inflation"


def _read_first(names):
    for n in names:
        p = PATH / n
        if p.exists():
            d = pd.read_csv(p, index_col=0)
            d.index = pd.to_datetime(d.index)
            return d
    raise FileNotFoundError(f"none of {names} in {PATH}")


def prepare_comparison(target, cutoff, fed_monthly):
    llama = _read_first([f"all_models_{cutoff}_{target}{LLAMA_SUFFIX}.csv",
                         f"all_models_{cutoff}_{target}_llama.csv"])
    llama_p = _read_first([f"df_pred_nowcasts_{cutoff}_{target}{LLAMA_SUFFIX}.csv",
                           f"df_pred_nowcasts_{cutoff}_{target}_llama.csv"])
    llm_p = _read_first([f"df_pred_nowcasts_{cutoff}_{target}{LLM_SUFFIX}.csv",
                         f"df_pred_nowcasts_{cutoff}_{target}.csv"])
    llama_p = llama_p.rename(columns={"aggregated_pred": "aggregated_pred_llama"})
    fed = fed_monthly[cutoff][[fed_column(target)]].rename(
        columns={fed_column(target): "fed"}) / 100
    return pd.concat([
        llama[["inflation", "pred_ar"]],
        llama_p[["aggregated_pred_llama"]],
        llm_p[["aggregated_pred"]],
        fed,
    ], axis=1).dropna()[EVAL_START:EVAL_END]


def metrics_and_dm(df):
    """RMSE and MAE ratios vs AR(1), each with its DM star."""
    e_b = df.inflation - df.pred_ar
    out = {}
    for col, _, _ in SERIES:
        e_m = df.inflation - df[col]
        rmse = np.sqrt((e_m ** 2).mean() / (e_b ** 2).mean())
        mae = e_m.abs().mean() / e_b.abs().mean()
        _, p2 = dm_test(e_m ** 2, e_b ** 2, h=1)
        _, p1 = dm_test(e_m.abs(), e_b.abs(), h=1)
        out[col] = {"rmse": rmse, "rmse_star": star(p2, rmse < 1),
                    "mae": mae, "mae_star": star(p1, mae < 1)}
    return out


# ============================================================
# FIGURES
# ============================================================
def plot_nowcast_fluctuation(df, target, cutoff, m):
    actual, y_ar = df["inflation"], df["pred_ar"]
    P = len(actual)
    start = 2 * (m // 2) - 1
    plt.figure(figsize=(8, 4))
    for col, label, color in SERIES:
        F, _ = fluctuation_test(df[col], y_ar, actual, m, P)
        x = pd.to_datetime(actual.index[start:start + len(F)])
        plt.plot(x, F, lw=2, color=color,
                 label=f"F-statistics, {label.replace(' nowcast', '')}")
    plt.axhline(CRITICAL[m], color="black", lw=1.8, ls="--", alpha=.8,
                label="critical value")
    plt.axhline(0, color=OKABE_ITO["grey"], lw=1.4, ls="--", alpha=.9)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"F-statistic, {MU_LABEL[m]}", fontsize=12)
    plt.title(f"Nowcasts comparison. Cutoff = {cutoff} days. {TARGET_SHORT[target]}.",
              fontsize=13)
    plt.legend(frameon=False, fontsize=9)
    plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout()
    out = OUT_DIR / f"fluctuation_test_nowcasts_{target}_{cutoff}_{m}_new.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    _maybe_show()
    plt.close()
    return out


def plot_nowcast_loss(df, target, cutoff, metrics, window=10):
    """10-period centred moving average of the nowcast error minus that of AR(1)
    (the construction of the original notebook)."""
    base = (df["inflation"] - df["pred_ar"]).rolling(window, center=True).mean()
    plt.figure(figsize=(8, 4))
    for col, label, color in SERIES:
        y = (df["inflation"] - df[col]).rolling(window, center=True).mean() - base
        r = metrics[col]
        plt.plot(df.index, y, lw=2, color=color,
                 label=f"{label}: RMSE = {r['rmse']:.3f}{r['rmse_star']}, "
                       f"MAE = {r['mae']:.3f}{r['mae_star']}")
    plt.axhline(0, color="black", lw=1, ls="--")
    plt.xlabel("Date")
    plt.legend(frameon=False, fontsize=8)
    plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout()
    out = OUT_DIR / f"cumloss_nowcasts_{target}_{cutoff}_new.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    _maybe_show()
    plt.close()
    return out


def run_nowcast_spec(target, spec, fed_monthly):
    df = prepare_comparison(target, spec["cutoff"], fed_monthly)
    met = metrics_and_dm(df)
    files = [plot_nowcast_fluctuation(df, target, spec["cutoff"], m)
             for m in spec["m_list"]]
    files.append(plot_nowcast_loss(df, target, spec["cutoff"], met))
    return files


def print_metrics(fed_monthly):
    for target, cutoffs in TABLE_CUTOFFS.items():
        print(f"\n{target}: loss ratios vs AR(1), {EVAL_START[:7]} to {EVAL_END[:7]}")
        print(f"{'cutoff':>7} {'model':<22} {'RMSE':>9} {'MAE':>9}")
        for c in cutoffs:
            met = metrics_and_dm(prepare_comparison(target, c, fed_monthly))
            for col, label, _ in SERIES:
                r = met[col]
                print(f"{'+' + str(c):>7} {label.replace(' nowcast', ''):<22} "
                      f"{r['rmse']:.3f}{r['rmse_star']:<3} {r['mae']:.3f}{r['mae_star']:<3}")


# ============================================================
def run(nowcast_dir=None, out_dir=None, show=None, metrics=True):
    global PATH, OUT_DIR, SHOW
    if nowcast_dir is not None:
        PATH = Path(nowcast_dir)
    if out_dir is not None:
        OUT_DIR = Path(out_dir)
    if show is not None:
        SHOW = show
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    daily = load_fed_nowcast_daily(PATH)
    fed_monthly = {c: resample_with_front_extension(daily, c) for c in FED_CUTOFFS}
    if metrics:
        print_metrics(fed_monthly)
    files = []
    for target, spec in NOWCAST_SPECS.items():
        files += run_nowcast_spec(target, spec, fed_monthly)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nowcast-dir", default=str(PATH))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--show", action="store_true", help="also open the figures")
    args = ap.parse_args()
    files = run(args.nowcast_dir, args.out, args.show)
    print("\nsaved:")
    for f in files:
        print(" ", f)


if __name__ == "__main__":
    main()
