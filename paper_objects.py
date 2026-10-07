"""
paper_objects.py  --  rebuild the tables and figures of the main text in one pass
================================================================================

Reads only what the forecast and nowcast scripts have already written (and the
MCS results shipped in data/mcs/) and produces, for CPI (headline) and PCE (core):

  TABLES (printed as text and as LaTeX)
    tab:rmse_cpi / tab:rmse_pce   Forecast results (RMSE ratios), 7 rows x 9 horizons
    tab:nowc_joint                Nowcast results, RMSE and MAE ratios, by cutoff

  FIGURES (saved under the file names the manuscript includes)
    Figure 6   fig:cssed_pce_cpi_h1       rev/cssed_{1,6}_{target}.jpg
    Figure 7   fig:fluctuation_tests      rev/fluctuation_test_1_{target}.jpg
    Figure 8   fig:mcs-cpi-pce-combined   mcs/MCS_heatmap_CPI_PCE_h1_h18_new.png
    Figure 9   fig:f_stat_nowcast         fluctuation_test_nowcasts_{target}_{cutoff}_{10,25}_new.pdf
    Figure 10  fig:cumloss_nowcast        cumloss_nowcasts_{target}_{cutoff}_new.pdf

The figures are drawn by forecast_codes/forecast_charts.py,
forecast_codes/mcs_figures.py and nowcast_codes/nowcast_charts.py, which can
also be run on their own. Figure 4 is in images/.

Cutoffs: CPI is released mid-month, so only +5/+10/+14 keep the Reddit
information strictly prior to the release; PCE is released later and also
allows +22. That is why the CPI panel has no +22 column.

Usage
-----
    python paper_objects.py              # tables + figures, figures shown on screen
    python paper_objects.py --no-show    # save the figures without opening them
"""

import argparse
import glob
import os
import sys
import warnings

import matplotlib
# Figures are shown on screen as well as saved. Pass --no-show (or run without a
# display) to skip the windows; the backend must be chosen before pyplot loads.
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "nowcast_codes"))
sys.path.insert(0, os.path.join(_HERE, "forecast_codes"))
from _common import MEND, dm_test, ensure_dirs, star  # noqa: E402
import forecast_charts  # noqa: E402
import mcs_figures  # noqa: E402
import nowcast_charts  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))

HORIZONS = [1, 2, 3, 4, 5, 6, 9, 12, 18]
TARGETS = [("CPIAUCSL", "CPI (headline)"), ("PCEPILFE", "PCE (core)")]

# cutoffs that keep Reddit information strictly before the official release
CUTOFFS = {"CPIAUCSL": [5, 10, 14], "PCEPILFE": [5, 10, 14, 22]}

AGG = {"aggregated_pred", "aggregated_pred_swap", "aggregated_pred_exp",
       "aggregated_pred_llama", "aggregated_pred_sent", "pred_ar"}

ROWS = ["Michigan Survey", "1-Y Inflation Swap", "Best sentiment",
        "Aggr-RIS", "Best fine-tuned LLM",
        "Aggr-LLaMA70B", "Best LLaMA 70B"]

GROUPS = [("Expectations", ROWS[0:2]),
          ("Lexicon-based sentiment", ROWS[2:3]),
          ("RIS-based models", ROWS[3:7])]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def rmse_ratio(loss, bench):
    return np.sqrt(loss.astype(float).mean()) / np.sqrt(bench.astype(float).mean())


def dm_star(loss, bench, h):
    loss = loss.astype(float)
    dm, p = dm_test(loss, bench, h=h)
    return star(p, loss.mean() < bench.astype(float).mean())


def best_individual(loss, bench, exclude_prefixes=()):
    cand = [c for c in loss.columns
            if c not in AGG and not any(c.startswith(p) for p in exclude_prefixes)]
    r = {c: rmse_ratio(loss[c], bench) for c in cand}
    b = min(r, key=r.get)
    return b, r[b]


def fmt(v, s):
    return f"{v:.3f}{s}"


# --------------------------------------------------------------------------
# 1. forecast tables
# --------------------------------------------------------------------------
def forecast_table(results, target, tag):
    vals = {r: [] for r in ROWS}
    stars = {r: [] for r in ROWS}
    picks = {r: [] for r in ["Best sentiment", "Best fine-tuned LLM", "Best LLaMA 70B"]}

    for h in HORIZONS:
        L = pd.read_csv(f"{results}/loss_{h}_{target}{tag}.csv", index_col=0)
        S = pd.read_csv(f"{results}/loss_sentiment_{h}_{target}{tag}.csv", index_col=0)
        M = pd.read_csv(f"{results}/loss_{h}_{target}_llama70{tag}.csv", index_col=0)
        b, bs, bm = L["pred_ar"], S["pred_ar"], M["pred_ar"]

        def put(row, series, bench):
            vals[row].append(rmse_ratio(series, bench))
            stars[row].append(dm_star(series, bench, h))

        put("Michigan Survey", L["aggregated_pred_exp"], b)
        put("1-Y Inflation Swap", L["aggregated_pred_swap"], b)
        put("Aggr-RIS", L["aggregated_pred"], b)
        put("Aggr-LLaMA70B", M["aggregated_pred"], bm)

        n, _ = best_individual(S, bs)
        picks["Best sentiment"].append(n); put("Best sentiment", S[n], bs)

        n, _ = best_individual(L, b, ("expect_", "swap_"))
        picks["Best fine-tuned LLM"].append(n); put("Best fine-tuned LLM", L[n], b)

        n, _ = best_individual(M, bm, ("expect_", "swap_"))
        picks["Best LLaMA 70B"].append(n); put("Best LLaMA 70B", M[n], bm)

    return vals, stars, picks


def print_forecast_table(target, nice, vals, stars, picks):
    print()
    print("=" * 113)
    print(f"  Forecast results (RMSE ratios) for {nice}      [tab:rmse_"
          f"{'cpi' if target == 'CPIAUCSL' else 'pce'}]")
    print("=" * 113)
    print("".ljust(32) + "".join(f"h={h}".rjust(9) for h in HORIZONS))
    print("-" * 113)
    for gname, grows in GROUPS:
        print(f"  -- {gname} --")
        for r in grows:
            print("  " + r.ljust(30)
                  + "".join(fmt(v, s).rjust(9) for v, s in zip(vals[r], stars[r])))
    print("-" * 113)
    print("  Best model picked at each horizon:")
    for k, v in picks.items():
        print(f"    {k}:")
        print("      " + ", ".join(f"h{h}: {n}" for h, n in zip(HORIZONS, v)))


def latex_forecast_table(target, nice, vals, stars):
    lab = "cpi" if target == "CPIAUCSL" else "pce"
    out = [r"\begin{table}",
           rf"\caption{{Forecast results (RMSE ratios) for \textit{{{nice}}}. \label{{tab:rmse_{lab}}}}}",
           r"\centering", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{lccccccccc}", r"\hline\hline",
           " & " + " & ".join(f"$h={h}$" for h in HORIZONS) + r" \\", r"\hline"]
    for gname, grows in GROUPS:
        out.append(rf"\multicolumn{{10}}{{c}}{{\textit{{{gname}}}}} \\")
        out.append(r"\hline")
        for r in grows:
            cells = " & ".join(
                f"${v:.3f}" + (rf"^{{{s}}}$" if s else "$")
                for v, s in zip(vals[r], stars[r]))
            out.append(f"{r} & {cells} " + r"\\")
    out += [r"\hline\hline", r"\end{tabular}", "}", r"\end{table}"]
    return "\n".join(out)


# --------------------------------------------------------------------------
# 2. nowcast table + the frame the nowcast figures are drawn from
# --------------------------------------------------------------------------
def load_fed(nowcast_dir):
    files = glob.glob(os.path.join(nowcast_dir, "Year-Over-YearPercentChange-*.csv"))
    if not files:
        raise FileNotFoundError(f"no Fed nowcast csv in {nowcast_dir}")
    dfs = []
    for f in files:
        parts = os.path.basename(f).replace(".csv", "").split("-")
        year, month = parts[-2], parts[-1].zfill(2)
        d = pd.read_csv(f)
        d = d[d["Label"].str.startswith(month + "/")].copy()
        d["date"] = pd.to_datetime(year + "/" + d["Label"], format="%Y/%m/%d")
        dfs.append(d.set_index("date"))
    return pd.concat(dfs).sort_index().drop(columns=["Label"], errors="ignore")


def comparison_frame(df_fed, nowcast_dir, target, cutoff, llm_sfx, llama_sfx,
                     start, end):
    shifted = df_fed.copy()
    shifted.index = shifted.index - pd.to_timedelta(cutoff, unit="D")
    sel = shifted.resample(MEND).last()

    llama = pd.read_csv(f"{nowcast_dir}/all_models_{cutoff}_{target}{llama_sfx}.csv",
                        index_col=0)
    llama_p = pd.read_csv(
        f"{nowcast_dir}/df_pred_nowcasts_{cutoff}_{target}{llama_sfx}.csv", index_col=0)
    llm_p = pd.read_csv(
        f"{nowcast_dir}/df_pred_nowcasts_{cutoff}_{target}{llm_sfx}.csv", index_col=0)
    for d in (llama, llama_p, llm_p):
        d.index = pd.to_datetime(d.index)
    llama_p.columns = llama_p.columns + "_llama"

    return pd.concat([
        llama[["inflation", "pred_ar"]],
        llama_p[["aggregated_pred_llama"]],
        llm_p["aggregated_pred"],
        sel[[f"Core {target[:3]} Inflation", f"{target[:3]} Inflation"]] / 100,
    ], axis=1).dropna()[start:end]


def nowcast_table(df_fed, nowcast_dir, llm_sfx, llama_sfx, start, end):
    res = {}
    for target, _ in TARGETS:
        res[target] = {"RMSE": {}, "MAE": {}}
        for metric in ("RMSE", "MAE"):
            for row in ("Aggr-RIS", "Aggr-LLaMA70B"):
                res[target][metric][row] = []
        for cutoff in CUTOFFS[target]:
            cmp_ = comparison_frame(df_fed, nowcast_dir, target, cutoff,
                                    llm_sfx, llama_sfx, start, end)
            e_b = cmp_.inflation - cmp_.pred_ar
            for row, col in [("Aggr-RIS", "aggregated_pred"),
                             ("Aggr-LLaMA70B", "aggregated_pred_llama")]:
                e_m = cmp_.inflation - cmp_[col]
                r = np.sqrt((e_m ** 2).mean() / (e_b ** 2).mean())
                m = e_m.abs().mean() / e_b.abs().mean()
                dm, p = dm_test(e_m ** 2, e_b ** 2, h=1)
                res[target]["RMSE"][row].append(
                    (r, star(p, (e_m ** 2).mean() < (e_b ** 2).mean())))
                dm, p = dm_test(e_m.abs(), e_b.abs(), h=1)
                res[target]["MAE"][row].append(
                    (m, star(p, e_m.abs().mean() < e_b.abs().mean())))
    return res


def print_nowcast_table(res):
    print()
    print("=" * 113)
    print("  Nowcast results: RMSE and MAE ratios relative to the AR(1) benchmark"
          "      [tab:nowc_joint]")
    print("=" * 113)
    allc = [5, 10, 14, 22]
    for panel, (target, nice) in zip("AB", TARGETS):
        print(f"\n  Panel {panel} — {nice}")
        print("    " + "Loss / Model".ljust(38)
              + "".join(f"+{c} days".rjust(12) for c in allc))
        print("    " + "-" * 86)
        for metric in ("RMSE", "MAE"):
            for j, row in enumerate(("Aggr-RIS",
                                     "Aggr-LLaMA70B")):
                cells = ""
                for c in allc:
                    if c in CUTOFFS[target]:
                        v, s = res[target][metric][row][CUTOFFS[target].index(c)]
                        cells += fmt(v, s).rjust(12)
                    else:
                        cells += "n.a.".rjust(12)
                print("    " + (metric if j == 0 else "").ljust(6)
                      + row.ljust(32) + cells)
        print("    " + "-" * 86)
    print("\n  n.a. = cutoff not used: CPI is released mid-month, so a +22-day")
    print("  cutoff would use Reddit content published after the release.")


def latex_nowcast_table(res):
    allc = [5, 10, 14, 22]
    out = [r"\begin{table}",
           r"\caption{Nowcast results for CPI and PCE: RMSE and MAE ratios relative "
           r"to the AR(1) benchmark. \label{tab:nowc_joint}}",
           r"\centering", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{llcccc}", r"\toprule"]
    for panel, (target, nice) in zip("AB", TARGETS):
        out.append(rf"{{Panel {panel}}} & & \multicolumn{{4}}{{c}}{{\textbf{{{nice}}}}} \\")
        out.append(r"\midrule")
        if panel == "A":
            out.append("Loss & Model & " + " & ".join(f"$+{c}$ days" for c in allc)
                       + r" \\")
            out.append(r"\midrule")
        for metric in ("RMSE", "MAE"):
            out.append(rf"\multirow{{2}}{{*}}{{{metric}}}")
            for row in ("Aggr-RIS", "Aggr-LLaMA70B"):
                cells = []
                for c in allc:
                    if c in CUTOFFS[target]:
                        v, s = res[target][metric][row][CUTOFFS[target].index(c)]
                        cells.append(f"{v:.3f}{s}")
                    else:
                        cells.append(r"\NA")
                out.append(f"& {row}      & " + " & ".join(cells) + r" \\")
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"]
    return "\n".join(out)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--nowcast-dir", default=os.path.join(ROOT, "data_fed_nowcast"))
    ap.add_argument("--mcs-dir", default=os.path.join(ROOT, "data", "mcs"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "figures_paper"))
    ap.add_argument("--tag", default="_new")
    ap.add_argument("--llm-suffix", default="_new")
    ap.add_argument("--llama-suffix", default="_llama_new")
    ap.add_argument("--start", default="2013-08-31")
    ap.add_argument("--end", default="2025-05-31")
    ap.add_argument("--no-latex", action="store_true")
    ap.add_argument("--no-show", action="store_true",
                    help="save the figures without opening them on screen")
    args = ap.parse_args()

    ensure_dirs(args.fig_dir)
    show = not args.no_show

    # ---- tables ----
    latex_blocks = []
    for target, nice in TARGETS:
        vals, stars_, picks = forecast_table(args.results, target, args.tag)
        print_forecast_table(target, nice, vals, stars_, picks)
        latex_blocks.append(latex_forecast_table(target, nice, vals, stars_))

    df_fed = load_fed(args.nowcast_dir)
    res = nowcast_table(df_fed, args.nowcast_dir, args.llm_suffix, args.llama_suffix,
                        args.start, args.end)
    print_nowcast_table(res)
    latex_blocks.append(latex_nowcast_table(res))

    # ---- figures ----
    print()
    print("=" * 113)
    print("  FIGURES")
    print("=" * 113)
    fc = forecast_charts.run(args.results, args.fig_dir, args.tag, show)
    if (fc.status != "ok").any():
        print(fc.to_string(index=False))
    mcs = mcs_figures.run(args.mcs_dir, os.path.join(args.results, "test_MCS"),
                          args.fig_dir, show, verbose=False)
    nowcast_charts.LLM_SUFFIX, nowcast_charts.LLAMA_SUFFIX = args.llm_suffix, args.llama_suffix
    nowcast_charts.EVAL_START, nowcast_charts.EVAL_END = args.start, args.end
    nc = nowcast_charts.run(args.nowcast_dir, args.fig_dir, show, metrics=False)

    rel = lambda p: os.path.relpath(str(p), ROOT)
    print("  Figure 6  fig:cssed_pce_cpi_h1")
    for h in (1, 6):
        for t, _ in TARGETS:
            print("    " + rel(os.path.join(args.fig_dir, "rev", f"cssed_{h}_{t}.jpg")))
    print("  Figure 7  fig:fluctuation_tests")
    for t, _ in TARGETS:
        print("    " + rel(os.path.join(args.fig_dir, "rev", f"fluctuation_test_1_{t}.jpg")))
    print("  Figure 8  fig:mcs-cpi-pce-combined")
    for p in mcs:
        print("    " + rel(p))
    print("  Figure 9  fig:f_stat_nowcast")
    for p in nc:
        if "fluctuation" in str(p):
            print("    " + rel(p))
    print("  Figure 10 fig:cumloss_nowcast")
    for p in nc:
        if "cumloss" in str(p):
            print("    " + rel(p))

    if not args.no_latex:
        print()
        print("=" * 113)
        print("  LATEX")
        print("=" * 113)
        for b in latex_blocks:
            print()
            print(b)


if __name__ == "__main__":
    main()
