"""
paper_objects.py  --  rebuild the paper's tables and figures in one pass
========================================================================

Reads only what the forecast and nowcast scripts have already written and
produces, for CPI (headline) and PCE (core):

  TABLES
    tab:rmse_cpi / tab:rmse_pce   Forecast results (RMSE ratios), 7 rows x 9 horizons
    tab:nowc_joint                Nowcast results, RMSE and MAE ratios, by cutoff

  FIGURES
    fig:fluctuation_tests         forecast fluctuation test at h = 1, both targets
    fig:f_stat_nowcast            nowcast fluctuation test, CPI +14d and PCE +22d,
                                  each at mu = 0.1 and mu = 0.2
    fig:cumloss_nowcast           10-period moving average of the nowcast loss
                                  differential vs AR(1), CPI +14d and PCE +22d

Each table is printed as text and as a LaTeX body; each figure is written both
under the file name the manuscript includes and as a combined preview png.

Cutoffs: CPI is released mid-month, so only +5/+10/+14 keep the Reddit
information strictly prior to the release; PCE is released later and also
allows +22. That is why the CPI panel has no +22 column.

Usage
-----
    python paper_objects.py
    python paper_objects.py --results results --nowcast-dir data_fed_nowcast
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "forecast_codes"))
from _common import MEND, dm_test, ensure_dirs, fluctuation_test, star  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))

HORIZONS = [1, 2, 3, 4, 5, 6, 9, 12, 18]
TARGETS = [("CPIAUCSL", "CPI (headline)"), ("PCEPILFE", "PCE (core)")]

# cutoffs that keep Reddit information strictly before the official release
CUTOFFS = {"CPIAUCSL": [5, 10, 14], "PCEPILFE": [5, 10, 14, 22]}
# cutoff used in the figures, per target
FIG_CUTOFF = {"CPIAUCSL": 14, "PCEPILFE": 22}

AGG = {"aggregated_pred", "aggregated_pred_swap", "aggregated_pred_exp",
       "aggregated_pred_llama", "aggregated_pred_sent", "pred_ar"}

ROWS = ["Michigan Survey", "1-Y Inflation Swap", "Best sentiment",
        "LLM forecast aggregation", "Best fine-tuned LLM",
        "Llama70B forecast aggregation", "Best Llama70B"]

GROUPS = [("Expectations", ROWS[0:2]),
          ("Reddit-sentiment", ROWS[2:3]),
          ("Reddit-LLM", ROWS[3:7])]


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
    picks = {r: [] for r in ["Best sentiment", "Best fine-tuned LLM", "Best Llama70B"]}

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
        put("LLM forecast aggregation", L["aggregated_pred"], b)
        put("Llama70B forecast aggregation", M["aggregated_pred"], bm)

        n, _ = best_individual(S, bs)
        picks["Best sentiment"].append(n); put("Best sentiment", S[n], bs)

        n, _ = best_individual(L, b, ("expect_", "swap_"))
        picks["Best fine-tuned LLM"].append(n); put("Best fine-tuned LLM", L[n], b)

        n, _ = best_individual(M, bm, ("expect_", "swap_"))
        picks["Best Llama70B"].append(n); put("Best Llama70B", M[n], bm)

    return vals, stars, picks


def print_forecast_table(target, nice, vals, stars, picks):
    print()
    print("=" * 104)
    print(f"  Forecast results (RMSE ratios) for {nice}      [tab:rmse_"
          f"{'cpi' if target == 'CPIAUCSL' else 'pce'}]")
    print("=" * 104)
    print("".ljust(32) + "".join(f"h={h}".rjust(8) for h in HORIZONS))
    print("-" * 104)
    for gname, grows in GROUPS:
        print(f"  -- {gname} --")
        for r in grows:
            print("  " + r.ljust(30)
                  + "".join(fmt(v, s).rjust(8) for v, s in zip(vals[r], stars[r])))
    print("-" * 104)
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
            for row in ("LLM forecast aggregation", "Llama70B forecast aggregation"):
                res[target][metric][row] = []
        for cutoff in CUTOFFS[target]:
            cmp_ = comparison_frame(df_fed, nowcast_dir, target, cutoff,
                                    llm_sfx, llama_sfx, start, end)
            e_b = cmp_.inflation - cmp_.pred_ar
            for row, col in [("LLM forecast aggregation", "aggregated_pred"),
                             ("Llama70B forecast aggregation", "aggregated_pred_llama")]:
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
    print("=" * 104)
    print("  Nowcast results: RMSE and MAE ratios relative to the AR(1) benchmark"
          "      [tab:nowc_joint]")
    print("=" * 104)
    allc = [5, 10, 14, 22]
    for panel, (target, nice) in zip("AB", TARGETS):
        print(f"\n  Panel {panel} — {nice}")
        print("    " + "Loss / Model".ljust(38)
              + "".join(f"+{c} days".rjust(12) for c in allc))
        print("    " + "-" * 86)
        for metric in ("RMSE", "MAE"):
            for j, row in enumerate(("LLM forecast aggregation",
                                     "Llama70B forecast aggregation")):
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
            for row in ("LLM forecast aggregation", "Llama70B forecast aggregation"):
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
# 3. figures
# --------------------------------------------------------------------------
def _finish(path, show):
    plt.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close()


def fig_fluctuation_forecast(results, tag, figdir, h=1, m=10, show=True):
    paths = []
    for target, _ in TARGETS:
        res = pd.read_csv(f"{results}/llm/df_tuning_{h}_{target}{tag}.csv", index_col=0)
        res_l = pd.read_csv(f"{results}/llm/df_tuning_{h}_{target}_llama70{tag}.csv",
                            index_col=0)
        res_ar = pd.read_csv(f"{results}/sentiment/df_sentiment_{h}_{target}{tag}.csv",
                             index_col=0)
        res_sw = pd.read_csv(f"{results}/swap/df_tuning_swap_{h}_{target}{tag}.csv",
                             index_col=0)
        res_ex = pd.read_csv(
            f"{results}/expectation/df_tuning_exp_{h}_{target}{tag}.csv", index_col=0)
        true = pd.read_csv(f"{results}/df_true_{h}_{target}{tag}.csv", index_col=0)

        actual, y1 = true["inflation"], res_ar["pred_ar"]
        P = res_ar["pred_ar"].shape[0]
        F_swap, _ = fluctuation_test(res_sw["aggregated_pred_swap"], y1, actual, m, P)
        F_exp, _ = fluctuation_test(res_ex["aggregated_pred_exp"], y1, actual, m, P)
        F_llm, _ = fluctuation_test(res["aggregated_pred"], y1, actual, m, P)
        F_lla, _ = fluctuation_test(res_l["aggregated_pred"], y1, actual, m, P)

        x = pd.to_datetime(actual.index[m - 1:])
        plt.figure(figsize=(8, 4))
        plt.plot(x, F_swap, marker=".", linestyle=":", label="F-statistics, Inflation-swap")
        plt.plot(x, F_exp, marker=".", linestyle="-", label="F-statistics, Michigan expectations")
        plt.plot(x, F_llm, marker=".", linestyle=":", label="F-statistics, Reddit-LLM")
        plt.plot(x, F_lla, marker=".", linestyle=":", label="F-statistics, Llama70B")
        plt.axhline(y=3.176, linestyle="--", color="red", alpha=0.7, label="critical value")
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("F-statistic", fontsize=14)
        plt.title(f"{h}-month-ahead Fluctuation Test. Target = {target[:3]}", fontsize=14)
        plt.legend(frameon=False, loc="best", fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p = f"{figdir}/fluctuation_test_{h}_{target}.jpg"
        _finish(p, show); paths.append(p)
    return paths


def fig_nowcast(df_fed, nowcast_dir, figdir, llm_sfx, llama_sfx, start, end,
                fed_mae_headline=True, show=True):
    fluct, cumloss = [], []
    for target, _ in TARGETS:
        cutoff = FIG_CUTOFF[target]
        cmp_ = comparison_frame(df_fed, nowcast_dir, target, cutoff,
                                llm_sfx, llama_sfx, start, end)
        col_fed = (f"Core {target[:3]} Inflation" if target == "PCEPILFE"
                   else f"{target[:3]} Inflation")
        actual, y1 = cmp_["inflation"], cmp_["pred_ar"]
        P = y1.shape[0]

        for m in (10, 25):
            F_llm, _ = fluctuation_test(cmp_["aggregated_pred"], y1, actual, m, P)
            F_lla, _ = fluctuation_test(cmp_["aggregated_pred_llama"], y1, actual, m, P)
            F_fed, _ = fluctuation_test(cmp_[col_fed], y1, actual, m, P)
            off = m - 1 if m == 10 else m - 2
            x = pd.to_datetime(actual.index[off:])
            plt.figure(figsize=(8, 4))
            plt.plot(x, F_fed, marker=".", linestyle=":", label="F-statistics, FED Nowcast")
            plt.plot(x, F_llm, marker=".", linestyle=":", color="green",
                     label="F-statistics, Reddit-LLM")
            plt.plot(x, F_lla, marker=".", linestyle=":", color="red",
                     label="F-statistics, Llama70B")
            plt.axhline(y=3.393 if m == 10 else 3.179, linestyle="--", color="red",
                        alpha=0.7, label="critical value")
            plt.xlabel("Date", fontsize=14)
            plt.ylabel(r"F-statistic, $\mathbf{\mu = 0.1}$" if m == 10
                       else r"F-statistic, $\mathbf{\mu = 0.2}$", fontsize=14)
            plt.title(f"Nowcasts comparison. Cutoff = {cutoff} days. {target[:3]}.",
                      fontsize=14)
            plt.legend(frameon=False, loc="best", fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            p = f"{figdir}/fluctuation_test_nowcasts_{target}_{cutoff}_{m}_new.pdf"
            _finish(p, show); fluct.append(p)

        # cumulative-loss chart
        base = (actual - y1).rolling(10, center=True).mean()
        e = lambda c: (actual - cmp_[c]).rolling(10, center=True).mean() - base
        rr = lambda c, q: (np.sqrt(((actual - cmp_[c]) ** 2).mean() /
                                   ((actual - y1) ** 2).mean()) if q == 2
                           else (actual - cmp_[c]).abs().mean() / (actual - y1).abs().mean())
        # The notebook computes the Fed RMSE on `Core PCE Inflation` but the Fed
        # MAE on headline `PCE Inflation`. Kept by default so the figure matches
        # the published one (it is what gives MAE = 2.524 in the PCE panel);
        # --fix-fed-mae uses the core series for both.
        lab = {}
        for c in (col_fed, "aggregated_pred", "aggregated_pred_llama"):
            c_mae = (f"{target[:3]} Inflation"
                     if (c == col_fed and fed_mae_headline) else c)
            r2 = rr(c, 2)
            r1 = (actual - cmp_[c_mae]).abs().mean() / (actual - y1).abs().mean()
            dm2, p2 = dm_test((actual - cmp_[c]) ** 2, (actual - y1) ** 2, h=1)
            dm1, p1 = dm_test((actual - cmp_[c_mae]).abs(), (actual - y1).abs(), h=1)
            lab[c] = (f"RMSE = {r2:.3f}{star(p2, r2 < 1)}, "
                      f"MAE = {r1:.3f}{star(p1, r2 < 1)}")

        plt.figure(figsize=(8, 4))
        plt.plot(cmp_.index, e(col_fed), label=f"FED Nowcast: {lab[col_fed]}")
        plt.plot(cmp_.index, e("aggregated_pred"), color="green",
                 label=f"llm nowcast: {lab['aggregated_pred']}")
        plt.plot(cmp_.index, e("aggregated_pred_llama"), color="red",
                 label=f"llama nowcast: {lab['aggregated_pred_llama']}")
        plt.axhline(0, color="black", linewidth=1, linestyle="--")
        plt.xlabel("Date")
        plt.legend(frameon=False)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p = f"{figdir}/cumloss_nowcasts_{target}_{cutoff}_new.pdf"
        _finish(p, show); cumloss.append(p)
    return fluct, cumloss


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--nowcast-dir", default=os.path.join(ROOT, "data_fed_nowcast"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "figures_paper"))
    ap.add_argument("--tag", default="_new")
    ap.add_argument("--llm-suffix", default="_new")
    ap.add_argument("--llama-suffix", default="_llama_new")
    ap.add_argument("--start", default="2013-08-31")
    ap.add_argument("--end", default="2025-05-31")
    ap.add_argument("--no-latex", action="store_true")
    ap.add_argument("--no-show", action="store_true",
                    help="save the figures without opening them on screen")
    ap.add_argument("--fix-fed-mae", action="store_true",
                    help="use the core series for the Fed MAE in the PCE panel of "
                         "fig:cumloss_nowcast (the notebook uses headline there)")
    args = ap.parse_args()

    ensure_dirs(args.fig_dir)

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
    print("=" * 104)
    print("  FIGURES")
    print("=" * 104)
    show = not args.no_show
    p1 = fig_fluctuation_forecast(args.results, args.tag, args.fig_dir, show=show)
    print("  fig:fluctuation_tests")
    for p in p1:
        print("    " + os.path.relpath(p, ROOT))
    f2, f3 = fig_nowcast(df_fed, args.nowcast_dir, args.fig_dir,
                         args.llm_suffix, args.llama_suffix, args.start, args.end,
                         fed_mae_headline=not args.fix_fed_mae, show=show)
    print("  fig:f_stat_nowcast")
    for p in f2:
        print("    " + os.path.relpath(p, ROOT))
    print("  fig:cumloss_nowcast")
    for p in f3:
        print("    " + os.path.relpath(p, ROOT))

    if not args.no_latex:
        print()
        print("=" * 104)
        print("  LATEX")
        print("=" * 104)
        for b in latex_blocks:
            print()
            print(b)


if __name__ == "__main__":
    main()
