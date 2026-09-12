"""
fed_nowcast_results.py  --  nowcast evaluation against the Fed nowcast and AR(1)
================================================================================

Terminal version of nowcast_codes/fed_nowcast_results.ipynb.

Reads the Cleveland Fed inflation nowcasts (the
`Year-Over-YearPercentChange-YYYY-M.csv` files in data_fed_nowcast/) together
with the output that new_codes_real_time.py and new_codes_real_time_llama70B.py
write there, and reports, for every cutoff, the RMSE and MAE of the Fed
nowcast, of the Reddit-LLM nowcast and of the Llama-70B nowcast relative to the
AR(1) benchmark, each with a Diebold-Mariano test. It also saves the
fluctuation-test and nowcast charts as pdf.

This script uses no Reddit data of any kind.

Usage
-----
    python fed_nowcast_results.py                     # core PCE
    python fed_nowcast_results.py --target CPIAUCSL
    python fed_nowcast_results.py --no-figures
"""

import argparse
import glob
import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

# the shared helpers live next door, in forecast_codes/
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "forecast_codes"))

from _common import MEND, dm_test, ensure_dirs, fluctuation_test, star

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_fed_nowcasts(path):
    """Append the monthly Cleveland Fed nowcast files into one daily frame."""
    all_files = glob.glob(os.path.join(path, "Year-Over-YearPercentChange-*.csv"))
    if not all_files:
        raise FileNotFoundError(
            f"no Year-Over-YearPercentChange-*.csv found in {path}"
        )

    dfs = []
    for file in all_files:
        base = os.path.basename(file)
        parts = base.replace(".csv", "").split("-")
        year = parts[-2]
        month = parts[-1].zfill(2)

        df = pd.read_csv(file)
        df = df[df["Label"].str.startswith(month + "/")].copy()
        df["date"] = pd.to_datetime(year + "/" + df["Label"], format="%Y/%m/%d")
        df = df.set_index("date")
        dfs.append(df)

    df_final = pd.concat(dfs).sort_index()
    return df_final.drop(columns=["Label"], errors="ignore")


def resample_with_front_extension(df, days, freq=MEND, agg="last"):
    """days: how many first days of each month count towards the previous month."""
    df_shifted = df.copy()
    df_shifted.index = df_shifted.index - pd.to_timedelta(days, unit="D")
    return getattr(df_shifted.resample(freq), agg)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="PCEPILFE", choices=["PCEPILFE", "CPIAUCSL"])
    ap.add_argument("--nowcast-dir", default=os.path.join(ROOT, "data_fed_nowcast"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 14, 22])
    ap.add_argument("--llm-suffix", default="_new",
                    help="suffix of the fine-tuned-LLM nowcast files")
    ap.add_argument("--llama-suffix", default="_llama_new",
                    help="suffix of the Llama-70B nowcast files")
    ap.add_argument("--start", default="2013-08-31")
    ap.add_argument("--end", default="2025-05-31")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    target = args.target
    P_ = args.nowcast_dir

    if args.no_figures:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dirs(args.fig_dir)

    df_final = load_fed_nowcasts(P_)
    dfs = {c: resample_with_front_extension(df_final, c) for c in args.cutoffs}

    column_fed = (
        f"Core {target[:3]} Inflation" if target == "PCEPILFE" else f"{target[:3]} Inflation"
    )
    column_llama = "aggregated_pred_llama"
    column_llm = "aggregated_pred"

    for cutoff in args.cutoffs:
        df_to_select = dfs[cutoff].copy()

        df_llama = pd.read_csv(
            f"{P_}/all_models_{cutoff}_{target}{args.llama_suffix}.csv", index_col=0
        )
        df_llama_pred = pd.read_csv(
            f"{P_}/df_pred_nowcasts_{cutoff}_{target}{args.llama_suffix}.csv", index_col=0
        )
        df_llama.index = pd.to_datetime(df_llama.index)
        df_llama_pred.index = pd.to_datetime(df_llama_pred.index)

        df_llm_pred = pd.read_csv(
            f"{P_}/df_pred_nowcasts_{cutoff}_{target}{args.llm_suffix}.csv", index_col=0
        )
        df_llm_pred.index = pd.to_datetime(df_llm_pred.index)

        df_llama_pred.columns = df_llama_pred.columns + "_llama"

        df_comparison = pd.concat(
            [
                df_llama[["inflation", "pred_ar"]],
                df_llama_pred[[column_llama]],
                df_llm_pred[column_llm],
                df_to_select[[f"Core {target[:3]} Inflation", f"{target[:3]} Inflation"]] / 100,
            ],
            axis=1,
        ).dropna()[args.start : args.end]

        def _rel(col, power):
            num = ((df_comparison.inflation - df_comparison[col]).abs() ** power).mean()
            den = ((df_comparison.inflation - df_comparison.pred_ar).abs() ** power).mean()
            return np.sqrt(num / den) if power == 2 else num / den

        rmse_fed, rmse_llama, rmse_llm = (_rel(c, 2) for c in (column_fed, column_llama, column_llm))
        rmae_fed = _rel(f"{target[:3]} Inflation", 1)
        rmae_llama, rmae_llm = _rel(column_llama, 1), _rel(column_llm, 1)

        print("\n")
        print(f"{cutoff}")

        loss_ar_sq = (df_comparison.inflation - df_comparison.pred_ar) ** 2
        dm_fed_rmse, p_fed_rmse = dm_test(
            (df_comparison.inflation - df_comparison[column_fed]) ** 2, loss_ar_sq, h=1)
        dm_llama_rmse, p_llama_rmse = dm_test(
            (df_comparison.inflation - df_comparison[column_llama]) ** 2, loss_ar_sq, h=1)
        dm_llm_rmse, p_llm_rmse = dm_test(
            (df_comparison.inflation - df_comparison[column_llm]) ** 2, loss_ar_sq, h=1)

        loss_ar_abs = (df_comparison.inflation - df_comparison.pred_ar).abs()
        dm_fed_mae, p_fed_mae = dm_test(
            (df_comparison.inflation - df_comparison[column_fed]).abs(), loss_ar_abs, h=1)
        dm_llama_mae, p_llama_mae = dm_test(
            (df_comparison.inflation - df_comparison[column_llama]).abs(), loss_ar_abs, h=1)
        dm_llm_mae, p_llm_mae = dm_test(
            (df_comparison.inflation - df_comparison[column_llm]).abs(), loss_ar_abs, h=1)

        print("\n=== RMSE (relative to AR) with DM stars ===")
        print(f"Fed:   {rmse_fed:.3f}{star(p_fed_rmse, rmse_fed < 1)}  "
              f"(DM={dm_fed_rmse:.2f},   p={p_fed_rmse:.3f})")
        print(f"Llama: {rmse_llama:.3f}{star(p_llama_rmse, rmse_llama < 1)}  "
              f"(DM={dm_llama_rmse:.2f}, p={p_llama_rmse:.3f})")
        print(f"LLM:   {rmse_llm:.3f}{star(p_llm_rmse, rmse_llm < 1)}  "
              f"(DM={dm_llm_rmse:.2f},   p={p_llm_rmse:.3f})")

        print("\n=== MAE (relative to AR) with DM stars ===")
        print(f"Fed:   {rmae_fed:.3f}{star(p_fed_mae, rmae_fed < 1)}  "
              f"(DM={dm_fed_mae:.2f},   p={p_fed_mae:.3f})")
        print(f"Llama: {rmae_llama:.3f}{star(p_llama_mae, rmae_llama < 1)}  "
              f"(DM={dm_llama_mae:.2f}, p={p_llama_mae:.3f})")
        print(f"LLM:   {rmae_llm:.3f}{star(p_llm_mae, rmae_llm < 1)}  "
              f"(DM={dm_llm_mae:.2f},   p={p_llm_mae:.3f})")

        if args.no_figures:
            continue

        F = args.fig_dir
        for m in (10, 25):
            P = df_comparison["pred_ar"].shape[0]
            actual = df_comparison["inflation"]
            y1 = df_comparison["pred_ar"]

            plt.figure(figsize=(8, 4))
            F_stat_llm, _ = fluctuation_test(df_comparison[column_llm], y1, actual, m, P)
            F_stat_llama, _ = fluctuation_test(df_comparison[column_llama], y1, actual, m, P)
            F_stat_fed, _ = fluctuation_test(df_comparison[column_fed], y1, actual, m, P)

            off = m - 1 if m == 10 else m - 2
            x = pd.to_datetime(actual.index[off:])
            plt.plot(x, F_stat_fed, marker=".", linestyle=":",
                     label="F-statistics, FED Nowcast")
            plt.plot(x, F_stat_llm, marker=".", linestyle=":", color="green",
                     label="F-statistics, Reddit-LLM")
            plt.plot(x, F_stat_llama, marker=".", linestyle=":", color="red",
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
            plt.savefig(f"{F}/fluctuation_test_nowcasts_{target}_{cutoff}_{m}_new.pdf",
                        dpi=300)
            plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(df_comparison.index, df_comparison[column_fed], label="FED Nowcast")
        plt.plot(df_comparison.index, df_comparison["inflation"], label="Inflation",
                 color="black")
        plt.plot(df_comparison.index, df_comparison[column_llm], label="llm nowcast",
                 color="green")
        plt.plot(df_comparison.index, df_comparison[column_llama], label="llama nowcast",
                 color="red")
        plt.xlabel("Date")
        plt.ylabel("Rate")
        plt.legend(frameon=False)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{F}/chart_nowcasts_{target}_{cutoff}_new.pdf", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 4))
        base = (df_comparison["inflation"] - df_comparison["pred_ar"]).rolling(
            10, center=True).mean()
        for col, lbl, color, r, s in [
            (column_fed, "FED Nowcast", None, rmse_fed,
             f"{star(p_fed_rmse, rmse_fed < 1)}, MAE = {rmae_fed:.3f}"
             f"{star(p_fed_mae, rmse_fed < 1)}"),
            (column_llm, "llm nowcast", "green", rmse_llm,
             f"{star(p_llm_rmse, rmse_llm < 1)}, MAE = {rmae_llm:.3f}"
             f"{star(p_llm_mae, rmse_llm < 1)}"),
            (column_llama, "llama nowcast", "red", rmse_llama,
             f"{star(p_llama_rmse, rmse_llama < 1)}, MAE = {rmae_llama:.3f}"
             f"{star(p_llama_mae, rmse_llama < 1)}"),
        ]:
            y = (df_comparison["inflation"] - df_comparison[col]).rolling(
                10, center=True).mean() - base
            plt.plot(df_comparison.index, y, color=color,
                     label=f"{lbl}: RMSE = {r:.3f}{s}")
        plt.axhline(0, color="black", linewidth=1, linestyle="--")
        plt.xlabel("Date")
        plt.legend(frameon=False)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{F}/cumloss_nowcasts_{target}_{cutoff}_new.pdf", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
