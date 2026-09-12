"""
check_results_forecast.py  --  evaluation of the multi-horizon forecasts
========================================================================

Terminal version of forecast_codes/check_results_forecast.ipynb.

Reads what new_codes.py, new_codes_llama70B.py and new_codes_sentiments.py
write in results/ and reports, for every horizon:

  * the RMSE ratio against the AR benchmark, with a Diebold-Mariano test, for
    the four combined forecasts (Reddit-LLM, Llama-70B, inflation swap,
    Michigan expectations);
  * the best individual models by RMSE ratio, and the best sentiment models;
  * the best models by relative MAD and by relative MAE (--mad / --mae);
  * for h = 1, the cumulative-loss and fluctuation-test charts.

It also merges the three loss files into loss_jae_{h}_{target}.xlsx (--export).

This script uses no Reddit data; it only reads the result files.

Usage
-----
    python check_results_forecast.py --target CPIAUCSL
    python check_results_forecast.py --target PCEPILFE --mad --mae
    python check_results_forecast.py --no-figures --export
"""

import argparse
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd

from _common import dm_test, ensure_dirs, fluctuation_test, star

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

HORIZONS = [1, 2, 3, 4, 5, 6, 9, 12, 18]


def top_k_rel_mad(df_preds, y_true, k=2, drop_cols=None):
    """Top-k columns by MAD (mean absolute deviation of forecast errors)."""
    if drop_cols:
        df_preds = df_preds.drop(columns=drop_cols, errors="ignore")
    errors = df_preds.sub(y_true, axis=0)
    mad = (errors.sub(errors.mean(), axis=1)).abs().mean()
    return mad.sort_values().head(k)


def top_k_rel_mae(df_preds, y_true, k=2, drop_cols=None):
    """Top-k columns by MAE."""
    if drop_cols:
        df_preds = df_preds.drop(columns=drop_cols, errors="ignore")
    mae = (df_preds.sub(y_true, axis=0)).abs().mean()
    return mae.sort_values().head(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="CPIAUCSL", choices=["PCEPILFE", "CPIAUCSL"])
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--tag", default="_new", help="suffix of the result files")
    ap.add_argument("--export", action="store_true",
                    help="write the merged loss_jae_*.xlsx files")
    ap.add_argument("--mad", action="store_true", help="also report relative MAD")
    ap.add_argument("--mae", action="store_true", help="also report relative MAE")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    target, tag, R = args.target, args.tag, args.results

    if args.no_figures:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dirs(args.fig_dir)

    for horizon in args.horizons:
        h_step = horizon

        df_results = pd.read_csv(
            f"{R}/llm/df_tuning_{horizon}_{target}{tag}.csv", index_col=0)
        df_results_llama = pd.read_csv(
            f"{R}/llm/df_tuning_{horizon}_{target}_llama70{tag}.csv", index_col=0)
        df_results_ar = pd.read_csv(
            f"{R}/sentiment/df_sentiment_{horizon}_{target}{tag}.csv", index_col=0)
        df_results_swap = pd.read_csv(
            f"{R}/swap/df_tuning_swap_{horizon}_{target}{tag}.csv", index_col=0)
        df_results_expect = pd.read_csv(
            f"{R}/expectation/df_tuning_exp_{horizon}_{target}{tag}.csv", index_col=0)
        df_true = pd.read_csv(
            f"{R}/df_true_{horizon}_{target}{tag}.csv", index_col=0)

        loss_df = pd.read_csv(f"{R}/loss_{horizon}_{target}{tag}.csv", index_col=0)
        loss_sent = pd.read_csv(
            f"{R}/loss_sentiment_{horizon}_{target}{tag}.csv", index_col=0)
        loss_llama = pd.read_csv(
            f"{R}/loss_{horizon}_{target}_llama70{tag}.csv", index_col=0)
        loss_df.index = pd.to_datetime(loss_df.index)
        loss_sent.index = pd.to_datetime(loss_sent.index)
        loss_llama.index = pd.to_datetime(loss_llama.index)
        loss_llama.rename(columns={"aggregated_pred": "aggregated_pred_llama"},
                          inplace=True)

        if args.export:
            merged = pd.concat(
                [loss_df, loss_llama.filter(like="llama"), loss_sent], axis=1)
            merged.to_excel(f"{R}/loss_jae_{horizon}_{target}{tag}.xlsx")

        loss_df = pd.concat([loss_df, loss_llama.filter(like="llama")], axis=1)

        # ---------------- figures ----------------
        # The notebook drew these only for h = 1 (`if horizon == 1 and plot == 1`),
        # but the paper uses fluctuation_test_{h}_{target}.jpg for every horizon,
        # so they are produced for each horizon in --horizons.
        if not args.no_figures:
            F = args.fig_dir

            plt.figure(figsize=(8, 4))
            plt.plot(loss_df.index,
                     (loss_df["expect_30"].cumsum() - loss_df["pred_ar"]),
                     linestyle=":", linewidth=2, label="Michigan expectations")
            plt.plot(loss_df.index,
                     (loss_df["swap_30"].cumsum() - loss_df["pred_ar"]),
                     linestyle="-", linewidth=2, label="Inflation-swap")
            plt.plot(loss_df.index,
                     (loss_df["aggregated_pred"] - loss_df["pred_ar"]).cumsum(),
                     marker="*", linewidth=2, markersize=8, label="Reddit-LLM")
            plt.plot(loss_df.index,
                     (loss_df["aggregated_pred_llama"] - loss_df["pred_ar"]).cumsum(),
                     marker="*", linewidth=2, markersize=8, label="Reddit-llama70B")
            plt.axhline(0, color="black", linewidth=1, linestyle="--")
            plt.title(f"Cumulative loss over time for {target}. h = {horizon}",
                      fontsize=14)
            plt.xlabel("Date", fontsize=12)
            plt.legend(frameon=False, loc="best", fontsize=8)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{F}/cumloss_forecast_{horizon}_{target}.jpg", dpi=300)
            plt.close()

            m = 10
            P = df_results_ar["pred_ar"].shape[0]
            actual = df_true["inflation"]
            y1 = df_results_ar["pred_ar"]

            F_stat_llm, _ = fluctuation_test(
                df_results["aggregated_pred"], y1, actual, m, P)
            F_stat_llama, _ = fluctuation_test(
                df_results_llama["aggregated_pred"], y1, actual, m, P)
            F_stat_swap, _ = fluctuation_test(
                df_results_swap["aggregated_pred_swap"], y1, actual, m, P)
            F_stat_exp, _ = fluctuation_test(
                df_results_expect["aggregated_pred_exp"], y1, actual, m, P)

            x = pd.to_datetime(actual.index[m - 1:])
            plt.figure(figsize=(8, 4))
            plt.plot(x, F_stat_swap, marker=".", linestyle=":",
                     label="F-statistics, Inflation-swap")
            plt.plot(x, F_stat_exp, marker=".", linestyle="-",
                     label="F-statistics, Michigan expectations")
            plt.plot(x, F_stat_llm, marker=".", linestyle=":",
                     label="F-statistics, Reddit-LLM")
            plt.plot(x, F_stat_llama, marker=".", linestyle=":",
                     label="F-statistics, Llama70B")
            plt.axhline(y=3.176, linestyle="--", color="red", alpha=0.7,
                        label="critical value")
            plt.xlabel("Date", fontsize=14)
            plt.ylabel("F-statistic", fontsize=14)
            plt.title(f"{h_step}-month-ahead Fluctuation Test. Target = {target[:3]}",
                      fontsize=14)
            plt.legend(frameon=False, loc="best", fontsize=8)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{F}/fluctuation_test_{horizon}_{target}.jpg", dpi=300)
            plt.close()

        # ---------------- tables ----------------
        print(f"**********h = {horizon}, {target} **********")

        bench = loss_df["pred_ar"]
        for model in ["aggregated_pred", "aggregated_pred_swap",
                      "aggregated_pred_exp", "aggregated_pred_llama"]:
            lm = loss_df[model].astype(float)
            lm_a, bench_a = lm.align(bench, join="inner")
            mse_ratio = np.sqrt(lm_a.mean()) / np.sqrt(bench_a.mean())
            dm_stat, p = dm_test(lm, bench, h=h_step)
            better = lm_a.mean() < bench_a.mean()
            print(f"{model}: MSE_ratio={mse_ratio:.3f}  DM={dm_stat:.3f}  "
                  f"p={p:.3f} {star(p, better)} ")

        print("\n")

        rmse_df = {m_: np.sqrt(loss_df[m_].astype(float).mean())
                   for m_ in loss_df.columns}
        for model in sorted(rmse_df, key=rmse_df.get)[:25]:
            lm = loss_df[model].astype(float)
            lm_a, bench_a = lm.align(bench, join="inner")
            mse_ratio = np.sqrt(lm_a.mean()) / np.sqrt(bench_a.mean())
            dm_stat, p = dm_test(lm, bench, h=h_step)
            better = lm_a.mean() < bench_a.mean()
            print(f"{model}: MSE_ratio={mse_ratio:.3f}  DM={dm_stat:.3f}  "
                  f"p={p:.3f} {star(p, better)} ")

        print("\n")

        rmse_sent = {m_: np.sqrt(loss_sent[m_].astype(float).mean())
                     for m_ in loss_sent.columns}
        for model in sorted(rmse_sent, key=rmse_sent.get)[:7]:
            lm = loss_sent[model].astype(float)
            lm_a, bench_a = lm.align(bench, join="inner")
            mse_ratio = np.sqrt(lm_a.mean()) / np.sqrt(bench_a.mean())
            dm_stat, p = dm_test(lm, bench, h=h_step)
            better = lm_a.mean() < bench_a.mean()
            print(f"{model}: MSE_ratio={mse_ratio:.3f}  DM={dm_stat:.3f}  "
                  f"p={p:.3f} {star(p, better)} ")

        print("\n")

        # ---------------- relative MAD / MAE ----------------
        y_true = df_true.iloc[:, 0]

        if args.mad:
            err_bench = df_results_ar["pred_ar"] - y_true
            mad_bench = (err_bench - err_bench.mean()).abs().mean()
            print(f"\n========== Horizon {horizon} (Rel. MAD vs pred_ar) ==========")
            print(f"Benchmark MAD (pred_ar): {mad_bench:.6f}\n")
            for label, df_, k, drop in [
                ("SWAP (best)", df_results_swap, 1, None),
                ("EXPECT (best)", df_results_expect, 1, None),
                ("LLAMA70 (best, 2nd)", df_results_llama, 30, None),
                ("LLM (best, 2nd)", df_results, 30, None),
                ("SENTIMENT (best, 2nd) [excluding pred_ar]", df_results_ar, 2,
                 ["pred_ar"]),
            ]:
                best = top_k_rel_mad(df_, y_true, k=k, drop_cols=drop) / mad_bench
                print(f"{label}:")
                print(best.rename("Rel_MAD").to_frame(), "\n")

        if args.mae:
            mae_bench = (df_results_ar["pred_ar"] - y_true).abs().mean()
            print(f"\n========== Horizon {horizon} (Rel. MAE vs pred_ar) ==========")
            print(f"Benchmark MAE (pred_ar): {mae_bench:.6f}\n")
            for label, df_, k, drop in [
                ("SWAP (best)", df_results_swap, 1, None),
                ("EXPECT (best)", df_results_expect, 1, None),
                ("LLAMA70 (best, 2nd)", df_results_llama, 20, None),
                ("LLM (best, 2nd)", df_results, 30, None),
                ("SENTIMENT (best, 2nd) [excluding pred_ar]", df_results_ar, 2,
                 ["pred_ar"]),
            ]:
                best = top_k_rel_mae(df_, y_true, k=k, drop_cols=drop) / mae_bench
                print(f"{label}:")
                print(best.rename("Rel_MAE").to_frame(), "\n")

        print("\n")


if __name__ == "__main__":
    main()
