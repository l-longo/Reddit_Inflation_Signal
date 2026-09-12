"""
new_codes.py  --  multi-horizon inflation forecasts from the fine-tuned LLM signals
===================================================================================

Terminal version of forecast_codes/new_codes.ipynb.

The notebook cells that read the raw labelled Reddit files (they carry post
titles and comment bodies and cannot be redistributed) are replaced by a single
read of data/daily_signals_finetuned.csv, which holds the very same daily
aggregated signals the notebook used to rebuild in memory. Everything from the
recursive estimation onwards is unchanged.

Usage
-----
    python new_codes.py                          # PCE core, all horizons
    python new_codes.py --target CPIAUCSL
    python new_codes.py --horizons 1 6 12
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from _common import (
    agg_with_discount,
    dm_test,
    ensure_dirs,
    load_daily_signals,
    load_macro,
    load_macro_fred,
    monthly_merge,
    star,
)

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="PCEPILFE", choices=["PCEPILFE", "CPIAUCSL"])
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument(
        "--horizons", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 9, 12, 18]
    )
    ap.add_argument("--fred-key", default=None, help="pull the macro block from FRED")
    ap.add_argument("--tag", default="_new", help="suffix appended to output files")
    args = ap.parse_args()

    target = args.target
    tag = args.tag

    ensure_dirs(
        args.out,
        os.path.join(args.out, "llm"),
        os.path.join(args.out, "swap"),
        os.path.join(args.out, "expectation"),
        os.path.join(ROOT, "data_fed_nowcast"),
    )

    # ---------------- data ----------------
    df_reg = load_daily_signals(os.path.join(args.data, "daily_signals_finetuned.csv"))

    if args.fred_key:
        df_monthly = load_macro_fred(args.fred_key, target)
    else:
        df_monthly = load_macro(os.path.join(args.data, "df_jae_reddit.xlsx"), target)

    print(f"signals: {df_reg.shape[1]} series, "
          f"{df_reg.index.min():%Y-%m-%d} .. {df_reg.index.max():%Y-%m-%d}")
    print(f"target : {target}\n")

    MA_seq = [1, 5, 10, 30, 90, 120, 180, 360]

    for horizon in args.horizons:
        h_step = horizon

        ma_window = 90
        df_merged = monthly_merge(df_monthly, df_reg, ma_window)

        if h_step > 0:
            data_g = df_merged.shift(+h_step).copy()
            data_g.iloc[:, 0] = df_merged.iloc[:, 0].copy()
            df_merged_new = data_g.iloc[h_step:].copy()

        df0_initial = df_merged_new.copy()

        df0_initial = df0_initial["2009-01-01":]
        ins = 152
        T_test = df0_initial.shape[0] - ins

        skip_cols = ["infl", "inflation"]
        target_cols = [c for c in df0_initial.columns if c not in skip_cols]

        new_cols = {
            f"{col}_{ma}": np.zeros(len(df0_initial))
            for col in target_cols
            for ma in MA_seq
        }
        df0_initial = df0_initial.assign(**new_cols)
        df0_initial["pred_ar"] = np.zeros(df0_initial.shape[0])

        # positional writes (the notebook used chained assignment, which newer
        # pandas silently drops under copy-on-write)
        col_pos = {c: i for i, c in enumerate(df0_initial.columns)}

        # The design matrices do not depend on the recursive origin, so they are
        # built once per moving-average window instead of inside the tr loop.
        # Numerically identical to the notebook, just much faster.
        df0_by_ma = {}
        for w in MA_seq:
            dfm = monthly_merge(df_monthly, df_reg, w)
            dfm[df_reg.columns] = dfm[df_reg.columns].diff(12)
            if h_step > 0:
                data_g = dfm.shift(+h_step).copy()
                data_g.iloc[:, 0] = dfm.iloc[:, 0].copy()
                dfm_new = data_g.iloc[h_step:].copy()
            d0 = dfm_new.copy()
            d0["inflation_lag"] = d0.inflation.shift(h_step + 1)
            df0_by_ma[w] = d0["2009-01-01":].fillna(0)

        for tr in range(df0_initial.shape[0] - T_test - 1):
            for ma_window in MA_seq:
                df0 = df0_by_ma[ma_window]
                df_regression = df0.iloc[: T_test + tr]

                for var in range(2, df_regression.shape[1] - 1):
                    X = df_regression[["inflation_lag"]]
                    X["signal"] = df_regression.iloc[:, var]
                    X = sm.add_constant(X)
                    y = df_regression["inflation"]
                    model = sm.OLS(y, X).fit()

                    df_pred = df0.iloc[T_test + tr :]
                    X_pred = df_pred[["inflation_lag"]]
                    X_pred["signal"] = df_pred.iloc[:, var]
                    X_pred = sm.add_constant(X_pred, has_constant="add")
                    df0_initial.iloc[
                        T_test + tr,
                        col_pos[f"{df_regression.columns[var]}_{ma_window}"],
                    ] = model.predict(X_pred)[0]

            X = df_regression[["inflation_lag"]]
            X = sm.add_constant(X)
            y = df_regression["inflation"]
            model = sm.OLS(y, X).fit()

            df_pred = df0.iloc[T_test + tr :]
            X_pred = df_pred[["inflation_lag"]]
            X_pred = sm.add_constant(X_pred)
            df0_initial.iloc[T_test + tr, col_pos["pred_ar"]] = model.predict(X_pred)[0]

        pattern = "|".join(str(ma) for ma in MA_seq)
        df_filtered = df0_initial.filter(regex=pattern)
        df_filtered["pred_ar"] = df0_initial.pred_ar

        start = T_test
        end = df0_initial.shape[0] - 0
        loss = {}
        for k in range(df_filtered.shape[1]):
            loss[f"{df_filtered.columns[k]}"] = (
                df0_initial["inflation"].iloc[start:end] - df_filtered.iloc[start:end, k]
            ) ** 2

        loss_df = pd.DataFrame(loss)

        N_models = df0.shape[1] - 1 + len(MA_seq) * 3
        N_models_last = df0_initial.shape[1] - 1
        N_loss = len(MA_seq) * 3
        df_tuning = df0_initial.iloc[start:end, N_models:N_models_last]
        loss_tuning = loss_df.iloc[:, N_loss:]

        df_tuning_swap = df0_initial.iloc[
            start:end, N_models - len(MA_seq) * 2 : N_models - len(MA_seq)
        ]
        df_tuning_exp = df0_initial.iloc[
            start:end, N_models - N_loss : N_models - len(MA_seq) * 2
        ]

        common_cols = df_tuning.columns.intersection(loss_tuning.columns)
        common_cols_swap = df_tuning_swap.columns.intersection(loss_df.columns)
        common_cols_exp = df_tuning_exp.columns.intersection(loss_df.columns)

        df_tuning, weights_llm = agg_with_discount(
            df_tuning, loss_tuning, common_cols, delta=0.95, eps=1e-8,
            colname="aggregated_pred", burn_in=30, return_weights=True,
        )
        df_tuning_swap, weights_swap = agg_with_discount(
            df_tuning_swap, loss_df, common_cols_swap, delta=0.95, eps=1e-8,
            colname="aggregated_pred_swap", burn_in=10, return_weights=True,
        )
        df_tuning_exp, weights_exp = agg_with_discount(
            df_tuning_exp, loss_df, common_cols_exp, delta=0.95, eps=1e-8,
            colname="aggregated_pred_exp", burn_in=10, return_weights=True,
        )

        loss_df["aggregated_pred"] = (
            df0_initial["inflation"].iloc[start:end] - df_tuning["aggregated_pred"]
        ) ** 2
        loss_df["aggregated_pred_swap"] = (
            df0_initial["inflation"].iloc[start:end]
            - df_tuning_swap["aggregated_pred_swap"]
        ) ** 2
        loss_df["aggregated_pred_exp"] = (
            df0_initial["inflation"].iloc[start:end]
            - df_tuning_exp["aggregated_pred_exp"]
        ) ** 2

        bench = loss_df["pred_ar"]

        print(f"{horizon}")
        print("\n")

        for model in ["aggregated_pred", "aggregated_pred_swap", "aggregated_pred_exp"]:
            lm = loss_df[model].astype(float)
            lm_a, bench_a = lm.align(bench, join="inner")
            mse_ratio = lm_a.mean() / bench_a.mean()
            dm_stat, p = dm_test(lm, bench, h=h_step)
            better = lm_a.mean() < bench_a.mean()
            print(f"{model}: MSE_ratio={mse_ratio:.3f}  DM={dm_stat:.3f}  "
                  f"p={p:.3f} {star(p, better)}")

        print("\n")

        for model in loss_df.columns:
            lm = loss_df[model].astype(float)
            lm_a, bench_a = lm.align(bench, join="inner")
            mse_ratio = lm_a.mean() / bench_a.mean()
            dm_stat, p = dm_test(lm, bench, h=h_step)
            better = lm_a.mean() < bench_a.mean()
            if mse_ratio < 1:
                print(f"{model}: MSE_ratio={mse_ratio:.3f}  DM={dm_stat:.3f}  "
                      f"p={p:.3f} {star(p, better)} ")

        O = args.out
        weights_llm.to_csv(f"{O}/llm/weights_llm_{h_step}_{target}{tag}.csv")
        weights_swap.to_csv(f"{O}/swap/weights_swap_{h_step}_{target}{tag}.csv")
        weights_exp.to_csv(f"{O}/expectation/weights_exp_{h_step}_{target}{tag}.csv")

        df_tuning.to_csv(f"{O}/llm/df_tuning_{h_step}_{target}{tag}.csv")
        df_tuning_swap.to_csv(f"{O}/swap/df_tuning_swap_{h_step}_{target}{tag}.csv")
        df_tuning_exp.to_csv(f"{O}/expectation/df_tuning_exp_{h_step}_{target}{tag}.csv")

        loss_df.to_csv(f"{O}/loss_{h_step}_{target}{tag}.csv")

        df0_initial.to_csv(
            os.path.join(ROOT, "data_fed_nowcast", f"aaa_all_models{tag}.csv")
        )
        print("")


if __name__ == "__main__":
    main()
