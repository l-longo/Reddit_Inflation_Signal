"""
new_codes_sentiments.py  --  multi-horizon forecasts from the lexicon sentiment signals
=======================================================================================

Terminal version of forecast_codes/new_codes_sentiments.ipynb.

The notebook scored TextBlob / VADER / Loughran-McDonald sentiment on the raw
submission titles. Those titles cannot be redistributed, so the scoring and the
daily aggregation are done once by _build/build_daily_signals.py and the result
is read here from data/daily_signals_sentiment.csv. Everything from the
recursive estimation onwards is unchanged.

Note: the notebook gives the Loughran-McDonald count column the same name
('sentiment_lm_economics_count') for all three subreddits. Those duplicate
names are preserved on purpose so that the predictions kept in df0_initial --
and hence the results -- match the published ones.

Usage
-----
    python new_codes_sentiments.py                    # PCE core, all horizons
    python new_codes_sentiments.py --target CPIAUCSL
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
    ap.add_argument("--fred-key", default=None)
    ap.add_argument("--tag", default="_new")
    args = ap.parse_args()

    target = args.target
    tag = args.tag

    ensure_dirs(args.out, os.path.join(args.out, "sentiment"))

    # ---------------- data ----------------
    df_reg = load_daily_signals(os.path.join(args.data, "daily_signals_sentiment.csv"))

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

        # the initial frame uses the raw daily signals, without moving average
        df_merged = monthly_merge(df_monthly, df_reg, None)

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

        col_pos = {c: i for i, c in enumerate(df0_initial.columns)}

        # no diff(12) here -- the sentiment block works on the levels
        df0_by_ma = {}
        for w in MA_seq:
            dfm = monthly_merge(df_monthly, df_reg, w)
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
        df_filtered = df_filtered.filter(like="sentiment")

        start = T_test
        end = df0_initial.shape[0] - 0
        df_filtered = df_filtered[start:end]

        loss = {}
        for k in range(df_filtered.shape[1]):
            loss[f"{df_filtered.columns[k]}"] = (
                df0_initial["inflation"].iloc[start:end] - df_filtered.iloc[:, k]
            ) ** 2

        loss_df = pd.DataFrame(loss)

        df_filtered["pred_ar"] = df0_initial.pred_ar[start:end]
        loss_df["pred_ar"] = (
            df0_initial["inflation"].iloc[start:end] - df_filtered["pred_ar"]
        ) ** 2

        bench = loss_df["pred_ar"]

        print(f"{horizon}")
        print("\n")
        for model in loss_df.columns:
            lm = loss_df[model].astype(float)
            lm_a, bench_a = lm.align(bench, join="inner")
            mse_ratio = lm_a.mean() / bench_a.mean()
            dm_stat, p = dm_test(lm, bench, h=h_step)
            better = lm_a.mean() < bench_a.mean()
            print(f"{model}: MSE_ratio={mse_ratio:.3f}  DM={dm_stat:.3f}  "
                  f"p={p:.3f} {star(p, better)}")

        print("\n")

        N_models = df0.shape[1] - 1 + len(MA_seq) * 3
        N_models_last = df0_initial.shape[1] - 1
        N_loss = len(MA_seq) * 3
        df_tuning = df0_initial.iloc[start:end, N_models:N_models_last]
        loss_tuning = loss_df.iloc[:, N_loss:]

        common_cols = df_tuning.columns.intersection(loss_tuning.columns)

        df_filtered, weights_llm = agg_with_discount(
            df_filtered, loss_tuning, common_cols, delta=0.95, eps=1e-8,
            colname="aggregated_pred_sent", burn_in=30, return_weights=True,
        )

        loss_df["aggregated_pred_sent"] = (
            df0_initial["inflation"].iloc[start:end]
            - df_filtered["aggregated_pred_sent"]
        ) ** 2

        for model in ["aggregated_pred_sent"]:
            lm = loss_df[model].astype(float)
            lm_a, bench_a = lm.align(bench, join="inner")
            mse_ratio = lm_a.mean() / bench_a.mean()
            dm_stat, p = dm_test(lm, bench, h=h_step)
            better = lm_a.mean() < bench_a.mean()
            print(f"{model}: MSE_ratio={mse_ratio:.3f}  DM={dm_stat:.3f}  "
                  f"p={p:.3f} {star(p, better)}")

        print("\n")

        O = args.out
        df_filtered.to_csv(f"{O}/sentiment/df_sentiment_{h_step}_{target}{tag}.csv")
        df0_initial["inflation"].iloc[start:end].to_csv(
            f"{O}/df_true_{h_step}_{target}{tag}.csv"
        )
        loss_df.to_csv(f"{O}/loss_sentiment_{h_step}_{target}{tag}.csv")


if __name__ == "__main__":
    main()
