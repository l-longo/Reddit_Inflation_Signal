"""
new_codes_real_time.py  --  real-time nowcasts from the fine-tuned LLM signals
==============================================================================

Terminal version of nowcast_codes/new_codes_real_time.ipynb.

Two things make this exercise real-time:

* the Reddit signal is aggregated so that every item enters at its own date and
  the thread signal is an expanding mean (no backdating of comments to the
  submission date). That aggregation is done once by
  _build/build_daily_signals_realtime.py; here we read its output,
  data/daily_signals_rt_finetuned.csv;
* the inflation target is taken from the ALFRED vintage file
  data/{target}_2.xlsx, so at each origin the regression only sees the numbers
  that had actually been published.

`cutoff` is the day of the month before which a daily observation is still
assigned to the previous month's bucket.

Usage
-----
    python new_codes_real_time.py                       # core PCE
    python new_codes_real_time.py --target CPIAUCSL
    python new_codes_real_time.py --cutoffs 22
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

# the shared helpers live next door, in forecast_codes/
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "forecast_codes"))

from _common import (
    MEND,
    agg_with_discount,
    dm_test,
    ensure_dirs,
    load_daily_signals,
    load_macro,
    load_macro_fred,
    star,
)

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

VINTAGE_SHEET = {
    "CPIAUCSL": "Vintages Starting 2013-01-16",
    "PCEPILFE": "Vintages Starting 2013-01-31",
}


def bucket_monthly(df_reg, ma_window, cutoff):
    """Rolling mean, then re-date each day to its month bucket and average.

    Days with day-of-month <= cutoff belong to the previous month end, the
    others to the current month end.
    """
    df_reddit_daily = df_reg.rolling(ma_window).mean()

    idx = df_reddit_daily.index
    cur_me = idx + pd.offsets.MonthEnd(0)
    prev_me = idx - pd.offsets.MonthEnd(1)
    bucket = np.where(idx.day <= cutoff, prev_me, cur_me)

    df_daily_new_idx = df_reddit_daily.copy()
    df_daily_new_idx["month_bucket"] = pd.to_datetime(bucket)
    df_daily_new_idx.index = pd.to_datetime(df_daily_new_idx.month_bucket)
    df_daily_new_idx = df_daily_new_idx.drop(columns="month_bucket")

    return df_daily_new_idx.resample(MEND).mean()


def build_parser(signals="finetuned"):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--signals",
        default=signals,
        choices=["finetuned", "llama70b"],
        help="which real-time daily signal file to use",
    )
    ap.add_argument("--target", default="PCEPILFE", choices=["PCEPILFE", "CPIAUCSL"])
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--nowcast-out", default=os.path.join(ROOT, "data_fed_nowcast"))
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 14, 22])
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--fred-key", default=None)
    ap.add_argument("--tag", default="_new")
    ap.add_argument("--save", action="store_true", help="write the result csv files")
    ap.add_argument(
        "--ma-init",
        type=int,
        default=90,
        help="initial ma_window for the stored signal levels; see the note in "
             "main(). Use 30 when resuming a run at a cutoff other than the first.",
    )
    return ap


def main(args=None):
    if args is None:
        args = build_parser().parse_args()

    target = args.target
    tag = args.tag
    # the llama run labels its output files with an extra '_llama'
    lbl = "_llama" if args.signals == "llama70b" else ""

    ensure_dirs(
        args.out,
        os.path.join(args.out, "llm"),
        os.path.join(args.out, "swap"),
        os.path.join(args.out, "expectation"),
        args.nowcast_out,
    )

    # ---------------- data ----------------
    df_reg = load_daily_signals(
        os.path.join(args.data, f"daily_signals_rt_{args.signals}.csv")
    )

    if args.fred_key:
        df_monthly0 = load_macro_fred(args.fred_key, target)
    else:
        df_monthly0 = load_macro(os.path.join(args.data, "df_jae_reddit.xlsx"), target)
    df_monthly0 = df_monthly0["2005":]

    df_real_time = pd.read_excel(
        os.path.join(args.data, f"{target}_2.xlsx"),
        sheet_name=VINTAGE_SHEET[target],
        index_col=0,
    )
    df_real_time.index = pd.to_datetime(df_real_time.index)
    df_real_time = df_real_time.resample(MEND).last()

    print(f"signals: {df_reg.shape[1]} series, "
          f"{df_reg.index.min():%Y-%m-%d} .. {df_reg.index.max():%Y-%m-%d}")
    print(f"target : {target}, {df_real_time.shape[1]} vintages\n")

    h_step = args.horizon
    MA_seq = [1, 2, 3, 4, 5, 10, 20, 30]

    # `ma_window` is deliberately initialised OUTSIDE the cutoff loop, as in the
    # notebook. It is rebound by the inner MA loop, so from the second cutoff on
    # the frame below is built with MA_seq[-1] rather than 90. This only affects
    # the raw signal levels stored in all_models_*.csv -- no regression uses
    # them -- but it is kept so the saved files match the published ones. Run
    # the default --cutoffs 5 10 14 22 in one go to reproduce them, or pass
    # --ma-init 30 when resuming at a later cutoff.
    ma_window = args.ma_init

    for cutoff in args.cutoffs:
        print(cutoff)

        df_merged = pd.concat(
            [df_monthly0.resample(MEND).last(), bucket_monthly(df_reg, ma_window, cutoff)],
            axis=1,
        )

        if h_step > 0:
            data_g = df_merged.shift(+h_step).copy()
            data_g.iloc[:, 0] = df_merged.iloc[:, 0].copy()
            df_merged_new = data_g.iloc[h_step:].copy()

        df0_initial = df_merged_new.copy()
        df0_initial = df0_initial["2009-01-01":]
        ins = 153
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

        # the bucketed monthly signals do not depend on the recursive origin
        sig_by_ma = {w: bucket_monthly(df_reg, w, cutoff) for w in MA_seq}

        for tr in range(df0_initial.shape[0] - T_test - 3):
            for ma_window in MA_seq:
                date_refer = df0_initial.index[T_test + tr]
                nxt = date_refer + pd.offsets.MonthEnd(1)

                df_monthly = df_monthly0.resample(MEND).last()

                # real-time vintage available at this origin
                new_df_real = df_real_time.iloc[:, tr]["2005":].dropna()[:date_refer]
                new_df_real_infl = np.log(new_df_real).diff(12)

                new_df_real1 = df_real_time.iloc[:, tr + 1]["2005":].dropna()
                new_df_real1_infl = np.log(new_df_real1).diff(12)

                df_monthly.loc[:date_refer, "inflation"] = new_df_real_infl
                df_monthly.loc[nxt, "inflation"] = new_df_real1_infl[nxt]
                df0_initial.loc[nxt, "inflation"] = df_monthly.loc[nxt, "inflation"]

                df_merged = pd.concat(
                    [df_monthly.resample(MEND).last(), sig_by_ma[ma_window]], axis=1
                )
                df_merged[df_reg.columns] = df_merged[df_reg.columns].diff(12)

                if h_step > 0:
                    data_g = df_merged.shift(+h_step).copy()
                    data_g.iloc[:, 0] = df_merged.iloc[:, 0].copy()
                    df_merged_new = data_g.iloc[h_step:].copy()

                df0 = df_merged_new.copy()
                df0["inflation_lag"] = df0.inflation.shift(h_step)
                df0 = df0["2009-01-01":].fillna(0)

                df_regression = df0.iloc[: T_test + tr]

                for var in range(2, df_regression.shape[1] - 1):
                    X = df_regression[["inflation_lag"]]
                    X["signal"] = df_regression.iloc[:, var]
                    X = sm.add_constant(X, has_constant="add")
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
            X = sm.add_constant(X, has_constant="add")
            y = df_regression["inflation"]
            model = sm.OLS(y, X).fit()

            df_pred = df0.iloc[T_test + tr :]
            X_pred = df_pred[["inflation_lag"]]
            X_pred = sm.add_constant(X_pred, has_constant="add")
            df0_initial.iloc[T_test + tr, col_pos["pred_ar"]] = model.predict(X_pred)[0]

        pattern = "|".join(str(ma) for ma in MA_seq)
        df_filtered = df0_initial.filter(regex=pattern)
        df_filtered["pred_ar"] = df0_initial.pred_ar

        start = T_test
        end = df0_initial.shape[0] - 1
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

        print(f"{h_step}")
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

        if args.save:
            O, N = args.out, args.nowcast_out
            weights_llm.to_csv(f"{O}/llm/weights_llm_{h_step}_{target}{lbl}_real{tag}.csv")
            weights_swap.to_csv(
                f"{O}/swap/weights_swap_{h_step}_{target}{lbl}_real{tag}.csv"
            )
            weights_exp.to_csv(
                f"{O}/expectation/weights_exp_{h_step}_{target}{lbl}_real{tag}.csv"
            )
            df_tuning.to_csv(f"{O}/llm/df_tuning_{h_step}_{target}{lbl}_real{tag}.csv")
            df_tuning_swap.to_csv(
                f"{O}/swap/df_tuning_swap_{h_step}_{target}{lbl}_real{tag}.csv"
            )
            df_tuning_exp.to_csv(
                f"{O}/expectation/df_tuning_exp_{h_step}_{target}{lbl}_real{tag}.csv"
            )
            loss_df.to_csv(f"{O}/loss_{h_step}_{target}{lbl}_real{tag}.csv")

            if not lbl:
                df0_initial.pred_ar.to_csv(f"{N}/pred_ar_{cutoff}_{target}{tag}.csv")
            df0_initial.to_csv(f"{N}/all_models_{cutoff}_{target}{lbl}{tag}.csv")
            df_tuning.to_csv(f"{N}/df_pred_nowcasts_{cutoff}_{target}{lbl}{tag}.csv")
        print("")


if __name__ == "__main__":
    main()
