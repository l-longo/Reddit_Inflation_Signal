"""
build_daily_signals_realtime.py
===============================

RUN-ONCE script, companion to build_daily_signals.py, for the NOWCAST exercise.

The nowcast uses a different aggregation from the forecast one: instead of
dating a whole thread at the submission date (which backdates comments and was
flagged as look-ahead by Referee 1), every item enters at its own date and the
thread signal is an EXPANDING mean of the items available up to t, discretised
with the same threshold. This is the aggregation of cell 3 of
new_codes_real_time.ipynb and new_codes_real_time_llama70B.ipynb, copied
verbatim.

Output (no Reddit text in either file):

    data/daily_signals_rt_finetuned.csv   -> df_reg of new_codes_real_time.py
    data/daily_signals_rt_llama70b.csv    -> df_reg of new_codes_real_time_llama70B.py

Usage:
    python build_daily_signals_realtime.py --raw /path/to/labelled_data --out ../data
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

METHOD_FT = "xqdora_trend"
FILES_FT = ["medium_part1", "medium_part2", "llama_3b", "small", "bert", "gemma_27"]
SUBREDDITS_LLAMA = ["economics", "wallstreetbets", "economy"]
THRESHOLD = 0


def _disc(x, thr=THRESHOLD):
    return 1 if x > thr else (-1 if x < -thr else 0)


def _to_naive_index(x):
    """Coerce a Series/Index/array of timestamps to a tz-naive DatetimeIndex."""
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(x).values, utc=True, errors="coerce"))
    return idx.tz_convert("UTC").tz_localize(None)


# --------------------------------------------------------------------------
# 1. fine-tuned LLM signals, real-time dated
# --------------------------------------------------------------------------
def _realtime_thread_daily(df_sub_models, df_com_models, suffix):
    """Each item at its own date; thread signal = expanding mean; real-time dated."""
    signal_cols = [c for c in df_sub_models.columns if c not in ["id_sub", "id_com"]]

    sub = df_sub_models.copy()
    sub.index = _to_naive_index(sub.index)
    sub["date"] = sub.index

    com = df_com_models.copy()
    com.index = _to_naive_index(com.index)
    com["date"] = com.index

    both = pd.concat([sub, com], axis=0)
    both = both[both["date"].notna()].sort_values("date")

    per_thread = []
    for _id, g in both.groupby("id_sub"):
        g = g.sort_values("date")
        running = g[signal_cols].astype(float).expanding().mean()  # real-time mean
        disc = running.applymap(_disc)                             # UP/DOWN/NEUTRAL
        disc.index = pd.to_datetime(g["date"].values)
        per_thread.append(disc.resample("D").last())               # realised label / day

    if not per_thread:
        out = pd.DataFrame()
    else:
        daily = pd.concat(per_thread, axis=0)
        out = daily.groupby(daily.index).sum()                     # sum threads by day

    out.columns = [c + suffix for c in out.columns]
    return out


def build_finetuned_rt(path):
    df_all = {}

    for name in FILES_FT:
        print(f"  [rt-ft] {name}", flush=True)
        df1 = pd.read_csv(f"{path}all_final_jae_{name}.csv", index_col=0, low_memory=False)
        df1.index = _to_naive_index(df1.index)

        df1_economics = df1[df1.reddit == "Economics"]
        if name == "bert":
            cols = df1_economics.filter(like="trend").columns.to_list()
        else:
            cols = df1_economics.filter(like=METHOD_FT).columns.to_list()
        cols += ["id_sub", "id_com"]
        cols = [c for c in cols if c in df1_economics.columns]

        df1_economics_models = df1_economics[cols]
        df1_economy_models = df1[df1.reddit == "economy"][cols]
        df1_wsb_models = df1[df1.reddit == "wallstreetbets"][cols]

        # only the columns the aggregation touches are read: the comment files
        # are ~240 MB each and reading them whole exhausts memory on most
        # machines. This changes nothing in the output.
        keep = sorted(set(cols) | {"reddit", "created_utc_com"})
        df1_comments = pd.read_csv(
            f"{path}all_comments_final_jae_{name}.csv",
            low_memory=False,
            usecols=keep,
        )
        df1_comments.index = _to_naive_index(df1_comments["created_utc_com"])

        df1_comments_economics_models = df1_comments[df1_comments.reddit == "Economics"][cols]
        df1_comments_economy_models = df1_comments[df1_comments.reddit == "economy"][cols]
        df1_comments_wsb_models = df1_comments[df1_comments.reddit == "wallstreetbets"][cols]

        df1_signals_economics = _realtime_thread_daily(
            df1_economics_models, df1_comments_economics_models, "_economics"
        )
        df1_signals_economy = _realtime_thread_daily(
            df1_economy_models, df1_comments_economy_models, "_economy"
        )
        df1_signals_wsb = _realtime_thread_daily(
            df1_wsb_models, df1_comments_wsb_models, "_wsb"
        )

        df_all[name] = pd.concat(
            [df1_signals_economics, df1_signals_economy, df1_signals_wsb], axis=1
        )

        del df1, df1_comments

    df_reg = pd.concat([df_all[n] for n in FILES_FT], axis=1)
    df_reg.index = df_reg.index.tz_localize(None)
    df_reg = df_reg.rename(columns=lambda c: re.sub(r"\W|^(?=\d)", "_", c))
    return df_reg


# --------------------------------------------------------------------------
# 2. Llama-3.3-70B signals, real-time dated
# --------------------------------------------------------------------------
def build_llama70_rt(path_llama):
    df_all0 = {}

    for name in SUBREDDITS_LLAMA:
        print(f"  [rt-llama70] {name}", flush=True)
        df1 = pd.read_csv(
            f"{path_llama}df_{name}_submissions_inflation_signal_llama.csv", index_col=0
        )
        df1.index = pd.to_datetime(df1.index)
        df1.rename(
            columns={"signal_llama-3.3-70b-instruct": "llama70_economics"}, inplace=True
        )

        df1_comm = pd.read_csv(
            f"{path_llama}df_{name}_comments_inflation_signal_llama.csv", index_col=0
        )
        df1_comm.index = pd.to_datetime(
            df1_comm["created_utc_com"], utc=True, errors="raise", format="mixed"
        )
        df1_comm.rename(
            columns={"signal_llama-3.3-70b-instruct": "llama70_economics"}, inplace=True
        )

        df_signals = pd.concat([df1, df1_comm], axis=0)
        d = pd.to_datetime(df_signals.index)
        try:
            d = d.tz_convert(None)
        except (TypeError, AttributeError):
            pass
        df_signals["date"] = d.floor("D")
        df_signals = df_signals[["date", "id_sub", "llama70_economics"]]

        df_signals = df_signals.sort_values("date")
        per_thread = []
        for _id, g in df_signals.groupby("id_sub"):
            g = g.sort_values("date")
            running = g[["llama70_economics"]].astype(float).expanding().mean()
            disc = running.applymap(_disc)
            disc.index = pd.to_datetime(g["date"].values)
            per_thread.append(disc.resample("D").last())

        daily = pd.concat(per_thread, axis=0)
        sig = daily.groupby(daily.index).sum()
        sig.columns = sig.columns + f"_{name}"

        df_all0[name] = sig.resample("D").sum()

        del df1, df1_comm

    df_reg = pd.concat([df_all0[n] for n in SUBREDDITS_LLAMA], axis=1)
    try:
        df_reg.index = df_reg.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return df_reg


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="ft,llama")
    args = ap.parse_args()

    path = os.path.join(args.raw, "")
    path_llama = os.path.join(args.raw, "llama70B", "")
    os.makedirs(args.out, exist_ok=True)
    blocks = [b.strip() for b in args.blocks.split(",")]

    if "llama" in blocks:
        print("real-time llama-70B signals ...", flush=True)
        df = build_llama70_rt(path_llama)
        df.to_csv(os.path.join(args.out, "daily_signals_rt_llama70b.csv"))
        print("  ->", df.shape, df.index.min(), df.index.max(), flush=True)

    if "ft" in blocks:
        print("real-time fine-tuned signals ...", flush=True)
        df = build_finetuned_rt(path)
        df.to_csv(os.path.join(args.out, "daily_signals_rt_finetuned.csv"))
        print("  ->", df.shape, df.index.min(), df.index.max(), flush=True)

    print("done.")


if __name__ == "__main__":
    main()
