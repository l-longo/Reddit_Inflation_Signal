"""
build_daily_signals.py
======================

RUN-ONCE script. Turns the *raw* labelled Reddit files (which contain post
titles and comment bodies, and therefore CANNOT be redistributed) into the
daily, fully aggregated signal series that the replication scripts consume.

Output (all published, none of them contains any Reddit text):

    data/daily_signals_finetuned.csv   -> df_reg of new_codes.py
    data/daily_signals_llama70b.csv    -> df_reg of new_codes_llama70B.py
    data/daily_signals_sentiment.csv   -> df_reg of new_codes_sentiments.py

The aggregation logic below is copied verbatim from the original notebooks
(cells that were removed from the published scripts), so the CSVs written here
are bit-for-bit the `df_reg` objects the notebooks used to build in memory.

Usage:
    python build_daily_signals.py --raw /path/to/labelled_data --out ../data
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

METHOD_FT = "xqdora_trend"

FILES_FT = [
    "medium_part1",
    "medium_part2",
    "llama_3b",
    "small",
    "bert",
    "gemma_27",
]

# the sentiment block of the original notebook does NOT use gemma_27
FILES_SENT = [
    "medium_part1",
    "medium_part2",
    "llama_3b",
    "small",
    "bert",
]

SUBREDDITS_LLAMA = ["economics", "wallstreetbets", "economy"]


# --------------------------------------------------------------------------
# 1. fine-tuned LLM signals  (submissions + comments)
# --------------------------------------------------------------------------
def build_finetuned(path):
    df_all = {}

    for name in FILES_FT:
        print(f"  [ft] {name}", flush=True)
        df1 = pd.read_csv(f"{path}all_final_jae_{name}.csv", index_col=0)
        df1.index = pd.to_datetime(df1.index)
        df1_economics = df1[df1.reddit == "Economics"]
        df1_economy = df1[df1.reddit == "economy"]
        df1_wsb = df1[df1.reddit == "wallstreetbets"]
        if name == "bert":
            cols = df1_economics.filter(like="trend").columns.to_list()
        else:
            cols = df1_economics.filter(like=METHOD_FT).columns.to_list()
        cols += ["id_sub", "id_com"]
        cols = [c for c in cols if c in df1_economics.columns]
        df1_economics_models = df1_economics[cols]
        df1_economy_models = df1_economy[cols]
        df1_wsb_models = df1_wsb[cols]

        df1_comments = pd.read_csv(f"{path}all_comments_final_jae_{name}.csv")
        df1_comments.index = pd.to_datetime(
            df1_comments["created_utc_com"], utc=True, errors="raise", format="mixed"
        )
        df1_comments_economics = df1_comments[df1_comments.reddit == "Economics"]
        df1_comments_economics_models = df1_comments_economics[cols]
        df1_comments_economy = df1_comments[df1_comments.reddit == "economy"]
        df1_comments_economy_models = df1_comments_economy[cols]
        df1_comments_wsb = df1_comments[df1_comments.reddit == "wallstreetbets"]
        df1_comments_wsb_models = df1_comments_wsb[cols]

        df_signals_economics = pd.concat(
            [df1_economics_models, df1_comments_economics_models], axis=0
        )
        df_signals_economics["date"] = df_signals_economics.index

        df_signals_economy = pd.concat(
            [df1_economy_models, df1_comments_economy_models], axis=0
        )
        df_signals_economy["date"] = df_signals_economy.index

        df_signals_wsb = pd.concat([df1_wsb_models, df1_comments_wsb_models], axis=0)
        df_signals_wsb["date"] = df_signals_wsb.index

        threshold = 0
        out = {}
        for key, dfs in [
            ("economics", df_signals_economics),
            ("economy", df_signals_economy),
            ("wsb", df_signals_wsb),
        ]:
            result = dfs.groupby("id_sub", as_index=False).agg(
                {
                    **{"date": "first"},
                    **{c: "mean" for c in dfs.columns if c not in ["id_sub", "date"]},
                }
            )
            result.index = pd.to_datetime(result.date)
            result = result.sort_index()
            sig = result.iloc[:, 2:].copy()
            sig = sig.applymap(
                lambda x: 1 if x > threshold else -1 if x < -threshold else 0
            )
            sig.columns = sig.columns + f"_{key}"
            out[key] = sig

        df_all[name] = pd.concat(
            [
                out["economics"].resample("D").sum(),
                out["economy"].resample("D").sum(),
                out["wsb"].resample("D").sum(),
            ],
            axis=1,
        )

        del df1, df1_comments

    df_reg = pd.concat([df_all[n] for n in FILES_FT], axis=1)
    df_reg.index = df_reg.index.tz_localize(None)
    df_reg = df_reg.rename(columns=lambda c: re.sub(r"\W|^(?=\d)", "_", c))
    return df_reg


# --------------------------------------------------------------------------
# 2. Llama-3.3-70B signals  (submissions + comments)
# --------------------------------------------------------------------------
def build_llama70(path_llama):
    df_all0 = {}

    for name in SUBREDDITS_LLAMA:
        print(f"  [llama70] {name}", flush=True)
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
        df_signals["date"] = pd.to_datetime(df_signals.index)
        df_signals["date"] = (
            pd.to_datetime(df_signals["date"]).dt.tz_convert(None).dt.floor("D")
        )
        df_signals = df_signals[["date", "id_sub", "llama70_economics"]]

        threshold = 0
        result = df_signals.groupby("id_sub", as_index=False).agg(
            {
                **{"date": "first"},
                **{
                    c: "mean"
                    for c in df_signals.columns
                    if c not in ["id_sub", "date"]
                },
            }
        )
        result.index = pd.to_datetime(result.date)
        result = result.sort_index()
        sig = result.iloc[:, 2:].copy()
        sig = sig.applymap(lambda x: 1 if x > threshold else -1 if x < -threshold else 0)
        sig.columns = sig.columns + f"_{name}"

        df_all0[name] = sig.resample("D").sum()

        del df1, df1_comm

    df_reg = pd.concat([df_all0[n] for n in SUBREDDITS_LLAMA], axis=1)
    df_reg.index = df_reg.index.tz_localize(None)
    return df_reg


# --------------------------------------------------------------------------
# 3. dictionary / lexicon sentiment signals  (submission titles only)
# --------------------------------------------------------------------------
def build_sentiment(path):
    from textblob import TextBlob
    import nltk
    import pysentiment2 as ps

    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity

    def get_vader_sentiment(text):
        return sia.polarity_scores(text)["compound"]

    df_all = {}

    for name in FILES_SENT:
        print(f"  [sent] {name}", flush=True)
        df1 = pd.read_csv(f"{path}all_final_jae_{name}.csv", index_col=0)
        df1["sentiment"] = df1["title_sub"].ffill().apply(get_sentiment)
        df1["sentiment_vader"] = df1["title_sub"].ffill().apply(get_vader_sentiment)
        lm = ps.LM()
        df1["tokens_lm"] = df1["title_sub"].astype(str).apply(lm.tokenize)
        df1["sentiment_lm"] = df1["tokens_lm"].apply(lm.get_score)

        df1.index = pd.to_datetime(df1.index)
        df1_economics = df1[df1.reddit == "Economics"]
        df1_economy = df1[df1.reddit == "economy"]
        df1_wsb = df1[df1.reddit == "wallstreetbets"]
        if name == "bert":
            cols = df1_economics.filter(like="trend").columns.to_list()
        else:
            cols = df1_economics.filter(like=METHOD_FT).columns.to_list()

        cols_sub = cols.copy()
        cols_sub += ["sentiment", "sentiment_vader", "sentiment_lm"]
        cols_sub = [c for c in cols_sub if c in df1_economics.columns]

        df_signals_economics = df1_economics[cols_sub]
        df_signals_economy = df1_economy[cols_sub]
        df_signals_wsb = df1_wsb[cols_sub]

        df_signals_economics.columns = df_signals_economics.columns + "_economics"
        df_signals_economy.columns = df_signals_economy.columns + "_economy"
        df_signals_wsb.columns = df_signals_wsb.columns + "_wsb"

        # NB: the original notebook gives all three subreddits the same
        # '..._economics_count' name; kept as is so that column order / names
        # (duplicates included) match the published results exactly.
        df_signals_economics["sentiment_lm_economics_count"] = df_signals_economics[
            "sentiment_lm_economics"
        ].apply(lambda x: x["Positive"] - x["Negative"] if isinstance(x, dict) else None)
        df_signals_economics = df_signals_economics.drop(
            columns=["sentiment_lm_economics"]
        )

        df_signals_economy["sentiment_lm_economics_count"] = df_signals_economy[
            "sentiment_lm_economy"
        ].apply(lambda x: x["Positive"] - x["Negative"] if isinstance(x, dict) else None)
        df_signals_economy = df_signals_economy.drop(columns=["sentiment_lm_economy"])

        df_signals_wsb["sentiment_lm_economics_count"] = df_signals_wsb[
            "sentiment_lm_wsb"
        ].apply(lambda x: x["Positive"] - x["Negative"] if isinstance(x, dict) else None)
        df_signals_wsb = df_signals_wsb.drop(columns=["sentiment_lm_wsb"])

        df_all[name] = pd.concat(
            [
                df_signals_economics.resample("D").sum(),
                df_signals_economy.resample("D").sum(),
                df_signals_wsb.resample("D").sum(),
            ],
            axis=1,
        )

        del df1

    df_reg = pd.concat([df_all[n] for n in FILES_SENT], axis=1)
    df_reg.index = df_reg.index.tz_localize(None)
    df_reg = df_reg.rename(columns=lambda c: re.sub(r"\W|^(?=\d)", "_", c))
    df_reg = df_reg.filter(like="sentiment")
    return df_reg


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder holding the raw labelled_data")
    ap.add_argument("--out", required=True, help="folder where the CSVs are written")
    ap.add_argument(
        "--blocks",
        default="ft,llama,sent",
        help="comma separated subset of {ft,llama,sent}",
    )
    args = ap.parse_args()

    path = os.path.join(args.raw, "")
    path_llama = os.path.join(args.raw, "llama70B", "")
    os.makedirs(args.out, exist_ok=True)
    blocks = [b.strip() for b in args.blocks.split(",")]

    if "ft" in blocks:
        print("fine-tuned signals ...", flush=True)
        df = build_finetuned(path)
        df.to_csv(os.path.join(args.out, "daily_signals_finetuned.csv"))
        print("  ->", df.shape, df.index.min(), df.index.max(), flush=True)

    if "llama" in blocks:
        print("llama-70B signals ...", flush=True)
        df = build_llama70(path_llama)
        df.to_csv(os.path.join(args.out, "daily_signals_llama70b.csv"))
        print("  ->", df.shape, df.index.min(), df.index.max(), flush=True)

    if "sent" in blocks:
        print("lexicon sentiment signals ...", flush=True)
        df = build_sentiment(path)
        df.to_csv(os.path.join(args.out, "daily_signals_sentiment.csv"))
        print("  ->", df.shape, df.index.min(), df.index.max(), flush=True)

    print("done.")


if __name__ == "__main__":
    main()
