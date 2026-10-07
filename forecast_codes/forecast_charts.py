"""
forecast_charts.py  --  point-forecast figures of the paper
==========================================================

Terminal version of section A of JAE_revision_chart_refresh_new.ipynb, limited
to the figures of the main text. Reads the result files written by
new_codes.py, new_codes_llama70B.py and new_codes_sentiments.py and produces,
for CPI and core PCE:

  Figure 6  CSSED, h = 1 and h = 6          rev/cssed_{1,6}_{target}.jpg
  Figure 7  fluctuation test, h = 1         rev/fluctuation_test_1_{target}.jpg
Display names: the fine-tuned aggregate is "Aggr-RIS", the Llama-70B aggregate
"Aggr-LLaMA70B".

The plotting functions are the notebook's, unchanged; only the paths, the
display switch and the horizon list are set from the command line.

Usage
-----
    python forecast_charts.py                 # save the figures, show nothing
    python forecast_charts.py --show          # also open them
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib

# the backend must be chosen before pyplot is imported
if __name__ == "__main__" and "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent

# ============================================================
# CONFIGURATION (set by configure())
# ============================================================
RESULTS_DIR = ROOT / "results"
REV_DIR = ROOT / "figures_paper" / "rev"
FILE_SUFFIX = "_new"
TARGETS = ["CPIAUCSL", "PCEPILFE"]
TARGET_LABELS = {
    "CPIAUCSL": "CPI inflation",
    "PCEPILFE": "Core PCE inflation",
}

SHOW = False


def configure(results_dir=None, out_dir=None, tag=None, show=None):
    global RESULTS_DIR, REV_DIR, FILE_SUFFIX, SHOW
    if results_dir is not None:
        RESULTS_DIR = Path(results_dir)
    if out_dir is not None:
        REV_DIR = Path(out_dir) / "rev"
    if tag is not None:
        FILE_SUFFIX = tag
    if show is not None:
        SHOW = show
    REV_DIR.mkdir(parents=True, exist_ok=True)


def _maybe_show(path=None):
    if SHOW:
        plt.show()


# ============================================================
# IMPORTS + SHARED HELPERS
# ============================================================
# Same color-blind-friendly palette used for the point-forecast figures.
OKABE_ITO = {
    'black': '#000000',
    'orange': '#E69F00',
    'sky': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'verm': '#D55E00',
    'purple': '#CC79A7',
    'grey': '#999999',
}
def cb_color(name, default='#333333'):
    return OKABE_ITO.get(name, default)

def target_label(target):
    return TARGET_LABELS.get(target, target)

def ensure_series(x):
    return x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x

def collapse_duplicate_columns(df):
    if df.columns.is_unique:
        return df
    out = {}
    for col in pd.unique(df.columns):
        block = df.loc[:, df.columns == col]
        out[col] = block if isinstance(block, pd.Series) else block.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(out, index=df.index)

def read_csv_first_existing(paths, index_col=0, parse_dates=False):
    tried=[]
    for p in paths:
        p=Path(p); tried.append(str(p))
        if p.exists():
            df=pd.read_csv(p, index_col=index_col)
            if parse_dates:
                df.index=pd.to_datetime(df.index)
            return df,p
    raise FileNotFoundError('None of the candidate files exists:\n' + '\n'.join(tried))

def file_variants(path):
    path=str(path)
    if path.endswith(FILE_SUFFIX + '.csv'):
        return [path, path[:-(len(FILE_SUFFIX)+4)] + '.csv']
    if path.endswith('.csv'):
        return [path[:-4] + FILE_SUFFIX + '.csv', path]
    return [path + FILE_SUFFIX + '.csv', path + '.csv']

def dm_test(loss_model, loss_bench, h=1):
    m,b=ensure_series(loss_model).astype(float).align(ensure_series(loss_bench).astype(float), join='inner')
    d=(m-b).to_numpy(); d=d[~np.isnan(d)]; T=len(d)
    if T < 3: return np.nan,np.nan
    lag=max(int(h)-1,0); dm0=d.mean(); dc=d-dm0
    var=np.dot(dc,dc)/T
    for k in range(1,min(lag,T-1)+1):
        cov=np.dot(dc[:-k],dc[k:])/T
        w=1-k/(lag+1.0) if lag>0 else 0.0
        var += 2*w*cov
    if var <= 0: return np.nan,np.nan
    stat=dm0/np.sqrt(var/T)
    if h>1:
        stat /= np.sqrt((T+1-2*h+(h*(h-1))/T)/T)
    return stat, norm.cdf(stat)

def fluctuation_test(y1, y2, actual, m, P=None):
    y1=ensure_series(y1).astype(float); y2=ensure_series(y2).astype(float); actual=ensure_series(actual).astype(float)
    z=pd.concat([y1.rename('y1'), y2.rename('y2'), actual.rename('actual')], axis=1).dropna()
    L1=(z.actual-z.y1)**2; L2=(z.actual-z.y2)**2; dL=L2-L1
    arr=dL.to_numpy(); P=min(len(arr), P) if P is not None else len(arr); arr=arr[:P]
    if P <= m: return np.array([]), dL.iloc[:0]
    dvar=np.sqrt(np.sum((arr-arr.mean())**2)/P)
    if dvar==0: return np.zeros(P-int(m/2)*2+1), dL
    F=np.zeros(P-int(m/2)*2+1)
    for j in range(int(m/2), P-int(m/2)+1):
        F[j-int(m/2)] = (1/(dvar*np.sqrt(m))) * np.sum(arr[j-int(m/2):j+int(m/2)-1])
    return F,dL

def rmse_of_errors(x):
    a=pd.to_numeric(ensure_series(x), errors='coerce').to_numpy(dtype=float)
    return float(np.sqrt(np.nanmean(a*a)))


# ============================================================
# FILE PATHS + LOADERS
# ============================================================
def loss_path(h,t): return RESULTS_DIR / f'loss_{h}_{t}{FILE_SUFFIX}.csv'
def loss_sent_path(h,t): return RESULTS_DIR / f'loss_sentiment_{h}_{t}{FILE_SUFFIX}.csv'
def loss_llama_path(h,t): return RESULTS_DIR / f'loss_{h}_{t}_llama70{FILE_SUFFIX}.csv'
def pred_llm_path(h,t): return RESULTS_DIR / 'llm' / f'df_tuning_{h}_{t}{FILE_SUFFIX}.csv'
def pred_llama_path(h,t): return RESULTS_DIR / 'llm' / f'df_tuning_{h}_{t}_llama70{FILE_SUFFIX}.csv'
def pred_ar_path(h,t): return RESULTS_DIR / 'sentiment' / f'df_sentiment_{h}_{t}{FILE_SUFFIX}.csv'
def pred_swap_path(h,t): return RESULTS_DIR / 'swap' / f'df_tuning_swap_{h}_{t}{FILE_SUFFIX}.csv'
def pred_exp_path(h,t): return RESULTS_DIR / 'expectation' / f'df_tuning_exp_{h}_{t}{FILE_SUFFIX}.csv'
def true_path(h,t): return RESULTS_DIR / f'df_true_{h}_{t}{FILE_SUFFIX}.csv'

def load_loss_bundle(h,target):
    a,_=read_csv_first_existing(file_variants(loss_path(h,target)), parse_dates=True)
    s,_=read_csv_first_existing(file_variants(loss_sent_path(h,target)), parse_dates=True)
    l,_=read_csv_first_existing(file_variants(loss_llama_path(h,target)), parse_dates=True)
    if 'aggregated_pred' in l.columns:
        l=l.rename(columns={'aggregated_pred':'aggregated_pred_llama'})
    merged=pd.concat([a, s.drop(columns=['pred_ar'],errors='ignore'), l.drop(columns=['pred_ar'],errors='ignore')], axis=1)
    return collapse_duplicate_columns(merged),a,s,l

def load_prediction_bundle(h,target):
    specs={
        'df_results':pred_llm_path(h,target), 'df_results_llama':pred_llama_path(h,target),
        'df_results_ar':pred_ar_path(h,target), 'df_results_swap':pred_swap_path(h,target),
        'df_results_expect':pred_exp_path(h,target), 'df_true':true_path(h,target),
    }
    out={}
    for k,p in specs.items():
        d,_=read_csv_first_existing(file_variants(p))
        d.index=pd.to_datetime(d.index); out[k]=d
    return out


# ============================================================
# A. CSSED + FLUCTUATION-TEST CHARTS
# ============================================================
def plot_cssed_single(loss_merged,target,h,out_path,sample_end=None):
    # IMPORTANT: loss_* files already contain squared forecast losses.
    # CSSED is therefore cumulative (benchmark loss - competing-model loss); do NOT square again.
    d=loss_merged.copy()
    if sample_end is not None:
        d=d.loc[d.index <= pd.Timestamp(sample_end)]
    plt.figure(figsize=(8,4))
    plt.plot(d.index,(d['pred_ar']-d['expect_30']).cumsum(),lw=2,color=cb_color('sky'),label='Michigan expectations')
    plt.plot(d.index,(d['pred_ar']-d['swap_30']).cumsum(),lw=2,color=cb_color('verm'),label='Inflation-swap')
    plt.plot(d.index,(d['pred_ar']-d['aggregated_pred']).cumsum(),lw=2,color=cb_color('green'),label='Aggr-RIS')
    if 'aggregated_pred_llama' in d:
        plt.plot(d.index,(d['pred_ar']-d['aggregated_pred_llama']).cumsum(),lw=2,color=cb_color('orange'),label='Aggr-LLaMA70B')
    plt.axhline(0,color='black',lw=1,ls='--')
    suffix=' (pre-Covid sample)' if sample_end is not None else ''
    plt.title(f'CSSED for {target_label(target)} - h = {h}{suffix}',fontsize=13)
    plt.xlabel('Date'); plt.legend(frameon=False,fontsize=8); plt.grid(True,ls='--',alpha=.5)
    plt.tight_layout(); plt.savefig(out_path,dpi=300,bbox_inches='tight'); _maybe_show(out_path); plt.close()

def plot_fluctuation_single(bundle,target,h,out_path,m=10,sample_end=None):
    b={k:v.copy() for k,v in bundle.items()}
    if sample_end is not None:
        cutoff=pd.Timestamp(sample_end)
        for k in b: b[k]=b[k].loc[b[k].index<=cutoff]
    actual=ensure_series(b['df_true'].iloc[:,0]); y_ar=ensure_series(b['df_results_ar']['pred_ar']); P=len(actual)
    swap_col='aggregated_pred_swap' if 'aggregated_pred_swap' in b['df_results_swap'] else 'swap_30'
    exp_col='aggregated_pred_exp' if 'aggregated_pred_exp' in b['df_results_expect'] else 'expect_30'
    F_swap,_=fluctuation_test(b['df_results_swap'][swap_col],y_ar,actual,m,P)
    F_exp,_=fluctuation_test(b['df_results_expect'][exp_col],y_ar,actual,m,P)
    F_ris,_=fluctuation_test(b['df_results']['aggregated_pred'],y_ar,actual,m,P)
    F_llama,_=fluctuation_test(b['df_results_llama']['aggregated_pred'],y_ar,actual,m,P)
    n=min(len(F_swap),len(F_exp),len(F_ris),len(F_llama)); start=2*(m//2)-1
    x=pd.to_datetime(actual.index[start:start+n])
    plt.figure(figsize=(8,4))
    plt.plot(x,F_swap[:n],lw=2,color=cb_color('verm'),label='F-statistics, Inflation-swap')
    plt.plot(x,F_exp[:n],lw=2,color=cb_color('sky'),label='F-statistics, Michigan expectations')
    plt.plot(x,F_ris[:n],lw=2,color=cb_color('green'),label='F-statistics, Aggr-RIS')
    plt.plot(x,F_llama[:n],lw=2,color=cb_color('orange'),label='F-statistics, Aggr-LLaMA70B')
    plt.axhline(3.176,color='black',lw=1.8,ls='--',alpha=.8,label='critical value')
    plt.axhline(0,color=cb_color('grey'),lw=1.5,ls='--')
    suffix=' (pre-Covid sample)' if sample_end is not None else ''
    plt.title(f'Fluctuation Test - h={h} - Target = {target_label(target)}{suffix}',fontsize=13)
    plt.xlabel('Date'); plt.ylabel('F-statistic'); plt.legend(frameon=False,fontsize=8); plt.grid(True,ls='--',alpha=.4)
    plt.tight_layout(); plt.savefig(out_path,dpi=300,bbox_inches='tight'); _maybe_show(out_path); plt.close()

CSSED_HORIZONS = [1, 6]        # Figure 6
FLUCTUATION_HORIZONS = [1]     # Figure 7


def run_main_text(targets=None):
    targets = targets or TARGETS
    rows = []
    for target in targets:
        for h in sorted(set(CSSED_HORIZONS) | set(FLUCTUATION_HORIZONS)):
            try:
                if h in CSSED_HORIZONS:
                    loss, *_ = load_loss_bundle(h, target)
                    plot_cssed_single(loss, target, h, REV_DIR / f"cssed_{h}_{target}.jpg")
                if h in FLUCTUATION_HORIZONS:
                    pred = load_prediction_bundle(h, target)
                    plot_fluctuation_single(pred, target, h,
                                            REV_DIR / f"fluctuation_test_{h}_{target}.jpg")
                rows.append({"target": target, "h": h, "status": "ok"})
            except Exception as e:
                warnings.warn(f"{target}, h={h}: {e}")
                rows.append({"target": target, "h": h, "status": f"failed: {e}"})
    return pd.DataFrame(rows)


# ============================================================
def run(results_dir=None, out_dir=None, tag=None, show=None):
    configure(results_dir, out_dir, tag, show)
    return run_main_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "figures_paper"))
    ap.add_argument("--tag", default="_new")
    ap.add_argument("--show", action="store_true", help="also open the figures")
    args = ap.parse_args()
    a = run(args.results, args.out, args.tag, args.show)
    print(a.to_string(index=False))
    failed = (a.status != "ok").sum()
    print(f"\n{'all ok' if failed == 0 else f'{failed} FAILED'} -> {REV_DIR}")


if __name__ == "__main__":
    main()
