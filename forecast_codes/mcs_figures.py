"""
mcs_figures.py  --  Model Confidence Set summaries and figures
=============================================================

Terminal version of build_MCS_summaries_and_heatmaps.ipynb (summary
workbooks) and of section C of JAE_revision_chart_refresh_MCS_audited_new.ipynb
(figures).

Input: the MCS result for each target and horizon, data/mcs/
test_mcs_1000_{h}_{target}_new.xlsx, one row per model left in the confidence
set (columns models, pvalues, status). B = 1,000 bootstrap samples.

Steps
  1. read the included models of every file;
  2. parse each model into family + subreddit + backward-looking MA window
     (the last numeric token of the name), stopping if any name cannot be
     parsed, so that no model is silently put in the wrong row;
  3. write the summary workbooks results/test_MCS/mcs-summary-{cpi,pce}-h1-h18.xlsx
     (Foglio1 = counts by family and subreddit, Foglio2 = full member lists),
     with Foglio1 derived mechanically from Foglio2;
  4. audit that the two sheets reconcile;
  5. draw the joint heatmap                       mcs/MCS_heatmap_CPI_PCE_h1_h18_new.png

Paper: Figure 8 (fig:mcs-cpi-pce-combined).
Display names: aggregated_pred -> Aggr-RIS, aggregated_pred_llama ->
Aggr-LLaMA70B, aggregated_pred_exp -> Mich. Exp., aggregated_pred_swap ->
1Y-Infl-Swap. aggregated_pred_sent is kept in Foglio2 for transparency but
excluded from Foglio1 and from the figures.

Usage
-----
    python mcs_figures.py
    python mcs_figures.py --show
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib

if __name__ == "__main__" and "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent

# ============================================================
# CONFIGURATION (set by configure())
# ============================================================
MCS_RESULTS_DIR = ROOT / "data" / "mcs"
SUMMARY_DIR = ROOT / "results" / "test_MCS"
FIG_DIR = ROOT / "figures_paper" / "mcs"

HORIZONS = [1, 2, 3, 4, 5, 6, 9, 12, 18]
TARGET_INFO = {
    'CPIAUCSL': {'dataset': 'CPI', 'slug': 'cpi'},
    'PCEPILFE': {'dataset': 'PCE', 'slug': 'pce'},
}
PREFER_NEW = True

SHOW = False


def configure(mcs_dir=None, summary_dir=None, out_dir=None, show=None):
    global MCS_RESULTS_DIR, SUMMARY_DIR, FIG_DIR, SHOW
    if mcs_dir is not None:
        MCS_RESULTS_DIR = Path(mcs_dir)
    if summary_dir is not None:
        SUMMARY_DIR = Path(summary_dir)
    if out_dir is not None:
        FIG_DIR = Path(out_dir) / "mcs"
    if show is not None:
        SHOW = show
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _maybe_show(path=None):
    if SHOW:
        plt.show()


# ============================================================
# 1. FIND AND READ ONE MCS RESULT FILE
# ============================================================
def find_mcs_file(target, horizon, base_dir=None):
    base_dir = Path(base_dir or MCS_RESULTS_DIR)
    stem = f'test_mcs_1000_{horizon}_{target}'
    if PREFER_NEW:
        names = [f'{stem}_new.csv', f'{stem}_new.xlsx', f'{stem}.csv', f'{stem}.xlsx']
    else:
        names = [f'{stem}.csv', f'{stem}.xlsx', f'{stem}_new.csv', f'{stem}_new.xlsx']
    for name in names:
        p = base_dir / name
        if p.exists():
            return p
    matches = []
    for ext in ('*.csv', '*.xlsx'):
        matches.extend(base_dir.glob(f'{stem}*{ext[1:]}'))
    matches = sorted(set(matches), key=lambda x: ('_new' not in x.stem, x.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f'No MCS file found for target={target}, h={horizon} in {base_dir}')


def read_mcs_result(path):
    """Read one result file and return the included models only."""
    suffix = path.suffix.lower()
    if suffix == '.csv':
        df = pd.read_csv(path)
    elif suffix in {'.xlsx', '.xls'}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f'Unsupported file type: {path}')
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {'models'} - set(df.columns)
    if missing:
        raise ValueError(f'{path.name}: missing required column(s): {sorted(missing)}')
    df['models'] = df['models'].astype('string').str.strip()
    df = df[df['models'].notna() & (df['models'] != '')].copy()
    if 'status' in df.columns:
        df['status'] = df['status'].astype('string').str.strip().str.lower()
        df = df[df['status'] == 'included'].copy()
    return df.reset_index(drop=True)


# ============================================================
# 2. MODEL-NAME PARSER: family + subreddit + backward-looking MA
# ============================================================
MODEL_ORDER = [
    'Sentiment', 'BERT base', 'FinBERT', 'InflaBERT',
    'LLaMA-1B', 'LLaMA-3B', 'LLaMA-8B',
    'Qwen-0.5B', 'Qwen-1.5B', 'Qwen-7B',
    'Gemma-2B', 'Gemma-9B', 'Gemma-27B',
    'LLaMA-70B',
    'Mich. Exp.', '1Y-Infl-Swap', 'Aggr-RIS', 'Aggr-LLaMA70B',
]
AGGREGATE_MODELS = ['Mich. Exp.', '1Y-Infl-Swap', 'Aggr-RIS', 'Aggr-LLaMA70B']
SUBREDDIT_COLS = ['r/Economics', 'r/economy', 'r/wallstreetbets']
EXCLUDED_FROM_FOGLIO1 = {'aggregated_pred_sent', 'aggregate_pred_sent'}


def normalize_model_name(name):
    return re.sub(r'\s+', ' ', str(name).strip()).lower()


def aggregate_family(name):
    x = normalize_model_name(name)
    if x in EXCLUDED_FROM_FOGLIO1:
        return 'EXCLUDED_SENTIMENT_AGGREGATE'
    if x in {'aggregated_pred_llama', 'aggregate_pred_llama'}:
        return 'Aggr-LLaMA70B'
    if x in {'aggregated_pred_exp', 'aggregate_pred_exp'}:
        return 'Mich. Exp.'
    if x in {'aggregated_pred_swap', 'aggregate_pred_swap', 'aggregated_swap', 'aggregate_swap'}:
        return '1Y-Infl-Swap'
    if x in {'aggregated_pred', 'aggregate_pred',
             'aggregated_pred (llm ensemble)', 'aggregate_pred (llm ensemble)'}:
        return 'Aggr-RIS'
    return None


def model_family(name):
    x = normalize_model_name(name)
    if x.startswith('sentiment_'):
        return 'Sentiment'
    if x.startswith('bert_base') or re.match(r'^bert(_|$)', x):
        return 'BERT base'
    if x.startswith('finbert'):
        return 'FinBERT'
    if x.startswith('inflabert'):
        return 'InflaBERT'
    if 'llama3_2_1b' in x or 'llama3_1b' in x:
        return 'LLaMA-1B'
    if 'llama3_2_3b' in x or 'llama3_3b' in x:
        return 'LLaMA-3B'
    if 'llama3_1_8b' in x or 'llama3_8b' in x:
        return 'LLaMA-8B'
    if 'qwen2_5_0_5b' in x or 'qwen_0_5' in x or 'qwen0.5' in x:
        return 'Qwen-0.5B'
    if 'qwen2_5_1_5b' in x or 'qwen_1_5' in x or 'qwen1.5' in x:
        return 'Qwen-1.5B'
    if 'qwen2_5_7b' in x or 'qwen_7b' in x or 'qwen7' in x:
        return 'Qwen-7B'
    if 'gemma2_2b' in x:
        return 'Gemma-2B'
    if 'gemma2_9b' in x:
        return 'Gemma-9B'
    if 'gemma2_27b' in x:
        return 'Gemma-27B'
    if x.startswith('llama70_') or 'llama70' in x:
        return 'LLaMA-70B'
    return None


def parse_subreddit(name):
    """Right-most subreddit token, e.g. llama70_economics_economy_90 -> r/economy."""
    tokens = normalize_model_name(name).replace('-', '_').split('_')
    mapping = {'economics': 'r/Economics', 'economy': 'r/economy',
               'wsb': 'r/wallstreetbets', 'wallstreetbets': 'r/wallstreetbets'}
    for tok in reversed(tokens):
        if tok in mapping:
            return mapping[tok]
    return None


def parse_ma_days(name):
    """The final numeric token is the backward-looking RIS smoothing MA."""
    tokens = normalize_model_name(name).replace('-', '_').split('_')
    if tokens and re.fullmatch(r'\d+', tokens[-1]):
        return int(tokens[-1])
    return None


def parse_model(name):
    agg = aggregate_family(name)
    if agg == 'EXCLUDED_SENTIMENT_AGGREGATE':
        return {'model': name, 'family': agg, 'subreddit': 'all', 'ma_days': None,
                'is_aggregate': True, 'include_foglio1': False}
    if agg is not None:
        return {'model': name, 'family': agg, 'subreddit': 'all', 'ma_days': None,
                'is_aggregate': True, 'include_foglio1': True}
    return {'model': name, 'family': model_family(name), 'subreddit': parse_subreddit(name),
            'ma_days': parse_ma_days(name), 'is_aggregate': False, 'include_foglio1': True}


def audit_inputs():
    all_parsed_rows, audit_rows = [], []
    for target in TARGET_INFO:
        for h in HORIZONS:
            try:
                p = find_mcs_file(target, h)
                raw = read_mcs_result(p)
            except Exception as exc:
                audit_rows.append({'target': target, 'horizon': h, 'status': 'FILE ERROR',
                                   'detail': str(exc)})
                continue
            parsed = pd.DataFrame([parse_model(x) for x in raw['models']])
            parsed['target'] = target
            parsed['horizon'] = h
            parsed['source_file'] = p.name
            all_parsed_rows.append(parsed)
            nonagg = parsed[~parsed['is_aggregate']]
            bad = (nonagg['family'].isna().sum() + nonagg['subreddit'].isna().sum()
                   + nonagg['ma_days'].isna().sum())
            audit_rows.append({
                'target': target, 'horizon': h, 'source_file': p.name,
                'raw_MCS_members': len(parsed),
                'excluded_sentiment_aggregate': int((~parsed['include_foglio1']).sum()),
                'Foglio1_members': int(parsed['include_foglio1'].sum()),
                'unparsed': int(bad), 'status': 'OK' if bad == 0 else 'CHECK',
            })
    parsed_all = pd.concat(all_parsed_rows, ignore_index=True) if all_parsed_rows else pd.DataFrame()
    audit_df = pd.DataFrame(audit_rows)
    problem = parsed_all[(~parsed_all['is_aggregate']) & (
        parsed_all['family'].isna() | parsed_all['subreddit'].isna() | parsed_all['ma_days'].isna())]
    if len(problem):
        print(problem[['target', 'horizon', 'model', 'family', 'subreddit', 'ma_days']].to_string())
    assert problem.empty, 'Fix parser rules before continuing: some model names are not fully parsed.'
    return audit_df


# ============================================================
# 3. BUILD FOGLIO1 + FOGLIO2
# ============================================================
BLOCK_HEADERS = ['r/Economics', 'r/economy', 'r/wallstreetbets', 'all', 'total']


def build_summary_tables(target):
    parsed_parts, foglio2_cols = [], {}
    for h in HORIZONS:
        p = find_mcs_file(target, h)
        raw = read_mcs_result(p)
        models = sorted(raw['models'].dropna().astype(str).tolist(), key=lambda z: z.lower())
        foglio2_cols[f'h={h}'] = pd.Series(models, dtype='object')
        parsed = pd.DataFrame([parse_model(x) for x in raw['models']])
        parsed['target'] = target
        parsed['horizon'] = h
        parsed['source_file'] = p.name
        parsed_parts.append(parsed)
    parsed_models = pd.concat(parsed_parts, ignore_index=True)
    chart_models = parsed_models[parsed_models['include_foglio1']].copy()

    count_rows = []
    for h in HORIZONS:
        dh = chart_models[chart_models['horizon'] == h]
        for fam in MODEL_ORDER:
            dfam = dh[dh['family'] == fam]
            if fam in AGGREGATE_MODELS:
                rec = {'target': target, 'horizon': h, 'family': fam,
                       'r/Economics': 0, 'r/economy': 0, 'r/wallstreetbets': 0,
                       'all': int(len(dfam))}
            else:
                rec = {'target': target, 'horizon': h, 'family': fam,
                       'r/Economics': int((dfam['subreddit'] == 'r/Economics').sum()),
                       'r/economy': int((dfam['subreddit'] == 'r/economy').sum()),
                       'r/wallstreetbets': int((dfam['subreddit'] == 'r/wallstreetbets').sum()),
                       'all': 0}
            rec['total'] = rec['r/Economics'] + rec['r/economy'] + rec['r/wallstreetbets'] + rec['all']
            count_rows.append(rec)
    tidy_counts = pd.DataFrame(count_rows)

    for h in HORIZONS:
        expected = int((chart_models['horizon'] == h).sum())
        got = int(tidy_counts.loc[tidy_counts['horizon'] == h, 'total'].sum())
        assert got == expected, f'{target} h={h}: Foglio1 total {got} != parsed Foglio2 total {expected}'

    ncols = 1 + 5 * len(HORIZONS)
    row0, row1 = [None] * ncols, [None] * ncols
    row1[0] = 'Model'
    for j, h in enumerate(HORIZONS):
        start = 1 + 5 * j
        row0[start] = f'h={h}'
        row1[start:start + 5] = BLOCK_HEADERS
    matrix = [row0, row1]
    for fam in MODEL_ORDER:
        row = [None] * ncols
        row[0] = fam
        for j, h in enumerate(HORIZONS):
            start = 1 + 5 * j
            rec = tidy_counts[(tidy_counts['family'] == fam) & (tidy_counts['horizon'] == h)].iloc[0]
            vals = [int(rec['r/Economics']), int(rec['r/economy']), int(rec['r/wallstreetbets']),
                    int(rec['all']), int(rec['total'])]
            row[start:start + 5] = [None if v == 0 else v for v in vals[:4]] + [vals[4]]
        matrix.append(row)
    bottom = [None] * ncols
    for j, h in enumerate(HORIZONS):
        bottom[1 + 5 * j + 4] = int(tidy_counts.loc[tidy_counts['horizon'] == h, 'total'].sum())
    matrix.append(bottom)

    return matrix, pd.DataFrame(foglio2_cols), tidy_counts, parsed_models


# ============================================================
# 4. WRITE THE SUMMARY WORKBOOKS
# ============================================================
def write_summary_workbook(summary_objects, target, out_path):
    matrix, foglio2_df, _, _ = summary_objects[target]
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        pd.DataFrame(matrix).to_excel(writer, sheet_name='Foglio1', header=False, index=False)
        foglio2_df.to_excel(writer, sheet_name='Foglio2', index=False)
    return out_path


def verify(summary_objects):
    rows = []
    for target, info in TARGET_INFO.items():
        _, _, tidy_counts, parsed_models = summary_objects[target]
        for h in HORIZONS:
            raw_total = int((parsed_models['horizon'] == h).sum())
            excluded = int(((parsed_models['horizon'] == h) & (~parsed_models['include_foglio1'])).sum())
            paper_total = int(tidy_counts.loc[tidy_counts['horizon'] == h, 'total'].sum())
            rows.append({'dataset': info['dataset'], 'horizon': h,
                         'Foglio2 MCS members': raw_total,
                         'excluded aggregated_pred_sent': excluded,
                         'Foglio1 / paper members': paper_total,
                         'check': 'OK' if paper_total == raw_total - excluded else 'ERROR'})
    df = pd.DataFrame(rows)
    assert (df['check'] == 'OK').all()
    return df


# ============================================================
# 5. FIGURES  (section C of the chart-refresh notebook, audited version)
# ============================================================
OKABE_ITO = {
    'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73',
    'yellow': '#F0E442', 'blue': '#0072B2', 'verm': '#D55E00', 'purple': '#CC79A7',
    'grey': '#999999',
}
def cb_color(name, default='#333333'):
    return OKABE_ITO.get(name, default)

SUB_COLS=['r/Economics','r/economy','r/wallstreetbets']
MCS_DISPLAY_RENAME={'Aggr-Exp':'Mich. Exp.','Aggr-Infl-Swap-1Y':'1Y-Infl-Swap','Aggr-LLM':'Aggr-RIS','Aggr-LLaMA70B':'Aggr-LLaMA70B'}
AGGR_MODELS=AGGREGATE_MODELS
DATASET_LABELS={}

def parse_foglio1_tidy(path,sheet='Foglio1'):
    raw=pd.read_excel(path,sheet_name=sheet,header=None); starts=sorted([i for i,v in raw.iloc[0].items() if isinstance(v,str) and v.strip().startswith('h=')]); rows=[r for r in range(2,len(raw)) if isinstance(raw.iloc[r,0],str) and raw.iloc[r,0].strip()]
    rec=[]; dataset=DATASET_LABELS.get(str(path),str(path))
    for s in starts:
        h=int(re.findall(r'\d+',str(raw.iloc[0,s]))[0])
        for r in rows:
            mr=str(raw.iloc[r,0]).strip(); model=MCS_DISPLAY_RENAME.get(mr,mr)
            vals=[pd.to_numeric(raw.iloc[r,s+k],errors='coerce') for k in range(5)]
            rec.append({'dataset':dataset,'model':model,'horizon':h,'r/Economics':vals[0],'r/economy':vals[1],'r/wallstreetbets':vals[2],'all':vals[3],'total':vals[4]})
    out=pd.DataFrame(rec)
    for c in SUB_COLS+['all','total']: out[c]=out[c].fillna(0).astype(int)
    return out

def parse_mcs_member(name):
    """Parse a Foglio2 MCS member into family, subreddit and RIS MA window.

    The final numeric suffix is always interpreted as the backward-looking
    moving-average window used to smooth the RIS. Aggregate models have no MA.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    s=name.strip(); low=s.lower()
    aggregate_map={
        'aggregated_pred':'Aggr-RIS', 'aggregate_pred':'Aggr-RIS',
        'aggregated_pred (llm ensemble)':'Aggr-RIS',
        'aggregated_pred_llama':'Aggr-LLaMA70B', 'aggregate_pred_llama':'Aggr-LLaMA70B',
        'aggregated_pred_exp':'Mich. Exp.', 'aggregate_pred_exp':'Mich. Exp.',
        'aggregated_pred_swap':'1Y-Infl-Swap', 'aggregate_pred_swap':'1Y-Infl-Swap',
        'aggregate_swap':'1Y-Infl-Swap',
    }
    if low in aggregate_map:
        return {'raw':s,'family':aggregate_map[low],'subreddit':'all','ma_days':None,'aggregate':True,'excluded':False}
    if low in {'aggregated_pred_sent','aggregate_pred_sent'}:
        return {'raw':s,'family':'Aggr-Sent','subreddit':'all','ma_days':None,'aggregate':True,'excluded':True}
    if low.startswith('bert_base_'): family='BERT base'
    elif low.startswith('finbert_'): family='FinBERT'
    elif low.startswith('inflabert_'): family='InflaBERT'
    elif low.startswith('llama3_2_1b_'): family='LLaMA-1B'
    elif low.startswith('llama3_2_3b_'): family='LLaMA-3B'
    elif low.startswith('llama3_1_8b_'): family='LLaMA-8B'
    elif low.startswith('qwen2_5_0_5b_'): family='Qwen-0.5B'
    elif low.startswith('qwen2_5_1_5b_'): family='Qwen-1.5B'
    elif low.startswith('qwen2_5_7b_'): family='Qwen-7B'
    elif low.startswith('gemma2_2b_'): family='Gemma-2B'
    elif low.startswith('gemma2_9b_'): family='Gemma-9B'
    elif low.startswith('gemma2_27b_'): family='Gemma-27B'
    elif low.startswith('llama70_'): family='LLaMA-70B'
    elif low.startswith('sentiment_'): family='Sentiment'
    else: family=None
    if 'wallstreetbets' in low or re.search(r'(^|_)wsb(_|$)',low): subreddit='r/wallstreetbets'
    elif re.search(r'(^|_)economy(_|$)',low): subreddit='r/economy'
    elif re.search(r'(^|_)economics(_|$)',low): subreddit='r/Economics'
    else: subreddit=None
    ma_match=re.search(r'_(\d+)$',low)
    ma_days=int(ma_match.group(1)) if ma_match else None
    return {'raw':s,'family':family,'subreddit':subreddit,'ma_days':ma_days,'aggregate':False,'excluded':False}

def _foglio2_horizon_members(path,sheet='Foglio2'):
    df=pd.read_excel(path,sheet_name=sheet,header=None)
    def is_h(v): return isinstance(v,str) and re.match(r'^\s*h\s*=\s*\d+\s*$',v)
    hc=[(j,int(re.findall(r'\d+',str(v))[0])) for j,v in enumerate(df.iloc[0].tolist()) if is_h(v)]
    if hc:
        return [(h,v) for j,h in hc for v in df.iloc[1:,j].dropna() if isinstance(v,str) and v.strip()]
    hr=[(i,int(re.findall(r'\d+',str(v))[0])) for i,v in enumerate(df.iloc[:,0].tolist()) if is_h(v)]
    return [(h,v) for i,h in hr for v in df.iloc[i,1:].dropna() if isinstance(v,str) and v.strip()]

def summarize_foglio2(path,sheet='Foglio2'):
    """Rebuild the Foglio1 counts directly from Foglio2 memberships."""
    dataset=DATASET_LABELS.get(str(path),str(path)); counts={}
    for h,v in _foglio2_horizon_members(path,sheet):
        p=parse_mcs_member(v)
        if not p or p['excluded']:
            continue
        if p['family'] is None or p['subreddit'] is None:
            raise ValueError(f'Cannot classify Foglio2 member: {v}')
        key=(h,p['family'])
        if key not in counts:
            counts[key]={'r/Economics':0,'r/economy':0,'r/wallstreetbets':0,'all':0,'total':0}
        counts[key][p['subreddit']]+=1
        counts[key]['total']+=1
    horizons=sorted({h for h,_ in counts})
    rows=[]
    for h in horizons:
        for model in MODEL_ORDER:
            c=counts.get((h,model),{'r/Economics':0,'r/economy':0,'r/wallstreetbets':0,'all':0,'total':0})
            rows.append({'dataset':dataset,'model':model,'horizon':h,**c})
    return pd.DataFrame(rows)

def audit_foglio1_vs_foglio2(path):
    """Cell-level discrepancies between the Excel summary and raw Foglio2 memberships."""
    f1=parse_foglio1_tidy(path)
    f2=summarize_foglio2(path)
    keys=['dataset','model','horizon']; metrics=SUB_COLS+['all','total']
    m=f1.merge(f2,on=keys,how='outer',suffixes=('_Foglio1','_Foglio2')).fillna(0)
    diffs=[]
    for _,r in m.iterrows():
        for metric in metrics:
            a=int(r[f'{metric}_Foglio1']); b=int(r[f'{metric}_Foglio2'])
            if a!=b:
                diffs.append({'dataset':r['dataset'],'horizon':int(r['horizon']),'model':r['model'],'field':metric,'Foglio1':a,'Foglio2_expected':b})
    return pd.DataFrame(diffs)

def heat_obj(df,dataset):
    w=df[df.dataset==dataset].pivot_table(index='model',columns='horizon',values='total',aggfunc='sum',fill_value=0); present=list(w.index); w=w.reindex([m for m in MODEL_ORDER if m in present]+[m for m in present if m not in MODEL_ORDER]); H=[h for h in range(1,19) if h in w.columns]; mat=w.reindex(columns=H).values.astype(float); ag=np.array([m in AGGR_MODELS for m in w.index]); std=mat.copy(); std[ag,:]=np.nan; ov=np.where(np.isnan(mat),0,np.clip(mat,0,1)).astype(float); ov[~ag,:]=np.nan; return w,H,mat,ag,std,ov

def plot_two_heatmaps(df,out):
    A=heat_obj(df,'CPI'); B=heat_obj(df,'PCE'); finite=np.concatenate([x[4][np.isfinite(x[4])] for x in [A,B]]); vmax=float(np.nanmax(finite)) if finite.size else 1; acm=ListedColormap(['#F0E442','#D55E00']); fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12.8,4.8),sharey=True)
    def draw(ax,obj,title):
        w,H,mat,ag,std,ov=obj; im=ax.imshow(std,aspect='auto',cmap='YlOrBr',vmin=0,vmax=vmax); ax.imshow(ov,aspect='auto',cmap=acm,vmin=0,vmax=1); ax.set_xticks(range(len(H))); ax.set_xticklabels([f'h={h}' for h in H],fontsize=9); ax.set_yticks(range(len(w))); ax.set_yticklabels(w.index,fontsize=9); ax.set_title(title,fontsize=11)
        if ag.any(): ax.axhline(np.where(ag)[0][0]-.5,color='black',lw=2)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v=mat[i,j]
                if ag[i] and v>=1: ax.text(j,i,'1',ha='center',va='center',fontsize=8)
                elif not ag[i] and np.isfinite(v) and v!=0:
                    rgba=im.cmap(im.norm(v)); lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]; ax.text(j,i,str(int(v)),ha='center',va='center',fontsize=8,color='black' if lum>.6 else 'white')
        return im
    im=draw(ax1,A,'CPI — Models in the Model Confidence Set'); draw(ax2,B,'PCE — Models in the Model Confidence Set'); plt.subplots_adjust(left=.30,right=.93,wspace=.18,top=.90,bottom=.10); cax=fig.add_axes([.945,.18,.012,.68]); fig.colorbar(im,cax=cax).set_label('Total'); fig.savefig(out,dpi=300,bbox_inches='tight'); _maybe_show(out); plt.close(fig)

def run_mcs_figures(cpi_path, pce_path):
    DATASET_LABELS.clear()
    DATASET_LABELS.update({str(cpi_path): 'CPI', str(pce_path): 'PCE'})
    audit=pd.concat([audit_foglio1_vs_foglio2(cpi_path),audit_foglio1_vs_foglio2(pce_path)],ignore_index=True)
    if len(audit):
        print('MCS audit found Foglio1/Foglio2 discrepancies; charts will use the Foglio2-reconstructed counts:')
        print(audit.to_string(index=False))
        f1=pd.concat([summarize_foglio2(cpi_path),summarize_foglio2(pce_path)],ignore_index=True)
    else:
        print('MCS audit passed: Foglio1 matches Foglio2.')
        f1=pd.concat([parse_foglio1_tidy(cpi_path),parse_foglio1_tidy(pce_path)],ignore_index=True)
    o1=FIG_DIR/'MCS_heatmap_CPI_PCE_h1_h18_new.png'
    plot_two_heatmaps(f1,o1)
    return [o1]


# ============================================================
def run(mcs_dir=None, summary_dir=None, out_dir=None, show=None, verbose=True):
    configure(mcs_dir, summary_dir, out_dir, show)
    audit_df = audit_inputs()
    summary_objects = {t: build_summary_tables(t) for t in TARGET_INFO}
    paths = {}
    for target, info in TARGET_INFO.items():
        out = SUMMARY_DIR / f"mcs-summary-{info['slug']}-h1-h18.xlsx"
        paths[target] = write_summary_workbook(summary_objects, target, out)
    ver = verify(summary_objects)
    if verbose:
        print('MCS input files:')
        print(audit_df.to_string(index=False))
        print('\nFoglio1 totals reconcile with Foglio2:')
        print(ver.to_string(index=False))
    figs = run_mcs_figures(paths['CPIAUCSL'], paths['PCEPILFE'])
    return figs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcs-dir", default=str(ROOT / "data" / "mcs"))
    ap.add_argument("--summary-dir", default=str(ROOT / "results" / "test_MCS"))
    ap.add_argument("--out", default=str(ROOT / "figures_paper"))
    ap.add_argument("--show", action="store_true", help="also open the figures")
    args = ap.parse_args()
    figs = run(args.mcs_dir, args.summary_dir, args.out, args.show)
    print('\nfigures:')
    for f in figs:
        print('  ' + str(f))


if __name__ == "__main__":
    main()
