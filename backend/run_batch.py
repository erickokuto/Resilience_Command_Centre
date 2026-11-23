"""Validated frozen-model inference; local trusted model artefacts only."""
import argparse, os, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
KEY=['po_id','item_id']
OUTCOMES=['receipt_date','actual_lead_days','late_flag']
AUDIT=['dataset_split','disruption_regime','order_status']

def validate(path):
    d=pd.read_csv(path,dtype={k:str for k in KEY},keep_default_na=True)
    schema=pd.read_csv(ROOT/'data/Data_Dictionary.csv')
    required=[c for c in schema.Variable if c not in OUTCOMES+AUDIT]
    missing=set(required)-set(d)
    if missing: raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    if not 1<=len(d)<=10000: raise ValueError('Upload must contain 1–10,000 records.')
    for c in OUTCOMES+AUDIT:
        if c not in d: d[c]=np.nan
    for c in required:
        if d[c].isna().any(): raise ValueError('Missing values in '+c)
    for _,r in schema.iterrows():
        c,t=r['Variable'],r['Type / unit']
        if t=='ISO date':
            s=pd.to_datetime(d[c],format='%Y-%m-%d',errors='coerce'); bad=d[c].notna() & s.isna()
            if bad.any(): raise ValueError('Invalid ISO date in '+c)
        elif t!='text':
            n=pd.to_numeric(d[c],errors='coerce'); bad=d[c].notna() & (~np.isfinite(n) | (n<0))
            if t in ['integer','0/1']: bad |= n.notna() & (n%1!=0)
            if t=='0/1': bad |= n.notna() & ~n.isin([0,1])
            if '0–1' in t: bad |= n> (100 if '100' in t else 1)
            if bad.any(): raise ValueError('Invalid numeric values in '+c)
            d[c]=n
        elif c in required:
            if d[c].astype(str).str.strip().eq('').any(): raise ValueError('Blank '+c)
    if d.duplicated(KEY).any(): raise ValueError('Duplicate order keys')
    if (d[['daily_demand','planned_lead_days','order_qty','alternate_lead_days']]<=0).any().any(): raise ValueError('Demand, quantity and lead times must be positive')
    if not d.transport_mode.isin(['Air','Road','Rail','Sea']).all(): raise ValueError('Unknown transport mode')
    if not d.material_criticality.isin(['Low','Medium','High']).all(): raise ValueError('Unknown criticality')
    if (d.historical_receipts_n>30).any(): raise ValueError('Historical receipt count exceeds 30')
    order=pd.to_datetime(d.order_date); cutoff=pd.to_datetime(d.as_of_date); signal=pd.to_datetime(d.signal_date); promised=pd.to_datetime(d.promised_date)
    if (order!=cutoff).any() or (signal>cutoff).any(): raise ValueError('Order-time model requires order_date=as_of_date and signal_date <= as_of_date')
    if ((promised-order).dt.days!=d.planned_lead_days).any(): raise ValueError('Promised dates disagree with planned lead days')
    has=d[OUTCOMES].notna().sum(axis=1)
    if (~has.isin([0,3])).any(): raise ValueError('Outcomes must all be present or all absent per row')
    h=d[has==3]; y=(pd.to_datetime(h.receipt_date)-pd.to_datetime(h.order_date)).dt.days
    if (y!=h.actual_lead_days).any() or (y<0).any() or ((y>h.planned_lead_days).astype(int)!=h.late_flag).any(): raise ValueError('Inconsistent receipt outcomes')
    return d[schema.Variable.tolist()]

def run(path,out,decisions=False):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    d=validate(path)
    if decisions and len(d)>1000: raise ValueError('Decision simulation requires at most 1,000 purchase order lines. Use Scoring.csv or Upload_Example.csv, or untick decision simulation in the Analytics Workflow.')
    clean=out/'validated.csv'; d.to_csv(clean,index=False)
    quality=pd.DataFrame({'column':d.columns,'missing':d.isna().sum().values,'distinct':d.nunique().values,'type':d.dtypes.astype(str).values})
    quality.to_csv(out/'quality.csv',index=False)
    def call(phase,*args):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        subprocess.run([sys.executable,str(ROOT/'backend'/{2:'Supplier_Intelligence',4:'Uncertainty_Quantification',5:'Prescriptive_Analytics'}[phase]/'score_new.py'),*map(str,args)],check=True,env=env,capture_output=True,text=True)
    call(2,clean,out/'unsupervised.csv')
    call(4,clean,out/'uncertainty.csv')
    p=pd.read_csv(out/'uncertainty.csv',dtype={k:str for k in KEY}); u=pd.read_csv(out/'unsupervised.csv',dtype={k:str for k in KEY})
    p=p.drop(columns=[c for c in p if c in d and c not in KEY]); u=u[['po_id','item_id','supplier_cluster','anomaly_score','anomaly_flag']]
    merged=d.merge(p,on=KEY,validate='one_to_one').merge(u,on=KEY,validate='one_to_one')
    merged['risk_band_raw']=pd.cut(merged.raw_late_probability,[-.001,.4,.75,1.001],labels=['Low','Review','High'],right=False)
    if decisions:
        if len(d)>1000: raise ValueError('Decision simulation limited to 1,000 rows per run.')
        call(5,clean,out/'uncertainty.csv',ROOT/'evidence/Prescriptive_Analytics/tables/validation_residual_source.csv',out/'decisions.csv')
        z=pd.read_csv(out/'decisions.csv',dtype={k:str for k in KEY}); merged=merged.merge(z,on=KEY,validate='one_to_one')
    merged.to_csv(out/'results.csv',index=False)
    print(f'Validated and scored {len(d)} records; {int(merged.anomaly_flag.sum())} anomalies; {int(merged.unseen_category.sum())} unseen categories.')
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('input'); a.add_argument('output'); a.add_argument('--decisions',action='store_true'); args=a.parse_args()
    try: run(args.input,args.output,args.decisions)
    except subprocess.CalledProcessError as e: sys.exit('Scoring failed: '+e.stderr[-3000:])
    except Exception as e: sys.exit(str(e))
