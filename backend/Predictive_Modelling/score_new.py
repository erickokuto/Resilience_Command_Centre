"""Score compatible CSV uploads with the frozen Predictive Modelling models; no retraining."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from phase3 import features, CAT
if len(sys.argv)!=3: raise SystemExit('Usage: python score_new.py input.csv output.csv')
d=pd.read_csv(sys.argv[1],dtype={'item_id':str,'po_id':str}); keys=['po_id','item_id']; x=features(d)
if d[keys].isna().any().any() or d.duplicated(keys).any(): raise ValueError('Missing/duplicate keys')
b=joblib.load(Path(__file__).parent/'models'/'phase3.joblib'); m=b['models']; enc=m['regression'].steps[0][1].named_transformers_['cat']; p=d[keys].copy(); p['predicted_lead_days']=np.maximum(0,m['regression'].predict(x)); p['late_probability']=m['classification'].predict_proba(x)[:,1]; p['late_prediction']=(p.late_probability>=b['threshold']).astype(int); p['predicted_deviation_days']=p.predicted_lead_days-d.planned_lead_days; p['unseen_category']=np.logical_or.reduce([~x[c].isin(v) for c,v in zip(CAT,enc.categories_) ]); p.to_csv(sys.argv[2],index=False)
