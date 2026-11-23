'''Score compatible order-time CSV data with frozen Uncertainty Quantification models.'''
import sys, joblib, pandas as pd, numpy as np
from phase4 import score, R
from phase3 import features, CAT
if len(sys.argv)!=3: raise SystemExit('Usage: python score_new.py input.csv output.csv')
d=pd.read_csv(sys.argv[1],dtype={'po_id':str,'item_id':str}); keys=['po_id','item_id']
if d[keys].isna().any().any() or d.duplicated(keys).any(): raise ValueError('Missing or duplicate keys')
b=joblib.load(R/'models/phase4.joblib'); p=score(d,b); x=features(d); enc=b['point']['regression'].steps[0][1].named_transformers_['cat']; p['unseen_category']=np.logical_or.reduce([~x[c].isin(v) for c,v in zip(CAT,enc.categories_)]); p.to_csv(sys.argv[2],index=False)
