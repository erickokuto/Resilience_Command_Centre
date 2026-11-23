import sys, math, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, mean_pinball_loss
from phase3 import features, pipe, SEED
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
levels=[.8,.9]
def group(d): return np.where(d.transport_mode=='Sea','Sea','Non-sea')
def conformal(s,c): return float(np.sort(s)[math.ceil((len(s)+1)*c)-1])
def logits(p):
 p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p)).reshape(-1,1)
def quantiles(b,x): return np.sort(np.column_stack([b['quantile'][q].predict(x) for q in [.1,.5,.9]]),axis=1)
def score(d,b):
 x=features(d); m=b['point']; raw=m['classification'].predict_proba(x)[:,1]; point=np.maximum(0,m['regression'].predict(x)); qs=quantiles(b,x); g=group(d)
 p=d[['po_id','item_id','supplier_id','order_date','transport_mode','planned_lead_days']].copy(); p['predicted_lead_days']=point; p['raw_late_probability']=raw; p['calibrated_late_probability']=b['platt'].predict_proba(logits(raw))[:,1]
 for i,q in enumerate([10,50,90]): p[f'quantile_p{q}']=np.maximum(0,qs[:,i])
 for lev in levels:
  t=str(int(lev*100)); pooled=b['pooled'][lev]; radius=np.array([b['group'][lev][a] for a in g]); r=np.array([b['cqr'][lev][a] for a in g])
  for name,lo,hi in [('pooled',point-pooled,point+pooled),('group',point-radius,point+radius),('cqr',qs[:,0]-r,qs[:,2]+r)]:
   p[f'{name}_lower_{t}']=np.maximum(0,lo); p[f'{name}_upper_{t}']=np.maximum(p[f'{name}_lower_{t}'],hi)
 p['risk_band']=pd.cut(p.calibrated_late_probability,[-.001,.4,.75,1.001],labels=['Low','Review','High'],right=False).astype(str)
 for c in ['actual_lead_days','late_flag']:
  if c in d: p[c]=d[c]
 return p

def main(src):
 ds={s:pd.read_csv(Path(src)/f'{s}.csv',dtype={'item_id':str}) for s in ['Train','Validation','Test','Scoring']}; m=joblib.load(R/'models/phase3.joblib')['models']; b={'point':m,'quantile':{},'pooled':{},'group':{},'cqr':{}}; x=features(ds['Train']); y=ds['Train'].actual_lead_days
 for q in [.1,.5,.9]:
  model=pipe(GradientBoostingRegressor(loss='quantile',alpha=q,n_estimators=80,max_depth=3,min_samples_leaf=15,random_state=SEED)); model.fit(x,y); b['quantile'][q]=model; print('Fitted',q,flush=True)
 v=ds['Validation']; xv=features(v); pv=np.maximum(0,m['regression'].predict(xv)); qv=quantiles(b,xv); e=np.abs(v.actual_lead_days-pv); ce=np.maximum(0,np.maximum(qv[:,0]-v.actual_lead_days,v.actual_lead_days-qv[:,2])); g=group(v)
 for lev in levels:
  b['pooled'][lev]=conformal(e,lev); b['group'][lev]={a:conformal(e[g==a],lev) for a in ['Sea','Non-sea']}; b['cqr'][lev]={a:conformal(ce[g==a],lev) for a in ['Sea','Non-sea']}
 raw=m['classification'].predict_proba(xv)[:,1]; b['platt']=LogisticRegression(C=1e6).fit(logits(raw),v.late_flag)
 joblib.dump(b,R/'models/phase4.joblib',compress=3)
 rows=[]; probs=[]; bins=[]; qrows=[]
 for s in ['Validation','Test','Scoring']:
  p=score(ds[s],b); p.to_csv(R/'tables'/f'{s.lower()}_uncertainty.csv',index=False)
  if s=='Scoring': continue
  for label,h in [('All',p),*list(p.groupby('transport_mode'))]:
   y=h.actual_lead_days
   for lev in levels:
    t=str(int(lev*100))
    for method in ['pooled','group','cqr']:
     lo=h[f'{method}_lower_{t}']; hi=h[f'{method}_upper_{t}']; width=hi-lo; isc=width+2/(1-lev)*(np.maximum(lo-y,0)+np.maximum(y-hi,0)); rows.append(dict(split=s,transport=label,n=len(h),method=method,nominal=lev,coverage=((y>=lo)&(y<=hi)).mean(),mean_width=width.mean(),interval_score=isc.mean()))
  for col in ['raw_late_probability','calibrated_late_probability']:
   a=p[col]; y=p.late_flag; probs.append(dict(split=s,method=col,brier=brier_score_loss(y,a),log_loss=log_loss(y,a),auc=roc_auc_score(y,a),mean_probability=a.mean(),observed=y.mean()))
   for k,h in p.assign(bin=pd.cut(a,np.linspace(0,1,11),include_lowest=True)).groupby('bin',observed=True): bins.append(dict(split=s,method=col,bin=str(k),n=len(h),mean_probability=h[col].mean(),observed=h.late_flag.mean()))
  for q in [.1,.5,.9]: qrows.append(dict(split=s,quantile=q,pinball=mean_pinball_loss(p.actual_lead_days,p[f'quantile_p{int(q*100)}'],alpha=q)))
 for name,data in [('interval_metrics',rows),('calibration_metrics',probs),('reliability_bins',bins),('quantile_metrics',qrows)]: pd.DataFrame(data).to_csv(R/'tables'/f'{name}.csv',index=False)
 pd.DataFrame([dict(level=l,pooled=b['pooled'][l],**{f'group_{k}':v for k,v in b['group'][l].items()},**{f'cqr_{k}':v for k,v in b['cqr'][l].items()}) for l in levels]).to_csv(R/'tables/calibration_radii.csv',index=False)
 p=pd.read_csv(R/'tables/scoring_uncertainty.csv'); p.sort_values('calibrated_late_probability',ascending=False).head(25).to_csv(R/'tables/priority_orders.csv',index=False)
 t=pd.read_csv(R/'tables/test_uncertainty.csv').sort_values('predicted_lead_days').iloc[::40]; fig,ax=plt.subplots(figsize=(9,4)); a=np.arange(len(t)); ax.fill_between(a,t.group_lower_80,t.group_upper_80,alpha=.25,label='80% interval'); ax.plot(a,t.predicted_lead_days,label='Point prediction'); ax.scatter(a,t.actual_lead_days,s=12,label='Actual'); ax.set(xlabel='Selected test orders (sorted by forecast)',ylabel='Days'); ax.legend(); fig.tight_layout(); fig.savefig(R/'charts/intervals.png',dpi=160); plt.close(fig)
 z=pd.DataFrame(bins); fig,ax=plt.subplots(); ax.plot([0,1],[0,1],'--',color='grey');
 for k,h in z[z.split=='Test'].groupby('method'): ax.plot(h.mean_probability,h.observed,'o-',label=k.replace('_late_probability',''))
 ax.set(xlabel='Mean predicted probability',ylabel='Observed late fraction',title='Test reliability diagram'); ax.legend(); fig.tight_layout(); fig.savefig(R/'charts/reliability.png',dpi=160); plt.close(fig)
 print(pd.DataFrame(rows).query("split=='Test' and transport=='All'").to_string(index=False)); print(pd.DataFrame(probs).to_string(index=False)); print(p[['predicted_lead_days','calibrated_late_probability','group_lower_80','group_upper_80']].mean()); print(p.risk_band.value_counts())
if __name__=='__main__': main(sys.argv[1])
