import sys, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import *
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from xgboost import XGBRegressor,XGBClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
SEED=20261115
CAT='supplier_id material_group plant_id origin_country transport_mode route_id material_criticality'.split()
NUM='order_qty planned_lead_days supplier_otd_rate prior_lt_mean prior_lt_sd port_congestion weather_severity geopolitical_risk strike_flag supplier_defect_rate supplier_utilisation demand_volatility energy_cost_index production_load historical_receipts_n'.split()
EXT='port_congestion weather_severity geopolitical_risk strike_flag energy_cost_index'.split()
REQ=CAT+NUM+['order_date']
def features(d):
 x=d[REQ].copy(); t=pd.to_datetime(x.pop('order_date'),format='%Y-%m-%d'); x['month_sin']=np.sin(2*np.pi*t.dt.month/12); x['month_cos']=np.cos(2*np.pi*t.dt.month/12); x['weekday']=t.dt.dayofweek
 for c in NUM: x[c]=pd.to_numeric(x[c],errors='raise')
 if x.isna().any().any() or not np.isfinite(x[NUM]).all().all(): raise ValueError('Invalid predictor')
 return x
def pipe(m,ext=True):
 nums=[c for c in NUM if ext or c not in EXT]+['month_sin','month_cos','weekday']; return make_pipeline(ColumnTransformer([('num',StandardScaler(),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)]),m)
def rm(y,p): return dict(MAE=mean_absolute_error(y,p),RMSE=root_mean_squared_error(y,p),R2=r2_score(y,p),Bias=np.mean(p-y))
def cm(y,p):
 z=p>=.5; tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel(); return dict(AUC=roc_auc_score(y,p),AP=average_precision_score(y,p),Brier=brier_score_loss(y,p),Accuracy=accuracy_score(y,z),Precision=precision_score(y,z,zero_division=0),Recall=recall_score(y,z),F1=f1_score(y,z),TN=tn,FP=fp,FN=fn,TP=tp)
def main():
 src,out=map(Path,sys.argv[1:3]); [ (out/i).mkdir(parents=True,exist_ok=True) for i in ['tables','models','charts'] ]; ds={s:pd.read_csv(src/f'{s}.csv',dtype={'item_id':str}) for s in ['Train','Validation','Test','Scoring']}; X={s:features(d) for s,d in ds.items()}
 for a,b in zip(['Train','Validation','Test'],['Validation','Test','Scoring']): assert pd.to_datetime(ds[a].receipt_date).max()<pd.to_datetime(ds[b].order_date).min()
 bundle={}; preds={s:{} for s in ds}; chosen={}
 for task in ['regression','classification']:
  r=task=='regression'; target='actual_lead_days' if r else 'late_flag'; y={s:d[target] for s,d in ds.items()}; met=rm if r else cm; metric='MAE' if r else 'Brier'; pred=lambda m,x: np.maximum(0,m.predict(x)) if r else m.predict_proba(x)[:,1]; baseline=lambda s: ds[s].planned_lead_days.to_numpy() if r else np.full(len(ds[s]),y['Train'].mean()); bn='ERP planned lead' if r else 'Training prevalence'
  models={'Ridge' if r else 'Logistic':Ridge(alpha=10) if r else LogisticRegression(C=.3,max_iter=1000),'Random Forest':(RandomForestRegressor if r else RandomForestClassifier)(n_estimators=80,min_samples_leaf=5,max_features=.5,n_jobs=2,random_state=SEED)}
  for depth in [3,5]: models[f'XGBoost d{depth}']=(XGBRegressor if r else XGBClassifier)(n_estimators=150,max_depth=depth,learning_rate=.04,subsample=.85,colsample_bytree=.85,random_state=SEED,n_jobs=2)
  rows=[dict(model=bn,split='Validation',**met(y['Validation'],baseline('Validation')))]; fitted={}; scores={}
  for name,m in models.items():
   p=pipe(m); p.fit(X['Train'],y['Train']); fitted[name]=p; v=met(y['Validation'],pred(p,X['Validation'])); scores[name]=v[metric]; rows.append(dict(model=name,split='Validation',**v))
  best=min(scores,key=scores.get); chosen[task]=best; m=fitted[best]; bundle[task]=m; print(task,best,scores,flush=True)
  ab=pipe(clone(models[best]),False); ab.fit(X['Train'],y['Train']); rows.append(dict(model=best+' (no external signals)',split='Validation',**met(y['Validation'],pred(ab,X['Validation']))))
  rows.extend([dict(model=bn,split='Test',**met(y['Test'],baseline('Test'))),dict(model=best,split='Test',**met(y['Test'],pred(m,X['Test'])))])
  pd.DataFrame(rows).to_csv(out/'tables'/f'{task}_metrics.csv',index=False)
  for s in ds: preds[s][task]=pred(m,X[s])
  imp=permutation_importance(m,X['Validation'],y['Validation'],scoring='neg_mean_absolute_error' if r else 'neg_brier_score',n_repeats=3,random_state=SEED); pd.DataFrame({'feature':X['Validation'].columns,'importance':imp.importances_mean,'sd':imp.importances_std}).sort_values('importance',ascending=False).to_csv(out/'tables'/f'{task}_importance.csv',index=False)
 joblib.dump(dict(models=bundle,selected=chosen,threshold=.5,required=REQ),out/'models'/'phase3.joblib',compress=3)
 for s,d in ds.items():
  p=d[['po_id','item_id','supplier_id','order_date','transport_mode','planned_lead_days','actual_lead_days','late_flag']].copy(); p['predicted_lead_days']=preds[s]['regression']; p['late_probability']=preds[s]['classification']; p['late_prediction']=(p.late_probability>=.5).astype(int); p['predicted_deviation_days']=p.predicted_lead_days-p.planned_lead_days; p.to_csv(out/'tables'/f'{s.lower()}_predictions.csv',index=False)
 test=pd.read_csv(out/'tables'/'test_predictions.csv'); score=pd.read_csv(out/'tables'/'scoring_predictions.csv'); score.sort_values('late_probability',ascending=False).head(25).to_csv(out/'tables'/'priority_orders.csv',index=False)
 pd.DataFrame([dict(transport_mode=k,n=len(g),**rm(g.actual_lead_days,g.predicted_lead_days),**cm(g.late_flag,g.late_probability)) for k,g in test.groupby('transport_mode')]).to_csv(out/'tables'/'test_by_transport.csv',index=False)
 fig,ax=plt.subplots(); ax.scatter(test.actual_lead_days,test.predicted_lead_days,s=8,alpha=.3); ax.plot([0,100],[0,100],'--',color='grey'); ax.set(xlabel='Actual days',ylabel='Predicted days',title='Test: actual vs predicted'); fig.tight_layout(); fig.savefig(out/'charts'/'actual_predicted.png',dpi=160); plt.close(fig)
 fpr,tpr,_=roc_curve(test.late_flag,test.late_probability); fig,ax=plt.subplots(); ax.plot(fpr,tpr,label=f'AUC {roc_auc_score(test.late_flag,test.late_probability):.3f}'); ax.plot([0,1],[0,1],'--',color='grey'); ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Test ROC'); ax.legend(); fig.tight_layout(); fig.savefig(out/'charts'/'roc.png',dpi=160); plt.close(fig)
 print(chosen, 'scoring >=.5',sum(score.late_probability>=.5),'>=.75',sum(score.late_probability>=.75),flush=True)
if __name__=='__main__': main()
