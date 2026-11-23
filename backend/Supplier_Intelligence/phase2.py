import os
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import sys, pathlib, json
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
import joblib
from scipy.cluster.hierarchy import linkage, dendrogram
D=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'data'); O=pathlib.Path(sys.argv[2] if len(sys.argv)>2 else 'Supplier_Intelligence');
for x in ['tables','charts','models']: (O/x).mkdir(parents=True,exist_ok=True)
def save(d,n): d.to_csv(O/'tables'/f'{n}.csv',index=False)
def chart(n): plt.tight_layout(); plt.savefig(O/'charts'/f'{n}.png',dpi=150); plt.close()
T=pd.read_csv(D/'Train.csv',dtype={'item_id':str}); V=pd.read_csv(D/'Validation.csv',dtype={'item_id':str}); S=pd.read_csv(D/'Scoring.csv',dtype={'item_id':str}); H=pd.read_csv(D/'Test.csv',dtype={'item_id':str}); M=pd.read_csv(D/'Supplier_Master.csv')
# Supplier profiles use only order-time measurements from training; no delivery outcomes.
F=['supplier_otd_rate','prior_lt_mean','prior_lt_sd','supplier_defect_rate','supplier_utilisation','geopolitical_risk','weather_severity','strike_flag']
P=T.groupby('supplier_id')[F].mean(); P['orders']=T.groupby('supplier_id').size(); X=P[F]; scaler=StandardScaler().fit(X); Z=scaler.transform(X); pca=PCA().fit(Z); scores=pca.transform(Z)
load=pd.DataFrame(pca.components_.T,index=F,columns=[f'PC{i+1}' for i in range(len(F))]); save(load.reset_index(names='feature'),'pca_loadings'); ev=pd.DataFrame({'component':range(1,len(F)+1),'variance_ratio':pca.explained_variance_ratio_,'cumulative':np.cumsum(pca.explained_variance_ratio_)}); save(ev,'pca_variance')
# Compare partitions and repeat on random 80% supplier subsamples; evaluate against full training partition.
rng=np.random.default_rng(42); rows=[]; fits={}
for k in range(2,8):
 km=KMeans(n_clusters=k,n_init=30,random_state=42).fit(Z); lab=km.labels_; ward=AgglomerativeClustering(n_clusters=k).fit_predict(Z); ar=[]
 for j in range(40):
  idx=rng.choice(len(Z),int(.8*len(Z)),replace=False); sub=KMeans(n_clusters=k,n_init=15,random_state=j).fit(Z[idx]); ar.append(adjusted_rand_score(lab,sub.predict(Z)))
 rows.append(dict(k=k,silhouette=silhouette_score(Z,lab),davies_bouldin=davies_bouldin_score(Z,lab),inertia=km.inertia_,min_cluster=np.bincount(lab).min(),ward_silhouette=silhouette_score(Z,ward),ward_agreement=adjusted_rand_score(lab,ward),stability_mean=np.mean(ar),stability_p10=np.quantile(ar,.1))); fits[k]=km
K=pd.DataFrame(rows); save(K,'cluster_selection'); eligible=K[K.min_cluster>=5]; k=int(eligible.sort_values('silhouette',ascending=False).iloc[0].k); km=fits[k]; P['cluster']=km.labels_+1; P['ward_cluster']=AgglomerativeClustering(n_clusters=k).fit_predict(Z)+1; P['PC1']=scores[:,0]; P['PC2']=scores[:,1];
# Descriptive retrospective outcome check restricted to receipts known before validation begins.
cut=pd.to_datetime(V.order_date).min(); known=T[pd.to_datetime(T.receipt_date)<cut]; obs=known.groupby('supplier_id').agg(known_receipts=('late_flag','size'),observed_late_rate=('late_flag','mean'),observed_lead_days=('actual_lead_days','mean')); P=P.join(obs).reset_index().merge(M,on='supplier_id',how='left'); save(P,'supplier_profiles'); prof=P.groupby('cluster').agg(suppliers=('supplier_id','size'),orders=('orders','sum'),**{f:(f,'mean') for f in F},observed_late_rate=('observed_late_rate','mean'),observed_lead_days=('observed_lead_days','mean')).reset_index(); save(prof,'cluster_profiles'); save(pd.crosstab(P.cluster,P.ward_cluster).reset_index(),'ward_comparison');
plt.plot(ev.component,ev.variance_ratio,'o-',label='Individual'); plt.plot(ev.component,ev.cumulative,'s-',label='Cumulative'); plt.xlabel('Principal component'); plt.ylabel('Explained variance'); plt.legend(); chart('pca_scree')
plt.figure(figsize=(8,5)); sns.scatterplot(data=P,x='PC1',y='PC2',hue='cluster',palette='tab10'); chart('supplier_pca')
plt.figure(figsize=(10,5)); sns.heatmap(pd.DataFrame(Z,columns=F).assign(cluster=km.labels_+1).groupby('cluster').mean(),center=0,cmap='vlag',annot=True,fmt='.1f'); chart('cluster_heatmap')
plt.figure(figsize=(10,5)); dendrogram(linkage(Z,method='ward'),truncate_mode='lastp',p=25); plt.ylabel('Ward distance'); chart('ward_dendrogram')
plt.plot(K.k,K.silhouette,'o-',label='K-means'); plt.plot(K.k,K.ward_silhouette,'s-',label='Ward'); plt.xlabel('Clusters'); plt.ylabel('Silhouette'); plt.legend(); chart('cluster_selection')
# Order anomaly inputs: normalise lead and congestion relative to training route/mode, not actual delivery.
A=['supplier_otd_rate','prior_lt_sd','supplier_defect_rate','supplier_utilisation','weather_severity','geopolitical_risk','strike_flag','demand_volatility','production_load','planned_route_deviation','congestion_route_deviation','inventory_coverage_ratio']
base=T.groupby('route_id')[['planned_lead_days','port_congestion']].median(); fallback=T.groupby('transport_mode')[['planned_lead_days','port_congestion']].median(); glob=T[['planned_lead_days','port_congestion']].median()
def features(d):
 x=d.copy();
 for c,new in [('planned_lead_days','planned_route_deviation'),('port_congestion','congestion_route_deviation')]:
  ref=x.route_id.map(base[c]).fillna(x.transport_mode.map(fallback[c])).fillna(glob[c]); x[new]=x[c]-ref
 x['inventory_coverage_ratio']=x.inventory_on_hand/x.daily_demand/x.planned_lead_days
 return x[A]
iso=IsolationForest(n_estimators=300,contamination=.05,random_state=42,n_jobs=1).fit(features(T)); maps=P.set_index('supplier_id').cluster; allout=[]
for name,d in [('Train',T),('Validation',V),('Test',H),('Scoring',S)]:
 a=d[['po_id','item_id','supplier_id','order_date','transport_mode']].copy(); a['dataset_split']=name; a['supplier_cluster']=a.supplier_id.map(maps).astype('Int64'); a['anomaly_score']=-iso.decision_function(features(d)); a['anomaly_flag']=(a.anomaly_score>0).astype(int); allout.append(a); save(a,f'{name.lower()}_unsupervised_scores')
 out=pd.concat(allout,ignore_index=True)
summ=out.groupby('dataset_split').agg(orders=('po_id','size'),flagged=('anomaly_flag','sum'),flag_rate=('anomaly_flag','mean'),unknown_cluster=('supplier_cluster',lambda x:x.isna().sum())).reset_index(); save(summ,'anomaly_summary'); save(out,'all_unsupervised_scores'); save(out[out.dataset_split=='Scoring'].nlargest(50,'anomaly_score'),'scoring_priority_anomalies')
plt.figure(figsize=(8,5)); sns.histplot(data=out,x='anomaly_score',hue='dataset_split',element='step',stat='density',common_norm=False,fill=False); plt.axvline(0,color='black',linestyle='--'); chart('anomaly_distribution')
joblib.dump(dict(features=F,scaler=scaler,pca=pca,kmeans=km,supplier_map=maps,anomaly_features=A,route_medians=base,mode_medians=fallback,global_medians=glob,isolation_forest=iso),O/'models'/'phase2_models.joblib');
print('cutoff',cut,'known',len(known),'suppliers',len(P),'min orders',P.orders.min(),'k',k); print(ev.to_string(index=False)); print(K.to_string(index=False)); print(prof.to_string(index=False)); print(summ.to_string(index=False))
