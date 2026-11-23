"""Score a new CSV with the frozen Supplier Intelligence model. Usage: python score_new.py input.csv output.csv [model.joblib]. Only load trusted model files."""
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
m=joblib.load(sys.argv[3] if len(sys.argv)>3 else Path(__file__).parent/'models/phase2_models.joblib')
d=pd.read_csv(sys.argv[1],dtype={'item_id':str}); x=d.copy()
required=set(m['anomaly_features'])-{'planned_route_deviation','congestion_route_deviation','inventory_coverage_ratio'}
required|={'po_id','item_id','supplier_id','route_id','transport_mode','planned_lead_days','port_congestion','inventory_on_hand','daily_demand','order_date'}
missing=required-set(x.columns)
if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
if x[list(required)].isna().any().any(): raise ValueError('Missing required input values')
if x.duplicated(['po_id','item_id']).any(): raise ValueError('Duplicate order keys')
if (x[['daily_demand','planned_lead_days']]<=0).any().any(): raise ValueError('Demand and planned lead time must be positive')
for c,new in [('planned_lead_days','planned_route_deviation'),('port_congestion','congestion_route_deviation')]:
 ref=x.route_id.map(m['route_medians'][c]).fillna(x.transport_mode.map(m['mode_medians'][c])).fillna(m['global_medians'][c]); x[new]=x[c]-ref
x['inventory_coverage_ratio']=x.inventory_on_hand/x.daily_demand/x.planned_lead_days
f=x[m['anomaly_features']].astype(float)
if not np.isfinite(f).all().all(): raise ValueError('Non-finite model inputs')
a=d[['po_id','item_id','supplier_id','order_date','transport_mode']].copy(); a['supplier_cluster']=a.supplier_id.map(m['supplier_map']).astype('Int64'); a['anomaly_score']=-m['isolation_forest'].decision_function(f); a['anomaly_flag']=(a.anomaly_score>0).astype(int); a.to_csv(sys.argv[2],index=False)
