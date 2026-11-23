"""Prescriptive Analytics: scenario-conditioned procurement decisions. All inputs synthetic."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
KEY=['po_id','item_id']
ACTIONS=['Maintain','Buffer','Expedite','Alternate','Expedite + buffer','Alternate + buffer']

def load(data,phase4,split):
 a=pd.read_csv(data/f'{split.capitalize()}.csv',dtype={'item_id':str})
 b=pd.read_csv(phase4/f'{split}_uncertainty.csv',dtype={'item_id':str})
 cols=KEY+['predicted_lead_days','raw_late_probability','group_upper_80','group_upper_90']
 return a.merge(b[cols],on=KEY,validate='one_to_one')

def simulate(d,residuals,n=600,seed=51,stress=0,actual=False):
 rng=np.random.default_rng(seed); N=len(d)
 def col(k): return d[k].to_numpy(float)[:,None]
 sea=d.transport_mode.eq('Sea').to_numpy(); err=np.empty((N,n))
 for group in [False,True]:
  ix=np.flatnonzero(sea==group); err[ix]=rng.choice(residuals[group],size=(len(ix),n),replace=True)
 lead=np.maximum(1,(col('actual_lead_days') if actual else col('predicted_lead_days')+err)+stress*np.where(sea[:,None],7,3)) + np.zeros((N,n))
 cv=col('demand_volatility'); sig=np.sqrt(np.log1p(cv**2)); demand=col('daily_demand')*np.exp(rng.normal(size=(N,n))*sig-sig**2/2)*(1+.15*stress)
 # Alternative quoted timing is not an observation: assume 20% lognormal variation and common shock.
 alt=np.maximum(1,col('alternate_lead_days')*np.exp(rng.normal(size=(N,n))*.2-.02)+stress*3)
 exp=np.maximum(1,lead-col('expedite_days_saved')*rng.uniform(.5,1.5,size=(N,n)))
 return lead,demand,alt,exp

def evaluate(d,draws, action,buffer):
 lead,dem,alt,exp=draws
 def c(k): return d[k].to_numpy(float)[:,None]
 action=np.asarray(action); buffer=np.asarray(buffer,float); usealt=np.isin(action,['Alternate','Alternate + buffer']); useexp=np.isin(action,['Expedite','Expedite + buffer'])
 L=np.where(usealt[:,None],alt,np.where(useexp[:,None],exp,lead)); stock=c('inventory_on_hand')+buffer[:,None]
 shortage=np.maximum(dem*L-stock,0); t=np.minimum(L,stock/dem)
 hold=(stock*t-.5*dem*t*t)*c('holding_cost')
 incremental=np.where(usealt,c('order_qty')[:,0]*(c('alternate_unit_cost')[:,0]-c('unit_cost')[:,0]),0)+np.where(useexp,c('expedite_cost')[:,0],0)+buffer*c('unit_cost')[:,0]*.2
 cost=hold+shortage*c('stockout_cost')+incremental[:,None]
 return {'cost':cost.mean(1),'shortage_units':shortage.mean(1),'stockout_probability':(shortage>0).mean(1),'late_probability_scenario':(L>c('planned_lead_days')).mean(1),'lead_days':L.mean(1),'holding_cost':hold.mean(1),'intervention_cost':incremental}

def optimise(d,residuals,n=600,seed=51):
 draws=simulate(d,residuals,n,seed); N=len(d); best=np.full(N,np.inf); chosen=np.full(N,'Maintain',dtype=object); units=np.zeros(N)
 # full switch only; cap emergency buffer at 14 days of expected demand, assumed available immediately.
 feasible=d.alternate_capacity.ge(d.order_qty).to_numpy() & d.alternate_supplier_id.ne(d.supplier_id).to_numpy()
 maxbuf=14*d.daily_demand.to_numpy(); inv=d.inventory_on_hand.to_numpy(); dd=d.daily_demand.to_numpy()
 for action in ACTIONS:
  base=draws[2] if action.startswith('Alternate') else draws[3] if action.startswith('Expedite') else draws[0]
  for q in ([.5,.8,.9,.95] if 'buffer' in action.lower() else [0]):
   b=np.minimum(maxbuf,np.maximum(0,np.ceil(np.quantile(base*draws[1],q,axis=1)-inv))) if q else np.zeros(N)
   v=evaluate(d,draws,np.repeat(action,N),b)['cost']; valid=feasible if action.startswith('Alternate') else np.ones(N,bool)
   improve=valid & (v<best-1e-8); best[improve]=v[improve]; chosen[improve]=action; units[improve]=b[improve]
 return chosen,units

def policies(d,residuals):
 n=len(d); inv=d.inventory_on_hand.to_numpy(); dd=d.daily_demand.to_numpy(); cap=14*dd
 conventional=np.minimum(cap,np.maximum(0,np.ceil(dd*d.planned_lead_days.to_numpy()+d.safety_stock_units.to_numpy()-inv)))
 uncertainty=np.minimum(cap,np.maximum(0,np.ceil(dd*d.group_upper_90.to_numpy()-inv)))
 act,buf=optimise(d,residuals)
 return {'Maintain':(np.repeat('Maintain',n),np.zeros(n)), 'ERP buffer':(np.where(conventional>0,'Buffer','Maintain'),conventional),'Uncertainty buffer':(np.where(uncertainty>0,'Buffer','Maintain'),uncertainty),'Optimised':(act,buf)}

def run(data,p4,out):
 (out/'tables').mkdir(parents=True,exist_ok=True); (out/'charts').mkdir(exist_ok=True)
 val=load(data,p4,'validation'); residuals={g:(val.actual_lead_days-val.predicted_lead_days)[val.transport_mode.eq('Sea').eq(g)].to_numpy() for g in [False,True]}
 rows=[]; actions=[]; results={}
 for split in ['test','scoring']:
  d=load(data,p4,split); pol=policies(d,residuals)
  for stress in [0,1,2]:
   draws=simulate(d,residuals,n=1000,seed=85,stress=stress,actual=split=='test')
   for name,(a,b) in pol.items():
    metrics=evaluate(d,draws,a,b); rows.append({'split':split,'stress':stress,'policy':name,'orders':len(d),**{k:float(np.mean(v)) for k,v in metrics.items()},'total_cost':float(metrics['cost'].sum()),'buffer_units':float(b.sum())})
    if stress==0 and name=='Optimised':
     base=evaluate(d,draws,*pol['Maintain']); output=d[KEY+['supplier_id','transport_mode','material_criticality','planned_lead_days','predicted_lead_days','raw_late_probability','group_upper_90','inventory_on_hand','daily_demand','alternate_supplier_id','alternate_capacity','order_qty']].copy(); output['action']=a; output['buffer_units']=b
     for k,v in metrics.items(): output[k]=v
     output['baseline_cost']=base['cost']; output['estimated_savings']=base['cost']-metrics['cost']; output['baseline_stockout_probability']=base['stockout_probability']; output.to_csv(out/'tables'/f'{split}_decisions.csv',index=False); results[split]=output
     for action,count in pd.Series(a).value_counts().items(): actions.append({'split':split,'action':action,'orders':count})
  if split=='scoring':
   assert ((~pd.Series(pol['Optimised'][0]).str.startswith('Alternate')) | d.alternate_capacity.ge(d.order_qty)).all()
   assert (pol['Optimised'][1]<=14*d.daily_demand).all()
 summary=pd.DataFrame(rows); summary.to_csv(out/'tables/policy_comparison.csv',index=False); pd.DataFrame(actions).to_csv(out/'tables/action_counts.csv',index=False)
 results['scoring'].sort_values('estimated_savings',ascending=False).head(50).to_csv(out/'tables/priority_orders.csv',index=False)
 for metric,label in [('cost','Expected incremental cost (EUR/order)'),('stockout_probability','Expected stockout exposure')]:
  x=summary.query("split=='scoring'").pivot(index='stress',columns='policy',values=metric); ax=x.plot(marker='o',figsize=(8,4.5)); ax.set(xlabel='Stress level',ylabel=label,xticks=[0,1,2]); ax.set_xticklabels(['Base','Moderate','Severe']); ax.legend(fontsize=8); ax.figure.tight_layout(); ax.figure.savefig(out/'charts'/f'{metric}.png',dpi=150); plt.close(ax.figure)
 print(summary.to_string(index=False)); print(pd.DataFrame(actions).to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('data',type=Path); p.add_argument('phase4_tables',type=Path); p.add_argument('--out',type=Path,default=Path(__file__).parent); a=p.parse_args(); run(a.data,a.phase4_tables,a.out)
