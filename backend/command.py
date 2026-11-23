"""Frozen scoring and fixed-policy stress evaluation."""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from run_batch import run, ROOT, KEY
sys.path.insert(0,str(ROOT/'backend/Prescriptive_Analytics'))
from phase5 import simulate,evaluate,policies

def command(source,out):
    out=Path(out); run(source,out,True)
    d=pd.read_csv(out/'validated.csv',dtype={k:str for k in KEY})
    p=pd.read_csv(out/'uncertainty.csv',dtype={k:str for k in KEY})
    d=d.merge(p[KEY+['predicted_lead_days','raw_late_probability','group_upper_90']],on=KEY,validate='one_to_one')
    v=pd.read_csv(ROOT/'evidence/Prescriptive_Analytics/tables/validation_residual_source.csv')
    residuals={g:(v.actual_lead_days-v.predicted_lead_days)[v.transport_mode.eq('Sea').eq(g)].to_numpy() for g in [False,True]}
    pol=policies(d,residuals); z=pd.read_csv(out/'decisions.csv',dtype={k:str for k in KEY})
    assert np.array_equal(pol['Optimised'][0],z.action.to_numpy())
    assert np.allclose(pol['Optimised'][1],z.buffer_units)
    rows=[]
    for stress in [0,1,2]:
        draws=simulate(d,residuals,n=1000,seed=85,stress=stress)
        for name,(a,b) in pol.items():
            m=evaluate(d,draws,a,b)
            t=d[KEY].copy(); t['stress']=stress;t['policy']=name;t['action']=a;t['buffer_units']=b
            for k,val in m.items(): t[k]=val
            rows.append(t)
    s=pd.concat(rows,ignore_index=True);s.to_csv(out/'scenarios.csv',index=False)
    assert len(s)==len(d)*12
    base=s[(s.stress==0)&(s.policy=='Optimised')]
    assert np.allclose(base.cost,z.cost) and np.allclose(base.stockout_probability,z.stockout_probability)
    print(f'Command centre ready: {len(d)} orders, {len(s)} order-policy-scenario rows.')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('input');a.add_argument('output');args=a.parse_args()
    try: command(args.input,args.output)
    except Exception as e: sys.exit(str(e))
