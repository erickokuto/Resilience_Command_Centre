"""Session-only model refresh from ONE authorised order-level CSV.

Training, validation and test are partitions of the labelled records in the upload.
Unlabelled (open) orders are scored, not used to fit or evaluate the models.
No model is written into the application installation. This protects the approved
historical models and avoids accidental cross-user state on a hosted Shiny server.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, brier_score_loss, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from run_batch import validate  # original application schema + integrity checks
from Predictive_Modelling.phase3 import CAT, NUM, features

KEYS = ['po_id', 'item_id']
SEED = 20261123


def create_model(classification=False):
    numeric = NUM + ['month_sin', 'month_cos', 'weekday']
    return Pipeline([
        ('transform', ColumnTransformer([
            ('numeric', StandardScaler(), numeric),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), CAT),
        ])),
        ('model', LogisticRegression(max_iter=1200, class_weight=None, random_state=SEED)
         if classification else Ridge(alpha=10.0)),
    ])


def split_positions(labels, train_pct, validation_pct, method):
    n = len(labels)
    counts = [int(np.floor(n * train_pct / 100)),
              int(np.floor(n * validation_pct / 100))]
    counts.append(n - sum(counts))
    if min(counts) < 10:
        raise ValueError('Each labelled partition requires at least 10 orders; adjust percentages or upload more completed orders.')
    if method == 'chronological':
        order = np.argsort(pd.to_datetime(labels['order_date']).to_numpy(), kind='stable')
    else:
        order = np.random.default_rng(SEED).permutation(n)
    return order[:counts[0]], order[counts[0]:sum(counts[:2])], order[sum(counts[:2]):]


def quality_scores(y, reg, p):
    result = {'n': len(y), 'MAE_days': float(mean_absolute_error(y['actual_lead_days'], reg)),
              'RMSE_days': float(np.sqrt(mean_squared_error(y['actual_lead_days'], reg))),
              'Brier': float(brier_score_loss(y['late_flag'].astype(int), p)),
              'Accuracy_50pct': float(accuracy_score(y['late_flag'].astype(int), p >= .5))}
    result['AUC'] = (float(roc_auc_score(y['late_flag'].astype(int), p))
                     if y['late_flag'].nunique() == 2 else np.nan)
    return result


def run(csv, output, train_pct, validation_pct, method):
    if not (10 <= train_pct <= 80 and 10 <= validation_pct <= 80
            and 10 <= 100-train_pct-validation_pct <= 80):
        raise ValueError('Training, validation and testing must each be 10–80% and add to 100%.')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    data = validate(csv)
    labelled = data.loc[data['late_flag'].notna() & data['actual_lead_days'].notna()].copy()
    if len(labelled) < 60:
        raise ValueError(f'Found {len(labelled)} completed orders. At least 60 labelled orders are required to retrain; open orders are predictions only.')
    train_idx, val_idx, test_idx = split_positions(labelled, train_pct, validation_pct, method)
    pieces = [labelled.iloc[idx].copy() for idx in (train_idx,val_idx,test_idx)]
    train, valid, test = pieces
    # Regression is fitted only on training records. Categorical encoding and
    # scaling are also fitted only on training to avoid information leakage.
    if train['late_flag'].nunique() < 2:
        raise ValueError('Training subset contains only one late-delivery class; change the split or upload more balanced observations.')
    model_reg, model_cls = create_model(), create_model(True)
    model_reg.fit(features(train), train['actual_lead_days'].astype(float))
    model_cls.fit(features(train), train['late_flag'].astype(int))

    def pred(df):
        x = features(df)
        lead = np.maximum(0.0, model_reg.predict(x))
        risk = np.clip(model_cls.predict_proba(x)[:,1], 0, 1)
        return lead, risk

    # Interval widths are empirically calculated from independent validation
    # residuals (symmetric, global split-conformal approximation).
    v_lead, v_prob = pred(valid)
    errors = np.abs(valid['actual_lead_days'].to_numpy() - v_lead)
    # Conservative finite-sample quantile; no interval guarantee under drift.
    def radius(coverage):
        rank = min(len(errors), int(np.ceil((len(errors)+1)*coverage)))
        return float(np.partition(errors, rank-1)[rank-1])
    rad80, rad90 = radius(.8), radius(.9)
    rows=[]
    for name, subset in zip(('Training','Validation','Test'),pieces):
        a,b = pred(subset)
        rows.append({'Partition':name,**quality_scores(subset,a,b)})
    pd.DataFrame(rows).to_csv(out/'adaptive_metrics.csv',index=False)

    all_reg, all_prob = pred(data)
    scored = data[KEYS+['supplier_id','plant_id','transport_mode','order_date',
                       'planned_lead_days','inventory_on_hand','daily_demand',
                       'actual_lead_days','late_flag']].copy()
    scored['adaptive_predicted_lead_days'] = np.round(all_reg,3)
    scored['adaptive_late_probability'] = np.round(all_prob,6)
    for pct,r in [(80,rad80),(90,rad90)]:
        scored[f'adaptive_lower_{pct}'] = np.maximum(0,all_reg-r).round(3)
        scored[f'adaptive_upper_{pct}'] = (all_reg+r).round(3)
    scored['risk_priority'] = pd.cut(all_prob, bins=[-0.000001,.4,.75,1.000001],
                                     labels=['Low','Review','High'],right=False)
    scored['available_inventory_days'] = (data['inventory_on_hand']/data['daily_demand']).round(2)
    scored['indicative_next_step'] = np.select(
        [(all_prob>=.75), (all_prob>=.4), (all_reg>scored['available_inventory_days'].to_numpy())],
        ['Urgent human review of supplier and buffer options',
         'Review delivery risk and contingency options',
         'Check inventory coverage and lead-time assumptions'],
        default='Routine monitoring')
    scored['partition'] = 'Open / unlabelled'
    for tag, subset in zip(('Training','Validation','Test'),pieces):
        keys = pd.MultiIndex.from_frame(subset[KEYS].astype(str))
        ix = pd.MultiIndex.from_frame(scored[KEYS].astype(str)).isin(keys)
        scored.loc[ix,'partition'] = tag
    scored['model_version'] = 'Session retrain (not deployed)'
    scored.to_csv(out/'adaptive_predictions.csv',index=False)
    counts = {'total_rows':len(data),'labelled_rows':len(labelled),
              'unlabelled_rows':int(len(data)-len(labelled)),
              'training':len(train),'validation':len(valid),'test':len(test),
              'requested_train_percent':train_pct,
              'requested_validation_percent':validation_pct,
              'realised_test_percent':100-train_pct-validation_pct,
              'split_method':method,'validation_radius80_days':rad80,
              'validation_radius90_days':rad90,
              'warning':'Session-only candidate model. Historic optimisation/scenario outputs remain based on frozen models. Avoid interpreting training metrics as independent evaluation.'}
    (out/'adaptive_summary.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(f"Retrained on {len(train)} labelled orders; validation {len(valid)}, test {len(test)}; scored {len(data)} rows ({len(data)-len(labelled)} open).")


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('output_dir')
    parser.add_argument('--train',type=int,default=70)
    parser.add_argument('--validation',type=int,default=15)
    parser.add_argument('--method',choices=['chronological','random'],default='chronological')
    a=parser.parse_args()
    try:
        run(a.input_csv,a.output_dir,a.train,a.validation,a.method)
    except Exception as e:
        sys.exit('Adaptive workflow rejected the data: '+str(e))
