#!/usr/bin/env python3
"""Train a 3-day-ahead village flood-event model from explicitly village-level labels."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score, average_precision_score,
                             roc_auc_score, brier_score_loss, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_sample_weight
import joblib
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; MODEL=ROOT/'model'; MODEL.mkdir(exist_ok=True)
FEATURES=['latitude','longitude','elevation_m','month','rain_1d_mm','rain_3d_mm','rain_7d_mm','rain_14d_mm']

def main():
    weather_path=DATA/'village_daily_weather.csv'; labels_path=ROOT/'village_flood_labels.csv'
    if not weather_path.exists(): raise SystemExit('Missing data/village_daily_weather.csv. Run download_data.py first.')
    if not labels_path.exists(): raise SystemExit('NOT TRAINED: create village_flood_labels.csv from verified village-level records. See village_flood_labels_TEMPLATE.csv. IFI district labels alone are not valid village labels.')
    w=pd.read_csv(weather_path); l=pd.read_csv(labels_path)
    req={'village_id','date','flood_event','source','source_reference'}
    if req-set(l.columns): raise SystemExit(f'Labels need columns: {sorted(req)}')
    l['date']=pd.to_datetime(l.date,errors='coerce').dt.normalize(); w['date']=pd.to_datetime(w.date,errors='coerce').dt.normalize()
    l['flood_event']=pd.to_numeric(l.flood_event,errors='coerce')
    l=l.dropna(subset=['village_id','date','flood_event'])
    if not l['flood_event'].isin([0,1]).all(): raise SystemExit('flood_event must contain only 0 or 1.')
    if l.duplicated(['village_id','date']).any(): raise SystemExit('Duplicate village_id/date labels found; resolve them first.')
    if (l['source'].fillna('').str.strip()=='').any() or (l['source_reference'].fillna('').str.strip()=='').any():
        raise SystemExit('Every label row needs a source and source_reference. Do not use unverified or inferred labels.')
    w=w.sort_values(['village_id','date']).copy()
    # Rolling precipitation totals end on the feature date, not the future target window.
    w['precipitation_sum']=pd.to_numeric(w['precipitation_sum'],errors='coerce')
    for n in [1,3,7,14]:
        w[f'rain_{n}d_mm']=w.groupby('village_id')['precipitation_sum'].transform(lambda s:s.rolling(n,min_periods=n).sum())
    w['month']=w.date.dt.month
    # Future target: any verified event during the next 1–3 days. Require complete labels for all 3 future days.
    label_map={(str(r.village_id),r.date):int(r.flood_event) for r in l.itertuples()}
    ys=[]; complete=[]
    for r in w[['village_id','date']].itertuples(index=False):
        future=[label_map.get((str(r.village_id),r.date+pd.Timedelta(days=d))) for d in (1,2,3)]
        complete.append(all(v is not None for v in future))
        ys.append(int(any(v==1 for v in future)) if all(v is not None for v in future) else np.nan)
    w['target_flood_next_3d']=ys; w['labels_complete']=complete
    d=w[w.labels_complete & w.target_flood_next_3d.notna()].dropna(subset=['rain_14d_mm']).copy()
    if len(d)<100 or d.target_flood_next_3d.nunique()<2:
        raise SystemExit(f'NOT TRAINED: only {len(d)} complete labeled rows and {d.target_flood_next_3d.nunique()} target class(es). Need more verified positive and negative village-date labels across time.')
    # Chronological split, avoiding random day-level leakage.
    dates=sorted(d.date.unique()); split_idx=max(1,int(len(dates)*0.8)); cutoff=pd.Timestamp(dates[split_idx]) if split_idx<len(dates) else pd.Timestamp(dates[-1])
    train=d[d.date<cutoff]; test=d[d.date>=cutoff]
    if len(test)<10 or train.target_flood_next_3d.nunique()<2 or test.target_flood_next_3d.nunique()<2:
        raise SystemExit(f'NOT TRAINED: temporal split lacks enough examples/classes. Train rows={len(train)}, test rows={len(test)}. Add more locations/seasons/labels.')
    model=Pipeline([('imputer',SimpleImputer(strategy='median')),('rf',RandomForestClassifier(n_estimators=500,min_samples_leaf=3,class_weight='balanced_subsample',random_state=42,n_jobs=-1,max_features='sqrt'))])
    model.fit(train[FEATURES],train.target_flood_next_3d.astype(int),rf__sample_weight=compute_sample_weight('balanced',train.target_flood_next_3d.astype(int)))
    proba=model.predict_proba(test[FEATURES])[:,list(model.named_steps['rf'].classes_).index(1)]
    pred=(proba>=0.5).astype(int); y=test.target_flood_next_3d.astype(int).to_numpy()
    metrics={'status':'trained_research_prototype','target':'verified village flood event within next 1–3 days','n_total_complete_rows':int(len(d)),'n_train':int(len(train)),'n_test':int(len(test)),'n_villages':int(d.village_id.nunique()),'train_date_end':str(train.date.max().date()),'test_date_start':str(test.date.min().date()),'positive_rate_all':float(d.target_flood_next_3d.mean()),'positive_rate_test':float(y.mean()),'threshold':0.5,'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'average_precision':float(average_precision_score(y,proba)),'brier_score':float(brier_score_loss(y,proba)),'confusion_matrix_tn_fp_fn_tp':confusion_matrix(y,pred,labels=[0,1]).ravel().tolist(),'features':FEATURES,'label_source_note':'Requires village-level verified labels; district inventory is not converted into village labels.'}
    try: metrics['roc_auc']=float(roc_auc_score(y,proba))
    except Exception: metrics['roc_auc']=None
    joblib.dump(model,MODEL/'floodguard_rf.joblib')
    (MODEL/'metadata.json').write_text(json.dumps({'features':FEATURES,'metrics':metrics,'model':'RandomForestClassifier','version':1},indent=2),encoding='utf-8')
    pd.DataFrame({'date':test.date.astype(str),'village_id':test.village_id,'actual':y,'probability':proba,'predicted_at_0_5':pred}).to_csv(MODEL/'holdout_predictions.csv',index=False)
    (MODEL/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2)); print(f'\nModel saved: {MODEL/"floodguard_rf.joblib"}')
if __name__=='__main__': main()
