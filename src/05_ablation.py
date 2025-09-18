#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_ablation.py
- 확장 피처 중 특정 그룹을 제외한 버전으로 XGB LOSO 실행
- 결과 JSON + CSV 저장
"""

import argparse, json, numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

GLOBAL_CLASSES = np.array([0,1,2])

def evaluate(y_true, y_pred):
    return dict(acc=float(accuracy_score(y_true,y_pred)),
                f1=float(f1_score(y_true,y_pred,average="macro")))

def fit_xgb(X_tr, y_tr):
    uniq = np.unique(y_tr)
    remap = {c:i for i,c in enumerate(uniq)}
    y_tr_map = np.vectorize(remap.get)(y_tr)
    num_class = len(uniq)
    xgb = XGBClassifier(
        objective="multi:softprob" if num_class>2 else "binary:logistic",
        num_class=num_class if num_class>2 else None,
        max_depth=3, n_estimators=400, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, n_jobs=2,
        reg_lambda=1.0, use_label_encoder=False, eval_metric="mlogloss"
    )
    xgb.fit(X_tr, y_tr_map)
    return xgb, remap, uniq

def predict_xgb(model, remap, uniq, X_te):
    if model.get_params()["objective"] == "binary:logistic":
        p1 = model.predict_proba(X_te)[:,1:2]
        proba_local = np.hstack([1.0 - p1, p1])
    else:
        proba_local = model.predict_proba(X_te)
    N = proba_local.shape[0]
    proba_global = np.full((N,len(GLOBAL_CLASSES)),1e-9,dtype=float)
    inv_remap = {v:k for k,v in remap.items()}
    for li, gc in inv_remap.items():
        gp = int(np.where(GLOBAL_CLASSES==gc)[0][0])
        proba_global[:,gp] = proba_local[:,li]
    return proba_global.argmax(axis=1)

def run_loso(df, feat_cols):
    X = df[feat_cols].values
    y = df["y"].values.astype(int)
    subj = df["subj"].values
    subjects = sorted(np.unique(subj))

    folds=[]
    for s in subjects:
        te = (subj == s); tr = ~te
        if te.sum()==0 or tr.sum()==0: continue
        xgb, remap, uniq = fit_xgb(X[tr], y[tr])
        yhat = predict_xgb(xgb, remap, uniq, X[te])
        mets = evaluate(y[te], yhat)
        folds.append({"subject":str(s), **mets})

    accs=[f["acc"] for f in folds]; f1s=[f["f1"] for f in folds]
    return {"folds":folds, "mean":{"acc":float(np.mean(accs)),"f1":float(np.mean(f1s))}}

def main(args):
    df = pd.read_parquet(args.feat)
    base_feats = [
        "path_len","straight","mean_v","max_v","pause","fft_energy",
        "jerk_mean","jerk_std","bimanual_corr","bp_0_3","bp_3_6","disp_std"
    ]

    configs = {
        "full": base_feats,
        "no_jerk": [f for f in base_feats if not f.startswith("jerk")],
        "no_band": [f for f in base_feats if not f.startswith("bp_")],
        "no_corr": [f for f in base_feats if f!="bimanual_corr"],
    }

    res={}
    for name, cols in configs.items():
        res[name] = run_loso(df, cols)
        print(f"[{name}] acc={res[name]['mean']['acc']:.3f} f1={res[name]['mean']['f1']:.3f}")

    import os
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/ablation.json","w") as f: json.dump(res,f,indent=2)
    pd.DataFrame([{**{"config":k}, **v["mean"]} for k,v in res.items()])\
      .to_csv(f"{args.out}/ablation_summary.csv", index=False)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--feat", default="out/features_skill_stride1_subjid_exp.parquet")
    ap.add_argument("--out", default="out/ablation")
    main(ap.parse_args())
