
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

GLOBAL_CLASSES = np.array([0,1,2])  # 0=Nov,1=Int,2=Exp

def evaluate(y_true, y_pred):
    return dict(acc=float(accuracy_score(y_true,y_pred)),
                f1=float(f1_score(y_true,y_pred,average="macro")))

def fit_xgb_fold(X_tr, y_tr):
    # 훈련에 등장한 클래스만으로 임시 리맵
    uniq = np.unique(y_tr)
    remap = {c:i for i,c in enumerate(uniq)}           # ex) {1:0, 2:1}
    y_tr_map = np.vectorize(remap.get)(y_tr)
    num_class = len(uniq)

    xgb = XGBClassifier(
        objective="multi:softprob" if num_class>2 else "binary:logistic",
        num_class=num_class if num_class>2 else None,
        max_depth=3, n_estimators=400, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, n_jobs=2, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="mlogloss"
    )
    xgb.fit(X_tr, y_tr_map)
    return xgb, remap, uniq

def predict_xgb_global(model, remap, uniq, X_te):
    if model.get_params()["objective"] == "binary:logistic":
        p1 = model.predict_proba(X_te)[:,1:2]
        proba_local = np.hstack([1.0 - p1, p1])
    else:
        proba_local = model.predict_proba(X_te)

    N = proba_local.shape[0]
    proba_global = np.full((N, len(GLOBAL_CLASSES)), 1e-9, dtype=float)
    inv_remap = {v:k for k,v in remap.items()}  # ex) {0:1,1:2}
    for local_idx, global_cls in inv_remap.items():
        global_pos = int(np.where(GLOBAL_CLASSES==global_cls)[0][0])
        proba_global[:, global_pos] = proba_local[:, local_idx]
    y_pred = proba_global.argmax(axis=1)
    return y_pred

def main(args):
    df = pd.read_parquet(args.feat) if args.feat.endswith(".parquet") else pd.read_csv(args.feat)
    meta_cols = [c for c in ["task","subj","trial","skill","y"] if c in df.columns]
    X = df[[c for c in df.columns if c not in meta_cols]].values
    y = (df["y"].values if "y" in df.columns else df["skill"].values).astype(int)
    subj = df["subj"].values

    subjects = sorted(np.unique(subj))
    res = {"svm":{"folds":[]}, "xgb":{"folds":[]}}
    svm_pred_rows = []
    xgb_pred_rows = []

    for s in subjects:
        te = (subj == s)
        tr = ~te
        if te.sum()==0 or tr.sum()==0:
            continue
        te_idx = np.where(te)[0]

        # --- SVM ---
        svm=SVC(kernel="rbf", C=3, gamma="scale", class_weight="balanced", probability=False)
        svm.fit(X[tr], y[tr])
        svm_yhat = svm.predict(X[te])
        mets = evaluate(y[te], svm_yhat)
        res["svm"]["folds"].append({"subject":str(s), **mets})
        for idx, yhat in zip(te_idx, svm_yhat):
            svm_pred_rows.append({
                "subject": str(s),
                "trial": str(df.iloc[idx]["trial"]),
                "true": int(y[idx]),
                "pred": int(yhat)
            })

        # --- XGB ---
        xgb, remap, uniq = fit_xgb_fold(X[tr], y[tr])
        xgb_yhat = predict_xgb_global(xgb, remap, uniq, X[te])
        mets = evaluate(y[te], xgb_yhat)
        res["xgb"]["folds"].append({"subject":str(s), **mets})
        for idx, yhat in zip(te_idx, xgb_yhat):
            xgb_pred_rows.append({
                "subject": str(s),
                "trial": str(df.iloc[idx]["trial"]),
                "true": int(y[idx]),
                "pred": int(yhat)
            })

    for k in ["svm","xgb"]:
        accs=[f["acc"] for f in res[k]["folds"]]
        f1s =[f["f1"]  for f in res[k]["folds"]]
        res[k]["mean"]={"acc":float(np.mean(accs) if accs else 0.0),
                        "f1" :float(np.mean(f1s)  if f1s  else 0.0)}
    print(json.dumps(res, indent=2))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(f"{args.out}/ml_results_loso.json","w") as f: json.dump(res,f,indent=2)
    pd.DataFrame(svm_pred_rows).to_csv(f"{args.out}/preds_svm.csv", index=False)
    pd.DataFrame(xgb_pred_rows).to_csv(f"{args.out}/preds_xgb.csv", index=False)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--feat", default="out/features_skill_stride1_subjid.parquet")
    ap.add_argument("--out", default="out/ml_skill_stride1_subjid")
    main(ap.parse_args())
