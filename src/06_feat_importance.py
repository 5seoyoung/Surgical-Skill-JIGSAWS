#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_feat_importance.py
- XGB 피처 중요도 (gain/weight) 시각화
"""

import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from xgboost import XGBClassifier

def main(args):
    df = pd.read_parquet(args.feat)
    feat_cols = [
        "path_len","straight","mean_v","max_v","pause","fft_energy",
        "jerk_mean","jerk_std","bimanual_corr","bp_0_3","bp_3_6","disp_std"
    ]
    X = df[feat_cols].values
    y = df["y"].values.astype(int)

    model = XGBClassifier(
        objective="multi:softprob", num_class=3,
        max_depth=3, n_estimators=400, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, n_jobs=2,
        reg_lambda=1.0, use_label_encoder=False, eval_metric="mlogloss"
    )
    model.fit(X,y)

    imp_gain = model.get_booster().get_score(importance_type="gain")
    imp_weight = model.get_booster().get_score(importance_type="weight")

    # importance 정리
    df_imp = pd.DataFrame({
        "feature": feat_cols,
        "gain": [imp_gain.get(f"f{i}",0.0) for i in range(len(feat_cols))],
        "weight": [imp_weight.get(f"f{i}",0.0) for i in range(len(feat_cols))]
    })
    df_imp = df_imp.sort_values("gain", ascending=False)

    # plot
    df_imp.plot(x="feature", y="gain", kind="barh", legend=False)
    plt.title("XGB Feature Importance (gain)")
    plt.tight_layout()
    plt.savefig(f"{args.out}/xgb_feat_importance_gain.png"); plt.close()

    df_imp.plot(x="feature", y="weight", kind="barh", legend=False)
    plt.title("XGB Feature Importance (weight)")
    plt.tight_layout()
    plt.savefig(f"{args.out}/xgb_feat_importance_weight.png"); plt.close()

    df_imp.to_csv(f"{args.out}/xgb_feat_importance.csv", index=False)
    print(f"Saved feature importance under {args.out}")

if __name__=="__main__":
    import os
    ap=argparse.ArgumentParser()
    ap.add_argument("--feat", default="out/features_skill_stride1_subjid_exp.parquet")
    ap.add_argument("--out", default="out/featimp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    main(args)
