#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_summary_figure.py
- Ablation, Feature Importance, Confusion Matrix (XGB, LSTM) 하나의 Figure로 정리
"""

import argparse, os, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_ablation(ax, ablation_csv):
    df = pd.read_csv(ablation_csv)
    df = df.set_index("config")
    df[["acc","f1"]].plot(kind="bar", ax=ax, rot=0)
    ax.set_ylim(0,1)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study (XGB)")

def plot_featimp(ax, featimp_csv):
    df = pd.read_csv(featimp_csv).sort_values("pi_mean")
    sns.barplot(data=df, y="feature", x="pi_mean", ax=ax, color="steelblue")
    ax.set_xlabel("Permutation Importance (macro-F1 drop)")
    ax.set_ylabel("")
    ax.set_title("XGB Feature Importance")

def plot_cm(ax, csv_path, model_name):
    df = pd.read_csv(csv_path)
    y_true, y_pred = df["true"], df["pred"]
    labels = [0,1,2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Novice","Interm.","Expert"],
                yticklabels=["Novice","Interm.","Expert"],
                ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name} Confusion Matrix")

def main(args):
    fig, axes = plt.subplots(2,2, figsize=(12,10))

    # 1) Ablation
    plot_ablation(axes[0,0], os.path.join(args.ablation,"ablation_summary.csv"))

    # 2) Feature importance
    plot_featimp(axes[0,1], os.path.join(args.featimp,"xgb_permutation_importance.csv"))

    # 3) XGB confusion matrix
    plot_cm(axes[1,0], os.path.join(args.ml,"preds_xgb.csv"), "XGB")

    # 4) LSTM confusion matrix
    plot_cm(axes[1,1], os.path.join(args.dl,"preds_lstm.csv"), "LSTM")

    plt.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out,"summary_figure.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved summary figure: {out_path}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ablation", default="out/ablation")
    ap.add_argument("--featimp", default="out/featimp")
    ap.add_argument("--ml", default="out/ml_skill_stride1_subjid_exp")
    ap.add_argument("--dl", default="out/dl/lstm_skill_stride1_subjid_fixed")
    ap.add_argument("--out", default="out/figs")
    args = ap.parse_args()
    main(args)
