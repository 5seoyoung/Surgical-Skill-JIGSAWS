
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_json(path):
    with open(path) as f:
        return json.load(f)

def plot_subject_bar(folds, out_dir, model_name="model"):
    df = pd.DataFrame(folds)
    if "subject" not in df.columns:
        raise ValueError(f"No 'subject' in folds for {model_name}")
    ax = df.plot(x="subject", y=["acc","f1"], kind="bar", rot=0)
    plt.title(f"{model_name} per-subject performance")
    plt.ylabel("score"); plt.ylim(0,1); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{model_name}_per_subject.png")); plt.close()

def save_table(res, out_dir, model_name="model"):
    folds = res["folds"]; df = pd.DataFrame(folds)
    mean = res.get("mean", {})
    if mean:
        df.loc[len(df)] = {"subject":"mean","acc":mean.get("acc",None),"f1":mean.get("f1",None)}
    df.to_csv(os.path.join(out_dir, f"{model_name}_results.csv"), index=False)
    try:
        with open(os.path.join(out_dir, f"{model_name}_results.md"), "w") as f:
            f.write(df.to_markdown(index=False))
    except Exception as e:
        print(f"[INFO] skip markdown ({e})")

def plot_confusion_from_csv(csv_path, out_dir, model_name="model"):
    df = pd.read_csv(csv_path)
    y_true = df["true"].values
    y_pred = df["pred"].values
    labels = [0,1,2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots()
    im = ax.imshow(cmn, vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["Novice","Interm.","Expert"])
    ax.set_yticklabels(["Novice","Interm.","Expert"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]*100:.1f}%", ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name} Confusion Matrix (Normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{model_name}_cm.png")); plt.close()

def main(args):
    os.makedirs(args.out, exist_ok=True)

    # ML
    ml_json = load_json(os.path.join(args.ml, "ml_results_loso.json"))
    plot_subject_bar(ml_json["xgb"]["folds"], args.out, model_name="XGB")
    save_table(ml_json["xgb"], args.out, model_name="XGB")
    plot_confusion_from_csv(os.path.join(args.ml, "preds_xgb.csv"), args.out, model_name="XGB")

    # DL
    dl_json = load_json(os.path.join(args.dl, "dl_results_loso.json"))
    plot_subject_bar(dl_json["lstm_loso"]["folds"], args.out, model_name="LSTM")
    save_table(dl_json["lstm_loso"], args.out, model_name="LSTM")
    plot_confusion_from_csv(os.path.join(args.dl, "preds_lstm.csv"), args.out, model_name="LSTM")

    print(f"Figures and tables saved under {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml", default="out/ml_skill_stride1_subjid")
    ap.add_argument("--dl", default="out/dl/lstm_skill_stride1_subjid_fixed")
    ap.add_argument("--out", default="out/figs")
    args = ap.parse_args()
    main(args)
