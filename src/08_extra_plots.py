#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_extra_plots.py
- Feature distributions (box & violin)
- Dataset composition (skill pie + task bar)
- Ablation summary table (markdown + PNG)
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def feature_distributions(df, outdir):
    labs = df["y"].map({0:"Novice",1:"Intermediate",2:"Expert"})
    feats = [("jerk_mean","Jerk (mean)"),
             ("bp_3_6","Bandpower 3–6 Hz"),
             ("path_len","Path length")]

    # Boxplots (3개 가로)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax,(col,title) in zip(axes, feats):
        groups = [df.loc[df["y"]==k, col].values for k in [0,1,2]]
        ax.boxplot(groups, labels=["Nov","Int","Exp"], showfliers=False)
        ax.set_title(title); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{outdir}/feature_boxplots.png", dpi=300)
    plt.close(fig)

    # Violinplots (3개 가로)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax,(col,title) in zip(axes, feats):
        vals = [df.loc[df["y"]==k, col].values for k in [0,1,2]]
        parts = ax.violinplot(vals, showmeans=True, showextrema=False)
        ax.set_xticks([1,2,3]); ax.set_xticklabels(["Nov","Int","Exp"])
        ax.set_title(title); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{outdir}/feature_violins.png", dpi=300)
    plt.close(fig)

def dataset_composition(df, outdir):
    # Skill 분포
    skill_map = {0:"Novice",1:"Intermediate",2:"Expert"}
    skill_counts = df["y"].map(skill_map).value_counts().reindex(["Novice","Intermediate","Expert"]).fillna(0)
    # Task 분포 (열 이름: task)
    task_counts = df["task"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Pie
    axes[0].pie(skill_counts.values, labels=skill_counts.index, autopct="%1.0f%%", startangle=90)
    axes[0].set_title("Skill level distribution (N={})".format(int(skill_counts.sum())))
    # Bar
    axes[1].bar(task_counts.index, task_counts.values)
    axes[1].set_title("Task composition")
    axes[1].set_ylabel("Samples")
    axes[1].grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{outdir}/dataset_composition.png", dpi=300)
    plt.close(fig)

def ablation_table(ablation_csv, outdir):
    df = pd.read_csv(ablation_csv)
    # 정렬: full, no_jerk, no_band, no_corr 순서 맞추기
    order = ["full","no_jerk","no_band","no_corr"]
    df["config"] = pd.Categorical(df["config"], order)
    df = df.sort_values("config")
    # Markdown 저장
    md = df.to_markdown(index=False)
    with open(f"{outdir}/ablation_table.md","w") as f:
        f.write(md)

    # PNG 테이블로 저장 (matplotlib table)
    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.axis("off")
    tbl = ax.table(cellText=np.round(df[["acc","f1"]].values,3),
                   colLabels=["acc","f1"],
                   rowLabels=df["config"].tolist(),
                   loc="center", cellLoc="center")
    tbl.scale(1, 1.4)
    ax.set_title("Ablation summary (LOSO)", pad=8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/ablation_table.png", dpi=300)
    plt.close(fig)

def main(args):
    ensure_dir(args.out)
    df = pd.read_parquet(args.feat)
    feature_distributions(df, args.out)
    dataset_composition(df, args.out)
    ablation_table(args.ablation_csv, args.out)
    print("Saved:",
          f"{args.out}/feature_boxplots.png,",
          f"{args.out}/feature_violins.png,",
          f"{args.out}/dataset_composition.png,",
          f"{args.out}/ablation_table.md,",
          f"{args.out}/ablation_table.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="out/features_skill_stride1_subjid_exp.parquet")
    ap.add_argument("--ablation_csv", default="out/ablation/ablation_summary.csv")
    ap.add_argument("--out", default="out/figs")
    main(ap.parse_args())
