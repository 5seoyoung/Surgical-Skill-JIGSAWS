#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature distribution plots for Nov/Int/Exp:
- Box + jitter, Violin + jitter (matplotlib only)
- Features: jerk_mean, bp_3_6, path_len  (column names in your parquet)
- Outputs: PNGs + CSV (group stats) + TXT (p-values)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal

LABELS = {0:"Novice", 1:"Intermediate", 2:"Expert"}
ORDER  = [0,1,2]  # plotting order

FEATS = [
    ("jerk_mean", "Jerk mean (a.u.)"),
    ("bp_3_6",    "Bandpower 3–6 Hz (a.u.)"),
    ("path_len",  "Path length (a.u.)"),
]

def _group_arrays(df, col):
    return [df[df["y"]==k][col].dropna().to_numpy() for k in ORDER]

def _save_stats(df, out_dir):
    rows=[]
    for feat,_lab in FEATS:
        for k in ORDER:
            x = df[df["y"]==k][feat].dropna()
            rows.append({
                "feature": feat,
                "class": LABELS[k],
                "n": int(x.shape[0]),
                "mean": float(x.mean()),
                "std": float(x.std(ddof=1)),
                "median": float(x.median()),
                "q25": float(x.quantile(0.25)),
                "q75": float(x.quantile(0.75)),
            })
    pd.DataFrame(rows).to_csv(out_dir/"feature_group_stats.csv", index=False)

def _save_tests(df, out_dir):
    lines=[]
    for feat,_lab in FEATS:
        groups = _group_arrays(df, feat)
        stat, p = kruskal(*groups)  # 비모수 분산분석(집단 3개 이상)
        lines.append(f"{feat}: Kruskal–Wallis H={stat:.3f}, p={p:.3e}")
    (out_dir/"feature_kruskal_pvalues.txt").write_text("\n".join(lines))

def _jitter(n, scale=0.06, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    return rng.normal(0, scale, n)

def plot_box(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (feat, lab) in zip(axes, FEATS):
        data = _group_arrays(df, feat)
        bp = ax.boxplot(data, labels=[LABELS[k] for k in ORDER],
                        showmeans=False, patch_artist=True)
        # 박스 색 약간
        colors = ["#c7d9f1", "#cdeccf", "#f9d4c7"]
        for patch,c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.9)
        # 지터 점
        for i,k in enumerate(ORDER, start=1):
            y = df[df["y"]==k][feat].dropna().to_numpy()
            x = np.full_like(y, i, dtype=float) + _jitter(len(y), rng=rng)
            ax.plot(x, y, "o", ms=2.8, alpha=0.55)
        ax.set_title(lab, fontsize=11)
        ax.grid(alpha=0.25, linestyle=":")
    fig.suptitle("Feature distributions by skill (Box + jitter)", fontsize=12)
    fig.savefig(out_dir/"feat_box_jitter.png", dpi=300)
    plt.close(fig)

def plot_violin(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    rng = np.random.default_rng(1)
    for ax, (feat, lab) in zip(axes, FEATS):
        data = _group_arrays(df, feat)
        parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.9)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        # 중앙선 스타일
        parts['cmedians'].set_linewidth(1.2)
        # 지터 점
        pos = np.arange(1, len(ORDER)+1)
        for i, yvals in enumerate(data, start=1):
            x = np.full_like(yvals, i, dtype=float) + _jitter(len(yvals), rng=rng)
            ax.plot(x, yvals, "o", ms=2.4, alpha=0.45)
        ax.set_xticks(pos, [LABELS[k] for k in ORDER])
        ax.set_title(lab, fontsize=11)
        ax.grid(alpha=0.25, linestyle=":")
    fig.suptitle("Feature distributions by skill (Violin + jitter)", fontsize=12)
    fig.savefig(out_dir/"feat_violin_jitter.png", dpi=300)
    plt.close(fig)

def main(args):
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.features)
    # 안전장치: y라벨 확인
    if "y" not in df.columns:
        raise RuntimeError("column 'y' not found. Expected 0/1/2 labels for Nov/Int/Exp.")
    # 필요한 컬럼 체크
    miss = [c for c,_ in FEATS if c not in df.columns]
    if miss:
        raise RuntimeError(f"missing feature columns: {miss}")

    _save_stats(df, out_dir)
    _save_tests(df, out_dir)
    plot_box(df, out_dir)
    plot_violin(df, out_dir)
    print(f"Saved under: {out_dir}")
    print(f"- PNG: feat_box_jitter.png, feat_violin_jitter.png")
    print(f"- CSV: feature_group_stats.csv")
    print(f"- TXT: feature_kruskal_pvalues.txt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="out/features_skill_stride1_subjid_exp.parquet")
    ap.add_argument("--out", default="out/featdist")
    main(ap.parse_args())
