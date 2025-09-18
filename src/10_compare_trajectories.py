#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Novice vs Expert kinematic trajectory comparison (JIGSAWS)
- Input: two kinematics txt files (Novice, Expert) + column indices for tooltip xyz
- Output: side-by-side 3D trajectories + 2D density heatmaps (XY)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_txt(path: str) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim == 1: arr = arr[None, :]
    return arr

def parse_idx(s: str):
    vals = [int(v) for v in s.strip().split()]
    if len(vals) != 3:
        raise ValueError("xyz 인덱스는 공백 구분 3개 필요. 예) '10 11 12'")
    return vals

def center_scale(P: np.ndarray) -> np.ndarray:
    P = P - np.nanmean(P, axis=0, keepdims=True)
    std = np.nanstd(P, axis=0, keepdims=True) + 1e-6
    return P / std

def plot_compare(nov_xy, exp_xy, out_png):
    # —— Figure 1: 3D trajectories
    fig = plt.figure(figsize=(11, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1,1], wspace=0.15)
    ax1 = fig.add_subplot(gs[0,0], projection='3d')
    ax2 = fig.add_subplot(gs[0,1], projection='3d')
    for (p1,p2,ax,title) in [
        (nov_xy[0], nov_xy[1], ax1, "Novice 3D trajectory"),
        (exp_xy[0], exp_xy[1], ax2, "Expert 3D trajectory"),
    ]:
        ax.plot(p1[:,0], p1[:,1], p1[:,2], '-', lw=1.2, alpha=0.9, label='Tool 1')
        ax.plot(p2[:,0], p2[:,1], p2[:,2], '-', lw=1.2, alpha=0.9, label='Tool 2')
        ax.scatter(p1[0,0], p1[0,1], p1[0,2], s=20)  # start marker
        ax.scatter(p2[0,0], p2[0,1], p2[0,2], s=20)
        ax.set_title(title); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0,1.0))

    fig.tight_layout()
    fig.savefig(out_png.replace(".png", "_3d.png"), dpi=300)
    plt.close(fig)

    # —— Figure 2: 2D density (XY)
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for (p1,p2,ax,title) in [
        (nov_xy[0], nov_xy[1], axes[0], "Novice XY density"),
        (exp_xy[0], exp_xy[1], axes[1], "Expert XY density"),
    ]:
        allp = np.vstack([p1[:,:2], p2[:,:2]])
        ax.hexbin(allp[:,0], allp[:,1], gridsize=40, bins='log')
        ax.set_title(title); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png.replace(".png", "_xy_density.png"), dpi=300)
    plt.close(fig)

def main(args):
    nov = load_txt(args.novice)
    exp = load_txt(args.expert)
    i1 = parse_idx(args.psm1_xyz); i2 = parse_idx(args.psm2_xyz)

    nov_p1 = center_scale(nov[:, i1]); nov_p2 = center_scale(nov[:, i2])
    exp_p1 = center_scale(exp[:, i1]); exp_p2 = center_scale(exp[:, i2])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.novice).stem}_VS_{Path(args.expert).stem}"
    out_png = str(out / f"compare_{stem}.png")

    plot_compare((nov_p1, nov_p2), (exp_p1, exp_p2), out_png)
    print("Saved:",
          out_png.replace(".png","_3d.png"),
          out_png.replace(".png","_xy_density.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--novice", required=True)
    ap.add_argument("--expert", required=True)
    ap.add_argument("--psm1_xyz", required=True, help='"i j k" indices for PSM1 tooltip')
    ap.add_argument("--psm2_xyz", required=True, help='"i j k" indices for PSM2 tooltip')
    ap.add_argument("--out", default="out/compare")
    main(ap.parse_args())


