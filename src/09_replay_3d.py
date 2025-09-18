#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JIGSAWS kinematics 3D replay (two tooltips)
- Saves PNG (key frame) + MP4 (if ffmpeg) or GIF fallback
Usage:
  python src/09_replay_3d.py \
    --file data/JIGSAWS/Knot-Tying/kinematics/AllGestures/Knot_Tying_C003.txt \
    --psm1_xyz "10 11 12" --psm2_xyz "29 30 31" --skip 2 --out out/replay
"""
import argparse, os, shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

def load_kinematics(path):
    # 공백 구분 텍스트, 숫자만 읽기
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim == 1:  # 안전장치
        arr = arr[None, :]
    return arr

def norm_xyz(x):
    # 중앙정렬 + 표준화(스케일 조절)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True) + 1e-6
    return x / std

def draw_task_board(ax, task_hint: str):
    """간단 작업판(원판/바/펀칭패드) 그리기"""
    task_hint = task_hint.lower()
    if "knot" in task_hint:
        th = np.linspace(0, 2*np.pi, 240)
        R = 1.0
        ax.plot(R*np.cos(th), R*np.sin(th), 0*th, lw=1.0, alpha=0.4, color="gray")
    elif "needle" in task_hint:
        # 길쭉한 바
        ax.plot([-1.2, 1.2], [0, 0], [0, 0], lw=6, alpha=0.25, color="gray")
    elif "sutur" in task_hint:
        # 3x3 펀칭 패드
        xs = [-0.8, 0.0, 0.8]
        ys = [-0.8, 0.0, 0.8]
        th = np.linspace(0, 2*np.pi, 80)
        for x in xs:
            for y in ys:
                ax.plot(0.15*np.cos(th)+x, 0.15*np.sin(th)+y, 0*th, lw=0.8, alpha=0.3, color="gray")
    else:
        # 기본 바닥 격자
        for g in np.linspace(-1.5, 1.5, 7):
            ax.plot([-1.5, 1.5], [g, g], [0,0], lw=0.5, alpha=0.15, color="gray")
            ax.plot([g, g], [-1.5, 1.5], [0,0], lw=0.5, alpha=0.15, color="gray")

def parse_idx(s):
    vals = [int(v) for v in s.strip().split()]
    if len(vals) != 3:
        raise ValueError("--psm*_xyz 는 공백으로 구분된 3개 인덱스여야 합니다. 예) \"10 11 12\"")
    return vals

def main(args):
    path = Path(args.file)
    arr = load_kinematics(path)
    i1 = parse_idx(args.psm1_xyz)
    i2 = parse_idx(args.psm2_xyz)

    # 인덱스 범위 체크
    C = arr.shape[1]
    for i in i1 + i2:
        if i < 0 or i >= C:
            raise IndexError(f"column index {i} is out of range (0..{C-1})")

    p1 = arr[:, i1].astype(float)  # [T,3]
    p2 = arr[:, i2].astype(float)

    # NaN 방지
    if np.isnan(p1).any() or np.isnan(p2).any():
        # NaN을 선형보간
        for P in (p1, p2):
            for d in range(3):
                col = P[:, d]
                nans = np.isnan(col)
                if nans.any():
                    idx = np.arange(len(col))
                    col[nans] = np.interp(idx[nans], idx[~nans], col[~nans])
                P[:, d] = col

    # 정규화
    p1 = norm_xyz(p1)
    p2 = norm_xyz(p2)

    T = len(p1)
    step = max(1, int(args.skip))
    frames = list(range(0, T, step))
    if len(frames) == 0:
        frames = [0]

    # 3D Figure
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(args.title or path.stem)

    # 작업판
    draw_task_board(ax, path.stem)

    # 툴팁 + 잔상
    tip1, = ax.plot([], [], [], "o", ms=6)
    tip2, = ax.plot([], [], [], "o", ms=6)
    trail1, = ax.plot([], [], [], "-", lw=1.2, alpha=0.7)
    trail2, = ax.plot([], [], [], "-", lw=1.2, alpha=0.7)

    # 범위 자동 설정
    allp = np.vstack([p1, p2])
    m = np.nanmean(allp, axis=0)
    s = np.nanstd(allp, axis=0) * 3.0 + 1e-3
    ax.set_xlim(m[0]-s[0], m[0]+s[0])
    ax.set_ylim(m[1]-s[1], m[1]+s[1])
    ax.set_zlim(m[2]-s[2], m[2]+s[2])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    trail_len = max(30, int(2.0 / (args.dt or 0.02)))  # 최근 2초 잔상(대략)
    def init():
        return tip1, tip2, trail1, trail2

    def update(t):
        # set_data는 시퀀스 필요 → 리스트로 감싸기
        tip1.set_data([p1[t,0]], [p1[t,1]])
        tip1.set_3d_properties([p1[t,2]])
        tip2.set_data([p2[t,0]], [p2[t,1]])
        tip2.set_3d_properties([p2[t,2]])

        sidx = max(0, t - trail_len)
        sl = slice(sidx, t+1)
        trail1.set_data(p1[sl,0], p1[sl,1]); trail1.set_3d_properties(p1[sl,2])
        trail2.set_data(p2[sl,0], p2[sl,1]); trail2.set_3d_properties(p2[sl,2])
        return tip1, tip2, trail1, trail2

    ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                        interval=20, blit=True)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    stem = path.stem.replace(" ", "_")
    # 대표 프레임 저장
    fig.savefig(outdir / f"{stem}_frame.png", dpi=300)

    # 비디오 저장: ffmpeg 있으면 mp4, 없으면 GIF
    if shutil.which("ffmpeg") is not None:
        ani.save(outdir / f"{stem}.mp4", dpi=200, fps=args.fps)
        print(f"Saved: {outdir / f'{stem}.mp4'}")
    else:
        ani.save(outdir / f"{stem}.gif", writer=PillowWriter(fps=args.fps))
        print(f"Saved: {outdir / f'{stem}.gif'}")

    print(f"Saved: {outdir / f'{stem}_frame.png'}")
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="JIGSAWS kinematics txt")
    ap.add_argument("--psm1_xyz", required=True, help='"i j k" indices for PSM1 tooltip')
    ap.add_argument("--psm2_xyz", required=True, help='"i j k" indices for PSM2 tooltip')
    ap.add_argument("--skip", type=int, default=2, help="frame stride")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dt", type=float, default=0.02, help="sampling period (s), for trail length")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="out/replay")
    main(ap.parse_args())
