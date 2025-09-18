#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_features.py (extended features)
- cache npz (X,y,meta)에서 특징 추출 후 parquet/csv로 저장
"""

import argparse, numpy as np, pandas as pd

def feats_block(arr):
    """
    arr: [T, C], 여기서 C=76 (JIGSAWS kinematics features)
    - 여기서는 좌/우 손 위치(x,y,z) 앞 6차원만 사용한다고 가정
    """
    pos = arr[:, :6]                     # [T,6] left/right 3D
    vel = np.diff(pos, axis=0)
    spd = np.linalg.norm(vel, axis=1)    # 전체 속도 norm

    # 기존 특징 -------------------------------------------------
    path_len = spd.sum()
    straight = np.linalg.norm(pos[-1,:3] - pos[0,:3]) / (path_len+1e-6)
    mean_v, max_v = spd.mean(), spd.max()
    pause = np.mean(spd < 1e-3)

    fft = np.fft.rfft(spd - spd.mean())
    fft_energy = (np.abs(fft)**2).sum()

    feats = [path_len, straight, mean_v, max_v, pause, fft_energy]

    # 확장 특징 -------------------------------------------------
    # jerk (가속도의 변화량)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    if len(jerk) > 0:
        jerk_mag = np.linalg.norm(jerk, axis=1)
        feats += [jerk_mag.mean(), jerk_mag.std()]
    else:
        feats += [0.0, 0.0]

    # bimanual correlation (좌/우 손 속도 크기)
    left_spd = np.linalg.norm(vel[:, :3], axis=1)
    right_spd = np.linalg.norm(vel[:, 3:6], axis=1)
    if left_spd.std()>1e-6 and right_spd.std()>1e-6:
        corr = np.corrcoef(left_spd, right_spd)[0,1]
    else:
        corr = 0.0
    feats += [corr]

    # spectral bandpower
    freqs = np.fft.rfftfreq(len(spd), d=1/50.0)  # 50Hz 샘플링
    psd = np.abs(np.fft.rfft(spd))**2
    bp1 = psd[(freqs>=0.5)&(freqs<3)].sum()
    bp2 = psd[(freqs>=3)&(freqs<6)].sum()
    feats += [bp1, bp2]

    # trajectory spread (위치 좌표 표준편차)
    feats += [pos.std()]

    return np.array(feats)

def main(args):
    arr = np.load(args.cache, allow_pickle=True)
    X, y, meta = arr["X"], arr["y"], arr["meta"]

    feats, rows = [], []
    for i in range(len(X)):
        f = feats_block(X[i])
        feats.append(f)

        m = meta[i]
        if len(m) == 3:
            task, subj, trial = m
        elif len(m) == 4:
            task, subj, trial, _ = m  # skill 무시
        else:
            raise ValueError(f"Unexpected meta format: {m}")
        rows.append((task, subj, trial, int(y[i])))

    df = pd.DataFrame(
        feats,
        columns=[
            "path_len","straight","mean_v","max_v","pause","fft_energy",
            "jerk_mean","jerk_std","bimanual_corr","bp_0_3","bp_3_6","disp_std"
        ]
    )
    meta_df = pd.DataFrame(rows, columns=["task","subj","trial","y"])
    df = pd.concat([meta_df, df], axis=1)
    df.to_parquet(args.out, index=False)
    print(f"saved: {args.out}", df.shape)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="out/cache/arrays_skill_stride1_subjid.npz")
    ap.add_argument("--out", default="out/features_skill_stride1_subjid_exp.parquet")
    main(ap.parse_args())
