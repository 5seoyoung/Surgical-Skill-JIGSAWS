#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from pathlib import Path

# 하이픈 표기를 기준으로 사용
TASKS_HYPHEN = ["Knot-Tying", "Needle-Passing", "Suturing"]

def main(args):
    df = pd.read_parquet(args.features)

    # 안전장치: y/skill 둘 다 허용
    label_col = "y" if "y" in df.columns else ("skill" if "skill" in df.columns else None)
    if label_col is None:
        raise RuntimeError("No label column ('y' or 'skill') found.")

    if args.task not in TASKS_HYPHEN:
        raise ValueError(f"--task must be one of {TASKS_HYPHEN}")

    dft = df[df["task"] == args.task].copy()
    if dft.empty:
        print(f"[WARN] No rows for task={args.task}. Available:", df["task"].unique().tolist())
        return

    # trial 단위로 대표 라벨(다수결) 산출
    rep = dft.groupby("trial")[label_col].agg(lambda s: s.value_counts().idxmax())

    nov_trials = rep[rep == 0].index.tolist()
    exp_trials = rep[rep == 2].index.tolist()

    print(f"[{args.task}] Novice trials (n={len(nov_trials)}):", nov_trials)
    print(f"[{args.task}] Expert  trials (n={len(exp_trials)}):", exp_trials)

    # kinematics 경로(하이픈 디렉터리 구조)
    base = Path(args.root) / args.task / "kinematics" / "AllGestures"
    if nov_trials:
        print("Novice path example:", base / f"{nov_trials[0]}.txt")
    if exp_trials:
        print("Expert path example:", base / f"{exp_trials[0]}.txt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="out/features_skill_stride1_subjid_exp.parquet")
    ap.add_argument("--root", default="data/JIGSAWS")
    ap.add_argument("--task", default="Knot-Tying", choices=["Knot-Tying","Needle-Passing","Suturing"])
    main(ap.parse_args())
