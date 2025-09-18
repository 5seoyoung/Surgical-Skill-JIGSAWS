# src/00_prepare.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_prepare.py  (JIGSAWS skill-level labels + LOSO split)
- JIGSAWS 키네마틱스 파일을 읽어 다운샘플/윈도잉 후 NPZ 캐시로 저장
- 라벨: 과제(task)가 아니라 '숙련도(E/I/N)'로 추출 (파일명의 B/C/D 규칙)
- 검증: 피험자(Subject) 기준 LOSO 스플릿 생성
"""

import argparse, json, os, re
import numpy as np, pandas as pd
from pathlib import Path

# --- 유틸 ---
def sliding_window(arr, win, stride):
    """arr: [T,C] -> [N,win,C]"""
    X = []
    T = arr.shape[0]
    if T == 0:
        return np.zeros((0, win, arr.shape[1]))
    for s in range(0, max(1, T - win + 1), stride):
        chunk = arr[s:s+win]
        if len(chunk) < win:
            pad = np.zeros((win - len(chunk), arr.shape[1]), dtype=arr.dtype)
            chunk = np.vstack([chunk, pad])
        X.append(chunk)
    return np.stack(X) if X else np.zeros((0, win, arr.shape[1]))

def safe_read_table(p):
    """공백/탭/콤마 구분 텍스트 안전 로딩"""
    try:
        df = pd.read_csv(p, sep=r"\s+|\t|,", engine="python", header=None)
        # 빈 파일 방지
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError("empty table")
        return df.values.astype(float)
    except Exception as e:
        print(f"skip {p} ({e})")
        return None

# --- 파일명 파싱 규칙 ---
# 예: Knot_Tying_B001.txt, Suturing_C007.txt, Needle_Passing_D003.txt
# - 숙련도: B(초보/Novice)=0, C(중간/Intermediate)=1, D(숙련/Expert)=2
# - 피험자: 관례적으로 'B/C/D + 3자리 번호'에서 앞 글자(B/C/D)를 Subject 그룹으로 사용
SKILL_MAP = {"B": 0, "C": 1, "D": 2}
def parse_skill_and_subject(trial_stem: str):
    # 예: Knot_Tying_B001
    m = re.search(r"_([BCD])(\d{3})", trial_stem, flags=re.I)
    if not m:
        return -1, "UNK"
    letter = m.group(1).upper()
    num = m.group(2)       # 001, 002, …
    skill = SKILL_MAP.get(letter, -1)
    subj = num             # <-- 여기! 숫자로 subject ID
    return skill, subj


# --- 데이터 로딩 ---
def read_kinematics(root: Path):
    """
    표준 배치:
      root/
        Knot-Tying/kinematics/AllGestures/*.txt
        Needle-Passing/kinematics/AllGestures/*.txt
        Suturing/kinematics/AllGestures/*.txt
    """
    rows = []
    tasks = {
        "Knot-Tying": "Knot-Tying/kinematics/AllGestures",
        "Needle-Passing": "Needle-Passing/kinematics/AllGestures",
        "Suturing": "Suturing/kinematics/AllGestures",
    }
    for task, subpath in tasks.items():
        kin_dir = root / subpath
        if not kin_dir.exists():
            print(f"[WARN] {kin_dir} not found")
            continue
        count_before = len(rows)
        for p in kin_dir.glob("*.txt"):
            arr = safe_read_table(p)
            if arr is None:
                continue
            trial = p.stem  # ex) Knot_Tying_B001
            skill, subj = parse_skill_and_subject(trial)
            rows.append(dict(task=task, subj=subj, trial=trial, skill=skill, data=arr))
        print(f"[INFO] {task}: +{len(rows)-count_before} files")
    print(f"[INFO] Loaded total {len(rows)} kinematics files")
    return rows

def make_loso(subjects):
    """피험자(문자 ID) 기준 LOSO splits: [{train:[...], test:[subj]} ...]"""
    uniq = sorted(set(subjects))
    splits = []
    for s in uniq:
        train = [x for x in uniq if x != s]
        splits.append({"test": [s], "train": train})
    return splits

# --- 메인 ---
def main(args):
    os.makedirs(Path(args.out_npz).parent, exist_ok=True)

    root = Path(args.root)
    rows = read_kinematics(root)

    # 다운샘플링 스텝(기본 가정: 200Hz -> 50Hz면 step=4)
    # 실제 원 주파수 차이가 있어도 단순 스텝 샘플링으로 처리
    step = max(1, int(200 / args.hz))

    win = int(args.win_sec * args.hz)
    stride = int(args.stride_sec * args.hz)

    # 유효 라벨(숙련도)만 대상으로 진행
    valid_rows = [r for r in rows if r["skill"] in (0, 1, 2)]
    if len(valid_rows) == 0:
        raise RuntimeError("No valid skill-labeled trials found. Check filenames like *_B001/_C001/_D001.")

    subjects = [r["subj"] for r in valid_rows]
    splits = make_loso(subjects)

    X_all, y_all, meta = [], [], []
    for r in valid_rows:
        # 다운샘플
        arr = r["data"][::step, :]
        # 윈도잉
        Xw = sliding_window(arr, win=win, stride=stride)  # [N,win,C]
        if len(Xw) == 0:
            continue
        label = r["skill"]  # 0(Novice),1(Intermediate),2(Expert)
        X_all.append(Xw)
        y_all += [label] * len(Xw)
        # 메타: (task, subj, trial, skill)
        meta += [(r["task"], r["subj"], r["trial"], r["skill"])] * len(Xw)

    if not X_all:
        raise RuntimeError("No windows produced. Try smaller --win_sec or larger data.")

    X = np.concatenate(X_all, axis=0)  # [N,win,C]
    y = np.array(y_all, dtype=np.int64)
    meta_arr = np.array(meta, dtype=object)

    # 라벨 분포 출력
    uniq, cnts = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, cnts)}
    print(f"[INFO] Skill label distribution (0=Nov,1=Int,2=Exp): {dist}")

    # 저장
    np.savez_compressed(args.out_npz, X=X, y=y, meta=meta_arr)
    with open(args.splits, "w") as f:
        json.dump(splits, f, indent=2)
    print("saved:", args.out_npz, args.splits, X.shape, y.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data/JIGSAWS")
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument("--win_sec", type=int, default=10)
    ap.add_argument("--stride_sec", type=int, default=2)
    ap.add_argument("--out_npz", default="out/cache/arrays_skill.npz")
    ap.add_argument("--splits", default="out/cache/loso.json")
    args = ap.parse_args()
    main(args)
