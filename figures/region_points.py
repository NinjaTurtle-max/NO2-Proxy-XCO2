"""Fig. 1 · Fig. 2 입력: 2020–2024 품질 통과(L_train) 사운딩 위치(0.01° 반올림 고유 점)와 공간 폴드 0 시험 5° 칸.
출력: runs/region_pts.npz (ix, iy = 0.01° 정수 좌표 · cells = 시험 칸 (경도, 위도) 5° 번호 · n2 · n3 = OCO-2 · OCO-3 사운딩 수 · nt = 시험 사운딩 수)
실행: python figures/region_points.py [--idx-dir data/processed/index] [--out runs/region_pts.npz]"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--idx-dir", default="data/processed/index"); ap.add_argument("--out", default="runs/region_pts.npz"); a = ap.parse_args()
fs = sorted(f for f in glob.glob(os.path.join(a.idx_dir, "oco_*.parquet")) if "2020" <= os.path.basename(f)[4:8] <= "2024")
keys = []; n = {"OCO-2": 0, "OCO-3": 0}; cy = {}; nt = 0
for f in fs:
    d = pd.read_parquet(f, columns=["latitude", "longitude", "label", "fold_space", "satellite"]); d = d[d.label == "L_train"]
    for s in n: n[s] += int((d.satellite == s).sum())
    ix = np.round((d.longitude.values - 100) * 100).astype(np.int32); iy = np.round((d.latitude.values - 20) * 100).astype(np.int32)
    keys.append(np.unique(ix * 4000 + iy))
    c0 = d[d.fold_space == 0]; nt += len(c0); y = os.path.basename(f)[4:8]
    cy.setdefault(y, set()).update(zip(np.floor(c0.longitude / 5).astype(int), np.floor(c0.latitude / 5).astype(int)))
k = np.unique(np.concatenate(keys)); cells = sorted(set().union(*cy.values()))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez_compressed(a.out, ix=(k // 4000).astype(np.int16), iy=(k % 4000).astype(np.int16), cells=np.array(cells), n2=n["OCO-2"], n3=n["OCO-3"], nt=nt)
print("파일", len(fs), "· 사운딩", n["OCO-2"] + n["OCO-3"], n, "· 고유 점(0.01°)", len(k), "· 시험 칸", len(cells), "· 시험 사운딩", nt)
