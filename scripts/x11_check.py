"""결정 11 ① 12개월 대조 (학습 없음): W1 분할 수 · row_idx 유일성 · W2 주입 사운딩·장면 수 · W7 시차 구간 자기 재현.
실행: scripts/x11_run.sh scripts/x11_check.py [--out experiments/x11_check_20261110] (게이트: 우리 패턴 python 0 ∧ AC ∧ 덮개 열림 ∧ 스왑 < 80 % — 기준선 단계는 S-6 예외(U-5)).
산출: <out>/x11_check.csv (항목·값·기대·판정 칸), <out>/split_scene0.parquet (row_idx · ym · scene_id · role)."""
import argparse
import os

import numpy as np
import pandas as pd

import no2xco2
from no2xco2.data import loader as L
from no2xco2.data import splits as S
from no2xco2.train import nested_masks

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="experiments/x11_check_20261110"); ap.add_argument("--months", default=",".join(f"2020{m:02d}" for m in range(1, 13)))
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); months = a.months.split(",")
print(f"no2xco2.__file__={no2xco2.__file__}", flush=True)
cols = ["row_idx", "time", "label", "latitude", "longitude", "xco2", "satellite", "step_g", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3", "fold_time", "fold_space", "space_block", "year"]
dfs = [L.load_oco(m, L.DEFAULT_IDX, columns=cols) for m in months]; ym_row = np.concatenate([np.full(len(d), m) for d, m in zip(dfs, months)])
big = pd.concat(dfs, ignore_index=True); y = big.xco2.to_numpy(np.float64); lab = (big.label == "L_train").to_numpy()
rows = []


def rec(item, value, expect="", ok=""):
    rows.append(dict(항목=item, 값=value, 기대=expect, 일치=ok)); print(f"{item}: {value}" + (f" (기대 {expect} → {ok})" if expect != "" else ""), flush=True)


rec("개월 · 전체 행 · L_train", f"{len(months)} · {len(big):,} · {lab.sum():,}", "12 · - · 2,299,582 (12개월일 때)", "" if len(months) != 12 else str(int(lab.sum()) == 2299582))
rec("row_idx 유일 (전 개월)", bool(big.row_idx.is_unique), "True (아니면 ym 열 추가)", str(bool(big.row_idx.is_unique)))

role, sid = S.scene_holdout(big, lab, seed=a.seed); tr, val, te = role == "train", role == "val", role == "test"
exp = dict(train=1521459, val=357732, test=420391)
for k, m in (("train", tr), ("val", val), ("test", te)):
    rec(f"W1 장면 보류 seed {a.seed} {k}", f"{int(m.sum()):,}", f"{exp[k]:,} (seed 0, 12개월)", "" if (len(months) != 12 or a.seed != 0) else str(int(m.sum()) == exp[k]))
rec("W1 test 장면 수", int(np.unique(sid[te]).size), "371", "" if (len(months) != 12 or a.seed != 0) else str(int(np.unique(sid[te]).size) == 371))
rec("W1 L_train 장면 수", int(np.unique(sid[lab]).size), "1886", "" if len(months) != 12 else str(int(np.unique(sid[lab]).size) == 1886))
rec("W1 역할 합 = L_train", bool((tr | val | te).sum() == lab.sum() and not (tr & val).any() and not (tr & te).any() and not (val & te).any()), "True", "")
rec("W1 한 장면 한 역할", bool(pd.DataFrame(dict(s=sid[lab], r=role[lab])).groupby("s").r.nunique().max() == 1), "True", "")
pd.DataFrame(dict(row_idx=big.row_idx, ym=ym_row, scene_id=sid, role=role)).to_parquet(f"{a.out}/split_scene{a.seed}.parquet", index=False)

# W2: 장면 보류 (X1s 정상 · X1-Ls 누출) 와 시간 폴드 2 (X1t, D8 = val 행 있는 장면 전체 제외)
t_tr, t_val, t_te = nested_masks(big, "time", 2, lab, 1); sid_t = S.scene_ids(big, lab)
for name, trm, vam, tem, sd in (("장면 보류", tr, val, te, sid), ("시간 폴드 2", t_tr, t_val, t_te, sid_t)):
    mu = y[trm].mean(); inj = trm & ~np.isin(sd, np.unique(sd[vam]))
    for leak in (False, True):
        tab, st = L.xinj_table(big, inj, sd, lab, y - mu, months, leak)
        bad = {k: int((inj & m).sum()) for k, m in (("val", vam), ("test", tem), ("버퍼·비훈련 L_train", lab & ~trm & ~vam & ~tem), ("비 L_train", ~lab))}
        rec(f"W2 {name} {'누출(자기 스텝)' if leak else '정상(장면 마지막 스텝)'}", f"훈련 {int(trm.sum()):,} · 주입 후보 {int(inj.sum()):,} · 주입 {st['n_inj']:,} 사운딩 · {st['n_inj_scenes']:,} 장면 · 달 경계 넘는 장면 {st['n_cross_scenes']} ({st['n_cross_rows']:,} 행) · 버림 {st['n_drop_scenes']} 장면 ({st['n_drop_rows']:,} 행) · "
            f"val 장면 제외로 빠진 훈련 행 {int(trm.sum() - inj.sum()):,} · μ_train {mu:.4f} · 누출 검사 {bad}", "누출 검사 전부 0", str(not any(bad.values())))

# W7: 600 km 직전 시차 (P1 정의, 과거 방향, 훈련 장면 후보) — 기대 P1: 장면 보류 ≤0.5 21.7 · ≤1 54.1 · ≤2 78.2 · ≤7 94.6 %, 시간 폴드 2 0.0 · 0.0 · 0.0 · 49.7 %
for name, trm, tem, sd, expc in (("장면 보류", tr, te, sid, (21.7, 54.1, 78.2, 94.6)), ("시간 폴드 2", t_tr, t_te, sid_t, (0.0, 0.0, 0.0, 49.7))):
    lag = S.upstream_lag(big, sd, trm, tem, 600.0)[tem]
    fr = [100 * float((lag <= k).mean()) for k in (0.5, 1, 2, 7)]
    rec(f"W7 {name} 600 km 시차 누적 비율 ≤0.5·≤1·≤2·≤7일 (%)", " · ".join(f"{v:.1f}" for v in fr) + f" · 직전 없음 {100 * float(np.isinf(lag).mean()):.1f} % (장면-칸 가중 아님, 사운딩 단위)",
        " · ".join(f"{v}" for v in expc) + " (P1 은 장면-칸 n 가중 = 사운딩 단위와 같음)", str(all(abs(f - e) < 0.1 for f, e in zip(fr, expc))))
pd.DataFrame(rows).to_csv(f"{a.out}/x11_check.csv", index=False); print(f"→ {a.out}/x11_check.csv")
