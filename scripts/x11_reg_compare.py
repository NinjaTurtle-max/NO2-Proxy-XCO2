"""W4 기본값 보존 회귀: x11 worktree 1개월 1에폭 기본 인자 런 vs 기준 런(reg6 주 · reg7 보조) — summary 열 · 값 · state 키 · params · pred.
사용: x11_reg_compare.py NEW_DIR BASE_DIR [--allow-key bg_extra]  (11-09 주 트리 재생성판: 기반이 A-6 코드라 주 기준 = reg7 (예외 없음), 보조 = reg6 은 --allow-key bg_extra — 열·state 키 차이가 이 키뿐이어야 통과)."""
import argparse

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser(); ap.add_argument("new"); ap.add_argument("base"); ap.add_argument("--allow-key", default="")
a = ap.parse_args(); tag = "time2_s0_phys"; allow = {a.allow_key} if a.allow_key else set()
sn, sb = pd.read_csv(f"{a.new}/summary.csv"), pd.read_csv(f"{a.base}/summary.csv")
stn, stb = torch.load(f"{a.new}/state_{tag}.pt", weights_only=False), torch.load(f"{a.base}/state_{tag}.pt", weights_only=False)
cn, cb = [c for c in sn.columns if c not in allow], [c for c in sb.columns if c not in allow]; kn, kb = [k for k in stn if k not in allow and k != "params"], [k for k in stb if k not in allow and k != "params"]
print(f"열 수 new {len(sn.columns)} (허용 제외 후 {len(cn)}) · base {len(sb.columns)} (허용 제외 후 {len(cb)}) · 열 같음(이름·순서) {cn == cb}")
print(f"state 키 new 허용 제외 후 {len(kn)} · base 허용 제외 후 {len(kb)} · 같음 {kn == kb} · 예외 키(new) {sorted(set(k for k in stn if k != 'params') - set(kn))} (base) {sorted(set(k for k in stb if k != 'params') - set(kb))}")
print(f"params 키 같음 {list(stn['params']) == list(stb['params'])} ({len(stn['params'])}개)")
num = [c for c in cn if c in cb and pd.api.types.is_numeric_dtype(sn[c]) and c != "sec_per_epoch"]
d = {c: abs(float(sn[c].iloc[0]) - float(sb[c].iloc[0])) for c in num if pd.notna(sn[c].iloc[0])}
worst = sorted(d.items(), key=lambda kv: -kv[1])[:4]; print(f"summary 수치 열 {len(num)}개 최대 |차| " + " · ".join(f"{c} {v:.2e}" for c, v in worst) + "  (sec_per_epoch 제외)")
neq = [c for c in cn if c in sb.columns and c not in num and str(sn[c].iloc[0]) != str(sb[c].iloc[0])]; print(f"비수치 열 값 불일치 {neq}")
pn, pb = pd.read_parquet(f"{a.new}/pred_{tag}.parquet"), pd.read_parquet(f"{a.base}/pred_{tag}.parquet")
print(f"pred 행 {len(pn):,}/{len(pb):,} · row_idx 같음 {bool((pn.row_idx.values == pb.row_idx.values).all())} · pred 최대 |차| {np.nanmax(np.abs(pn.pred.values - pb.pred.values)):.2e} · 결측 위치 같음 {bool((np.isnan(pn.pred.values) == np.isnan(pb.pred.values)).all())}")
pd_ = max(float((stn['params'][k] - stb['params'][k]).abs().max()) for k in stn['params'])
print(f"params 최대 |차| {pd_:.2e}")
