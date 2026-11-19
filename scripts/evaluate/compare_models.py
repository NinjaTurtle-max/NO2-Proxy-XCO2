"""같은 시험 사운딩에서 네 모델 비교 (Table 2 · Fig. 3b) 와 시험 칸 12개별 수송 효과 (Fig. 3a).

모델: GNN 수송 켬 · GNN 수송 끔 · 크리깅 · LightGBM (NO₂ + 바람).
지표: RMSE, R² (시험 행 안 1 − SSE/SST), 편향. ΔRMSE = RMSE(모델) − RMSE(GNN 켬), 95 % 구간 = UTC 일 블록 부트스트랩 (n 1,000, seed 0).
칸별: 공간 칸(5°)마다 ΔRMSE = RMSE(끔) − RMSE(켬) 과 같은 부트스트랩 구간.
출력: <out>/compare_models.csv · <out>/cells_transport.csv
사용: python scripts/evaluate/compare_models.py --runs runs --idx-dir data/processed/index --out runs/tables
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from no2xco2.train import block_bootstrap, day_blocks

MODELS = [("off", "GNN, transport off", "gnn_m60_off/pred_space0_s0_nophys.parquet"), ("krig", "Kriging", "kriging/pred_krig.parquet"),
          ("lgbm", "LightGBM (same inputs)", "lgbm_no2_wind/pred_lgbm_no2wind.parquet")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="runs"); ap.add_argument("--idx-dir", default="data/processed/index"); ap.add_argument("--out", default="runs/tables")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    on = pd.read_parquet(os.path.join(a.runs, "gnn_m60_on/pred_space0_s0_phys.parquet"), columns=["row_idx", "pred", "xco2", "test"])
    d = on[on.test.astype(bool)].drop(columns="test")
    for k, _, f in MODELS:
        d = d.merge(pd.read_parquet(os.path.join(a.runs, f), columns=["row_idx", "pred"]).rename(columns={"pred": k}), on="row_idx", validate="one_to_one")
    t = pd.concat([pd.read_parquet(f, columns=["row_idx", "time", "space_block"]) for f in sorted(glob.glob(os.path.join(a.idx_dir, "oco_*.parquet")))])
    d = d.merge(t, on="row_idx", validate="one_to_one").reset_index(drop=True)
    y = d.xco2.to_numpy(); days = day_blocks(d.time); sst = ((y - y.mean()) ** 2).sum(); e_on = d.pred.to_numpy() - y
    rm = lambda e: float(np.sqrt((e ** 2).mean()))
    rows = [dict(key="on", model="GNN, transport on", n=len(d), rmse=rm(e_on), r2=1 - (e_on ** 2).sum() / sst, bias=e_on.mean(), delta=np.nan, lo=np.nan, hi=np.nan)]
    for k, lab, _ in MODELS:
        e = d[k].to_numpy() - y; lo, hi = block_bootstrap(e_on, e, days)
        rows.append(dict(key=k, model=lab, n=len(d), rmse=rm(e), r2=1 - (e ** 2).sum() / sst, bias=e.mean(), delta=rm(e) - rm(e_on), lo=lo, hi=hi))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(a.out, "compare_models.csv"), index=False); print(T.round(4).to_string(index=False))
    cells = []
    for c, gi in d.groupby("space_block").groups.items():
        g = d.loc[gi]; ea = (g.pred - g.xco2).to_numpy(); eb = (g.off - g.xco2).to_numpy(); lo, hi = block_bootstrap(ea, eb, days[gi])
        cells.append(dict(cell=int(c), lat0=20 + 5 * (c // 10), lon0=100 + 5 * (c % 10), n=len(g), rmse_on=rm(ea), rmse_off=rm(eb), delta=rm(eb) - rm(ea), lo=lo, hi=hi))
    C = pd.DataFrame(cells).sort_values("delta"); C.to_csv(os.path.join(a.out, "cells_transport.csv"), index=False); print(C.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
