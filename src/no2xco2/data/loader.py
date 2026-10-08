"""5년 인덱스(build_index 산출) → 모델 입력 (D2·D3·D4 반영).

- NO₂ 표준화: idx_dir/no2_stats.json 의 전 기간 μ·σ 1쌍 (D2). 월별 μ·σ(구 prep) 는 쓰지 않는다.
- 시각: px·qry 는 월 상대 step_h 키 (ERA5 월 파일 인덱스). 전역 step_g 는 df 에 보존 (D3, 월 경계 warm start 는 run 이 h 를 넘겨 처리).
- 배경 입력 bg_raw [S,11] = BG_FEATS (lat, lon, sin_doy, cos_doy, tdays, hour, t2m, blh, sp, z850, thk) 원값 (결정 1 + 종관 지표, 2026-09-22). 표준화는 fit_bg_scaler(훈련 행) → apply_bg_scaler(qry) 로 학습 시 수행 (D4, 누수 방지).
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from no2xco2.config import INDEX_DIR, LOCAL_STAGE_OUT
from no2xco2.data.era5 import open_wind

T0 = pd.Timestamp("2020-01-01")
BG_FEATS = ["latitude", "longitude", "sin_doy", "cos_doy", "tdays", "hour", "t2m", "blh", "sp", "z850", "thk"]  # 결정 1 (2026-09-22): 연주기 sin/cos + 연속 시간(추세); 종관 z850·700–1000 두께 (승인 2026-09-22)
N_BG = len(BG_FEATS)


def n_bg(harmonics: int = 1) -> int:
    """배경 입력 수: harmonics = 1 이면 BG_FEATS 11 (현행), k = 2..harmonics 마다 sin/cos(2πk·DOY/365.25) 2열 추가 (TS-1 승인 09-28)."""
    return N_BG + 2 * (harmonics - 1)


def bg_features(df: pd.DataFrame, harmonics: int = 1) -> np.ndarray:
    """배경항 g 입력 [S, n_bg(harmonics)] 원값. sin/cos(2π·DOY/365.25) 로 연주기, tdays = 2020-01-01 기준 일수(연속)로 추세를 표현.
    harmonics > 1: 고차 조화 sin/cos(2πk·DOY/365.25), k = 2..harmonics 를 끝에 덧붙인다 (열 순서 불변 → harmonics = 1 은 현행과 동일)."""
    doy = df.doy.to_numpy(np.float32); ang = 2 * np.pi * doy / 365.25
    tdays = ((pd.to_datetime(df["time"]) - T0) / pd.Timedelta(days=1)).to_numpy().astype(np.float32)
    return np.column_stack([df.latitude.to_numpy(np.float32), df.longitude.to_numpy(np.float32), np.sin(ang), np.cos(ang), tdays,
                            df.hour.to_numpy(np.float32), df.bg_t2m.to_numpy(np.float32), df.bg_blh.to_numpy(np.float32), df.bg_sp.to_numpy(np.float32),
                            df.bg_z850.to_numpy(np.float32), df.bg_thk.to_numpy(np.float32)]
                           + [f(k * ang) for k in range(2, harmonics + 1) for f in (np.sin, np.cos)]).astype(np.float32)


def load_no2_stats(idx_dir: str) -> dict:
    """no2_stats.json 전체(n·mean·std·months·qa_hi) — 호출자가 mean/std 를 꺼내 쓰고 출처(n·months)를 기록한다 (QA R10)."""
    with open(os.path.join(idx_dir, "no2_stats.json")) as f:
        return json.load(f)


def month_h0(ym: str) -> int:
    return int((pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01") - T0) / pd.Timedelta(hours=1))


OCO_COLS = ["row_idx", "time", "latitude", "longitude", "label", "xco2", "step_g", "step_h", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3",
            "doy", "hour", "bg_t2m", "bg_blh", "bg_sp", "bg_z850", "bg_thk", "time_block", "space_block", "year", "fold_time", "fold_space"]


def load_oco(ym: str, idx_dir: str, columns=OCO_COLS) -> pd.DataFrame:
    """학습에 쓰는 열만 읽는다 (45열 전부 읽으면 5년 ≈ 11 GB)."""
    return pq.read_table(os.path.join(idx_dir, f"oco_{ym}.parquet"), columns=columns).to_pandas()


def _offsets(w1, w2, w3):
    ty = (w2 + w3).astype(np.float32); tx = (w1 + w3).astype(np.float32)
    return [(ty, tx), (ty, tx - 1), (ty - 1, tx), (ty - 1, tx - 1)]  # 노드 k 기준 화소 오프셋(셀 단위)


def load_px(ym: str, idx_dir: str, mu: float, sd: float, dev, qa_min: float = 0.75) -> dict:
    """trop_hi 월 파일 → {step_h: dict(node [E], edge [E,3], src [E,2])}."""
    assert qa_min >= 0.75, "trop_hi 는 qa ≥ 0.75 만 담고 있음 — qa_min < 0.75 는 인덱스 재생성 필요"
    e = pq.read_table(os.path.join(idx_dir, f"trop_hi_{ym}.parquet")).to_pandas()
    if qa_min > 0.75:
        e = e[e.qa >= qa_min]
    if len(e) == 0:  # 필터 후 화소 0 → 주입 없음 (빈 배열로 IndexError 나던 경로, QA X7)
        return {}
    e = e.sort_values("step_g", kind="stable").reset_index(drop=True)
    h0 = month_h0(ym); step = (e.step_g.to_numpy() - h0).astype(np.int32)
    off = _offsets(e.w1.to_numpy(), e.w2.to_numpy(), e.w3.to_numpy())
    z = ((e.no2.to_numpy(np.float32) - mu) / sd).astype(np.float32); qa = e.qa.to_numpy(np.float32)
    px = {}
    bounds = np.flatnonzero(np.diff(step)) + 1; starts = np.r_[0, bounds]; ends = np.r_[bounds, len(step)]
    for s0, s1 in zip(starts, ends):
        sl = slice(s0, s1); nodes, edge, src = [], [], []
        for k in range(4):
            nodes.append(e[f"n{k}"].to_numpy()[sl]); edge.append(np.stack([e[f"w{k}"].to_numpy(np.float32)[sl], off[k][0][sl], off[k][1][sl]], 1))
            src.append(np.stack([z[sl], qa[sl]], 1))
        px[int(step[s0])] = dict(node=torch.tensor(np.concatenate(nodes), device=dev), edge=torch.tensor(np.concatenate(edge), device=dev),
                                 src=torch.tensor(np.concatenate(src), device=dev))
    return px


def load_qry(df: pd.DataFrame, dev, harmonics: int = 1) -> dict:
    """oco 월 df → {step_h: dict(node [4S], edge [4S,3], seg [4S], n, bg_raw [S,n_bg(harmonics)], month [S], rows)}. rows = df 내 위치. month = UTC 월 0–11 (월별 β, 결정 7-a)."""
    off = _offsets(df.w1.to_numpy(), df.w2.to_numpy(), df.w3.to_numpy()); bg = bg_features(df, harmonics)
    mon = (pd.to_datetime(df["time"]).dt.month.to_numpy() - 1).astype(np.int64)
    qry = {}
    for s, g in df.groupby("step_h", sort=True):
        pos = df.index.get_indexer(g.index); nodes, edge = [], []
        for k in range(4):
            nodes.append(g[f"n{k}"].to_numpy()); edge.append(np.stack([g[f"w{k}"].to_numpy(np.float32), off[k][0][pos], off[k][1][pos]], 1))
        S = len(g)
        qry[int(s)] = dict(node=torch.tensor(np.concatenate(nodes), device=dev), edge=torch.tensor(np.concatenate(edge), device=dev),
                           seg=torch.tensor(np.tile(np.arange(S), 4), device=dev), n=S, bg_raw=torch.tensor(bg[pos], device=dev),
                           month=torch.tensor(mon[pos], device=dev), rows=pos)
    return qry


def load_wind(ym: str, era5_dir: str, dev):
    f = os.path.join(era5_dir, f"era5_wind_{ym}_z100.nc")
    if not os.path.exists(f):  # NAS fallback 제거 (QA R5): SMB 위 HDF5 는 세그폴트 이력 → 로컬 사본(rsync) 필수
        raise FileNotFoundError(f"ERA5 로컬 사본 없음: {f} — NAS era5_wind_z100 에서 rsync 후 실행")
    with open_wind(f, ym) as ds:  # 위도 오름차순 보장 + 월 길이·시작 시각 검사
        U = torch.tensor(ds["u_pbl"].values, device=dev); V = torch.tensor(ds["v_pbl"].values, device=dev)
    return U, V, U.shape[0]


def fit_bg_scaler(dfs: list[pd.DataFrame], train_masks: list[np.ndarray], harmonics: int = 1):
    x = np.concatenate([bg_features(d, harmonics)[m] for d, m in zip(dfs, train_masks)])
    return torch.tensor(x.mean(0)), torch.tensor(x.std(0) + 1e-6)


def apply_bg_scaler(qry: dict, mean, std) -> None:
    for q in qry.values():
        q["bg"] = (q["bg_raw"] - mean.to(q["bg_raw"].device)) / std.to(q["bg_raw"].device)


DEFAULT_IDX = INDEX_DIR; DEFAULT_ERA5 = LOCAL_STAGE_OUT
