"""인덱스(build_index 산출) → 모델 입력.

- NO₂ 표준화: idx_dir/no2_stats.json 의 전 기간 μ · σ 한 쌍.
- 시각: 화소 · 질의는 월 상대 step_h 를 키로 쓴다 (ERA5 월 파일의 시간 인덱스). 전역 step_g 는 표에 남긴다.
- 배경 입력 [S, 11] = BG_FEATS 원값. 표준화는 훈련 행으로 fit_bg_scaler → apply_bg_scaler (누수 방지).
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from no2xco2.config import ERA5_WIND_DIR, INDEX_DIR
from no2xco2.data.era5 import open_wind

T0 = pd.Timestamp("2020-01-01")
BG_FEATS = ["latitude", "longitude", "sin_doy", "cos_doy", "tdays", "hour", "t2m", "blh", "sp", "z850", "thk"]
N_BG = len(BG_FEATS)


def bg_features(df: pd.DataFrame) -> np.ndarray:
    """배경항 g 입력 [S, 11] 원값. sin · cos(2π·DOY/365.25) 로 연주기, tdays = 2020-01-01 기준 일수로 추세를 나타낸다."""
    doy = df.doy.to_numpy(np.float32); ang = 2 * np.pi * doy / 365.25
    tdays = ((pd.to_datetime(df["time"]) - T0) / pd.Timedelta(days=1)).to_numpy().astype(np.float32)
    return np.column_stack([df.latitude.to_numpy(np.float32), df.longitude.to_numpy(np.float32), np.sin(ang), np.cos(ang), tdays,
                            df.hour.to_numpy(np.float32), df.bg_t2m.to_numpy(np.float32), df.bg_blh.to_numpy(np.float32), df.bg_sp.to_numpy(np.float32),
                            df.bg_z850.to_numpy(np.float32), df.bg_thk.to_numpy(np.float32)]).astype(np.float32)


def load_no2_stats(idx_dir: str) -> dict:
    """no2_stats.json (n · mean · std · months)."""
    with open(os.path.join(idx_dir, "no2_stats.json")) as f:
        return json.load(f)


def month_h0(ym: str) -> int:
    return int((pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01") - T0) / pd.Timedelta(hours=1))


OCO_COLS = ["row_idx", "time", "latitude", "longitude", "label", "xco2", "step_g", "step_h", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3",
            "doy", "hour", "bg_t2m", "bg_blh", "bg_sp", "bg_z850", "bg_thk", "time_block", "space_block", "year", "fold_time", "fold_space"]


def load_oco(ym: str, idx_dir: str, columns=OCO_COLS) -> pd.DataFrame:
    """필요한 열만 읽는다 (전 열을 읽으면 5년 약 11 GB)."""
    return pq.read_table(os.path.join(idx_dir, f"oco_{ym}.parquet"), columns=columns).to_pandas()


def _offsets(w1, w2, w3):
    ty = (w2 + w3).astype(np.float32); tx = (w1 + w3).astype(np.float32)
    return [(ty, tx), (ty, tx - 1), (ty - 1, tx), (ty - 1, tx - 1)]  # 노드 k 기준 관측 위치 오프셋 (셀 단위)


def load_px(ym: str, idx_dir: str, mu: float, sd: float, dev, qa_min: float = 0.75) -> dict:
    """trop_hi 월 파일 → {step_h: dict(node [E], edge [E, 3], src [E, 2])}."""
    assert qa_min >= 0.75, "trop_hi 는 qa ≥ 0.75 화소만 담고 있다"
    e = pq.read_table(os.path.join(idx_dir, f"trop_hi_{ym}.parquet")).to_pandas()
    if qa_min > 0.75:
        e = e[e.qa >= qa_min]
    if len(e) == 0:
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


def load_qry(df: pd.DataFrame, dev) -> dict:
    """사운딩 월 표 → {step_h: dict(node [4S], edge [4S, 3], seg [4S], n, bg_raw [S, 11], rows)}. rows = 표 안 위치."""
    off = _offsets(df.w1.to_numpy(), df.w2.to_numpy(), df.w3.to_numpy()); bg = bg_features(df)
    qry = {}
    for s, g in df.groupby("step_h", sort=True):
        pos = df.index.get_indexer(g.index); nodes, edge = [], []
        for k in range(4):
            nodes.append(g[f"n{k}"].to_numpy()); edge.append(np.stack([g[f"w{k}"].to_numpy(np.float32), off[k][0][pos], off[k][1][pos]], 1))
        S = len(g)
        qry[int(s)] = dict(node=torch.tensor(np.concatenate(nodes), device=dev), edge=torch.tensor(np.concatenate(edge), device=dev),
                           seg=torch.tensor(np.tile(np.arange(S), 4), device=dev), n=S, bg_raw=torch.tensor(bg[pos], device=dev), rows=pos)
    return qry


def load_wind(ym: str, era5_dir: str, dev):
    f = os.path.join(era5_dir, f"era5_wind_{ym}_z100.nc")
    if not os.path.exists(f):
        raise FileNotFoundError(f"ERA5 바람 파일 없음: {f}")
    with open_wind(f, ym) as ds:  # 위도 오름차순 · 월 길이 · 시작 시각 검사
        U = torch.tensor(ds["u_pbl"].values, device=dev); V = torch.tensor(ds["v_pbl"].values, device=dev)
    return U, V, U.shape[0]


def fit_bg_scaler(dfs, train_masks):
    x = np.concatenate([bg_features(d)[m] for d, m in zip(dfs, train_masks)])
    return torch.tensor(x.mean(0)), torch.tensor(x.std(0) + 1e-6)


def apply_bg_scaler(qry: dict, mean, std) -> None:
    for q in qry.values():
        q["bg"] = (q["bg_raw"] - mean.to(q["bg_raw"].device)) / std.to(q["bg_raw"].device)


DEFAULT_IDX = INDEX_DIR; DEFAULT_ERA5 = ERA5_WIND_DIR
