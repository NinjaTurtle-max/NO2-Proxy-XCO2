"""
dataset.py — 길 B(XCO₂ 이상장 재구성) PINN 학습용 데이터셋
==========================================================
NC → (B,T,C,H,W) 텐서 + obs_mask. 설계 근거 = docs/research_synthesis_decision_20260611.html §7.

데이터 소스:
  · data/processed/monthly_grid_025deg_augmented.nc  — xco2_anomaly_A(관측픽셀만)·t2m·ndvi·nox_cams(dense)
  · data/raw/auxiliary/era5_transport_monthly.nc     — u10·v10·z500·div850 (dense, 2020–2023)
    └ era5_u10/v10(augmented)은 관측픽셀에만 존재 → dense 수송장은 transport 파일 사용.

핵심 설계:
  1. 타겟 = xco2_anomaly_A − BG(t).
     BG(t) = 월별 도메인 공간중앙값 (Hakkarainen et al. 2016; project_background_removal_fix).
     BG 미제거 시 전지구 IAV 오프셋(2023 −0.45ppm)이 val을 죽임 — 실증됨.
  2. dense NO₂ source: TROPOMI NO₂ dense 격자는 없음(관측픽셀만) →
     CAMS NOₓ(dense)를 train 관측픽셀에서 TROPOMI NO₂에 선형 정합(a·x+b, OLS)하여
     μmol m⁻² 등가 스케일로 변환. β₀=0.0081의 물리 단위 의미 보존.
  3. gap-filling 학습: 입력 채널의 관측을 랜덤 부분마스킹(keep_ratio~U)하고
     손실은 전체 관측픽셀에 부과 → 모델이 입력 복사가 아닌 공간 재구성을 학습.
     val은 고정 시드 50% 마스킹 → hidden-pixel R²가 1차 벤치마크(천장 0.70 대비).
  4. 시간 분할: train=2020-01~2022-12(36개월), val=2023(12개월). 2024 제외(dense 수송장 부재).
     val 샘플의 컨텍스트(t-2,t-1)가 2022 말을 포함하는 것은 입력 관측 재사용일 뿐 타겟 누수 아님.

x_seq 채널 (C=11 기본, 표준화는 train 기간 통계):
  0 y_in        타겟 이상장(부분마스킹된 관측, 결측=0) [ppm, 비표준화 — PDE와 단위 일치]
  1 in_mask     입력으로 준 관측 마스크 (0/1)
  2 no2_dense   NOₓ→NO₂ 등가 (표준화)
  3 u10         (표준화)
  4 v10         (표준화)
  5 t2m         (표준화)
  6 ndvi        (결측=0 후 표준화; 0≈해양)
  7 z500        (표준화)
  8 div850      (표준화)
  9 sin_month
 10 cos_month

lag_features=True 시 경향(1차 차분) 채널 4종 추가 (C=15, v4 스펙):
 11 d_u10       u10[τ]−u10[τ−1]   이류장 변동 (표준화)
 12 d_v10       v10[τ]−v10[τ−1]                (표준화)
 13 d_no2       no2[τ]−no2[τ−1]   배출 변동    (표준화)
 14 d_z500      z500[τ]−z500[τ−1] 종관 전환    (표준화)
  └ τ=0(2020-01)은 이전 달 부재 → 0. 수송 인과성은 농도가 아닌 변화율에 실림.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
AUG_NC = ROOT / "data" / "processed" / "monthly_grid_025deg_augmented.nc"
TRANSPORT_NC = ROOT / "data" / "raw" / "auxiliary" / "era5_transport_monthly.nc"

N_MONTHS = 48          # 2020-01 ~ 2023-12 (2024 제외: dense 수송장·ODIAC 부재)
TRAIN_END = 36         # train = 인덱스 [0,36) = 2020~2022, val = [36,48) = 2023
SEQ_LEN = 3            # 컨텍스트 t-2, t-1, t → 타겟 t
N_CHANNELS = 11

# 표준화 대상 채널 (y_in·in_mask·sin/cos 제외)
_STD_VARS = ["no2_dense", "u10", "v10", "t2m", "ndvi", "z500", "div850"]
# 경향(lag) 채널 원천 변수 — 프레임 간 1차 차분
_LAG_VARS = ["u10", "v10", "no2_dense", "z500"]
N_LAG_CHANNELS = len(_LAG_VARS)


def _fill_spatial_median(arr: np.ndarray) -> np.ndarray:
    """(T,H,W) 결측을 해당 월 공간중앙값으로 채움 (가장자리 소수 결측용)."""
    out = arr.copy()
    for t in range(arr.shape[0]):
        sl = out[t]
        if np.isnan(sl).any():
            sl[np.isnan(sl)] = np.nanmedian(sl)
    return out


def build_fields(verbose: bool = True) -> dict:
    """NC 2개를 읽어 (48,H,W) numpy 필드 사전과 정합/표준화 파라미터를 반환."""
    aug = xr.open_dataset(AUG_NC).isel(time=slice(0, N_MONTHS))
    tr = xr.open_dataset(TRANSPORT_NC)
    assert aug.sizes["lat"] == tr.sizes["lat"] == 120
    assert aug.sizes["time"] == tr.sizes["time"] == N_MONTHS

    H, W = aug.sizes["lat"], aug.sizes["lon"]
    months = aug["time"].dt.month.values  # (48,)

    # ── 타겟: anomaly_A − 월별 공간중앙값 배경 (Hakkarainen 2016) ──
    anom = aug["xco2_anomaly_A"].values.astype(np.float32)        # (48,H,W), 관측픽셀만
    bg = np.array([np.nanmedian(anom[t]) if np.isfinite(anom[t]).any() else 0.0
                   for t in range(N_MONTHS)], dtype=np.float32)
    y = anom - bg[:, None, None]
    obs_mask = np.isfinite(y).astype(np.float32)

    # ── dense NO₂: CAMS NOₓ → TROPOMI NO₂ 선형 정합 (train 관측픽셀 OLS) ──
    no2_obs = aug["tropomi_no2"].values.astype(np.float32)
    nox = _fill_spatial_median(aug["nox_cams"].values.astype(np.float32))
    sel = np.isfinite(no2_obs[:TRAIN_END]) & np.isfinite(nox[:TRAIN_END])
    x_fit, y_fit = nox[:TRAIN_END][sel], no2_obs[:TRAIN_END][sel]
    a, b = np.polyfit(x_fit, y_fit, 1)
    no2_dense = np.clip(a * nox + b, 0.0, None).astype(np.float32)
    fit_r = float(np.corrcoef(a * x_fit + b, y_fit)[0, 1])

    fields = {
        "y": y, "obs_mask": obs_mask,
        "no2_dense": no2_dense,
        "u10": tr["u10"].values.astype(np.float32),
        "v10": tr["v10"].values.astype(np.float32),
        "t2m": _fill_spatial_median(aug["t2m"].values.astype(np.float32)),
        "ndvi": np.nan_to_num(aug["ndvi"].values.astype(np.float32), nan=0.0),
        "z500": tr["z500"].values.astype(np.float32),
        "div850": tr["div850"].values.astype(np.float32),
    }

    # ── 경향(1차 차분) 필드: τ=0은 이전 달 부재 → 0 ──
    for k in _LAG_VARS:
        d = np.zeros_like(fields[k])
        d[1:] = fields[k][1:] - fields[k][:-1]
        fields["d_" + k] = d

    # ── 표준화 파라미터 (train 기간만 — 누수 방지) ──
    norm = {}
    for k in _STD_VARS + ["d_" + k for k in _LAG_VARS]:
        v = fields[k][:TRAIN_END]
        norm[k] = (float(np.nanmean(v)), float(np.nanstd(v) + 1e-8))

    # ── 좌표 (정규화 −1~1) ──
    lat = aug["lat"].values.astype(np.float32)
    lon = aug["lon"].values.astype(np.float32)
    lat_norm = ((lat - lat.mean()) / (lat.max() - lat.min()) * 2.0)[:, None] * np.ones((1, W), np.float32)
    lon_norm = ((lon - lon.mean()) / (lon.max() - lon.min()) * 2.0)[None, :] * np.ones((H, 1), np.float32)

    meta = {
        "bg": bg.tolist(), "months": months.tolist(),
        "nox2no2": {"a": float(a), "b": float(b), "fit_r": fit_r},
        "norm": norm,
        "obs_frac_train": float(obs_mask[:TRAIN_END].mean()),
        "obs_frac_val": float(obs_mask[TRAIN_END:].mean()),
        "y_std_train": float(np.nanstd(y[:TRAIN_END])),
    }
    if verbose:
        print(f"[dataset] NOₓ→NO₂ 정합: a={a:.4g}, b={b:.4g}, r={fit_r:.3f}")
        print(f"[dataset] 관측밀도 train={meta['obs_frac_train']*100:.1f}% "
              f"val={meta['obs_frac_val']*100:.1f}%, 타겟 σ(train)={meta['y_std_train']:.3f} ppm")
    aug.close(); tr.close()
    fields["lat_norm"], fields["lon_norm"] = lat_norm, lon_norm
    fields["months"] = months
    return fields, meta


class XCO2ReconDataset(Dataset):
    """월별 프레임 샘플. __getitem__(i) → 타겟 월 t의 시퀀스/마스크/물리장 텐서 사전.

    split='train': 입력 관측 keep_ratio~U(keep_lo, keep_hi) 랜덤 마스킹(에폭마다 변동).
    split='val'  : 고정 시드로 50% 마스킹 → hidden-pixel 평가 재현 가능.
                   eval_full=True면 관측 전부 입력(temporal-only 평가용).
    """

    def __init__(self, fields: dict, split: str = "train",
                 keep_lo: float = 0.3, keep_hi: float = 0.8,
                 val_keep: float = 0.5, eval_full: bool = False, seed: int = 42,
                 lag_features: bool = False):
        self.f = fields
        self.split = split
        self.lag_features = lag_features
        self.keep_lo, self.keep_hi = keep_lo, keep_hi
        self.val_keep = val_keep
        self.eval_full = eval_full
        self.seed = seed
        if split == "train":
            # 컨텍스트가 train 내부에 있어야 함 → 타겟 t ∈ [SEQ_LEN−1, 36)
            self.t_idx = list(range(SEQ_LEN - 1, TRAIN_END))
        else:
            self.t_idx = list(range(TRAIN_END, N_MONTHS))
        H, W = fields["y"].shape[1:]
        self.H, self.W = H, W

    def __len__(self):
        return len(self.t_idx)

    def _input_mask(self, t: int, obs: np.ndarray, item_seed: int | None) -> np.ndarray:
        """관측픽셀 중 입력으로 줄 부분집합 마스크."""
        if self.split != "train" and self.eval_full:
            return obs.copy()
        if self.split == "train":
            rng = np.random.default_rng(item_seed)
            keep = rng.uniform(self.keep_lo, self.keep_hi)
        else:
            rng = np.random.default_rng(self.seed * 10_000 + t)  # val 고정
            keep = self.val_keep
        u = rng.random(obs.shape).astype(np.float32)
        return (obs > 0) * (u < keep)

    def __getitem__(self, i: int):
        t = self.t_idx[i]
        f = self.f
        item_seed = None
        if self.split == "train":
            item_seed = int(np.random.randint(0, 2**31 - 1))  # DataLoader worker별 독립

        # 프레임별 채널 스택 (T=SEQ_LEN)
        frames, in_masks = [], []
        for tau in range(t - SEQ_LEN + 1, t + 1):
            obs = f["obs_mask"][tau]
            im = self._input_mask(tau, obs, item_seed)
            y_in = np.nan_to_num(f["y"][tau], nan=0.0) * im
            m = f["months"][tau]
            ch = [y_in, im]
            for k in _STD_VARS:
                ch.append(f[k][tau])  # 표준화는 아래 일괄 처리
            ch.append(np.full_like(y_in, np.sin(2 * np.pi * m / 12.0)))
            ch.append(np.full_like(y_in, np.cos(2 * np.pi * m / 12.0)))
            if self.lag_features:  # 경향 채널은 말미 고정(기본 채널 인덱스 보존)
                for k in _LAG_VARS:
                    ch.append(f["d_" + k][tau])
            frames.append(np.stack(ch, 0))
            in_masks.append(im)
        x_seq = np.stack(frames, 0).astype(np.float32)  # (T,C,H,W)

        # 표준화 (채널 2~8 = _STD_VARS, 11~14 = 경향 채널)
        for ci, k in enumerate(_STD_VARS, start=2):
            mu, sd = self.norm[k]
            x_seq[:, ci] = (x_seq[:, ci] - mu) / sd
        if self.lag_features:
            for ci, k in enumerate(["d_" + k for k in _LAG_VARS], start=N_CHANNELS):
                mu, sd = self.norm[k]
                x_seq[:, ci] = (x_seq[:, ci] - mu) / sd

        obs_t = f["obs_mask"][t]
        in_t = in_masks[-1]
        hidden_t = obs_t * (1.0 - in_t)                  # 입력에서 숨긴 관측 (gap-filling 평가)
        y_t = np.nan_to_num(f["y"][t], nan=0.0)

        def T2(a):  # (H,W) → (1,H,W)
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(0)

        return {
            "x_seq": torch.from_numpy(x_seq),            # (T,C,H,W)
            "target": T2(y_t),                           # ppm (관측 외 0; 손실은 obs_mask로 제한)
            "obs_mask": T2(obs_t),
            "in_mask": T2(in_t),
            "hidden_mask": T2(hidden_t),
            "lat_norm": T2(f["lat_norm"]),
            "lon_norm": T2(f["lon_norm"]),
            # PDE/source용 물리 단위 dense 장
            "no2": T2(f["no2_dense"][t]),                # μmol m⁻² 등가
            "u": T2(f["u10"][t]), "v": T2(f["v10"][t]),  # m s⁻¹
            "ndvi": T2(f["ndvi"][t]),
            "t_index": torch.tensor(t),
        }

    # build_fields의 norm 주입 (생성 후 1회)
    def attach_norm(self, norm: dict):
        self.norm = norm
        return self


def make_datasets(seed: int = 42, verbose: bool = True, lag_features: bool = False):
    """fields 1회 로드 → (train_ds, val_hidden_ds, val_full_ds, meta)."""
    fields, meta = build_fields(verbose=verbose)
    kw = dict(seed=seed, lag_features=lag_features)
    tr = XCO2ReconDataset(fields, "train", **kw).attach_norm(meta["norm"])
    va = XCO2ReconDataset(fields, "val", **kw).attach_norm(meta["norm"])
    va_full = XCO2ReconDataset(fields, "val", eval_full=True, **kw).attach_norm(meta["norm"])
    meta["n_channels"] = N_CHANNELS + (N_LAG_CHANNELS if lag_features else 0)
    meta["n_lag_channels"] = N_LAG_CHANNELS if lag_features else 0
    return tr, va, va_full, meta


if __name__ == "__main__":
    tr, va, va_full, meta = make_datasets()
    s = tr[0]
    print(f"train n={len(tr)}, val n={len(va)}")
    for k, v in s.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s} {tuple(v.shape)}")
    print(json.dumps(meta["nox2no2"], indent=2))
