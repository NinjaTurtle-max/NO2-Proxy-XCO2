"""
pipeline/02c_harmonic_baseline.py
=================================
역할: 위도1°bin × 연도별 조화함수 베이스라인 피팅 → XCO2 이상 산출

입력: data/raw/super_obs_025/super_obs_dataset_025.parquet
출력: data/raw/harmonic_params/anom_1d_harmonic_dual.parquet
      data/raw/harmonic_params/fitted_harmonic_params_A.json  (Scheme A 파라미터)
      data/raw/harmonic_params/fitted_harmonic_params_B.json  (Scheme B 파라미터)

방법론:
  Scheme A — 위도1°bin × 연도별 (lat_bin, year) 조화함수 피팅
    key    : (lat_bin_float, year_int)
    모델   : XCO2(doy) = C0 + A1·cos(2π·doy/T) + B1·sin(2π·doy/T)
                              + A2·cos(4π·doy/T) + B2·sin(4π·doy/T)
    T = 365.25 (연주기)
    최소 관측수: MIN_OBS_A = 30 per (lat_bin, year)

  Scheme B — EAIC sub-region × 연도별 (region, year) 조화함수 피팅
    key    : (eaic_region_str, year_int)
    모델   : 동일 2-harmonic 모델
    최소 관측수: MIN_OBS_B = 20 per (region, year)
    OUT 데이터는 Scheme B baseline NaN 처리

EAIC sub-region 정의:
  NCP : lat 34~41N, lon 113~122E  (화북평원)
  YRD : lat 28.5~33N, lon 118~123E (양쯔강 삼각주)
  KCR : lat 35~38.5N, lon 125~129E (수도권)
  JKT : lat 34.5~37N, lon 138.5~141E (관동)
  PRD : lat 21.5~24.5N, lon 112~115.5E (주강 삼각주)
  OUT : 위 5개 지역에 속하지 않는 나머지
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
IN_PARQUET  = ROOT / "data" / "raw" / "super_obs_025" / "super_obs_dataset_025.parquet"
OUT_DIR     = ROOT / "data" / "raw" / "harmonic_params"
OUT_PARQUET = OUT_DIR / "anom_1d_harmonic_dual.parquet"
OUT_PARAMS_A = OUT_DIR / "fitted_harmonic_params_A.json"
OUT_PARAMS_B = OUT_DIR / "fitted_harmonic_params_B.json"

# ---------------------------------------------------------------------------
# 조화함수 파라미터
# ---------------------------------------------------------------------------
T_PERIOD = 365.25   # 연주기 (일)
MIN_OBS_A = 30      # Scheme A 최소 관측수 (lat_bin, year) 당
MIN_OBS_B = 20      # Scheme B 최소 관측수 (region, year) 당

# ---------------------------------------------------------------------------
# EAIC sub-region 정의
# ---------------------------------------------------------------------------
EAIC_REGIONS = {
    "NCP": {"lat": (34.0,   41.0),   "lon": (113.0,  122.0)},
    "YRD": {"lat": (28.5,   33.0),   "lon": (118.0,  123.0)},
    "KCR": {"lat": (35.0,   38.5),   "lon": (125.0,  129.0)},
    "JKT": {"lat": (34.5,   37.0),   "lon": (138.5,  141.0)},
    "PRD": {"lat": (21.5,   24.5),   "lon": (112.0,  115.5)},
}


def assign_eaic_region(df: pd.DataFrame) -> pd.Series:
    """위도/경도 기반 EAIC sub-region 할당. 복수 해당 시 첫 번째 우선."""
    region = pd.Series("OUT", index=df.index)
    for name, bounds in EAIC_REGIONS.items():
        lat_ok = (df["latitude"] >= bounds["lat"][0]) & (df["latitude"] <= bounds["lat"][1])
        lon_ok = (df["longitude"] >= bounds["lon"][0]) & (df["longitude"] <= bounds["lon"][1])
        mask = lat_ok & lon_ok & (region == "OUT")
        region[mask] = name
    return region


# ---------------------------------------------------------------------------
# 조화함수 모델 (2-harmonic, trend 없음)
# ---------------------------------------------------------------------------
def harmonic_model(doy: np.ndarray, C0: float, A1: float, B1: float, A2: float, B2: float) -> np.ndarray:
    """
    XCO2(doy) = C0 + A1·cos(2π·doy/T) + B1·sin(2π·doy/T)
                   + A2·cos(4π·doy/T) + B2·sin(4π·doy/T)
    """
    t = doy
    return (C0
            + A1 * np.cos(2 * np.pi * t / T_PERIOD)
            + B1 * np.sin(2 * np.pi * t / T_PERIOD)
            + A2 * np.cos(4 * np.pi * t / T_PERIOD)
            + B2 * np.sin(4 * np.pi * t / T_PERIOD))


def fit_harmonic(doy: np.ndarray, xco2: np.ndarray, min_obs: int) -> tuple[np.ndarray | None, float | None]:
    """
    단일 그룹에 대한 조화함수 피팅.
    Returns (params 5-array, fit_residual_std) or (None, None).
    """
    valid = ~np.isnan(xco2)
    if valid.sum() < min_obs:
        return None, None
    try:
        p0 = [np.nanmean(xco2), 1.0, 1.0, 0.5, 0.5]
        popt, _ = curve_fit(harmonic_model, doy[valid], xco2[valid], p0=p0, maxfev=5000)
        residuals = xco2[valid] - harmonic_model(doy[valid], *popt)
        return popt, float(np.std(residuals))
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Scheme A: (lat_bin, year) 피팅
# ---------------------------------------------------------------------------
def run_scheme_a(df: pd.DataFrame) -> tuple[dict, pd.Series, pd.Series]:
    """
    Returns:
        params_A  : {(lat_bin, year): {'params': [...], 'n_samples': int, 'fit_residual_std': float}}
        baseline  : Series (xco2_baseline_A)
        resid_std : Series (fit_residual_std_A)
    """
    print("[Scheme A] (lat_bin, year) 조화함수 피팅 시작...")
    params_A = {}
    baseline  = pd.Series(np.nan, index=df.index)
    resid_std = pd.Series(np.nan, index=df.index)

    groups = df.groupby(["lat_bin", "year"])
    for (lat_bin, year), grp in groups:
        doy  = grp["doy"].values.astype(float)
        xco2 = grp["xco2"].values.astype(float)

        popt, std = fit_harmonic(doy, xco2, MIN_OBS_A)
        if popt is None:
            continue

        key = (float(lat_bin), int(year))
        params_A[key] = {
            "params"          : popt.tolist(),
            "n_samples"       : int(np.sum(~np.isnan(xco2))),
            "fit_residual_std": std,
        }

        fitted = harmonic_model(doy, *popt)
        baseline.loc[grp.index]  = fitted
        resid_std.loc[grp.index] = std

    n_fitted = len(params_A)
    print(f"[Scheme A] 피팅 완료: {n_fitted}/{groups.ngroups} 그룹")
    return params_A, baseline, resid_std


# ---------------------------------------------------------------------------
# Scheme B: (eaic_region, year) 피팅
# ---------------------------------------------------------------------------
def run_scheme_b(df: pd.DataFrame) -> tuple[dict, pd.Series, pd.Series]:
    """
    Returns:
        params_B  : {(region, year): {'params': [...], 'n_samples': int, 'fit_residual_std': float}}
        baseline  : Series (xco2_baseline_B) — OUT 행은 NaN
        resid_std : Series (fit_residual_std_B)
    """
    print("[Scheme B] (eaic_region, year) 조화함수 피팅 시작...")
    params_B = {}
    baseline  = pd.Series(np.nan, index=df.index)
    resid_std = pd.Series(np.nan, index=df.index)

    df_eaic = df[df["eaic_region"] != "OUT"]
    groups = df_eaic.groupby(["eaic_region", "year"])
    for (region, year), grp in groups:
        doy  = grp["doy"].values.astype(float)
        xco2 = grp["xco2"].values.astype(float)

        popt, std = fit_harmonic(doy, xco2, MIN_OBS_B)
        if popt is None:
            continue

        key = (str(region), int(year))
        params_B[key] = {
            "params"          : popt.tolist(),
            "n_samples"       : int(np.sum(~np.isnan(xco2))),
            "fit_residual_std": std,
        }

        fitted = harmonic_model(doy, *popt)
        baseline.loc[grp.index]  = fitted
        resid_std.loc[grp.index] = std

    n_fitted = len(params_B)
    print(f"[Scheme B] 피팅 완료: {n_fitted}/{groups.ngroups} 그룹")
    return params_B, baseline, resid_std


# ---------------------------------------------------------------------------
# JSON 직렬화 헬퍼 (numpy repr 키 포맷 유지)
# ---------------------------------------------------------------------------
def to_json_serializable(params: dict) -> dict:
    """키를 원본 저장 포맷과 동일하게 직렬화."""
    out = {}
    for (k1, k2), v in params.items():
        if isinstance(k1, float):
            key_str = f"(np.float32({k1}), np.int32({k2}))"
        else:
            key_str = f"('{k1}', np.int32({k2}))"
        out[key_str] = v
    return out


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------
def run(in_path: Path = IN_PARQUET, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[02c] 입력 로드: {in_path}")
    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["doy"]   = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month

    # 위도1°bin 할당
    df["lat_bin"] = np.floor(df["latitude"]).astype(np.float32)

    # EAIC region 할당
    df["eaic_region"] = assign_eaic_region(df)

    print(f"[02c] 총 {len(df):,}행 로드 완료")
    print(f"  eaic_region 분포:\n{df['eaic_region'].value_counts().to_string()}")

    # ---------------------------------------------------------------------------
    # Scheme A 피팅
    params_A, baseline_A, resid_A = run_scheme_a(df)
    df["xco2_baseline_A"]    = baseline_A
    df["fit_residual_std_A"] = resid_A
    df["xco2_anomaly_A"]     = df["xco2"] - df["xco2_baseline_A"]

    # ---------------------------------------------------------------------------
    # Scheme B 피팅
    params_B, baseline_B, resid_B = run_scheme_b(df)
    df["xco2_baseline_B"]    = baseline_B
    df["fit_residual_std_B"] = resid_B
    df["xco2_anomaly_B"]     = df["xco2"] - df["xco2_baseline_B"]

    # ---------------------------------------------------------------------------
    # 출력 컬럼 정렬 (원본 포맷 기준)
    out_cols = [
        "date", "lat_idx", "lon_idx", "xco2", "n_soundings",
        "tropomi_no2", "era5_wind_speed", "era5_blh", "era5_u10", "era5_v10",
        "latitude", "longitude", "population_density", "odiac_emission",
        "xco2_bootstrap_std",
        "year", "doy", "month", "lat_bin", "eaic_region",
        "xco2_baseline_A", "fit_residual_std_A", "xco2_anomaly_A",
        "xco2_baseline_B", "fit_residual_std_B", "xco2_anomaly_B",
    ]
    df_out = df[out_cols]

    # ---------------------------------------------------------------------------
    # 저장
    out_parquet = out_dir / "anom_1d_harmonic_dual.parquet"
    df_out.to_parquet(out_parquet, index=False)
    print(f"[02c] anomaly parquet 저장: {out_parquet}")

    out_params_a = out_dir / "fitted_harmonic_params_A.json"
    with open(out_params_a, "w") as f:
        json.dump(to_json_serializable(params_A), f, indent=2)
    print(f"[02c] Scheme A 파라미터 저장: {out_params_a}")

    out_params_b = out_dir / "fitted_harmonic_params_B.json"
    with open(out_params_b, "w") as f:
        json.dump(to_json_serializable(params_B), f, indent=2)
    print(f"[02c] Scheme B 파라미터 저장: {out_params_b}")

    # ---------------------------------------------------------------------------
    # 요약 통계
    print("\n=== Scheme A 이상 요약 ===")
    for reg in ["NCP", "YRD", "KCR", "JKT", "PRD"]:
        sub = df_out[df_out["eaic_region"] == reg]["xco2_anomaly_A"].dropna()
        print(f"  {reg}: n={len(sub):,}  mean={sub.mean():.3f}  std={sub.std():.3f} ppm")

    print("\n=== Scheme B 이상 요약 ===")
    for reg in ["NCP", "YRD", "KCR", "JKT", "PRD"]:
        sub = df_out[df_out["eaic_region"] == reg]["xco2_anomaly_B"].dropna()
        print(f"  {reg}: n={len(sub):,}  mean={sub.mean():.3f}  std={sub.std():.3f} ppm")


if __name__ == "__main__":
    run()
