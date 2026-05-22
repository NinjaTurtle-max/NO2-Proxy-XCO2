"""
pipeline/02c_harmonic_baseline.py
=================================
역할: 위도1°bin × 전기간 연속 조화함수 베이스라인 피팅 → XCO2 이상 산출

입력: data/raw/super_obs_025/super_obs_dataset_025.parquet
출력: data/raw/harmonic_params/anom_1d_harmonic_dual.parquet
      data/raw/harmonic_params/fitted_harmonic_params_A.json
      data/raw/harmonic_params/fitted_harmonic_params_B.json

방법론 (v2 — 전기간 연속 피팅):
  Scheme A — 위도1°bin × 전기간 (lat_bin, 2020-2024 통합)
    key    : lat_bin_float
    모델   : XCO2(t) = C0 + trend·t
                     + A1·cos(2π·t/T) + B1·sin(2π·t/T)
                     + A2·cos(4π·t/T) + B2·sin(4π·t/T)
    t      : 2020-01-01 기준 연속 일수 (연도 경계 없음)
    T      : 365.25 (연주기)
    장점   : 연도 경계 불연속 없음, 연간 CO2 성장 추세 명시 보존

  Scheme B — EAIC sub-region × 전기간 (region, 2020-2024 통합)
    key    : eaic_region_str
    모델   : 동일 6-파라미터 모델
    OUT    : Scheme B baseline NaN 처리

EAIC sub-region 정의:
  NCP : lat 34~41N,   lon 113~122E  (화북평원)
  YRD : lat 28.5~33N, lon 118~123E  (양쯔강 삼각주)
  KCR : lat 35~38.5N, lon 125~129E  (수도권)
  JKT : lat 34.5~37N, lon 138.5~141E (관동)
  PRD : lat 21.5~24.5N, lon 112~115.5E (주강 삼각주)
  OUT : 위 5개 지역 외
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent.parent
IN_PARQUET   = ROOT / "data" / "raw" / "super_obs_025" / "super_obs_dataset_025.parquet"
OUT_DIR      = ROOT / "data" / "raw" / "harmonic_params"
OUT_PARQUET  = OUT_DIR / "anom_1d_harmonic_dual.parquet"
OUT_PARAMS_A = OUT_DIR / "fitted_harmonic_params_A.json"
OUT_PARAMS_B = OUT_DIR / "fitted_harmonic_params_B.json"

# ---------------------------------------------------------------------------
# 모델 설정
# ---------------------------------------------------------------------------
T_PERIOD  = 365.25                          # 연주기 (일)
T0        = pd.Timestamp("2020-01-01")      # 연속 일수 기준점
MIN_OBS_A = 50                              # Scheme A 최소 관측수 (lat_bin 전기간)
MIN_OBS_B = 50                              # Scheme B 최소 관측수 (region 전기간)

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
    """위도/경도 기반 EAIC sub-region 할당."""
    region = pd.Series("OUT", index=df.index)
    for name, bounds in EAIC_REGIONS.items():
        lat_ok = (df["latitude"] >= bounds["lat"][0]) & (df["latitude"] <= bounds["lat"][1])
        lon_ok = (df["longitude"] >= bounds["lon"][0]) & (df["longitude"] <= bounds["lon"][1])
        mask = lat_ok & lon_ok & (region == "OUT")
        region[mask] = name
    return region


# ---------------------------------------------------------------------------
# 조화함수 모델 (6-파라미터: C0 + trend + 2-harmonic)
# ---------------------------------------------------------------------------
def harmonic_model(
    t: np.ndarray,
    C0: float, trend: float,
    A1: float, B1: float,
    A2: float, B2: float,
) -> np.ndarray:
    """
    XCO2(t) = C0 + trend·t
            + A1·cos(2π·t/T) + B1·sin(2π·t/T)
            + A2·cos(4π·t/T) + B2·sin(4π·t/T)
    t : 2020-01-01 기준 연속 일수
    """
    return (
        C0 + trend * t
        + A1 * np.cos(2 * np.pi * t / T_PERIOD)
        + B1 * np.sin(2 * np.pi * t / T_PERIOD)
        + A2 * np.cos(4 * np.pi * t / T_PERIOD)
        + B2 * np.sin(4 * np.pi * t / T_PERIOD)
    )


def fit_harmonic(
    t_days: np.ndarray,
    xco2: np.ndarray,
    min_obs: int,
) -> tuple[np.ndarray | None, float | None]:
    """전기간 연속 조화함수 피팅. Returns (params 6-array, residual_std) or (None, None)."""
    valid = ~np.isnan(xco2)
    if valid.sum() < min_obs:
        return None, None
    try:
        # 초기값: C0=관측평균, trend=2ppm/yr≈0.00548/일, 나머지 소진폭
        p0 = [np.nanmean(xco2), 2.0 / 365.25, 1.0, 1.0, 0.3, 0.3]
        popt, _ = curve_fit(
            harmonic_model,
            t_days[valid], xco2[valid],
            p0=p0, maxfev=10000,
        )
        resid = xco2[valid] - harmonic_model(t_days[valid], *popt)
        return popt, float(np.std(resid))
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Scheme A: lat_bin × 전기간
# ---------------------------------------------------------------------------
def run_scheme_a(df: pd.DataFrame) -> tuple[dict, pd.Series, pd.Series]:
    print("[Scheme A] lat_bin × 전기간 연속 피팅 시작...")
    params_A  = {}
    baseline  = pd.Series(np.nan, index=df.index, dtype=float)
    resid_std = pd.Series(np.nan, index=df.index, dtype=float)

    groups = df.groupby("lat_bin")
    for lat_bin, grp in groups:
        t    = grp["t_days"].values.astype(float)
        xco2 = grp["xco2"].values.astype(float)

        popt, std = fit_harmonic(t, xco2, MIN_OBS_A)
        if popt is None:
            continue

        params_A[float(lat_bin)] = {
            "params"          : popt.tolist(),
            "n_samples"       : int(np.sum(~np.isnan(xco2))),
            "fit_residual_std": std,
            "param_names"     : ["C0", "trend(ppm/day)", "A1", "B1", "A2", "B2"],
        }

        baseline.loc[grp.index]  = harmonic_model(t, *popt)
        resid_std.loc[grp.index] = std

    print(f"[Scheme A] 피팅 완료: {len(params_A)}/{groups.ngroups} lat_bin")
    return params_A, baseline, resid_std


# ---------------------------------------------------------------------------
# Scheme B: eaic_region × 전기간
# ---------------------------------------------------------------------------
def run_scheme_b(df: pd.DataFrame) -> tuple[dict, pd.Series, pd.Series]:
    print("[Scheme B] eaic_region × 전기간 연속 피팅 시작...")
    params_B  = {}
    baseline  = pd.Series(np.nan, index=df.index, dtype=float)
    resid_std = pd.Series(np.nan, index=df.index, dtype=float)

    df_eaic = df[df["eaic_region"] != "OUT"]
    groups  = df_eaic.groupby("eaic_region")
    for region, grp in groups:
        t    = grp["t_days"].values.astype(float)
        xco2 = grp["xco2"].values.astype(float)

        popt, std = fit_harmonic(t, xco2, MIN_OBS_B)
        if popt is None:
            continue

        params_B[str(region)] = {
            "params"          : popt.tolist(),
            "n_samples"       : int(np.sum(~np.isnan(xco2))),
            "fit_residual_std": std,
            "param_names"     : ["C0", "trend(ppm/day)", "A1", "B1", "A2", "B2"],
        }

        baseline.loc[grp.index]  = harmonic_model(t, *popt)
        resid_std.loc[grp.index] = std

    print(f"[Scheme B] 피팅 완료: {len(params_B)}/{groups.ngroups} region")
    return params_B, baseline, resid_std


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------
def run(in_path: Path = IN_PARQUET, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[02c v2] 입력 로드: {in_path}")
    df = pd.read_parquet(in_path)
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["doy"]   = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month

    # 연속 일수 (연도 경계 없는 시간축)
    df["t_days"] = (df["date"] - T0).dt.days.astype(float)

    # 위도1°bin / EAIC region
    df["lat_bin"]     = np.floor(df["latitude"]).astype(np.float32)
    df["eaic_region"] = assign_eaic_region(df)

    print(f"[02c v2] {len(df):,}행 로드")
    print(f"  기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  eaic_region:\n{df['eaic_region'].value_counts().to_string()}")

    # Scheme A
    params_A, baseline_A, resid_A = run_scheme_a(df)
    df["xco2_baseline_A"]    = baseline_A
    df["fit_residual_std_A"] = resid_A
    df["xco2_anomaly_A"]     = df["xco2"] - df["xco2_baseline_A"]

    # Scheme B
    params_B, baseline_B, resid_B = run_scheme_b(df)
    df["xco2_baseline_B"]    = baseline_B
    df["fit_residual_std_B"] = resid_B
    df["xco2_anomaly_B"]     = df["xco2"] - df["xco2_baseline_B"]

    # 저장
    out_cols = [
        "date", "lat_idx", "lon_idx", "xco2", "n_soundings",
        "tropomi_no2", "era5_wind_speed", "era5_blh", "era5_u10", "era5_v10",
        "latitude", "longitude", "population_density", "odiac_emission",
        "xco2_bootstrap_std",
        "year", "doy", "month", "t_days", "lat_bin", "eaic_region",
        "xco2_baseline_A", "fit_residual_std_A", "xco2_anomaly_A",
        "xco2_baseline_B", "fit_residual_std_B", "xco2_anomaly_B",
    ]
    df[out_cols].to_parquet(out_dir / "anom_1d_harmonic_dual.parquet", index=False)
    print(f"\n[02c v2] parquet 저장: {out_dir / 'anom_1d_harmonic_dual.parquet'}")

    with open(out_dir / "fitted_harmonic_params_A.json", "w") as f:
        json.dump(params_A, f, indent=2)
    with open(out_dir / "fitted_harmonic_params_B.json", "w") as f:
        json.dump(params_B, f, indent=2)
    print(f"[02c v2] 파라미터 저장 완료")

    # ── 진단 출력 ────────────────────────────────────────────────────────────
    print("\n=== Scheme A trend 파라미터 (ppm/day → ppm/yr) ===")
    print(f"  {'lat_bin':>8s}  {'C0':>8s}  {'trend(ppm/yr)':>14s}  {'resid_std':>10s}")
    for lb in sorted(params_A):
        p   = params_A[lb]["params"]
        std = params_A[lb]["fit_residual_std"]
        print(f"  {lb:>8.1f}  {p[0]:>8.3f}  {p[1]*365.25:>+14.4f}  {std:>10.4f}")

    print("\n=== 연도별 이상값 평균 (연속 피팅 후 경계 점프 확인) ===")
    from scipy.stats import pearsonr
    print(f"  {'지역':<6s}  {'2020':>8s}  {'2021':>8s}  {'2022':>8s}  {'2023':>8s}  {'2024':>8s}  {'전체r':>8s}")
    print("  " + "-" * 64)
    df_v = df.dropna(subset=["xco2_anomaly_A", "tropomi_no2"])
    for reg in ["NCP", "YRD", "KCR", "JKT", "PRD", "OUT"]:
        sub = df_v[df_v["eaic_region"] == reg]
        means = []
        for yr in range(2020, 2025):
            s = sub[sub["year"] == yr]["xco2_anomaly_A"]
            means.append(f"{s.mean():+.3f}" if len(s) > 0 else "  N/A ")
        r_all, _ = pearsonr(sub["tropomi_no2"], sub["xco2_anomaly_A"]) if len(sub) > 10 else (np.nan, np.nan)
        print(f"  {reg:<6s}  {'  '.join(means)}  {r_all:>+8.3f}")

    print("\n=== 연도 경계 불연속 진단 (NCP 12→1월 점프) ===")
    ncp = df_v[df_v["eaic_region"] == "NCP"]
    for yr1, yr2 in [(2020,2021),(2021,2022),(2022,2023),(2023,2024)]:
        dec = ncp[(ncp["year"]==yr1)&(ncp["month"]==12)]["xco2_anomaly_A"]
        jan = ncp[(ncp["year"]==yr2)&(ncp["month"]==1)]["xco2_anomaly_A"]
        print(f"  {yr1}-12 → {yr2}-01  점프: {jan.mean()-dec.mean():+.4f} ppm  "
              f"(12월 {dec.mean():+.4f} / 1월 {jan.mean():+.4f})")


if __name__ == "__main__":
    run()
