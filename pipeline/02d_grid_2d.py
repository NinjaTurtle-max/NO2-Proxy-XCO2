"""
pipeline/02d_grid_2d.py
=================================
역할: 1D super-obs parquet → 2D 월별 집계 격자 NetCDF 산출

입력: data/raw/harmonic_params/anom_1d_harmonic_dual.parquet
출력: data/processed/grid_2d_monthly_025deg.nc

격자 정의:
  해상도  : 0.25° × 0.25°
  위도 축 : 20.125 ~ 49.875°N  (120셀, lat_idx 0-119)
  경도 축 : 100.125 ~ 149.875°E (200셀, lon_idx 0-199)
  시간 축 : 2020-01 ~ 2024-12 (60개월, 월별 집계)

집계 방법:
  - 각 (월, lat_idx, lon_idx) 셀 내 관측치 평균
  - 1관측 이상이면 유효값, 0관측이면 NaN
  - n_obs 레이어도 함께 저장 (관측 밀도 진단용)

저장 변수:
  xco2, xco2_anomaly_A, xco2_anomaly_B
  tropomi_no2, era5_u10, era5_v10, era5_wind_speed, era5_blh
  n_obs                   (격자 내 관측수)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent
IN_PARQUET = ROOT / "data" / "raw" / "harmonic_params" / "anom_1d_harmonic_dual.parquet"
OUT_NC     = ROOT / "data" / "processed" / "grid_2d_monthly_025deg.nc"

# ---------------------------------------------------------------------------
# 격자 정의 (0.25° 중심좌표)
# ---------------------------------------------------------------------------
LAT_ORIGIN = 20.0       # 도메인 남단
LON_ORIGIN = 100.0      # 도메인 서단
RESOLUTION = 0.25
N_LAT      = 120        # lat_idx 0~119
N_LON      = 200        # lon_idx 0~199

LAT_CENTERS = LAT_ORIGIN + RESOLUTION / 2 + np.arange(N_LAT) * RESOLUTION  # 20.125 ~ 49.875
LON_CENTERS = LON_ORIGIN + RESOLUTION / 2 + np.arange(N_LON) * RESOLUTION  # 100.125 ~ 149.875

# 집계 대상 변수
AGG_VARS = [
    "xco2",
    "xco2_anomaly_A",
    "xco2_anomaly_B",
    "tropomi_no2",
    "era5_u10",
    "era5_v10",
    "era5_wind_speed",
    "era5_blh",
]


# ---------------------------------------------------------------------------
# 월별 2D 집계
# ---------------------------------------------------------------------------
def aggregate_monthly(df: pd.DataFrame) -> xr.Dataset:
    """
    1D 관측 → (time, lat, lon) 월별 평균 집계.
    Returns xarray.Dataset.
    """
    df = df.copy()
    df["date"]      = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")

    # 월 축 생성 (2020-01 ~ 2024-12)
    all_periods = pd.period_range("2020-01", "2024-12", freq="M")
    n_time      = len(all_periods)
    period_to_idx = {p: i for i, p in enumerate(all_periods)}

    print(f"  시간 축: {all_periods[0]} ~ {all_periods[-1]}  ({n_time}개월)")
    print(f"  공간 축: {N_LAT} lat × {N_LON} lon")

    # 결과 배열 초기화
    arrays = {var: np.full((n_time, N_LAT, N_LON), np.nan, dtype=np.float32) for var in AGG_VARS}
    n_obs  = np.zeros((n_time, N_LAT, N_LON), dtype=np.int16)

    # 집계: (year_month, lat_idx, lon_idx) 그룹
    group_cols = ["year_month", "lat_idx", "lon_idx"]
    agg_dict   = {var: "mean" for var in AGG_VARS if var in df.columns}
    agg_dict["xco2"] = "mean"    # n_obs 계산용 기준

    grouped = df.groupby(group_cols)

    # n_obs 계산
    n_obs_ser = df.groupby(group_cols)["xco2"].count()

    for (period, li, loi), cnt in n_obs_ser.items():
        ti = period_to_idx.get(period)
        if ti is None:
            continue
        li, loi = int(li), int(loi)
        n_obs[ti, li, loi] = int(cnt)

    # 변수별 평균 집계
    for var in AGG_VARS:
        if var not in df.columns:
            print(f"  [경고] 변수 없음: {var} — 건너뜀")
            continue
        ser = df.groupby(group_cols)[var].mean()
        for (period, li, loi), val in ser.items():
            ti = period_to_idx.get(period)
            if ti is None:
                continue
            arrays[var][ti, int(li), int(loi)] = val

    # xarray Dataset 구성
    time_coords = all_periods.to_timestamp()   # Period → Timestamp (월 첫날)

    data_vars = {}
    for var in AGG_VARS:
        if var not in arrays:
            continue
        attrs = _var_attrs(var)
        data_vars[var] = xr.DataArray(
            arrays[var],
            dims=["time", "lat", "lon"],
            attrs=attrs,
        )
    data_vars["n_obs"] = xr.DataArray(
        n_obs,
        dims=["time", "lat", "lon"],
        attrs={"long_name": "number of super-observations per grid cell per month", "units": "count"},
    )

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": time_coords,
            "lat" : ("lat", LAT_CENTERS, {"long_name": "latitude",  "units": "degrees_north"}),
            "lon" : ("lon", LON_CENTERS, {"long_name": "longitude", "units": "degrees_east"}),
        },
        attrs={
            "title"      : "OCO-2 XCO2 anomaly 2D monthly grid (0.25°)",
            "source"     : "pipeline/02c_harmonic_baseline.py → pipeline/02d_grid_2d.py",
            "baseline_A" : "(lat_bin, year) harmonic fitting — 2-harmonic, T=365.25",
            "baseline_B" : "(eaic_region, year) harmonic fitting — 2-harmonic, T=365.25",
            "resolution" : "0.25 degree",
            "lat_range"  : "20.0 ~ 50.0 N",
            "lon_range"  : "100.0 ~ 150.0 E",
        },
    )
    return ds


def _var_attrs(var: str) -> dict:
    attr_map = {
        "xco2"           : {"long_name": "XCO2 column-averaged CO2 mole fraction",  "units": "ppm"},
        "xco2_anomaly_A" : {"long_name": "XCO2 anomaly — Scheme A (lat_bin×year harmonic baseline)", "units": "ppm"},
        "xco2_anomaly_B" : {"long_name": "XCO2 anomaly — Scheme B (region×year harmonic baseline)",  "units": "ppm"},
        "tropomi_no2"    : {"long_name": "TROPOMI tropospheric NO2 column", "units": "mol/m2"},
        "era5_u10"       : {"long_name": "ERA5 10m zonal wind",       "units": "m/s"},
        "era5_v10"       : {"long_name": "ERA5 10m meridional wind",  "units": "m/s"},
        "era5_wind_speed": {"long_name": "ERA5 10m wind speed",       "units": "m/s"},
        "era5_blh"       : {"long_name": "ERA5 boundary layer height","units": "m"},
    }
    return attr_map.get(var, {"long_name": var})


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------
def run(in_path: Path = IN_PARQUET, out_nc: Path = OUT_NC) -> None:
    print(f"[02d] 입력 로드: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"  총 {len(df):,}행")

    print("[02d] 월별 2D 집계 중...")
    ds = aggregate_monthly(df)

    # 저장
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for v in ds.data_vars if v != "n_obs"}
    encoding["n_obs"] = {"zlib": True, "complevel": 4, "dtype": "int16"}
    ds.to_netcdf(out_nc, encoding=encoding)
    print(f"[02d] 저장 완료: {out_nc}")

    # 요약
    print("\n=== 출력 Dataset 요약 ===")
    print(ds)
    print()

    # 유효 커버리지 통계
    valid_A = (~np.isnan(ds["xco2_anomaly_A"].values)).sum()
    total   = ds["xco2_anomaly_A"].size
    print(f"xco2_anomaly_A 유효 셀 수: {valid_A:,} / {total:,} ({100*valid_A/total:.2f}%)")

    valid_B = (~np.isnan(ds["xco2_anomaly_B"].values)).sum()
    print(f"xco2_anomaly_B 유효 셀 수: {valid_B:,} / {total:,} ({100*valid_B/total:.2f}%)")

    # 월별 관측 커버리지
    n_obs_monthly = (ds["n_obs"].values > 0).sum(axis=(1, 2))
    print(f"\n월별 유효 셀 수 (min/mean/max): "
          f"{n_obs_monthly.min()} / {n_obs_monthly.mean():.0f} / {n_obs_monthly.max()}")


if __name__ == "__main__":
    run()
