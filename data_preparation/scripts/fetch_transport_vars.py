"""
=============================================================
 ERA5 수송(transport) 변수 취득 — 리뷰 디펜스 서브 실험용
 fetch_transport_vars.py

 목적: "대기 수송 변수를 넣으면 oracle 천장이 오르지 않나?"라는
       예상 리뷰에 대한 선제 디펜스(Supplementary Table) 데이터 취득.

 취득 대상 (모두 monthly means, 2020–2023, 동아시아 0.25°):
   1. z500   — 500 hPa geopotential → height(m)로 변환 (종관 패턴/블로킹)
   2. div850 — 850 hPa horizontal divergence (하층 수렴·발산 = 축적/환기)
   3. u10/v10 — 10 m 바람 성분 → 풍향 sin/cos 인코딩용
                (기존 era5_wind_speed 와 동일 레벨로 일관성 유지)

 출력: data/raw/auxiliary/era5_transport_monthly.nc
       변수: z500 [m], div850 [s-1], u10 [m s-1], v10 [m s-1]
       격자: pipeline/01_spatial_alignment.py 와 동일 (20–50N, 100–150E, 0.25°)

 실행: python data_preparation/scripts/fetch_transport_vars.py
 사전: ~/.cdsapirc (fetch_auxiliary_vars.py 와 동일)
=============================================================
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[2]
AUX_DIR = ROOT / "data" / "raw" / "auxiliary"
AUX_DIR.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0
RES = 0.25
# AUG_NC/parquet 좌표 = 셀 중심 (edge + RES/2) — 병합 키 일치 필수
LAT_GRID = np.arange(LAT_MIN, LAT_MAX, RES) + RES / 2
LON_GRID = np.arange(LON_MIN, LON_MAX, RES) + RES / 2

YEARS  = [str(y) for y in range(2020, 2024)]            # 기본 2020–2023
MONTHS = [f"{m:02d}" for m in range(1, 13)]
AREA   = [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX]            # [N, W, S, E]

G0 = 9.80665                                             # 표준 중력 [m s-2]

OUT_NC = AUX_DIR / "era5_transport_monthly.nc"
SUFFIX = ""                                              # --suffix 시 출력/raw 파일명에 부가


def _align(da: xr.DataArray) -> xr.DataArray:
    """ERA5 격자(내림차순 lat, 0.25 배수점) → 셀 중심 격자로 선형 보간."""
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"
    da = da.rename({lat_name: "lat", lon_name: "lon"}).sortby("lat")
    # sel/squeeze 잔류 스칼라 좌표(pressure_level, number, expver 등) 제거 — 병합 충돌 방지
    da = da.drop_vars([c for c in da.coords if c not in ("time", "lat", "lon")], errors="ignore")
    return da.interp(lat=LAT_GRID, lon=LON_GRID, method="linear")


def _open_time_fixed(path: Path) -> xr.Dataset:
    """CDS monthly means 의 time 축 이름(valid_time/date) 표준화."""
    ds = xr.open_dataset(path)
    for cand in ("valid_time", "date", "forecast_reference_time"):
        if cand in ds.dims or cand in ds.coords:
            ds = ds.rename({cand: "time"})
            break
    ds["time"] = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).to_period("M").to_timestamp()
    return ds


def fetch(overwrite: bool = False) -> Path:
    out_nc = AUX_DIR / f"era5_transport_monthly{SUFFIX}.nc"
    if out_nc.exists() and not overwrite:
        print(f"⏭️  스킵 (기존 파일): {out_nc}")
        return out_nc

    import cdsapi
    c = cdsapi.Client()

    raw_pl = AUX_DIR / f"_era5_transport_pl_raw{SUFFIX}.nc"   # pressure levels
    raw_sl = AUX_DIR / f"_era5_transport_sl_raw{SUFFIX}.nc"   # single levels

    # ① pressure-levels: geopotential(500) + divergence(850) — 단일 요청
    if not raw_pl.exists():
        print("📥 ERA5 pressure-levels monthly means (z500, div850) ...", flush=True)
        c.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type"  : ["monthly_averaged_reanalysis"],
                "variable"      : ["geopotential", "divergence"],
                "pressure_level": ["500", "850"],
                "year"          : YEARS,
                "month"         : MONTHS,
                "time"          : ["00:00"],
                "area"          : AREA,
                "data_format"   : "netcdf",
            },
            str(raw_pl),
        )
        print("   완료")

    # ② single-levels: 10m u/v — 단일 요청
    if not raw_sl.exists():
        print("📥 ERA5 single-levels monthly means (u10, v10) ...", flush=True)
        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable"    : ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year"        : YEARS,
                "month"       : MONTHS,
                "time"        : ["00:00"],
                "area"        : AREA,
                "data_format" : "netcdf",
            },
            str(raw_sl),
        )
        print("   완료")

    # ── 전처리 → 단일 NC ──────────────────────────────────────────
    print("🔧 전처리: 레벨 선택 + 격자 정렬 + 단위 변환")
    pl = _open_time_fixed(raw_pl)
    lev_name = next(d for d in ("pressure_level", "level", "isobaricInhPa") if d in pl.dims)
    z_key = next(k for k in pl.data_vars if k.lower() in ("z", "geopotential"))
    d_key = next(k for k in pl.data_vars if k.lower() in ("d", "divergence"))

    z500   = _align(pl[z_key].sel({lev_name: 500}).squeeze() / G0).astype(np.float32)
    div850 = _align(pl[d_key].sel({lev_name: 850}).squeeze()).astype(np.float32)

    sl = _open_time_fixed(raw_sl)
    u_key = next(k for k in sl.data_vars if k.lower() in ("u10", "10u"))
    v_key = next(k for k in sl.data_vars if k.lower() in ("v10", "10v"))
    u10 = _align(sl[u_key].squeeze()).astype(np.float32)
    v10 = _align(sl[v_key].squeeze()).astype(np.float32)

    out = xr.Dataset(
        {
            "z500"  : z500.assign_attrs(units="m",    long_name="ERA5 500 hPa geopotential height"),
            "div850": div850.assign_attrs(units="s-1", long_name="ERA5 850 hPa horizontal divergence"),
            "u10"   : u10.assign_attrs(units="m s-1", long_name="ERA5 10 m u-wind"),
            "v10"   : v10.assign_attrs(units="m s-1", long_name="ERA5 10 m v-wind"),
        },
        attrs={"source": "ERA5 monthly means (CDS)", "purpose": "transport-oracle defense experiment"},
    )
    out.to_netcdf(out_nc)
    pl.close(); sl.close()
    print(f"✅ 저장: {out_nc}  (time={out.sizes['time']}, lat={out.sizes['lat']}, lon={out.sizes['lon']})")
    return out_nc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", default=None,
                    help="예: --years 2024 (기본 2020–2023)")
    ap.add_argument("--suffix", default="",
                    help="출력/raw 파일명 부가 (예: _2024 → era5_transport_monthly_2024.nc)")
    a = ap.parse_args()
    if a.years:
        YEARS = [str(y) for y in a.years]
    SUFFIX = a.suffix
    fetch()
