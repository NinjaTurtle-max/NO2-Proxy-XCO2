"""
fetch_cams_xco2.py — CAMS 전지구 GHG 역산 XCO₂ 취득 (재구성 외부 검증용)
=========================================================================
목적: PINN 재구성장의 외부 검증(리뷰 디펜스). OCO-2 스와스 외 영역의
대규모 공간 패턴을 독립 모델장(CAMS inversion)과 비교한다.
주의: 해상도 ~1.9°×3.75°로 거칠어 줄무늬(0.25° 스케일) 검증은 불가 —
      대규모 패턴 상관 검증 전용. 줄무늬는 15_striping_audit.py가 담당.

산출: data/raw/auxiliary/cams_xco2_monthly.nc  (XCO2 [ppm], 2020–2024,
      동아시아 서브셋, 네이티브 격자 유지 — 비교 시 재구성장을 이 격자로 집계)

실행: python data_preparation/scripts/fetch_cams_xco2.py
사전: ~/.adsapirc (fetch_auxiliary_vars.py 와 동일)
"""
from __future__ import annotations

import os
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
AUX_DIR = ROOT / "data" / "raw" / "auxiliary"
AUX_DIR.mkdir(parents=True, exist_ok=True)

ADS_URL = "https://ads.atmosphere.copernicus.eu/api"

LAT_MIN, LAT_MAX = 18.0, 52.0      # 여유 포함 (네이티브 격자가 거칠어 경계 셀 확보)
LON_MIN, LON_MAX = 96.0, 154.0
YEARS = [str(y) for y in range(2020, 2025)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

OUT_NC = AUX_DIR / "cams_xco2_monthly.nc"


def _load_ads_key() -> str:
    key = os.environ.get("ADS_KEY", "")
    if key:
        return key
    rc = Path.home() / ".adsapirc"
    if rc.exists():
        for line in rc.read_text().splitlines():
            if line.strip().startswith("key:"):
                return line.split(":", 1)[1].strip()
    return ""


def fetch(overwrite: bool = False) -> Path:
    if OUT_NC.exists() and not overwrite:
        print(f"⏭️  스킵 (기존 파일): {OUT_NC}")
        return OUT_NC

    import cdsapi
    c = cdsapi.Client(url=ADS_URL, key=_load_ads_key())

    raw = AUX_DIR / "_cams_xco2_raw.zip"
    if not raw.exists():
        print("📥 CAMS global greenhouse gas inversion (XCO2 column) ...", flush=True)
        c.retrieve(
            "cams-global-greenhouse-gas-inversion",
            {
                "variable": "carbon_dioxide",
                "quantity": "mean_column",
                "input_observations": "surface",
                "time_aggregation": "instantaneous",
                "version": "latest",
                "year": YEARS,
                "month": MONTHS,
            },
            str(raw),
        )
        print("   완료")

    # ── zip 해제 → 병합 → 동아시아 서브셋 ──
    ex_dir = AUX_DIR / "_cams_xco2_nc"
    ex_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(raw) as z:
        z.extractall(ex_dir)
    files = sorted(ex_dir.glob("*.nc"))
    print(f"🔧 병합: {len(files)}개 파일")
    ds = xr.open_mfdataset(files, combine="by_coords")

    # 변수명 표준화 (XCO2 후보 탐색)
    var = next(k for k in ds.data_vars
               if "xco2" in k.lower() or "co2" in k.lower())
    da = ds[var]
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"
    da = da.rename({lat_name: "lat", lon_name: "lon"}).sortby("lat")
    # 경도 0–360 → −180–180 보정 (필요시)
    if float(da.lon.max()) > 180.0:
        da = da.assign_coords(lon=(("lon",), np.where(da.lon.values > 180,
                                                      da.lon.values - 360,
                                                      da.lon.values))).sortby("lon")
    da = da.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

    # 시간축 표준화 (월초)
    tname = next(d for d in ("time", "valid_time", "date") if d in da.dims or d in da.coords)
    if tname != "time":
        da = da.rename({tname: "time"})
    da["time"] = pd.DatetimeIndex(pd.to_datetime(da["time"].values)).to_period("M").to_timestamp()

    # 단위: kg/kg 또는 mol fraction → ppm 변환 시도
    units = str(da.attrs.get("units", "")).lower()
    arr = da.astype(np.float32)
    if "ppm" not in units:
        med = float(arr.median())
        if 3e-4 < med < 6e-4:          # mol/mol
            arr = arr * 1e6
            arr.attrs["units"] = "ppm"
        else:
            arr.attrs["units"] = units or "unknown"

    out = xr.Dataset({"xco2": arr},
                     attrs={"source": "CAMS global greenhouse gas inversion (ADS)",
                            "purpose": "PINN reconstruction external validation",
                            "native_variable": var})
    out.to_netcdf(OUT_NC)
    print(f"✅ 저장: {OUT_NC}  (time={out.sizes['time']}, "
          f"lat={out.sizes['lat']}, lon={out.sizes['lon']}, units={arr.attrs.get('units')})")
    return OUT_NC


if __name__ == "__main__":
    fetch()
