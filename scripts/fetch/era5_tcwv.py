"""ERA5 단일면 total_column_water_vapour(tcwv) 월 파일 — A-6 배경 입력 (사용자 승인 2026-10-08, docs/decision_a6_sif_gate_2026-10-08.md).

기존 era5_wind·era5_syn 파일은 건드리지 않는 별도 파일이다 (설계 검토 D2: 단일면 원본 요청에 변수를 더하면 이미 받은 달은 건너뛰고,
파생 wind 파일을 다시 만들면 이류 입력까지 바뀐다).
출력: <out>/era5_tcwv_YYYYMM.nc — tcwv [time, lat(오름차순, 격자 규약), lon] float32 kg m⁻², zlib. 도메인·격자는 scripts/fetch/era5.py 와 같다.
CDS 자격증명은 cdsapi 기본 설정을 그대로 쓴다(이 스크립트는 자격증명 파일을 읽지 않는다).

사용: python scripts/fetch/era5_tcwv.py                 # 2020 12개월
      python scripts/fetch/era5_tcwv.py --months 202001 202002
"""
import argparse
import calendar
import os
import time
import zipfile

import numpy as np
import xarray as xr

from no2xco2.config import ERA5_TCWV_DIR, GRID_LAT0, GRID_LON0
from no2xco2.data.era5 import NLAT, NLON

AREA = [50.0, 100.0, 20.0, 150.0]  # N, W, S, E — scripts/fetch/era5.py 와 같은 동아시아 도메인
GRID = [0.25, 0.25]
MONTHS_2020 = [f"2020{m:02d}" for m in range(1, 13)]


def _norm(ds: xr.Dataset) -> xr.Dataset:
    """CDS 버전별 좌표명 차이 흡수 (valid_time→time, latitude→lat, longitude→lon)."""
    ren = {k: v for k, v in (("valid_time", "time"), ("latitude", "lat"), ("longitude", "lon")) if k in ds.coords or k in ds.dims}
    return ds.rename(ren) if ren else ds


def _request(ym: str, path: str) -> None:
    import cdsapi
    y, m = int(ym[:4]), int(ym[4:]); nd = calendar.monthrange(y, m)[1]
    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels",
        {"product_type": "reanalysis", "variable": ["total_column_water_vapour"], "year": str(y), "month": f"{m:02d}",
         "day": [f"{d:02d}" for d in range(1, nd + 1)], "time": [f"{h:02d}:00" for h in range(24)],
         "area": AREA, "grid": GRID, "data_format": "netcdf"},
        path + ".part")
    os.replace(path + ".part", path)  # 원자적: 중단된 다운로드가 완성본으로 남지 않게


def fetch_month(ym: str, out: str) -> str:
    o = os.path.join(out, f"era5_tcwv_{ym}.nc")
    if os.path.exists(o):
        return "skip"
    raw = os.path.join(out, f"_raw_tcwv_{ym}.download")
    if not os.path.exists(raw):
        _request(ym, raw)
    src, extracted = raw, None
    if zipfile.is_zipfile(raw):  # CDS 가 netcdf 를 zip 으로 묶어 줄 때
        with zipfile.ZipFile(raw) as z:
            names = [n for n in z.namelist() if n.endswith(".nc")]
            assert len(names) == 1, f"{ym}: zip 안 nc {names}"
            z.extract(names[0], out); src = extracted = os.path.join(out, names[0])
    with xr.open_dataset(src) as ds0:
        ds = _norm(ds0)
        assert "tcwv" in ds.data_vars, f"{ym}: 변수 {list(ds.data_vars)}"
        a = ds["tcwv"]
        assert a.dims == ("time", "lat", "lon"), f"{ym}: 차원 {a.dims} (expver 등 추가 차원은 처리하지 않고 중단)"
        lat = ds.lat.values.astype(np.float64); lon = ds.lon.values.astype(np.float64); t = ds.time.values; A = a.values.astype(np.float32)
    if lat[0] > lat[-1]:  # 격자 규약: 위도 오름차순
        lat = lat[::-1]; A = A[:, ::-1, :]
    y, m = int(ym[:4]), int(ym[4:]); T = calendar.monthrange(y, m)[1] * 24
    assert A.shape == (T, NLAT, NLON), f"{ym}: 모양 {A.shape} ≠ {(T, NLAT, NLON)}"
    assert abs(lat[0] - GRID_LAT0) < 1e-6 and abs(lon[0] - GRID_LON0) < 1e-6, f"{ym}: 격자 시작 {lat[0]}, {lon[0]}"
    assert t[0] == np.datetime64(f"{y}-{m:02d}-01T00:00"), f"{ym}: 시작 시각 {t[0]}"
    assert np.isfinite(A).all(), f"{ym}: tcwv 비유한 {int((~np.isfinite(A)).sum())}"
    o_ds = xr.Dataset({"tcwv": (("time", "lat", "lon"), A)}, coords={"time": t, "lat": lat, "lon": lon},
                      attrs={"source": "ERA5 reanalysis-era5-single-levels total_column_water_vapour", "units": "kg m-2",
                             "lat_order": "ascending (grid convention)", "area_NWSE": str(AREA)})
    o_ds.to_netcdf(o + ".tmp", encoding={"tcwv": {"zlib": True, "complevel": 4, "chunksizes": (248, NLAT, NLON)}})
    os.replace(o + ".tmp", o)
    os.remove(raw)
    if extracted:
        os.remove(extracted)
    return f"ok {os.path.getsize(o) / 1e6:.0f} MB · tcwv {A.min():.2f}–{A.max():.2f} kg m⁻²"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=MONTHS_2020); ap.add_argument("--out", default=ERA5_TCWV_DIR)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    for ym in a.months:
        print(f"{ym}: {fetch_month(ym, a.out)} · {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
