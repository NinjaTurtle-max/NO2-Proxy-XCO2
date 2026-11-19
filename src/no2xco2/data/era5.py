"""ERA5 월 파일을 격자 규약(행 i ↔ lat = 20 + 0.25·i, 열 j ↔ lon = 100 + 0.25·j, 위도 오름차순)으로 연다.

CDS 에서 받은 파일은 위도 내림차순(50 → 20)이다. ERA5 를 읽는 모든 코드는 이 함수를 거쳐 오름차순 격자를 쓴다.
"""
import calendar
import numpy as np
import xarray as xr

from no2xco2.config import GRID_D, GRID_LAT0, GRID_LON0

NLAT, NLON = 121, 201


def open_wind(path: str, ym: str | None = None) -> xr.Dataset:
    """위도 오름차순 · 격자 규약 · 시간축 길이를 보장한 Dataset. ym(YYYYMM) 을 주면 시간축 길이 · 시작 시각도 검사."""
    ds = xr.open_dataset(path)
    if float(ds.lat[0]) > float(ds.lat[-1]):
        ds = ds.sortby("lat")
    lat = ds.lat.values; lon = ds.lon.values
    assert lat.size == NLAT and lon.size == NLON, f"격자 크기 {lat.size}×{lon.size} ≠ {NLAT}×{NLON}: {path}"
    assert abs(lat[0] - GRID_LAT0) < 1e-6 and abs(lat[1] - lat[0] - GRID_D) < 1e-6, f"위도 규약 위반 lat[0]={lat[0]}: {path}"
    assert abs(lon[0] - GRID_LON0) < 1e-6 and abs(lon[1] - lon[0] - GRID_D) < 1e-6, f"경도 규약 위반 lon[0]={lon[0]}: {path}"
    if ym is not None:
        y, m = int(ym[:4]), int(ym[4:]); T = calendar.monthrange(y, m)[1] * 24
        assert ds.time.size == T, f"시간축 {ds.time.size} ≠ {T} h: {path}"
        assert ds.time.values[0] == np.datetime64(f"{y}-{m:02d}-01T00:00"), f"시작 시각 {ds.time.values[0]} ≠ 월 초: {path}"
    return ds
