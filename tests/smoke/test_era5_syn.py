"""era5_syn 산출물 구조 스모크 (수치 기준값 없음 — 구조만 고정: 격자 규약·시간축 길이·변수·NaN).
전제: data/raw/era5_syn/era5_syn_202001.nc (없으면 skip). 2026-09-22 QA 검수에서 추가."""
import calendar
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
YM = "202001"; F = f"data/raw/era5_syn/era5_syn_{YM}.nc"


@pytest.mark.skipif(not os.path.exists(F), reason="era5_syn 산출물 없음")
def test_era5_syn_202001_structure():
    """격자 규약(결정 2: lat 20→50 오름차순, lon 100→150, 0.25°) · 시간축 = 일수×24 · 변수 3종 float32 · NaN 0."""
    import xarray as xr
    from no2xco2.config import GRID_D, GRID_LAT0, GRID_LON0
    with xr.open_dataset(F) as ds:
        lat = ds.lat.values; lon = ds.lon.values
        assert lat.size == 121 and lon.size == 201
        assert abs(lat[0] - GRID_LAT0) < 1e-6 and abs(lat[1] - lat[0] - GRID_D) < 1e-6 and lat[0] < lat[-1]
        assert abs(lon[0] - GRID_LON0) < 1e-6 and abs(lon[1] - lon[0] - GRID_D) < 1e-6
        y, m = int(YM[:4]), int(YM[4:])
        assert ds.time.size == calendar.monthrange(y, m)[1] * 24 and ds.time.values[0] == np.datetime64(f"{y}-{m:02d}-01T00:00")
        assert set(ds.data_vars) == {"z850", "thk", "z850_anom"}
        for v in ds.data_vars:
            a = ds[v].isel(time=slice(0, 24)).values
            assert a.dtype == np.float32 and a.shape == (24, 121, 201) and np.isfinite(a).all(), v
        assert ds.attrs.get("lat_order", "").startswith("ascending")
