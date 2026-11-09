"""700 hPa–지면 Δp 가중 평균풍 규칙 스모크 (합성 4격자점, NAS 불필요) + 산출물 구조 (있으면). 결정 11 ① D5 (가), 2026-10-27."""
import os

import numpy as np
import pytest

from no2xco2.data import era5_wmean as W

P = np.array([700, 775, 800, 850, 875, 900, 925, 950, 975, 1000], float) * 100
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F = os.path.join(ROOT, "data/raw/era5_wmean/era5_wmean_202001.nc")


def test_wmean_rules():
    sp = np.array([101300.0, 99000.0, 76000.0, 65000.0])
    w = W.weights(P, sp)
    assert np.allclose(w.sum(axis=0)[:3], sp[:3] - 70000) and w[:, 3].sum() == 0          # Σw = sp − 700 hPa, sp < 700 → 0
    assert np.allclose(w[:, 0] / 100, [37.5, 50, 37.5, 37.5, 25, 25, 25, 25, 25, 25.5])      # 중간값 경계 · 1000 hPa 층을 sp 까지
    assert w[9, 1] == 0 and np.isclose(w[8, 1] / 100, 27.5)                                   # 지면 아래 1000 hPa 가중 0, 975 hPa 를 sp 까지
    assert np.isclose(w[0, 2] / 100, 60) and w[1:, 2].sum() == 0                              # sp 760: 700 hPa 하나
    u = np.tile(np.arange(10.0)[:, None], (1, 4)); v = np.ones((10, 4))
    um, vm, fb = W.mean_wind(u, v, P, sp, np.full(4, -9.0), np.full(4, -8.0))
    assert np.isclose(um[2], 0.0) and np.allclose(vm[:3], 1.0) and (fb == [0, 0, 0, 1]).all() and um[3] == -9 and vm[3] == -8
    assert np.isclose(um[0], (w[:, 0] * np.arange(10)).sum() / w[:, 0].sum())


@pytest.mark.skipif(not os.path.exists(F), reason="평균풍 산출물 없음")
def test_wmean_202001_structure():
    import xarray as xr
    with xr.open_dataset(F) as ds:
        assert set(ds.data_vars) == {"u_mean", "v_mean", "fallback"} and ds.u_mean.dims == ("time", "lat", "lon")
        assert ds.lat.size == 121 and ds.lon.size == 201 and ds.lat.values[0] == 20.0 and ds.lat.values[-1] == 50.0 and ds.time.size == 744
        a = ds.u_mean.isel(time=slice(0, 24)).values; assert a.dtype == np.float32 and np.isfinite(a).all()
        assert int(ds.attrs["fallback_count"]) == int(ds.fallback.sum())
