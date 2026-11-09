"""SIF 월 격자 규칙 스모크 (합성 화소, NAS 불필요) — 격자점 배정 · 품질 조합 · 해양 0 · 앞뒤 달 채움 · 잔여 0 · NaN 0.
산출물(data/processed/sif/sif_202001.nc)이 있으면 구조(격자 규약·변수 1개·NaN 0)도 확인. 2026-10-08 A-6·C-1."""
import os

import numpy as np
import pandas as pd
import pytest

from no2xco2.data import sif_grid as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F = os.path.join(ROOT, "data/processed/sif/sif_202001.nc")


def _px(lat, lon, sif, qa, cf=0.1, lc=12):
    return {"lat": lat, "lon": lon, "sif743_dc": sif, "qa743": qa, "cloud_fraction": cf, "lc_mask": lc}


def test_sif_grid_rules():
    acc = S.new_accum()
    # 1월: 격자점 (0,0) 에 qa 1.0 두 화소(1, 3) + qa 0.5 한 화소(10) + 구름 0.5 화소(qa 1, 값 5) · 해양 화소(lc 0, sif NaN)
    jan = pd.DataFrame([_px(20.05, 100.1, 1.0, 1.0), _px(19.95, 99.9, 3.0, 1.0), _px(20.0, 100.0, 10.0, 0.5), _px(20.1, 100.0, 5.0, 1.0, cf=0.5),
                        _px(30.0, 120.0, np.nan, np.nan, lc=0)])
    S.accumulate(acc, jan, 1)
    # 3월: 격자점 (4,4) = 20.98–21.12 → (lat 21.0, lon 101.0) 에 값 6 · 1월에도 그 격자점 관측 2 → 2월 채움 = (2+6)/2
    S.accumulate(acc, pd.DataFrame([_px(21.0, 101.0, 6.0, 1.0)]), 3)
    S.accumulate(acc, pd.DataFrame([_px(21.0, 101.1, 2.0, 1.0)]), 1)
    # 7월만 관측된 격자점 (8,8) → 1월은 앞뒤 달 결측 → 0 처리
    S.accumulate(acc, pd.DataFrame([_px(22.0, 102.0, 4.0, 1.0)]), 7)
    grid, state, raw = S.finalize(acc, "qa10")
    assert np.isfinite(grid).all() and grid.dtype == np.float32 and grid.shape == (12, 121, 201)
    assert raw[0, 0, 0] == pytest.approx((1 + 3 + 5) / 3)                      # qa = 1.0 만, cf 거르기 없음
    g05, _, r05 = S.finalize(acc, "qa05"); assert r05[0, 0, 0] == pytest.approx((1 + 3 + 10 + 5) / 4)
    _, _, r02 = S.finalize(acc, "qa10_cf02"); assert r02[0, 0, 0] == pytest.approx(2.0)  # 구름 0.5 화소 제외
    assert state[1, 4, 4] == S.FILLED and grid[1, 4, 4] == pytest.approx(4.0)  # 2월 = 1·3월 평균
    assert state[1, 0, 0] == S.FILLED and grid[1, 0, 0] == pytest.approx(raw[0, 0, 0])  # 3월 결측 → 한쪽만
    assert state[0, 8, 8] == S.ZEROED and grid[0, 8, 8] == 0.0
    assert state[0, 40, 80] == S.OCEAN and grid[0, 40, 80] == 0.0              # lc 0 만 → 해양
    rows = S.month_summary(acc, state, "qa10")
    assert rows[0]["육지 격자점"] == 3 and rows[0]["관측"] == 2 and rows[0]["0 처리"] == 1 and rows[0]["화소_qa05"] == 5


@pytest.mark.skipif(not os.path.exists(F), reason="SIF 격자 산출물 없음")
def test_sif_202001_structure():
    import xarray as xr
    with xr.open_dataset(F) as ds:
        assert list(ds.data_vars) == ["sif743_dc"] and ds.sif743_dc.dims == ("lat", "lon")
        lat, lon = S.grid_coords()
        assert np.allclose(ds.lat.values, lat) and np.allclose(ds.lon.values, lon) and lat[0] < lat[-1]
        assert np.isfinite(ds.sif743_dc.values).all() and ds.attrs["units"] == S.UNITS
