"""ERA5 z100 월 파일을 격자 규약(결정 2: 행 i ↔ lat = 20 + 0.25·i, 열 j ↔ lon = 100 + 0.25·j, 즉 위도 오름차순)으로 연다.

배경: CDS/PC 파생 파일은 위도 내림차순(50 → 20)으로 저장돼 있다. 2026-09-22 검수에서 모든 소비 코드(bilinear 인덱스·Physics·배경 보간)가
파일을 그대로 인덱싱해 남북이 뒤집힌 장을 썼음이 확인됨 (2020-01: corr(bg_sp, ret_psurf) 0.564 → 뒤집으면 0.983).
이 모듈이 유일한 ERA5 진입점이다: 여기서 오름차순으로 정렬하고 규약을 assert 한다.
"""
import calendar
import os

import numpy as np
import xarray as xr

from no2xco2.config import GRID_D, GRID_LAT0, GRID_LON0, NAS_MOUNT

NLAT, NLON = 121, 201


def open_wind(path: str, ym: str | None = None) -> xr.Dataset:
    """위도 오름차순·격자 규약·시간축 길이를 보장한 Dataset. ym(YYYYMM) 을 주면 시간축 길이·시작 시각도 검사.
    SMB 마운트 위 경로는 거부한다 — HDF5 직접 읽기가 프로세스를 세그폴트로 죽인 이력(build_index: 2020-04·2021-01) 때문에 로컬 사본만 연다 (QA 2026-09-22)."""
    assert not os.path.abspath(path).startswith(NAS_MOUNT.rstrip("/") + "/"), f"SMB 마운트 위 HDF5 직접 읽기 금지 — 로컬 사본을 쓰세요: {path}"
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
