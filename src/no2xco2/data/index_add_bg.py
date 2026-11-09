"""인덱스 oco parquet 에 배경 열을 덧붙인다 (A-6, 사용자 승인 2026-10-08; docs/decision_a6_sif_gate_2026-10-08.md, 설계 검토 D2-4 (b)).

기존 열은 바꾸지 않는다. 저장된 step_h·n0–n3·w0–w3 로 build_index._interp4 와 같은 4점 이중선형 보간을 해 새 열 하나를 **끝에** 붙인다.
  tcwv : ERA5_TCWV_DIR/era5_tcwv_YYYYMM.nc — tcwv [T, lat 오름차순, lon], 사운딩 시각 = step_h → bg_tcwv (kg m⁻²)
  sif  : SIF_DIR/sif_YYYYMM.nc — 월 고정장 [lat 오름차순, lon] (시간 축 없음) → bg_sif. 파일 규약은 Data 산출(AC-C1g) 확정 후 맞춘다
월은 반드시 명시한다(--months; 60개월 기본값 없음 — 2021–2024 는 로컬에 없음, 설계 검토 D2-5).
쓰기 전 원본을 <idx>/_backup_pre_<var>_<실행 시각 YYYYMMDD_HHMMSS>/ 에 복사한다(실행마다 새 디렉토리 — 같은 날 옛 백업을 기준으로 쓰지 않음, QA §33 F2).
쓰기 후 기존 열 전부를 **이번 실행에서 읽은 원본**과 대조한다(NaN 은 같은 위치면 같음으로 봄, QA §33 F1). 불일치면 이번 실행의 백업으로 되돌리고 중단.

사용: python -m no2xco2.data.index_add_bg --var tcwv --months 202001 202002 …
"""
import argparse
import datetime
import os
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from no2xco2.config import ERA5_TCWV_DIR, GRID_LAT0, GRID_LON0, INDEX_DIR, SIF_DIR
from no2xco2.data.build_index import _interp4
from no2xco2.data.era5 import NLAT, NLON

NODE_COLS = ["step_h", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3"]


def _check_grid(ds: xr.Dataset, f: str) -> None:
    lat, lon = ds.lat.values, ds.lon.values
    assert lat.size == NLAT and lon.size == NLON, f"격자 {lat.size}×{lon.size}: {f}"
    assert abs(lat[0] - GRID_LAT0) < 1e-6 and lat[1] > lat[0] and abs(lon[0] - GRID_LON0) < 1e-6, f"격자 규약(위도 오름차순·시작점) 위반: {f}"


def field(var: str, ym: str) -> np.ndarray:
    """[T, NLAT, NLON] (tcwv) 또는 [1, NLAT, NLON] (sif 월 고정장)."""
    if var == "tcwv":
        f = os.path.join(ERA5_TCWV_DIR, f"era5_tcwv_{ym}.nc")
        with xr.open_dataset(f) as ds:
            _check_grid(ds, f)
            return ds["tcwv"].values.astype(np.float32)
    if var == "sif":
        f = os.path.join(SIF_DIR, f"sif_{ym}.nc")
        with xr.open_dataset(f) as ds:
            _check_grid(ds, f)
            vs = list(ds.data_vars)
            assert len(vs) == 1, f"SIF 파일 변수가 1개가 아님 {vs}: {f}"
            A = ds[vs[0]].values.astype(np.float32)
        assert A.shape == (NLAT, NLON), f"SIF 모양 {A.shape}: {f}"
        return A[None]
    raise ValueError(var)


def same_columns(new: pa.Table, orig: pa.Table) -> bool:
    """new 의 orig 열들이 orig 와 같은가: 이름·순서·행 수·arrow 타입이 같고, 값은 pandas DataFrame.equals (같은 위치 NaN = 같음).
    pyarrow ChunkedArray.equals 는 NaN ≠ NaN 이라 NaN 있는 float 열을 늘 다르다고 판정 (QA §33 F1)."""
    sub = new.select(orig.column_names)
    if new.num_rows != orig.num_rows or [f.type for f in sub.schema] != [f.type for f in orig.schema]:
        return False
    return sub.to_pandas().equals(orig.to_pandas())


def add_month(var: str, ym: str, idx: str, backup: str) -> str:
    p = os.path.join(idx, f"oco_{ym}.parquet"); col = f"bg_{var}"
    tab = pq.read_table(p)
    if col in tab.column_names:
        return f"건너뜀 ({col} 있음)"
    df = tab.select(NODE_COLS).to_pandas()
    A = field(var, ym)
    t = df.step_h.to_numpy() if A.shape[0] > 1 else np.zeros(len(df), np.int64)
    assert (t >= 0).all() and (t < A.shape[0]).all(), f"{ym}: step_h 범위 밖 {int(((t < 0) | (t >= A.shape[0])).sum())}"
    vals = _interp4(A, df, t)
    assert np.isfinite(vals).all(), f"{ym}: {col} 비유한 {int((~np.isfinite(vals)).sum())}"
    os.makedirs(backup, exist_ok=True); b = os.path.join(backup, os.path.basename(p))
    shutil.copy2(p, b)  # 이번 실행 디렉토리에 새로 복사 (옛 백업 재사용 없음)
    pq.write_table(tab.append_column(col, pa.array(vals, pa.float32())), p + ".tmp", compression="zstd"); os.replace(p + ".tmp", p)
    chk = pq.read_table(p)
    ok = chk.column_names == tab.column_names + [col] and same_columns(chk, tab)  # 기준 = 이번 실행에서 읽은 원본 tab (QA §33 F2)
    if not ok:
        shutil.copy2(b, p)
        raise AssertionError(f"{ym}: 기존 열 대조 실패 → 원본으로 되돌림")
    return f"{col} 추가 · 행 {chk.num_rows:,} · 기존 {len(tab.column_names)}열 동일 · 범위 {vals.min():.3f}–{vals.max():.3f} · 중앙 {np.median(vals):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True, choices=["tcwv", "sif"]); ap.add_argument("--months", nargs="+", required=True)
    ap.add_argument("--idx-dir", default=INDEX_DIR)
    a = ap.parse_args(); t0 = time.time()
    backup = os.path.join(a.idx_dir, f"_backup_pre_{a.var}_{datetime.datetime.now():%Y%m%d_%H%M%S}")  # 실행마다 새 백업 (QA §33 F2)
    for ym in a.months:
        print(f"{ym}: {add_month(a.var, ym, a.idx_dir, backup)} · {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
