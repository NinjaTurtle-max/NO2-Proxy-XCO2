"""ERA5 기압면 지오퍼텐셜 → 종관 지표 파생 (2026-09-22, 사용자 지시 "ERA5 z 먼저 파생").

입력: NAS era5_pl_raw/era5_pl_YYYYMM.nc (z [time, level, lat(내림차순), lon], m² s⁻², 10층 700–1000 hPa)
출력: <out>/era5_syn_YYYYMM.nc — 위도 오름차순(격자 규약), float32, zlib
  z850 : 850 hPa 지오퍼텐셜 고도 (gpm) — 기단·고기압 지표
  thk  : 700–1000 hPa 두께 (gpm) — 하층 기온(기단 온난/한랭) 지표
  z850_anom : z850 − 그 달 시간 평균 (gpm) — 종관 아노말리
SMB 위 HDF5 읽기가 세그폴트를 낼 수 있어 월마다 자식 프로세스로 격리하고 실패 시 재시도한다.
"""
import argparse
import calendar
import os
import subprocess
import sys
import time

import numpy as np
import xarray as xr

from no2xco2.config import ERA5_SYN_DIR, GRID_LAT0, GRID_LON0, NAS_ERA5_RAW, wait_nas
from no2xco2.data.era5 import NLAT, NLON

G0 = 9.80665
MONTHS = [f"{y}{m:02d}" for y in range(2020, 2025) for m in range(1, 13)]


def _norm(ds: xr.Dataset) -> xr.Dataset:
    """CDS 버전별 좌표명 차이 흡수 (valid_time→time, pressure_level→level) — scripts/fetch/era5._norm 과 같은 규칙 (QA R6 2026-09-22).
    60개월 원본은 전부 time/level 임을 확인(2023-03·04·06, 2024-06·12 표본)."""
    ren = {k: v for k, v in (("valid_time", "time"), ("pressure_level", "level")) if k in ds.coords or k in ds.dims}
    return ds.rename(ren) if ren else ds


def derive_month(ym: str, out: str) -> str:
    f = os.path.join(NAS_ERA5_RAW, f"era5_pl_{ym}.nc"); o = os.path.join(out, f"era5_syn_{ym}.nc")
    ds = _norm(xr.open_dataset(f))
    lev = ds.level.values.tolist(); z = ds["z"]
    z850 = z.isel(level=lev.index(850.0)).values / G0; thk = (z.isel(level=lev.index(700.0)).values - z.isel(level=lev.index(1000.0)).values) / G0
    lat = ds.latitude.values; lon = ds.longitude.values; t = ds.time.values; ds.close()
    if lat[0] > lat[-1]:  # 격자 규약(결정 2): 위도 오름차순으로 저장
        lat = lat[::-1]; z850 = z850[:, ::-1, :]; thk = thk[:, ::-1, :]
    assert abs(lat[0] - GRID_LAT0) < 1e-6 and abs(lon[0] - GRID_LON0) < 1e-6 and z850.shape[1:] == (NLAT, NLON)  # 격자 상수는 config·era5 공유 (QA C8)
    y, m = int(ym[:4]), int(ym[4:]); T = calendar.monthrange(y, m)[1] * 24  # 시간축 = 일수×24, 월 초 시작 (QA Q4; era5.open_wind 와 같은 검사)
    assert t.size == T and t[0] == np.datetime64(f"{y}-{m:02d}-01T00:00"), f"{ym}: 시간축 {t.size} ≠ {T} 또는 시작 {t[0]}"
    anom = z850 - z850.mean(axis=0, keepdims=True)
    o_ds = xr.Dataset({"z850": (("time", "lat", "lon"), z850.astype(np.float32)), "thk": (("time", "lat", "lon"), thk.astype(np.float32)),
                       "z850_anom": (("time", "lat", "lon"), anom.astype(np.float32))},
                      coords={"time": t, "lat": lat, "lon": lon},
                      attrs={"source": os.path.basename(f), "units": "gpm", "lat_order": "ascending (grid convention)", "g0": G0})
    enc = {v: {"zlib": True, "complevel": 4, "chunksizes": (248, NLAT, NLON)} for v in o_ds.data_vars}
    o_ds.to_netcdf(o + ".tmp", encoding=enc); os.replace(o + ".tmp", o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=MONTHS); ap.add_argument("--out", default=ERA5_SYN_DIR)
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a._child:
        derive_month(a._child, a.out); return
    wait_nas(); t0 = time.time()
    for ym in a.months:
        o = os.path.join(a.out, f"era5_syn_{ym}.nc")
        if os.path.exists(o):
            continue
        for k in range(4):
            r = subprocess.run([sys.executable, "-m", "no2xco2.data.era5_syn", "--out", a.out, "--_child", ym], capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(o):
                print(f"{ym} ok {os.path.getsize(o)/1e6:.0f} MB {time.time()-t0:.0f}s", flush=True); break
            print(f"  {ym} 실패 ({k+1}/4) rc={r.returncode}: {r.stderr.strip()[-160:]} → 재시도", flush=True); time.sleep(20); wait_nas()
        else:
            print(f"{ym} 포기", flush=True)


if __name__ == "__main__":
    main()
