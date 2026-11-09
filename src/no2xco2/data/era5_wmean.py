"""ERA5 기압면 u·v → 700 hPa–지면 Δp 가중 평균풍 (결정 11 ① D5 (가), 사용자 승인 2026-10-27 "승인 + W-c 수정").

입력: NAS era5_pl_raw/era5_pl_YYYYMM.nc (u·v, 10층 700–1000 hPa, 1시간 0.25°, 위도 내림차순)
      + data/raw/era5_wind/era5_wind_YYYYMM_z100.nc (sp · u_pbl · v_pbl, 같은 격자·시각)
출력: <out>/era5_wmean_YYYYMM.nc — u_mean · v_mean (float32, m/s) · fallback (uint8), (time, lat, lon), 위도 오름차순(격자 규약)
정의 (W-a·W-b): 층 경계 = 이웃 층 중간값, 위 경계 700 hPa. 지면 아래 층(p_k > sp)은 가중 0,
  지면 위 최하층의 층을 sp 까지 연장 → Σw = sp − 700 hPa.
W-c (사용자 수정): Σw = 0 (sp ≤ 700 hPa, 지면 위 층 두께 0) → 같은 파일의 경계층 대표풍 u_pbl·v_pbl 로 대체, fallback = 1.
  (sp 가 정확히 700 hPa 인 경우 포함 — 2026-10-27 연구책임자 승인, 수치 정의 불변)
--local-copy: 월 원본을 로컬 임시 폴더로 순차 복사 → 계산 → 복사본 삭제 (SMB 청크 읽기 회피, 한 번에 1개월분만). 복사·계산 초를 따로 출력.
절전 감지: 월 처리 중 pmset 로그에 Sleep 이 있으면 그 달 출력을 최종 이름으로 두지 않고(.slept) 실패로 멈춘다. 포기·중단 월이 있으면 종료 코드 1.
SMB 위 HDF5 읽기가 세그폴트를 낼 수 있어 월마다 자식 프로세스로 격리한다 (era5_syn 과 같은 구조).
읽기는 원본 청크(시간 232–248 × 층 3 × 위도 41 × 경도 67)에 맞춰 (월 전체 시간 아님) 시간 청크 × 위도 41 띠 단위.
"""
import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from no2xco2.config import GRID_LAT0, GRID_LON0, LOCAL_STAGE_RAW, NAS_ERA5_RAW, wait_nas
from no2xco2.data.era5 import NLAT, NLON

OUT_DIR = "data/raw/era5_wmean"  # config.ERA5_WMEAN_DIR (worktree) 와 같은 값
WIND_DIR = "data/raw/era5_wind"
P_TOP = 70000.0  # Pa
MONTHS = [f"2020{m:02d}" for m in range(1, 13)]


def layer_bounds(p):
    """p: 오름차순 층 기압(Pa). 층 k 의 (위 경계, 아래 경계). 위 = 700 hPa 또는 중간값, 아래 = 중간값 (최하층은 +inf → sp 에서 자름)."""
    mid = 0.5 * (p[:-1] + p[1:])
    return np.concatenate([[P_TOP], mid]), np.concatenate([mid, [np.inf]])


def weights(p, sp):
    """sp: (...) Pa → w: (len(p), ...) Pa. 지면 위 층(p_k ≤ sp)만, 최하 유효층은 sp 까지. sp < 700 hPa 이면 전부 0."""
    top, bot = layer_bounds(p)
    shape = (len(p),) + (1,) * np.ndim(sp)
    top, bot, pk = top.reshape(shape), bot.reshape(shape), p.reshape(shape)
    nvalid = (pk <= sp[None]).sum(axis=0)                     # 유효층 = 0..nvalid−1 (p 오름차순이라 연속)
    k = np.arange(len(p)).reshape(shape)
    lowest = k == (nvalid[None] - 1)
    return np.where(k < nvalid[None] - 1, bot - top, np.where(lowest, sp[None] - top, 0.0))


def mean_wind(u, v, p, sp, u_fb, v_fb):
    """u, v: (lev, ...) 오름차순 p 순서. 반환 (u_mean, v_mean, fallback)."""
    w = weights(p, sp).astype(np.float32); ws = w.sum(axis=0)  # float32: 띠 단위 메모리 절반
    fb = ws <= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        um = (w * u).sum(axis=0) / ws; vm = (w * v).sum(axis=0) / ws
    um = np.where(fb, u_fb, um); vm = np.where(fb, v_fb, vm)
    return um.astype(np.float32), vm.astype(np.float32), fb.astype(np.uint8)


def derive_month(ym, out, raw_dir=NAS_ERA5_RAW, wind_dir=WIND_DIR):
    import netCDF4 as nc
    import xarray as xr
    f = os.path.join(raw_dir, f"era5_pl_{ym}.nc"); fw = os.path.join(wind_dir, f"era5_wind_{ym}_z100.nc"); o = os.path.join(out, f"era5_wmean_{ym}.nc")
    with nc.Dataset(f) as d, nc.Dataset(fw) as dw:
        tn = "valid_time" if "valid_time" in d.variables else "time"; ln = "pressure_level" if "pressure_level" in d.variables else "level"
        lev = d[ln][:].astype(np.float64) * 100.0; order = np.argsort(lev); p = lev[order]
        lat, lon = d["latitude"][:], d["longitude"][:]
        assert np.allclose(lat, dw["lat"][:]) and np.allclose(lon, dw["lon"][:]), f"{ym}: 원본·wind 격자 불일치"
        t = nc.num2date(d[tn][:], d[tn].units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        tw = nc.num2date(dw["time"][:], dw["time"].units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        t = np.array(t, dtype="datetime64[ns]"); assert (t == np.array(tw, dtype="datetime64[ns]")).all(), f"{ym}: 원본·wind 시간축 불일치"
        T = t.size; tc = d["u"].chunking()[0]; lc = d["u"].chunking()[2]
        U = np.empty((T, NLAT, NLON), np.float32); V = np.empty_like(U); FB = np.empty((T, NLAT, NLON), np.uint8)
        for t0 in range(0, T, tc):
            for y0 in range(0, NLAT, lc):
                ts, ys = slice(t0, min(t0 + tc, T)), slice(y0, min(y0 + lc, NLAT))
                u = d["u"][ts, :, ys, :].filled(np.nan)[:, order].transpose(1, 0, 2, 3)
                v = d["v"][ts, :, ys, :].filled(np.nan)[:, order].transpose(1, 0, 2, 3)
                sp = dw["sp"][ts, ys, :].filled(np.nan).astype(np.float64)  # Pa
                U[ts, ys], V[ts, ys], FB[ts, ys] = mean_wind(u, v, p, sp, dw["u_pbl"][ts, ys, :].filled(np.nan), dw["v_pbl"][ts, ys, :].filled(np.nan))
    if lat[0] > lat[-1]:  # 격자 규약: 위도 오름차순 저장
        lat = lat[::-1]; U = U[:, ::-1]; V = V[:, ::-1]; FB = FB[:, ::-1]
    assert abs(lat[0] - GRID_LAT0) < 1e-6 and abs(lon[0] - GRID_LON0) < 1e-6 and U.shape[1:] == (NLAT, NLON)
    assert np.isfinite(U).all() and np.isfinite(V).all(), f"{ym}: 비유한 값"
    nfb = int(FB.sum())
    ds = xr.Dataset({"u_mean": (("time", "lat", "lon"), U), "v_mean": (("time", "lat", "lon"), V), "fallback": (("time", "lat", "lon"), FB)},
                    coords={"time": t, "lat": np.asarray(lat), "lon": np.asarray(lon)},
                    attrs={"title": "ERA5 700 hPa–지면 Δp 가중 평균풍 (결정 11 ① D5 (가), 2026-10-27 승인)", "units": "m s-1",
                           "levels_hPa": ",".join(f"{x / 100:g}" for x in p),
                           "def_weight": "층 경계 = 이웃 층 중간값, 위 경계 700 hPa; 지면 아래 층(p > sp) 가중 0; 지면 위 최하층의 층을 sp 까지 연장 → Σw = sp − 700 hPa",
                           "def_fallback": "Σw = 0 (sp ≤ 700 hPa, 지면 위 층 두께 0) → u_pbl·v_pbl (같은 월 era5_wind_YYYYMM_z100.nc, z_agl=max(BLH/2,100 m) 경계층 대표풍), fallback = 1",
                           "fallback_count": nfb, "fallback_frac": nfb / FB.size,
                           "source": f"{os.path.basename(f)} (u, v); {os.path.basename(fw)} (sp, u_pbl, v_pbl)",
                           "lat_order": "ascending (grid convention)"})
    enc = {"u_mean": {"zlib": True, "complevel": 4}, "v_mean": {"zlib": True, "complevel": 4}, "fallback": {"zlib": True, "complevel": 4}}
    ds.to_netcdf(o + ".tmp", encoding=enc); os.replace(o + ".tmp", o)
    return o, nfb


def slept_since(t0):
    """pmset 로그에서 시각 t0(epoch s) 이후 시스템 Sleep 진입 횟수 (macOS 외에는 0)."""
    try:
        log = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    since = datetime.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S")
    return sum(1 for ln in log.splitlines() if ln[:19] >= since and ln[26:].lstrip().startswith("Sleep "))


def child(ym, out, local_copy):
    if not local_copy:
        t1 = time.time(); _, nfb = derive_month(ym, out); print(f"fallback {nfb} · 계산 {time.time() - t1:.0f}s"); return
    os.makedirs(local_copy, exist_ok=True); src = os.path.join(NAS_ERA5_RAW, f"era5_pl_{ym}.nc"); dst = os.path.join(local_copy, os.path.basename(src))
    try:
        t1 = time.time(); shutil.copyfile(src, dst + ".part"); os.replace(dst + ".part", dst); tc = time.time() - t1
        assert os.path.getsize(dst) == os.path.getsize(src), f"{ym}: 복사 크기 불일치"
        t2 = time.time(); _, nfb = derive_month(ym, out, raw_dir=local_copy)
        print(f"fallback {nfb} · 복사 {tc:.0f}s ({os.path.getsize(dst) / 1e6 / max(tc, 1e-9):.1f} MB/s) · 계산 {time.time() - t2:.0f}s")
    finally:  # 한 번에 1개월분만 — 성공·실패 모두 복사본 삭제 (NAS 원본은 읽기만)
        for f in (dst, dst + ".part"):
            if os.path.exists(f):
                os.remove(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=MONTHS); ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--local-copy", nargs="?", const=LOCAL_STAGE_RAW, default=None, help="월 원본을 로컬로 복사 후 계산 (기본 폴더 config.LOCAL_STAGE_RAW)")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a._child:
        child(a._child, a.out, a.local_copy); return
    wait_nas(); t0 = time.time(); bad = []
    extra = ["--local-copy", a.local_copy] if a.local_copy else []
    for ym in a.months:
        o = os.path.join(a.out, f"era5_wmean_{ym}.nc")
        if os.path.exists(o):  # N-2: 덮어쓰지 않음
            print(f"{ym} 존재 → 건너뜀", flush=True); continue
        for k in range(3):
            t1 = time.time(); r = subprocess.run([sys.executable, "-m", "no2xco2.data.era5_wmean", "--out", a.out, "--_child", ym, *extra], capture_output=True, text=True)
            ns = slept_since(t1)
            if ns:  # 절전이 끼면 그 달은 실패로 두고 중단 (사용자 결정 10-27)
                if os.path.exists(o):
                    os.replace(o, o + ".slept")
                print(f"{ym} 절전 {ns}회 감지 → 실패로 두고 중단 (rc={r.returncode}, {time.time() - t1:.0f}s)", flush=True); sys.exit(1)
            if r.returncode == 0 and os.path.exists(o):
                print(f"{ym} ok {os.path.getsize(o) / 1e6:.0f} MB · {r.stdout.strip()} · {time.time() - t1:.0f}s (누적 {time.time() - t0:.0f}s)", flush=True); break
            print(f"  {ym} 실패 ({k + 1}/3) rc={r.returncode}: {r.stderr.strip()[-200:]} → 재시도", flush=True); time.sleep(20); wait_nas()
        else:
            print(f"{ym} 포기", flush=True); bad.append(ym)
    if bad:
        print(f"포기 {len(bad)}개월: {' '.join(bad)}", flush=True); sys.exit(1)


if __name__ == "__main__":
    main()
