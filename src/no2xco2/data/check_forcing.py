"""ERA5 z100 바람장·CarbonTracker 월 파일 60개월 전수 검증 (항목 8).

검사: 파일 존재·열림 / 시간축(길이·시작·끝·1 h 또는 3 h 등간격·중복) / 격자(121×201, 0.25°; CT 17×18) /
변수 존재 / NaN 수 / 값 범위 / 월 경계 연속성(전월 마지막 시각 → 당월 첫 시각 차이를 월내 시간차 중앙값과 비교) /
속성(z_min_m, source) / 원본 era5_sl 의 expver(ERA5T 잠정자료 여부).
판정 임계값은 물리적 불가능 범위만 사용(아래 RANGES). 그 외 수치는 기록만 한다.
출력: docs/results/era5_check.csv · docs/results/ct_check.csv · 표준출력 요약.
--stride 는 통계 표본 간격이나, z100 파일의 시간 청크가 (372, 61, 101) 이라 I/O 절감 효과는 없음(측정: stride 24 도 전체 청크 읽음).
"""
import argparse
import calendar
import os
import time

import numpy as np
import pandas as pd
import xarray as xr

from no2xco2.config import NAS_CT, NAS_ERA5_RAW, NAS_ERA5_WIND, wait_nas

MONTHS = [f"{y}{m:02d}" for y in range(2020, 2025) for m in range(1, 13)]
ERA5_VARS = ["u_pbl", "v_pbl", "blh", "sp", "t2m"]
# 물리적 불가능 범위 (이 밖이면 flag). 통계적 이상 임계는 미설정.
RANGES = {"u_pbl": (-80, 80), "v_pbl": (-80, 80), "blh": (0, 6000), "sp": (45000, 110000), "t2m": (190, 335), "co2": (300, 600)}


def _hours(ym: str) -> int:
    return calendar.monthrange(int(ym[:4]), int(ym[4:]))[1] * 24


def _time_checks(t: np.ndarray, ym: str, step_h: int, first_offset_h: float) -> dict:
    y, m = int(ym[:4]), int(ym[4:])
    n_exp = _hours(ym) // step_h
    t0_exp = np.datetime64(f"{y}-{m:02d}-01T00:00") + np.timedelta64(int(first_offset_h * 60), "m")
    d = np.diff(t).astype("timedelta64[m]").astype(int) if len(t) > 1 else np.array([])
    return {"n_time": len(t), "n_time_ok": len(t) == n_exp, "t_first_ok": bool(len(t) and t[0] == t0_exp),
            "t_last_ok": bool(len(t) and t[-1] == t0_exp + np.timedelta64((n_exp - 1) * step_h, "h")),
            "t_step_ok": bool(len(d) and (d == step_h * 60).all()), "t_dup": int(len(t) - len(np.unique(t)))}


def check_era5_month(ym: str, prev_last: dict | None, stride: int = 1) -> tuple[dict, dict | None]:
    """stride>1 이면 데이터 통계(NaN·범위·풍속)는 stride 시간마다 표본으로 계산(시간축·격자·속성은 전수). 월 경계 항은 항상 첫/마지막 시각."""
    f = os.path.join(NAS_ERA5_WIND, f"era5_wind_{ym}_z100.nc")
    r = {"ym": ym, "file": os.path.basename(f), "exists": os.path.exists(f), "stride": stride}
    if not r["exists"]:
        return r, None
    r["size_MB"] = os.path.getsize(f) / 1e6
    try:
        ds = xr.open_dataset(f)
    except Exception as e:  # noqa: BLE001
        r["open_error"] = str(e)[:120]; return r, None
    t = ds.time.values
    r.update(_time_checks(t, ym, 1, 0))
    r["grid_ok"] = bool(ds.lat.size == 121 and ds.lon.size == 201 and abs(float(ds.lat[0]) - 50) < 1e-6 and abs(float(ds.lat[-1]) - 20) < 1e-6
                        and abs(float(ds.lon[0]) - 100) < 1e-6 and abs(float(ds.lon[-1]) - 150) < 1e-6)
    r["vars_ok"] = all(v in ds for v in ERA5_VARS)
    r["z_min_m"] = ds.attrs.get("z_min_m"); r["src_ok"] = ym in str(ds.attrs.get("source", ""))
    last = {}
    for v in ERA5_VARS:
        if v not in ds:
            continue
        a = ds[v].isel(time=slice(0, None, stride)).values if stride > 1 else ds[v].values
        first, lastv = (ds[v].isel(time=0).values, ds[v].isel(time=-1).values) if stride > 1 else (a[0], a[-1])
        r[f"{v}_nan"] = int(np.isnan(a).sum()); lo, hi = RANGES[v]
        r[f"{v}_min"] = float(np.nanmin(a)); r[f"{v}_max"] = float(np.nanmax(a))
        r[f"{v}_range_ok"] = bool(r[f"{v}_min"] >= lo and r[f"{v}_max"] <= hi)
        if v in ("u_pbl", "v_pbl"):
            r[f"{v}_med_absdiff_h"] = float(np.nanmedian(np.abs(np.diff(a, axis=0))))  # 월내 연속 표본 시각 차 중앙값 (stride h 간격)
        last[v] = lastv
        if prev_last is not None and v in prev_last:
            r[f"{v}_boundary_absdiff"] = float(np.nanmean(np.abs(first - prev_last[v])))  # 전월 말 → 당월 초
    spd = np.hypot(ds["u_pbl"].isel(time=slice(0, None, stride)).values, ds["v_pbl"].isel(time=slice(0, None, stride)).values) if r["vars_ok"] else np.array([np.nan])
    r["spd_p50"] = float(np.nanpercentile(spd, 50)); r["spd_p95"] = float(np.nanpercentile(spd, 95)); r["spd_max"] = float(np.nanmax(spd))
    ds.close()
    # 원본 단일면 파일의 expver (ERA5T 잠정자료 '0005' 포함 여부)
    fs = os.path.join(NAS_ERA5_RAW, f"era5_sl_{ym}.nc")
    if os.path.exists(fs):
        try:
            with xr.open_dataset(fs) as s:
                r["sl_expver"] = ",".join(sorted(set(np.asarray(s.expver.values).astype(str)))) if "expver" in s else "-"
        except Exception as e:  # noqa: BLE001
            r["sl_expver"] = f"err:{str(e)[:40]}"
    else:
        r["sl_expver"] = "raw 없음"
    r["ok"] = bool(r["n_time_ok"] and r["t_first_ok"] and r["t_last_ok"] and r["t_step_ok"] and r["t_dup"] == 0 and r["grid_ok"] and r["vars_ok"]
                   and all(r.get(f"{v}_nan", 1) == 0 and r.get(f"{v}_range_ok", False) for v in ERA5_VARS))
    return r, last


def check_ct_month(ym: str, prev_last: np.ndarray | None) -> tuple[dict, np.ndarray | None]:
    f = os.path.join(NAS_CT, f"ct_ea_{ym}.nc")
    r = {"ym": ym, "file": os.path.basename(f), "exists": os.path.exists(f)}
    if not r["exists"]:
        return r, None
    r["size_MB"] = os.path.getsize(f) / 1e6
    try:
        ds = xr.open_dataset(f)
    except Exception as e:  # noqa: BLE001
        r["open_error"] = str(e)[:120]; return r, None
    r.update(_time_checks(ds.time.values, ym, 3, 1.5))
    r["grid_ok"] = bool(ds.level.size == 34 and ds.latitude.size == 17 and ds.longitude.size == 18)
    r["vars_ok"] = all(v in ds for v in ("co2", "pressure", "air_mass"))
    r["version"] = str(ds.attrs.get("source_version", ds.attrs.get("version", "")))[:40]
    co2 = ds["co2"].values; am = ds["air_mass"].values
    r["co2_nan"] = int(np.isnan(co2).sum()); r["co2_min"] = float(np.nanmin(co2)); r["co2_max"] = float(np.nanmax(co2))
    r["co2_range_ok"] = bool(RANGES["co2"][0] <= r["co2_min"] and r["co2_max"] <= RANGES["co2"][1])
    r["air_mass_nan"] = int(np.isnan(am).sum()); r["air_mass_min"] = float(np.nanmin(am))
    col = (co2 * am).sum(axis=1) / am.sum(axis=1)  # 질량가중 컬럼 평균 (time, lat, lon)
    r["colmean_p50"] = float(np.nanmedian(col)); r["col_med_absdiff_3h"] = float(np.nanmedian(np.abs(np.diff(col, axis=0))))
    if prev_last is not None:
        r["col_boundary_absdiff"] = float(np.nanmean(np.abs(col[0] - prev_last)))
    ds.close()
    r["ok"] = bool(r["n_time_ok"] and r["t_first_ok"] and r["t_last_ok"] and r["t_step_ok"] and r["t_dup"] == 0 and r["grid_ok"] and r["vars_ok"]
                   and r["co2_nan"] == 0 and r["co2_range_ok"] and r["air_mass_nan"] == 0 and r["air_mass_min"] > 0)
    return r, col[-1]


def _run_months(check, csv_path: str, label: str, t0: float, **kw) -> pd.DataFrame:
    """월별 검사를 재개 가능하게 실행: 기존 CSV 의 ok 행은 건너뛰고(단, 월 경계 항 계산을 위해 직전 월은 다시 읽음),
    월마다 CSV 저장, SMB 끊김(OSError·HDF error)은 wait_nas 후 최대 4회 재시도."""
    done = pd.read_csv(csv_path, dtype={"ym": str}) if os.path.exists(csv_path) else pd.DataFrame()
    done_ok = set(done[done.ok.fillna(False).astype(bool)].ym) if len(done) and "ok" in done else set()
    rows = {r["ym"]: r for r in done.to_dict("records")} if len(done) else {}
    prev = None
    for i, ym in enumerate(MONTHS):
        need_prev = (i + 1 < len(MONTHS)) and (MONTHS[i + 1] not in done_ok)  # 다음 달이 미완이면 이번 달 마지막 시각이 필요
        if ym in done_ok and not need_prev:
            prev = None; continue
        for k in range(4):
            try:
                r, prev = check(ym, prev, **kw); break
            except (OSError, RuntimeError) as e:
                print(f"  {label} {ym} 읽기 실패 ({k+1}/4): {str(e)[:80]} → NAS 대기", flush=True); time.sleep(20); wait_nas()
        else:
            r = {"ym": ym, "exists": True, "ok": False, "open_error": "4회 실패"}; prev = None
        if ym not in done_ok:
            rows[ym] = r
            print(f"{label} {ym} {'ok' if r.get('ok') else ('없음' if not r.get('exists') else 'FLAG')} {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame([rows[m] for m in MONTHS if m in rows]).to_csv(csv_path, index=False)
    return pd.DataFrame([rows[m] for m in MONTHS if m in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results"); ap.add_argument("--skip-era5", action="store_true"); ap.add_argument("--skip-ct", action="store_true"); ap.add_argument("--stride", type=int, default=1, help="ERA5 데이터 통계 표본 간격(h). 1=전수")
    a = ap.parse_args(); wait_nas(); t0 = time.time()
    if not a.skip_era5:
        e = _run_months(check_era5_month, f"{a.out}/era5_check.csv", "ERA5", t0, stride=a.stride)
        print(f"\nERA5 z100: 존재 {int(e.exists.sum())}/60 · ok {int(e.ok.fillna(False).sum())} · flag {int((e.exists & ~e.ok.fillna(False)).sum())}")
        print(e[e.exists][["ym", "n_time", "spd_p50", "spd_p95", "spd_max", "blh_max", "sp_min", "t2m_min", "t2m_max", "u_pbl_boundary_absdiff", "u_pbl_med_absdiff_h", "sl_expver", "ok"]].to_string(index=False))
    if not a.skip_ct:
        c = _run_months(check_ct_month, f"{a.out}/ct_check.csv", "CT", t0)
        print(f"\nCT: 존재 {int(c.exists.sum())}/60 · ok {int(c.ok.fillna(False).sum())} · flag {int((c.exists & ~c.ok.fillna(False)).sum())} ({time.time()-t0:.0f}s)")
        print(c[c.exists][["ym", "version", "n_time", "co2_min", "co2_max", "colmean_p50", "col_boundary_absdiff", "col_med_absdiff_3h", "ok"]].to_string(index=False))


if __name__ == "__main__":
    main()
