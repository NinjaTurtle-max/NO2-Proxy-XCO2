"""5년 전처리 인덱스 (설계 D1–D4, 2026-09-21 승인).

월별 산출 (<out>/):
  trop_hi_YYYYMM.parquet : qa ≥ QA_HI 화소 — step_g, qa, no2, n0..n3, w0..w3          (주입용, D1)
  trop_lo_YYYYMM.parquet : QA_LO ≤ qa < QA_HI 화소 — step_g, node(최대 가중 노드)      (결측 마스크용, D1)
  oco_YYYYMM.parquet     : 사운딩 — 식별·좌표·라벨·타깃·ret_*·기하 + step_g·step_h·n0..n3·w0..w3
                           + 배경 원값(doy, hour, t2m, blh, sp: ERA5 z100 4점 보간; z850, thk: era5_syn 4점 보간, 승인 2026-09-22) + 분할(time_block, space_block, year, fold_time, fold_space)  (D4)
  no2_stats_YYYYMM.json  : hi 화소 no2 의 n·sum·sumsq (전 기간 μ·σ 산출용, D2)
  index_summary.csv      : 월별 요약 (재개 판정에도 사용)
전역 시각 step_g = (obs_time − 2020-01-01 00:00 UTC) 시 (int32), 월 상대 step_h = step_g − 월 시작 시 (D3).
표준화는 여기서 하지 않는다 (배경 변수는 학습 시 훈련 폴드에서 fit; no2 는 finalize 의 전 기간 상수).
재개형: 월 산출물 3종이 있으면 건너뜀. SMB 끊김은 wait_nas 후 재시도.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from no2xco2.config import ERA5_SYN_DIR, INDEX_DIR, LOCAL_STAGE_OUT, NAS_ERA5_WIND, NAS_MOUNT, NAS_OCO_NODES, nas_tropomi_ea_v2, wait_nas
from no2xco2.data.loader import T0, month_h0
from no2xco2.data.era5 import open_wind
from no2xco2.data import splits as splits_mod
from no2xco2.data.encoder_index import NLON, bilinear

QA_HI, QA_LO = 0.75, 0.66
SYN_DIR = ERA5_SYN_DIR
OCO_INDEX_COLS_REQUIRED = ["bg_t2m", "bg_blh", "bg_sp", "bg_z850", "bg_thk", "fold_time", "fold_space"]  # 재개 스킵 시 스키마 검사 (QA S3)
OCO_KEEP = ["row_idx", "sounding_id", "time", "latitude", "longitude", "label", "satellite", "xco2", "xco2_uncertainty", "xco2_apriori",
            "ret_aod_dust", "ret_aod_ice", "ret_aod_total", "ret_aod_water", "ret_psurf", "ret_snow_flag", "ret_surface_type", "ret_tcwv", "ret_tcwv_uncertainty",
            "solar_zenith_angle", "sensor_zenith_angle", "snd_land_fraction", "snd_operation_mode"]
MONTHS = [f"{y}{m:02d}" for y in range(2020, 2025) for m in range(1, 13)]


def _retry(fn, what, tries=4, nas=True):
    """읽기 재시도. nas=True(NAS 원천)면 실패마다 마운트 대기, 로컬 원천이면 대기 없이 재시도 (QA K3: 로컬 일시 오류 1회로 최대 6 h 대기하던 문제).
    open_wind 의 AssertionError(격자·시간축 규약 위반)는 재시도 대상이 아니다 — 자료 결함이므로 즉시 종료."""
    for k in range(tries):
        try:
            return fn()
        except (OSError, RuntimeError) as e:
            print(f"  {what} 읽기 실패 ({k+1}/{tries}): {str(e)[:80]}" + (" → NAS 대기" if nas else ""), flush=True); time.sleep(15)
            if nas:
                wait_nas()
    raise OSError(f"{what}: {tries}회 실패")


def _dump_json(obj, path: str, **kw) -> None:
    """원자적 json 쓰기 — 중단 시 잘린 통계 파일이 재개 판정(존재)을 통과하지 않게 (QA X9)."""
    with open(path + ".tmp", "w") as fo:
        json.dump(obj, fo, **kw)
    os.replace(path + ".tmp", path)


def _on_nas(path: str) -> bool:
    return os.path.abspath(path).startswith(NAS_MOUNT.rstrip("/") + "/")


def _era5_local_ready(f: str, f_nas: str, ym: str) -> bool:
    """로컬 ERA5 사본이 완전한가. NAS 원본이 보이면 크기 대조(rsync 진행 중 판정), NAS 미마운트·순단이면 로컬 파일 자체를 검사
    (open_wind: 격자·시간축 길이·시작 시각 assert) — NAS 없이도 완전한 로컬 사본으로 재빌드 가능 (QA A4 2026-09-24).
    판독 실패(OSError·ValueError = 잘린·복사 중 파일)는 False → 대기. 규약 위반 AssertionError(자료 결함)는 삼키지 않고 그대로 올린다 (QA X2)."""
    if not os.path.exists(f):
        return False
    try:
        if os.path.exists(f_nas):
            return os.path.getsize(f) == os.path.getsize(f_nas)
    except OSError:  # SMB 순단 중 getsize 실패 → 로컬 검사로
        pass
    try:
        open_wind(f, ym).close()
        return True
    except (OSError, ValueError):
        return False


def _step_g(ts) -> np.ndarray:
    return ((pd.to_datetime(ts) - T0) / pd.Timedelta(hours=1)).astype(np.int64).to_numpy().astype(np.int32)


def build_tropomi_month(ym: str, tdir: str, out: str, qa_hi=QA_HI, qa_lo=QA_LO) -> dict:
    files = sorted(glob.glob(os.path.join(tdir, f"*NO2____{ym}*.parquet")))
    if not files:  # 빈 입력(경로 오류·SMB 순단)을 n=0 성공으로 기록하지 않는다 (QA A1)
        raise FileNotFoundError(f"TROPOMI {ym}: granule 0개 — {tdir}")
    hi_parts, lo_parts = [], []; n_all = n_empty = 0; s = np.zeros(3)  # n, sum, sumsq
    for f in files:
        if _retry(lambda: pq.read_metadata(f).num_rows, os.path.basename(f)[:40], nas=_on_nas(f)) == 0:
            n_empty += 1; continue
        t = _retry(lambda: pq.read_table(f, columns=["lat", "lon", "qa", "obs_time", "no2_tvcd"]), os.path.basename(f)[:40], nas=_on_nas(f))
        qa = t["qa"].to_numpy(); lat = t["lat"].to_numpy(); lon = t["lon"].to_numpy(); no2 = t["no2_tvcd"].to_numpy().astype(np.float32)
        n_all += len(qa); nodes, w, _ = bilinear(lat, lon); step = _step_g(t["obs_time"].to_numpy())
        ok = np.isfinite(no2) & np.isfinite(w).all(1); hi = (qa >= qa_hi) & ok; lo = (qa >= qa_lo) & ~hi & ok  # NaN 화소는 제외 (주입 시 NaN 전파 방지)
        if hi.any():
            x = no2[hi].astype(np.float64); s += [hi.sum(), np.nansum(x), np.nansum(x * x)]
            hi_parts.append(pa.table({"step_g": pa.array(step[hi], pa.int32()), "qa": pa.array(qa[hi].astype(np.float32)), "no2": pa.array(no2[hi]),
                                      **{f"n{k}": pa.array(nodes[hi, k], pa.int32()) for k in range(4)},
                                      **{f"w{k}": pa.array(w[hi, k].astype(np.float32)) for k in range(4)}}))
        if lo.any():
            lo_parts.append(pa.table({"step_g": pa.array(step[lo], pa.int32()), "node": pa.array(nodes[lo][np.arange(lo.sum()), w[lo].argmax(1)], pa.int32())}))
    hi_t = pa.concat_tables(hi_parts) if hi_parts else None; lo_t = pa.concat_tables(lo_parts) if lo_parts else None
    if hi_t is not None:
        pq.write_table(hi_t, os.path.join(out, f"trop_hi_{ym}.parquet"), compression="zstd")
    if lo_t is not None:
        pq.write_table(lo_t, os.path.join(out, f"trop_lo_{ym}.parquet"), compression="zstd")
    _dump_json({"n": int(s[0]), "sum": float(s[1]), "sumsq": float(s[2])}, os.path.join(out, f"no2_stats_{ym}.json"))
    steps = hi_t["step_g"].to_numpy() if hi_t is not None else np.array([], np.int32)
    return dict(granules=len(files), empty=n_empty, pix_all=n_all, hi=hi_t.num_rows if hi_t is not None else 0, lo=lo_t.num_rows if lo_t is not None else 0,
                steps_hi=len(np.unique(steps)))


def _interp4(A: np.ndarray, df: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    """[T, NLAT, NLON] 장을 사운딩 4점·해당 시각으로 이중선형 보간 (QA S7: 중복 제거)."""
    acc = np.zeros(len(df), np.float64)
    for k in range(4):
        n = df[f"n{k}"].to_numpy(); acc += df[f"w{k}"].to_numpy() * A[t, n // NLON, n % NLON]
    return acc.astype(np.float32)


def build_oco_month(ym: str, odir: str, era5_dir: str, out: str) -> dict:
    files = sorted(glob.glob(os.path.join(odir, f"year_month={ym}", "*.parquet"))) or sorted(glob.glob(os.path.join(odir, "*.parquet")))  # 파티션 없으면 디렉토리 직접 (파일럿 사본)
    if not files:  # 빈 입력을 oco=0 성공으로 기록하지 않는다 (QA A1)
        raise FileNotFoundError(f"OCO {ym}: parquet 0개 — {odir}")
    df = _retry(lambda: pq.read_table(files, columns=OCO_KEEP).to_pandas(), f"oco {ym}", nas=_on_nas(odir))
    h0 = month_h0(ym)  # loader 와 정의 공유 (QA S7)
    nodes, w, inside = bilinear(df.latitude.to_numpy(), df.longitude.to_numpy())
    df["step_g"] = _step_g(df["time"]); df["step_h"] = (df["step_g"] - h0).astype(np.int32)
    for k in range(4):
        df[f"n{k}"] = nodes[:, k].astype(np.int32); df[f"w{k}"] = w[:, k].astype(np.float32)
    df["inside"] = inside
    tt = pd.to_datetime(df["time"]); df["doy"] = tt.dt.dayofyear.astype(np.int16); df["hour"] = (tt.dt.hour + tt.dt.minute / 60).astype(np.float32)
    # 배경 기상: ERA5 z100 월 파일에서 4점·해당 시각 보간 (로컬 → NAS 순으로 탐색)
    f = os.path.join(era5_dir, f"era5_wind_{ym}_z100.nc"); f_nas = os.path.join(NAS_ERA5_WIND, f"era5_wind_{ym}_z100.nc")
    # ERA5 는 로컬 완전 사본만 연다 (SMB 위 HDF5 읽기가 세그폴트로 프로세스를 죽인 사례 2회: 2020-04, 2021-01).
    # 로컬 사본이 완전해질 때까지 대기 (rsync 가 순차 복사 중). 판정은 _era5_local_ready.
    t_wait = 0
    while not _era5_local_ready(f, f_nas, ym):
        if t_wait == 0:
            print(f"  ERA5 {ym} 로컬 사본 대기 (rsync 진행 중)", flush=True)
        time.sleep(30); t_wait += 30
        if t_wait > 4 * 3600:
            raise OSError(f"ERA5 {ym} 로컬 사본 4 h 대기 초과")
    ds = _retry(lambda: open_wind(f, ym), os.path.basename(f), nas=False); T = ds.time.size  # 위도 오름차순·월 길이 보장 (로컬 사본)
    assert ((df.step_h >= 0) & (df.step_h < T)).all(), f"{ym}: step_h 범위 밖 사운딩 {int(((df.step_h < 0) | (df.step_h >= T)).sum())}"
    t = df.step_h.to_numpy()
    for v in ("t2m", "blh", "sp"):
        df[f"bg_{v}"] = _interp4(_retry(lambda: ds[v].values, f"{v} {ym}", nas=False), df, t)
    ds.close()
    # 종관 지표 (승인 2026-09-22): era5_syn 로컬 파일에서 z850·thk 4점 보간 → bg_z850, bg_thk (gpm)
    fz = os.path.join(SYN_DIR, f"era5_syn_{ym}.nc")
    if not os.path.exists(fz):
        raise FileNotFoundError(f"era5_syn 없음: {fz} — python -m no2xco2.data.era5_syn 로 생성")
    with xr.open_dataset(fz) as dz:
        assert dz.time.size == T and abs(float(dz.lat[0]) - 20) < 1e-6, f"era5_syn {ym} 시간축/위도 규약 불일치"
        for v in ("z850", "thk"):
            df[f"bg_{v}"] = _interp4(dz[v].values, df, t)
    df = splits_mod.assign(df)
    df.to_parquet(os.path.join(out, f"oco_{ym}.parquet"), compression="zstd", index=False)
    lab = (df.label == "L_train").to_numpy()
    return dict(oco=len(df), L_train=int(lab.sum()), outside=int((~inside).sum()), steps_oco=int(df.step_g.nunique()), era5_T=int(T))


def finalize(out: str) -> dict:
    n = s = ss = 0.0
    for f in sorted(glob.glob(os.path.join(out, "no2_stats_*.json"))):
        d = json.load(open(f)); n += d["n"]; s += d["sum"]; ss += d["sumsq"]
    if n == 0:  # 통계 파일 없음 또는 전부 n=0 → ZeroDivisionError 대신 명시 (QA A1)
        raise ValueError(f"no2_stats 합계 n = 0: {out}")
    mu = s / n; sd = float(np.sqrt(max(ss / n - mu * mu, 0)))
    st = {"n": int(n), "mean": mu, "std": sd, "months": len(glob.glob(os.path.join(out, "no2_stats_*.json"))), "qa_hi": QA_HI}
    _dump_json(st, os.path.join(out, "no2_stats.json"), indent=1); return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=MONTHS, help="YYYYMM 목록 (기본 60개월)")
    ap.add_argument("--tropomi-dir", default=None, help="기본: NAS _tropomi_ea_v2")
    ap.add_argument("--oco-dir", default=NAS_OCO_NODES)
    ap.add_argument("--era5-dir", default=LOCAL_STAGE_OUT)
    ap.add_argument("--out", default=INDEX_DIR)
    ap.add_argument("--finalize-only", action="store_true"); ap.add_argument("--rebuild-oco", action="store_true", help="완료된 달도 OCO(배경) 부분을 다시 만든다")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.finalize_only:
        print(finalize(a.out)); return
    if _on_nas(a.oco_dir):  # NAS 원천을 쓸 때만 마운트 대기 (QA A4·S4). TROPOMI 원천은 새로 만들 달이 있을 때만 해석
        wait_nas()
    tdir = a.tropomi_dir
    summ_p = os.path.join(a.out, "index_summary.csv"); rows = pd.read_csv(summ_p, dtype={"ym": str}).to_dict("records") if os.path.exists(summ_p) else []
    done = {r["ym"] for r in rows}
    for ym in a.months:
        oco_p = os.path.join(a.out, f"oco_{ym}.parquet")
        if ym in done and not a.rebuild_oco and os.path.exists(os.path.join(a.out, f"trop_hi_{ym}.parquet")) and os.path.exists(oco_p) \
                and set(OCO_INDEX_COLS_REQUIRED) <= set(pq.read_schema(oco_p).names):  # 스키마 검사: 열 추가 후 구 인덱스 재사용 방지 (QA S3)
            continue
        t0 = time.time(); r = {"ym": ym}
        hi_p, st_p = os.path.join(a.out, f"trop_hi_{ym}.parquet"), os.path.join(a.out, f"no2_stats_{ym}.json")
        if os.path.exists(hi_p) and os.path.exists(st_p):  # TROPOMI 부분은 완료(파일 존재) → 재사용 (OCO 단계에서 죽은 경우)
            steps = pq.read_table(hi_p, columns=["step_g"])["step_g"].to_numpy(); lo_p = os.path.join(a.out, f"trop_lo_{ym}.parquet")
            prev = next((x for x in rows if x["ym"] == ym), {})
            r.update(granules=prev.get("granules", -1), empty=prev.get("empty", -1), pix_all=prev.get("pix_all", -1), hi=len(steps),
                     lo=pq.read_metadata(lo_p).num_rows if os.path.exists(lo_p) else 0, steps_hi=len(np.unique(steps)))
            print(f"{ym}: TROPOMI 산출물 재사용 (hi {r['hi']:,})", flush=True)
        else:
            if tdir is None or _on_nas(tdir):
                wait_nas(); tdir = tdir or nas_tropomi_ea_v2()
            r.update(build_tropomi_month(ym, tdir, a.out))
        r.update(build_oco_month(ym, a.oco_dir, a.era5_dir, a.out)); r["sec"] = round(time.time() - t0)
        rows = [x for x in rows if x["ym"] != ym] + [r]; pd.DataFrame(rows).sort_values("ym").to_csv(summ_p + ".tmp", index=False); os.replace(summ_p + ".tmp", summ_p)  # 원자적 갱신
        print(f"{ym}: granule {r['granules']} (빈 {r['empty']}) · 화소 {r['pix_all']:,} → hi {r['hi']:,} / lo {r['lo']:,} · 주입 스텝 {r['steps_hi']} · OCO {r.get('oco',0):,} (L_train {r.get('L_train',0):,}) · {r['sec']} s", flush=True)
    print("no2 전 기간 통계:", finalize(a.out))


if __name__ == "__main__":
    main()
