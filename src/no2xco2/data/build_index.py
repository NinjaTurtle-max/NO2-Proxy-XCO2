"""모델 입력 인덱스 (월별).

산출 (<out>/):
  trop_hi_YYYYMM.parquet : qa ≥ 0.75 TROPOMI 화소 — step_g, qa, no2, n0..n3, w0..w3 (격자 4점 인덱스 · 이중선형 가중)
  oco_YYYYMM.parquet     : OCO 사운딩 — 식별 · 좌표 · 라벨 · 타깃 · 반환 변수 · 기하 + step_g · step_h · n0..n3 · w0..w3
                           + 배경 원값 (doy, hour; t2m, blh, sp: ERA5 4점 보간; z850, thk: era5_syn 4점 보간)
                           + 분할 열 (time_block, space_block, year, fold_time, fold_space)
  no2_stats_YYYYMM.json  : qa ≥ 0.75 화소 NO₂ 의 n · 합 · 제곱합
  no2_stats.json         : 전 기간 μ · σ (finalize)
  index_summary.csv      : 월별 요약
시각: step_g = (관측 시각 − 2020-01-01 00 UTC) 시 (int32), step_h = step_g − 월 시작 시.
표준화는 여기서 하지 않는다 (배경 변수는 학습 때 훈련 행으로, NO₂ 는 전 기간 상수로).
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

from no2xco2.config import ERA5_SYN_DIR, ERA5_WIND_DIR, INDEX_DIR, OCO_NODES_DIR, TROPOMI_DIR
from no2xco2.data import splits as splits_mod
from no2xco2.data.encoder_index import NLON, bilinear
from no2xco2.data.era5 import open_wind
from no2xco2.data.loader import T0, month_h0

QA_HI = 0.75
OCO_KEEP = ["row_idx", "sounding_id", "time", "latitude", "longitude", "label", "satellite", "xco2", "xco2_uncertainty", "xco2_apriori",
            "ret_aod_dust", "ret_aod_ice", "ret_aod_total", "ret_aod_water", "ret_psurf", "ret_snow_flag", "ret_surface_type", "ret_tcwv", "ret_tcwv_uncertainty",
            "solar_zenith_angle", "sensor_zenith_angle", "snd_land_fraction", "snd_operation_mode"]
MONTHS = [f"{y}{m:02d}" for y in range(2020, 2025) for m in range(1, 13)]


def _dump_json(obj, path: str, **kw) -> None:
    with open(path + ".tmp", "w") as fo:
        json.dump(obj, fo, **kw)
    os.replace(path + ".tmp", path)


def _step_g(ts) -> np.ndarray:
    return ((pd.to_datetime(ts) - T0) / pd.Timedelta(hours=1)).astype(np.int64).to_numpy().astype(np.int32)


def build_tropomi_month(ym: str, tdir: str, out: str, qa_hi=QA_HI) -> dict:
    files = sorted(glob.glob(os.path.join(tdir, f"*NO2____{ym}*.parquet")))
    if not files:
        raise FileNotFoundError(f"TROPOMI {ym}: granule 0개 — {tdir}")
    parts = []; n_all = n_empty = 0; s = np.zeros(3)  # n, 합, 제곱합
    for f in files:
        if pq.read_metadata(f).num_rows == 0:
            n_empty += 1; continue
        t = pq.read_table(f, columns=["lat", "lon", "qa", "obs_time", "no2_tvcd"])
        qa = t["qa"].to_numpy(); lat = t["lat"].to_numpy(); lon = t["lon"].to_numpy(); no2 = t["no2_tvcd"].to_numpy().astype(np.float32)
        n_all += len(qa); nodes, w, _ = bilinear(lat, lon); step = _step_g(t["obs_time"].to_numpy())
        hi = (qa >= qa_hi) & np.isfinite(no2) & np.isfinite(w).all(1)
        if hi.any():
            x = no2[hi].astype(np.float64); s += [hi.sum(), np.nansum(x), np.nansum(x * x)]
            parts.append(pa.table({"step_g": pa.array(step[hi], pa.int32()), "qa": pa.array(qa[hi].astype(np.float32)), "no2": pa.array(no2[hi]),
                                   **{f"n{k}": pa.array(nodes[hi, k], pa.int32()) for k in range(4)},
                                   **{f"w{k}": pa.array(w[hi, k].astype(np.float32)) for k in range(4)}}))
    hi_t = pa.concat_tables(parts) if parts else None
    if hi_t is not None:
        pq.write_table(hi_t, os.path.join(out, f"trop_hi_{ym}.parquet"), compression="zstd")
    _dump_json({"n": int(s[0]), "sum": float(s[1]), "sumsq": float(s[2])}, os.path.join(out, f"no2_stats_{ym}.json"))
    steps = hi_t["step_g"].to_numpy() if hi_t is not None else np.array([], np.int32)
    return dict(granules=len(files), empty=n_empty, pix_all=n_all, hi=hi_t.num_rows if hi_t is not None else 0, steps_hi=len(np.unique(steps)))


def _interp4(A: np.ndarray, df: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    """[T, NLAT, NLON] 장을 사운딩 4점 · 해당 시각으로 이중선형 보간."""
    acc = np.zeros(len(df), np.float64)
    for k in range(4):
        n = df[f"n{k}"].to_numpy(); acc += df[f"w{k}"].to_numpy() * A[t, n // NLON, n % NLON]
    return acc.astype(np.float32)


def build_oco_month(ym: str, odir: str, era5_dir: str, syn_dir: str, out: str) -> dict:
    files = sorted(glob.glob(os.path.join(odir, f"year_month={ym}", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"OCO {ym}: parquet 0개 — {odir}")
    df = pq.read_table(files, columns=OCO_KEEP).to_pandas()
    h0 = month_h0(ym)
    nodes, w, inside = bilinear(df.latitude.to_numpy(), df.longitude.to_numpy())
    df["step_g"] = _step_g(df["time"]); df["step_h"] = (df["step_g"] - h0).astype(np.int32)
    for k in range(4):
        df[f"n{k}"] = nodes[:, k].astype(np.int32); df[f"w{k}"] = w[:, k].astype(np.float32)
    df["inside"] = inside
    tt = pd.to_datetime(df["time"]); df["doy"] = tt.dt.dayofyear.astype(np.int16); df["hour"] = (tt.dt.hour + tt.dt.minute / 60).astype(np.float32)
    with open_wind(os.path.join(era5_dir, f"era5_wind_{ym}_z100.nc"), ym) as ds:  # 위도 오름차순 · 월 길이 보장
        T = ds.time.size
        assert ((df.step_h >= 0) & (df.step_h < T)).all(), f"{ym}: step_h 범위 밖 사운딩 {int(((df.step_h < 0) | (df.step_h >= T)).sum())}"
        t = df.step_h.to_numpy()
        for v in ("t2m", "blh", "sp"):
            df[f"bg_{v}"] = _interp4(ds[v].values, df, t)
    with xr.open_dataset(os.path.join(syn_dir, f"era5_syn_{ym}.nc")) as dz:
        assert dz.time.size == T and abs(float(dz.lat[0]) - 20) < 1e-6, f"era5_syn {ym} 시간축 · 위도 규약 불일치"
        for v in ("z850", "thk"):
            df[f"bg_{v}"] = _interp4(dz[v].values, df, t)
    df = splits_mod.assign(df)
    df.to_parquet(os.path.join(out, f"oco_{ym}.parquet"), compression="zstd", index=False)
    lab = (df.label == "L_train").to_numpy()
    return dict(oco=len(df), L_train=int(lab.sum()), outside=int((~inside).sum()), steps_oco=int(df.step_g.nunique()), era5_T=int(T))


def finalize(out: str) -> dict:
    n = s = ss = 0.0; files = sorted(glob.glob(os.path.join(out, "no2_stats_*.json")))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        n += d["n"]; s += d["sum"]; ss += d["sumsq"]
    if n == 0:
        raise ValueError(f"no2_stats 합계 n = 0: {out}")
    mu = s / n; sd = float(np.sqrt(max(ss / n - mu * mu, 0)))
    st = {"n": int(n), "mean": mu, "std": sd, "months": len(files), "qa_hi": QA_HI}
    _dump_json(st, os.path.join(out, "no2_stats.json"), indent=1); return st


def main():
    ap = argparse.ArgumentParser(description="TROPOMI 화소 · OCO 사운딩 → 모델 입력 인덱스")
    ap.add_argument("--months", nargs="*", default=MONTHS, help="YYYYMM 목록 (기본 2020-01 – 2024-12)")
    ap.add_argument("--tropomi-dir", default=TROPOMI_DIR); ap.add_argument("--oco-dir", default=OCO_NODES_DIR)
    ap.add_argument("--era5-dir", default=ERA5_WIND_DIR); ap.add_argument("--syn-dir", default=ERA5_SYN_DIR); ap.add_argument("--out", default=INDEX_DIR)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    summ_p = os.path.join(a.out, "index_summary.csv"); rows = pd.read_csv(summ_p, dtype={"ym": str}).to_dict("records") if os.path.exists(summ_p) else []
    for ym in a.months:
        t0 = time.time(); r = {"ym": ym}
        r.update(build_tropomi_month(ym, a.tropomi_dir, a.out))
        r.update(build_oco_month(ym, a.oco_dir, a.era5_dir, a.syn_dir, a.out)); r["sec"] = round(time.time() - t0)
        rows = [x for x in rows if x["ym"] != ym] + [r]; pd.DataFrame(rows).sort_values("ym").to_csv(summ_p, index=False)
        print(f"{ym}: granule {r['granules']} (빈 {r['empty']}) · 화소 {r['pix_all']:,} → qa≥0.75 {r['hi']:,} · 주입 스텝 {r['steps_hi']} · OCO {r['oco']:,} (L_train {r['L_train']:,}) · {r['sec']} s", flush=True)
    print("NO₂ 전 기간 통계:", finalize(a.out))


if __name__ == "__main__":
    main()
