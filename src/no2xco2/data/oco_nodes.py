"""OCO 사운딩 노드 테이블 — integrated_dataset.nc(17.4M행) → NAS parquet (year_month 파티션).

결정 9·10 적용:
  - 센티넬 마스킹: population_density(−3.4e38) · xco2_qf_bitflag(−9999) · |v|>1e30
  - 제거: tropomi_no2 (최근접 채움) · era5_* 6종 (일단위 지표풍) → 엣지·재분석으로 대체
  - 위성 구분: 행 인덱스 경계 8,795,034 (OCO-2 | OCO-3). file_source 문자열 대신
  - 라벨: L_train = qf==0 & 380≤xco2≤450 & 도메인 / 그 외 L_masked (노드 유지, 라벨만 마스킹)
  - 잠재 격자 인덱스 (0.25°, ERA5 격자와 동일) — 인코더용 lat_idx/lon_idx
원칙: 값을 만들어내지 않는다(채움·집계 없음). 행 수는 입력과 동일.
"""
import os, sys, shutil, time
import numpy as np, pandas as pd, h5py
import pyarrow as pa, pyarrow.parquet as pq
from no2xco2.config import NAS_OCO_NODES, NAS_SRC_NC, LOCAL_NC, GRID_LAT0, GRID_LON0, GRID_D

CUT = 8_795_034
FILL = 1e30
DROP = {"tropomi_no2", "era5_u10", "era5_v10", "era5_wind_speed", "era5_wind_dir", "era5_blh", "file_source"}
BLOCK = 1_000_000
XCO2_LO, XCO2_HI = 380.0, 450.0
LAT_LO, LAT_HI, LON_LO, LON_HI = 20.0, 50.0, 100.0, 150.0


def main(out_dir: str):
    if not os.path.exists(LOCAL_NC):
        print(f"NC 로컬 복사 {NAS_SRC_NC} → {LOCAL_NC} (2.0 GB)", flush=True)
        t0 = time.time(); os.makedirs(os.path.dirname(LOCAL_NC), exist_ok=True)
        shutil.copy2(NAS_SRC_NC, LOCAL_NC + ".part"); os.replace(LOCAL_NC + ".part", LOCAL_NC); print(f"  {time.time()-t0:.0f}s", flush=True)  # 원자적 (QA X9)
    f = h5py.File(LOCAL_NC, "r"); n = f["time"].shape[0]
    cols = [k for k in f.keys() if isinstance(f[k], h5py.Dataset) and f[k].shape == (n,) and k not in DROP]
    print(f"행 {n:,} · 사용 칼럼 {len(cols)}: {cols}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        d = {k: f[k][s:e] for k in cols}
        df = pd.DataFrame(d)
        # 센티넬
        sent = {}
        for c in df.columns:
            if df[c].dtype.kind == "f":
                m = df[c].abs() > FILL
                if m.any(): sent[c] = sent.get(c, 0) + int(m.sum()); df.loc[m, c] = np.nan
        m9 = df["xco2_qf_bitflag"] == -9999
        sent["xco2_qf_bitflag=-9999"] = int(m9.sum()); df.loc[m9, "xco2_qf_bitflag"] = np.nan
        df.loc[df["sounding_id"] <= -9998, "sounding_id"] = -1
        # 파생
        idx = np.arange(s, e)
        df["satellite"] = np.where(idx < CUT, "OCO-2", "OCO-3")
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df["year_month"] = df["time"].dt.strftime("%Y%m").astype("int32")
        phys = df.xco2.between(XCO2_LO, XCO2_HI)
        geo = df.latitude.between(LAT_LO, LAT_HI) & df.longitude.between(LON_LO, LON_HI)
        qf0 = df.xco2_quality_flag == 0
        df["label"] = np.where(qf0 & phys & geo, "L_train", "L_masked")
        df["is_outlier"] = ~phys
        df["lat_idx"] = np.floor((df.latitude - GRID_LAT0) / GRID_D).astype("int16")
        df["lon_idx"] = np.floor((df.longitude - GRID_LON0) / GRID_D).astype("int16")
        df["row_idx"] = idx.astype("int32")   # 원본 nc 행 → provenance
        for c in df.columns:
            if df[c].dtype == "float64" and c != "time": df[c] = df[c].astype("float32")
        # 파티션별 append
        for ym, g in df.groupby("year_month", sort=False):
            pdir = os.path.join(out_dir, f"year_month={ym}"); os.makedirs(pdir, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(g.drop(columns=["year_month"]), preserve_index=False),
                           os.path.join(pdir, f"part-{s//BLOCK:03d}.parquet"), compression="zstd")
        # 통계
        for (sat, lab), k in df.groupby(["satellite", "label"]).size().items():
            stats[(sat, lab)] = stats.get((sat, lab), 0) + int(k)
        for c, k in sent.items(): stats[("sentinel", c)] = stats.get(("sentinel", c), 0) + k
        print(f"  {e:>10,}/{n:,}  {100*e/n:5.1f}%", flush=True)
    f.close()
    print("\n=== 결과 (우리 산출값 · 조건: qf==0 & 380–450 ppm & 20–50N/100–150E) ===")
    tot = 0
    for (a, b), k in sorted(stats.items()):
        print(f"  {a:18s} {b:28s} {k:>12,}"); tot += k if a in ("OCO-2", "OCO-3") else 0
    print(f"  {'합계':18s} {'':28s} {tot:>12,}  (입력 {n:,} — 동일해야 함)")
    print(f"저장: {out_dir}  파티션 {len(os.listdir(out_dir))}개")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else NAS_OCO_NODES)
