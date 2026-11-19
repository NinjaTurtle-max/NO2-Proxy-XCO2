"""OCO-2/3 사운딩 표 — integrated_dataset.nc (17,415,604행) → year_month 파티션 parquet.

  - 결측 표식 처리: |v| > 1e30 · xco2_qf_bitflag = −9999 → NaN
  - 쓰지 않는 열 제거: tropomi_no2 · era5_* 6종 · file_source
  - 위성: 행 번호 경계 8,795,034 (앞 OCO-2 | 뒤 OCO-3)
  - 라벨: L_train = quality flag 0 & 380 ≤ xco2 ≤ 450 & 20–50°N · 100–150°E, 그 외 L_masked (행은 남기고 라벨만 구분)
  - 0.25° 격자 인덱스 lat_idx · lon_idx
값을 채우거나 집계하지 않는다. 출력 행 수 = 입력 행 수.
사용: python -m no2xco2.data.oco_nodes [출력 폴더]  (입력 config.OCO_NC)
"""
import os, sys
import numpy as np, pandas as pd, h5py
import pyarrow as pa, pyarrow.parquet as pq
from no2xco2.config import GRID_D, GRID_LAT0, GRID_LON0, OCO_NC, OCO_NODES_DIR

CUT = 8_795_034
FILL = 1e30
DROP = {"tropomi_no2", "era5_u10", "era5_v10", "era5_wind_speed", "era5_wind_dir", "era5_blh", "file_source"}
BLOCK = 1_000_000
XCO2_LO, XCO2_HI = 380.0, 450.0
LAT_LO, LAT_HI, LON_LO, LON_HI = 20.0, 50.0, 100.0, 150.0


def main(out_dir: str):
    f = h5py.File(OCO_NC, "r"); n = f["time"].shape[0]
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
    print("\n=== 위성 · 라벨별 행 수 ===")
    tot = 0
    for (a, b), k in sorted(stats.items()):
        print(f"  {a:18s} {b:28s} {k:>12,}"); tot += k if a in ("OCO-2", "OCO-3") else 0
    print(f"  {'합계':18s} {'':28s} {tot:>12,}  (입력 {n:,} — 동일해야 함)")
    print(f"저장: {out_dir}  파티션 {len(os.listdir(out_dir))}개")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OCO_NODES_DIR)
