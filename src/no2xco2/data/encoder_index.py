"""결정 5 인코더·디코더 엣지: 관측(OCO 사운딩·TROPOMI 화소) → 0.25° 잠재 격자 최근접 4점 이중선형 인덱스.

격자: lat0=20, lon0=100, d=0.25 → 121×201 = 24,321 노드. node_id = i_lat*201 + i_lon.
관측 (lat, lon)에 대해 fy=(lat-20)/0.25, fx=(lon-100)/0.25, i0=floor(fy), j0=floor(fx),
  4점 = (i0,j0),(i0,j0+1),(i0+1,j0),(i0+1,j0+1), 가중 = (1-ty)(1-tx),(1-ty)tx,ty(1-tx),ty·tx.
경계(lat=50 또는 lon=150)는 i0/j0를 마지막 셀로 당김(가중은 그대로 합 1).
TROPOMI 화소는 obs_time(UTC)을 시간 스텝 index(월 시작 기준 h)로 변환 → 결정 4의 통과일 주입 스텝.

입력: --month YYYYMM, --tropomi-dir (granule parquet), --oco-dir (oco_nodes 파티션)
출력: <out>/enc_tropomi_YYYYMM.parquet, <out>/enc_oco_YYYYMM.parquet, 요약 표.
"""
import argparse, glob, os, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from no2xco2.config import GRID_LAT0, GRID_LON0, GRID_D

NLAT, NLON = 121, 201


def bilinear(lat: np.ndarray, lon: np.ndarray):
    fy = (lat - GRID_LAT0) / GRID_D; fx = (lon - GRID_LON0) / GRID_D
    i0 = np.clip(np.floor(fy).astype(np.int32), 0, NLAT - 2); j0 = np.clip(np.floor(fx).astype(np.int32), 0, NLON - 2)
    ty = (fy - i0).astype(np.float32); tx = (fx - j0).astype(np.float32)
    nodes = np.stack([i0 * NLON + j0, i0 * NLON + j0 + 1, (i0 + 1) * NLON + j0, (i0 + 1) * NLON + j0 + 1], 1)
    w = np.stack([(1 - ty) * (1 - tx), (1 - ty) * tx, ty * (1 - tx), ty * tx], 1)
    inside = (fy >= 0) & (fy <= NLAT - 1) & (fx >= 0) & (fx <= NLON - 1)
    return nodes, w, inside


def build_tropomi(files, month0: pd.Timestamp, out: str):
    parts = []; n_pix = 0; n_out = 0; t0 = time.time()
    for f in files:
        t = pq.read_table(f, columns=["lat", "lon", "qa", "obs_time", "no2_tvcd", "orbit"])
        lat = t["lat"].to_numpy(); lon = t["lon"].to_numpy(); n_pix += len(lat)
        nodes, w, inside = bilinear(lat, lon); n_out += int((~inside).sum())
        ts = pd.to_datetime(t["obs_time"].to_numpy())
        step = ((ts - month0) / pd.Timedelta(hours=1)).astype(np.int32)  # 월 시작 기준 시간 스텝
        parts.append(pa.table({
            "orbit": t["orbit"], "obs_time": t["obs_time"], "step_h": pa.array(step, pa.int32()),
            "qa": t["qa"], "no2_tvcd": t["no2_tvcd"],
            "n0": nodes[:, 0], "n1": nodes[:, 1], "n2": nodes[:, 2], "n3": nodes[:, 3],
            "w0": w[:, 0], "w1": w[:, 1], "w2": w[:, 2], "w3": w[:, 3], "inside": inside}))
    tab = pa.concat_tables(parts); pq.write_table(tab, out, compression="zstd")
    steps = tab["step_h"].to_numpy(); days = np.unique(steps // 24)
    return dict(files=len(files), pixels=n_pix, outside=n_out, edges=4 * n_pix, days=len(days),
                pix_per_day=n_pix / max(len(days), 1), steps_with_obs=len(np.unique(steps)), sec=time.time() - t0)


def build_oco(files, month0: pd.Timestamp, out: str):
    t = pq.read_table(files, columns=["row_idx", "latitude", "longitude", "time", "label", "satellite", "xco2"])
    step = ((pd.to_datetime(t["time"].to_numpy()) - month0) / pd.Timedelta(hours=1)).astype(np.int32)
    lat = t["latitude"].to_numpy(); lon = t["longitude"].to_numpy()
    nodes, w, inside = bilinear(lat, lon)
    tab = pa.table({"row_idx": t["row_idx"], "time": t["time"], "step_h": pa.array(step, pa.int32()), "label": t["label"], "satellite": t["satellite"], "xco2": t["xco2"],
                    "n0": nodes[:, 0], "n1": nodes[:, 1], "n2": nodes[:, 2], "n3": nodes[:, 3],
                    "w0": w[:, 0], "w1": w[:, 1], "w2": w[:, 2], "w3": w[:, 3], "inside": inside})
    pq.write_table(tab, out, compression="zstd")
    lab = t["label"].to_numpy() == "L_train"
    return dict(soundings=len(lat), outside=int((~inside).sum()), edges=4 * len(lat), L_train=int(lab.sum()),
                nodes_touched=len(np.unique(nodes)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="202001")
    ap.add_argument("--tropomi-dir", default="data/raw/pilot_202001/tropomi")
    ap.add_argument("--oco-dir", default="data/raw/pilot_202001/oco_nodes")
    ap.add_argument("--out", default="data/processed/encoder_index")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    month0 = pd.Timestamp(f"{a.month[:4]}-{a.month[4:]}-01")
    tf = sorted(glob.glob(os.path.join(a.tropomi_dir, f"*NO2____{a.month}*.parquet")))
    of = sorted(glob.glob(os.path.join(a.oco_dir, "*.parquet")))
    print(f"TROPOMI granule {len(tf)}개, OCO 파티션 파일 {len(of)}개")
    r1 = build_tropomi(tf, month0, os.path.join(a.out, f"enc_tropomi_{a.month}.parquet"))
    r2 = build_oco(of, month0, os.path.join(a.out, f"enc_oco_{a.month}.parquet"))
    print("\n[TROPOMI → 격자] " + ", ".join(f"{k}={v:,.0f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in r1.items()))
    print("[OCO → 격자]     " + ", ".join(f"{k}={v:,}" for k, v in r2.items()))
    print(f"격자 노드 {NLAT*NLON:,} (121×201)")
