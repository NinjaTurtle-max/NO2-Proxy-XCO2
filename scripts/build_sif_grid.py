"""TROPOSIF 궤도 parquet (NAS troposif_ea) → 0.25° 달력 월 평균 sif743_dc 격자 (A-6·C-1, 사용자 승인 2026-10-08).

단계 1 (NAS 읽기): 파일 단위로 열 6개만 읽어 월 누적(품질 4조합 합·개수 + 해양 판정용 화소 수)에 더한다. 50파일마다 체크포인트(재개형).
         파일마다 바이트·읽기 초를 <out>/sif_read_log_<연도>.csv 에 남긴다 (NAS 읽기 시간 측정).
단계 2 (격자): 누적 → 채움(D7) → <out>/sif_YYYYMM.nc (변수 sif743_dc 1개, (lat, lon), NaN 0) + <out>/sif_<연도>_summary.csv.
`--from-accum` 이면 단계 1 을 건너뛴다 (D4 조합 변경 시 NAS 재읽기 불필요).
출력이 있으면 중단 (N-2), `--overwrite` 로만 덮어씀. 실행 전 게이트(python 학습 0 · 스왑 < 80 %) 확인은 실행자 책임.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from no2xco2.config import NAS_ROOT, SIF_DIR, wait_nas
from no2xco2.data import sif_grid as S


def nas_files(d, year):
    """NAS 디렉토리 목록에서 해당 연도 궤도 parquet. troposif_summary.csv 에는 1행 누락이 있어(10-08 확인) 목록을 쓴다."""
    t0 = time.time(); names = sorted(f for f in os.listdir(d) if f.startswith(f"S5P_PAL__L2__SIF____{year}") and f.endswith(".parquet"))
    print(f"목록 {len(names):,}개 · {time.time() - t0:.1f} s", flush=True)
    return names


def read_one(path, tries=3):
    import pyarrow.parquet as pq
    for k in range(1, tries + 1):
        try:
            return pq.read_table(path, columns=S.COLS).to_pandas()
        except OSError as e:
            print(f"    읽기 실패({k}/{tries}) {os.path.basename(path)}: {e}", flush=True)
            if k == tries:
                raise
            wait_nas(); time.sleep(30 * k)


def stage_read(a, ck):
    wait_nas()
    d = a.nas_dir; names = nas_files(d, a.year)
    if a.limit:
        names = names[:: max(len(names) // a.limit, 1)][: a.limit]  # 측정용: 연중 고르게 표본
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=False); acc = {k: z[k] for k in z.files if k != "done"}; done = set(z["done"].tolist())
        print(f"체크포인트 재개: {len(done):,}개 완료", flush=True)
    else:
        acc = S.new_accum(); done = set()
    log = os.path.join(a.out, f"sif_read_log_{a.year}.csv"); new_log = not os.path.exists(log)
    t_all = time.time(); k = 0
    with open(log, "a", encoding="utf-8") as fl:
        if new_log:
            fl.write("file,bytes,read_s,rows\n")
        for f in names:
            if f in done:
                continue
            p = os.path.join(d, f); t0 = time.time(); df = read_one(p); dt = time.time() - t0
            S.accumulate(acc, df, int(f[24:26]))
            fl.write(f"{f},{os.path.getsize(p)},{dt:.3f},{len(df)}\n"); done.add(f); k += 1
            if k % 50 == 0:
                fl.flush(); np.savez(ck + ".tmp.npz", done=np.array(sorted(done)), **acc); os.replace(ck + ".tmp.npz", ck)
                print(f"  {len(done):,}/{len(names):,} · 경과 {time.time() - t_all:.0f} s", flush=True)
    np.savez(ck + ".tmp.npz", done=np.array(sorted(done)), **acc); os.replace(ck + ".tmp.npz", ck)
    print(f"읽기 완료 {len(done):,}개 · 이번 실행 {k:,}개 {time.time() - t_all:.0f} s", flush=True)
    return acc


def write_month(grid, ym, variant, path):
    import xarray as xr
    lat, lon = S.grid_coords()
    ds = xr.Dataset({"sif743_dc": (("lat", "lon"), grid)}, coords={"lat": lat, "lon": lon},
                    attrs={"units": S.UNITS, "source": "TROPOSIF L2 (S5P-PAL) troposif_ea parquet", "month": ym, "variant": variant,
                           "quality": "qa743 >= {}".format(S.VARIANTS[variant][0]) + ("" if S.VARIANTS[variant][1] is None else f" · cloud_fraction < {S.VARIANTS[variant][1]}"),
                           "fill": "ocean(lc_mask=0 all year)=0; land gap=mean of adjacent months; remaining=0",
                           "lat_order": "ascending (grid convention)"})
    ds.to_netcdf(path + ".tmp", encoding={"sif743_dc": {"zlib": True, "complevel": 4}}); os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2020"); ap.add_argument("--variant", default=S.DEFAULT_VARIANT, choices=list(S.VARIANTS))
    ap.add_argument("--nas-dir", default=os.path.join(NAS_ROOT, "troposif_ea")); ap.add_argument("--out", default=SIF_DIR)
    ap.add_argument("--from-accum", action="store_true", help="NAS 를 읽지 않고 기존 누적(npz)으로 격자만 생성")
    ap.add_argument("--limit", type=int, default=0, help="읽기 시간 측정용: 연중 고르게 N개만 (격자 쓰지 않음)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ck = os.path.join(a.out, f"sif_accum_{a.year}{'_limit' + str(a.limit) if a.limit else ''}.npz")
    outs = [os.path.join(a.out, f"sif_{a.year}{m:02d}.nc") for m in range(1, 13)]
    summ = os.path.join(a.out, f"sif_{a.year}_summary.csv")
    if not a.limit and not a.overwrite and any(os.path.exists(o) for o in outs + [summ]):
        ap.error(f"출력 존재 — {a.out}/sif_{a.year}*.nc (덮어쓰려면 --overwrite)")
    if a.from_accum:
        z = np.load(ck, allow_pickle=False); acc = {k: z[k] for k in z.files if k != "done"}
    else:
        acc = stage_read(a, ck)
    if a.limit:
        return
    grid, state, _ = S.finalize(acc, a.variant)
    for m, o in enumerate(outs):
        write_month(grid[m], f"{a.year}{m + 1:02d}", a.variant, o)
    t = pd.DataFrame(S.month_summary(acc, state, a.variant)); t.to_csv(summ, index=False)
    pd.set_option("display.width", 220); print(t.to_string(index=False)); print(f"→ {a.out}/sif_{a.year}MM.nc · {summ}")


if __name__ == "__main__":
    main()
