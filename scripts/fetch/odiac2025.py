"""ODIAC2025 1 km 월별 GeoTIFF 2024년분(.tif.gz.zip, 1.39 GB) → NAS/odiac2025/.

NIES 배포 페이지(DL_odiac2025.html)의 직접 링크. 기존 NAS 보유 odiac2024 zip(2020–2023)과 동일 포맷.
"""
import os, sys, shutil, time, urllib.request
from no2xco2.config import NAS_ODIAC2025, LOCAL_STAGE_DL, move_to_nas, wait_nas

BASE = "https://db.cger.nies.go.jp/nies_data/10.17595/20170411.001/odiac2025/1km_tiff/{y}/"
YEARS = [int(a) for a in sys.argv[1:]] or [2024]


def fetch(url, dst, tries=3):
    for k in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            return os.path.getsize(dst)
        except Exception as e:
            print(f"    실패({k}/{tries}) {e}", flush=True); time.sleep(30 * k)
    raise RuntimeError(url)


if __name__ == "__main__":
    wait_nas()
    os.makedirs(LOCAL_STAGE_DL, exist_ok=True); os.makedirs(NAS_ODIAC2025, exist_ok=True)
    for y in YEARS:
        zn = f"odiac2025_1km_excl_intl_{y}_allmonths.tif.gz.zip"; md = f"odiac2025_1km_checksum_{y}.md5.txt"
        dst = os.path.join(NAS_ODIAC2025, zn)
        if os.path.exists(dst) and os.path.getsize(dst) > 1e9:
            print(f"{y}: 이미 존재 → 스킵"); continue
        t0 = time.time(); loc = os.path.join(LOCAL_STAGE_DL, zn)
        n = os.path.getsize(loc) if os.path.exists(loc) and os.path.getsize(loc) == 1393104361 else fetch(BASE.format(y=y) + zn, loc); print(f"{y}: {n/1e9:.2f} GB 수신 {time.time()-t0:.0f}s", flush=True)
        fetch(BASE.format(y=y) + md, os.path.join(LOCAL_STAGE_DL, md))
        # md5 파일은 월별 .tif.gz 기준이라 zip 자체 검증은 크기만 (content-length 1,393,104,361 B, 2026-09-07 HEAD)
        move_to_nas(os.path.join(LOCAL_STAGE_DL, md), os.path.join(NAS_ODIAC2025, md))
        move_to_nas(loc, dst); print(f"{y}: NAS 이동 완료 {os.path.getsize(dst)} B", flush=True)
    print("완료")
