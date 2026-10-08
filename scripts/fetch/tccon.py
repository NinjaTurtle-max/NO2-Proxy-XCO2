"""TCCON GGG2020 공개 파일(.public.qc.nc) 5개 사이트 → NAS.

사이트: 연구영역(20–50°N, 100–150°E) 안의 GGG2020 공개 레코드 (CaltechDATA 검색 2026-09-07).
  Burgos(18.5°N)는 영역 밖, Anmyeondo는 GGG2020 레코드 미검색 → 제외.
로컬 스테이징에 받고 크기 대조 후 NAS로 이동한다.
"""
import os, shutil, time, urllib.request
from no2xco2.config import NAS_TCCON, LOCAL_STAGE_DL, move_to_nas, wait_nas

API = "https://data.caltech.edu/api/records/{rec}/files/{fn}/content"
SITES = {  # site: (record, filename, size_bytes(API 보고))
    "hefei":     ("etz11-jpg19", "hf20151102_20251230.public.qc.nc", 57_400_000),
    "saga":      ("dy9h2-6gc10", "js20110728_20231213.public.qc.nc", 131_300_000),
    "tsukuba":   ("2ve20-pr498", "tk20140328_20210331.public.qc.nc", 66_400_000),
    "rikubetsu": ("ksrr6-jqh95", "rj20140624_20250501.public.qc.nc", 50_900_000),
    "xianghe":   ("6ywxa-yk431", "xh20180614_20241231.public.qc.nc", 78_000_000),
}


def fetch(url, dst, tries=3):
    for k in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research fetch)"})  # 기본 UA는 403
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            return os.path.getsize(dst)
        except Exception as e:
            print(f"    실패({k}/{tries}) {e}", flush=True); time.sleep(20 * k)
    raise RuntimeError(url)


if __name__ == "__main__":
    wait_nas()
    os.makedirs(LOCAL_STAGE_DL, exist_ok=True); os.makedirs(NAS_TCCON, exist_ok=True)
    for site, (rec, fn, approx) in SITES.items():
        dst = os.path.join(NAS_TCCON, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0.9 * approx:
            print(f"{site}: 이미 존재 {os.path.getsize(dst)/1e6:.1f} MB → 스킵"); continue
        loc = os.path.join(LOCAL_STAGE_DL, fn)
        t0 = time.time(); n = fetch(API.format(rec=rec, fn=fn), loc)
        print(f"{site}: {n/1e6:.1f} MB 수신 {time.time()-t0:.0f}s", flush=True)
        for extra in ("README.txt", "LICENSE.txt"):
            fetch(API.format(rec=rec, fn=extra), os.path.join(LOCAL_STAGE_DL, f"{site}_{extra}"))
            move_to_nas(os.path.join(LOCAL_STAGE_DL, f"{site}_{extra}"), os.path.join(NAS_TCCON, f"{site}_{extra}"))
        move_to_nas(loc, dst); print(f"{site}: NAS 이동 완료", flush=True)
    print("완료:", sorted(os.listdir(NAS_TCCON)))
