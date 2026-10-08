"""CarbonTracker 3시간 몰분율(글로벌 3°×2°, 34층) → 동아시아 부분집합 월별 nc → NAS/carbonttracker_ea.

출처: NOAA GML https://gml.noaa.gov/aftp/products/carbontracker/co2/
  CT2022           molefrac_glb3x2 2000-01-01 ~ 2021-02-28 (97 MB/일)
  CT-NRT.v2025-1   molefrac_glb3x2 2021-01-01 ~ 2024-12-31 (70 MB/일)
  → 2020-01 ~ 2021-02 CT2022, 2021-03 ~ 2024-12 CT-NRT.v2025-1 (중복 구간은 정식 릴리스 우선)
보존 변수: co2, pressure(경계 35), air_mass, specific_humidity, gph, blh, orography  (temperature·u·v 제외: ERA5 보유)
부분집합: lat 19~51 (2° 중심 17행), lon 100.5~151.5 (3° 중심 18열). 일 1.09 MB(zlib) → 5년 ≈ 2 GB.
"""
import os, sys, shutil, time, urllib.request, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import xarray as xr
from no2xco2.config import NAS_CT, LOCAL_STAGE_DL, move_to_nas, wait_nas

BASE = "https://gml.noaa.gov/aftp/products/carbontracker/co2/{ver}/molefractions/co2_total/{ver}.molefrac_glb3x2_{d}.nc"
KEEP = ["co2", "pressure", "air_mass", "specific_humidity", "gph", "blh", "orography"]
CT2022_END = dt.date(2021, 2, 28)


def version(d):
    return "CT2022" if d <= CT2022_END else "CT-NRT.v2025-1"


def fetch(d, tries=4):
    ver = version(d); url = BASE.format(ver=ver, d=d.isoformat())
    loc = os.path.join(LOCAL_STAGE_DL, f"ct_{d.isoformat()}.nc")
    for k in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(loc, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
                expect = int(r.headers.get("Content-Length") or 0)
            got = os.path.getsize(loc)
            if expect and got != expect:
                raise OSError(f"수신 크기 불일치 {got} != {expect}")
            with xr.open_dataset(loc) as ds:  # 열어서 HDF 손상 검사
                ds["co2"].isel(time=0, level=0).load()
            return d, loc, ver
        except Exception as e:
            try: os.remove(loc)
            except OSError: pass
            print(f"    {d} 실패({k}/{tries}) {e}", flush=True); time.sleep(30 * k)
    raise RuntimeError(url)


def subset(path):
    with xr.open_dataset(path) as ds:
        return ds[KEEP].sel(latitude=slice(19, 51), longitude=slice(98.5, 151.5)).load()


def run_month(y, m, workers=3):
    out = os.path.join(NAS_CT, f"ct_ea_{y}{m:02d}.nc")
    if os.path.exists(out):
        print(f"{y}-{m:02d}: 이미 존재 → 스킵", flush=True); return
    loc_out = os.path.join(LOCAL_STAGE_DL, os.path.basename(out))
    if os.path.exists(loc_out):
        move_to_nas(loc_out, out); print(f"{y}-{m:02d}: 로컬 잔여분 NAS 이동", flush=True); return
    d0 = dt.date(y, m, 1); d1 = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1))
    days = [d0 + dt.timedelta(i) for i in range((d1 - d0).days + 1)]
    t0 = time.time(); parts = []; vers = set()
    with ThreadPoolExecutor(workers) as ex:
        for d, loc, ver in ex.map(fetch, days):
            parts.append(subset(loc)); vers.add(ver); os.remove(loc)
    ds = xr.concat(parts, dim="time")
    ds.attrs["source_version"] = ",".join(sorted(vers)); ds.attrs["subset"] = "lat 19-51, lon 100.5-151.5; vars " + ",".join(KEEP)
    loc_out = os.path.join(LOCAL_STAGE_DL, os.path.basename(out))
    ds.to_netcdf(loc_out, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    move_to_nas(loc_out, out)
    print(f"{y}-{m:02d}: {len(days)}일 {ds.sizes['time']}스텝 {sorted(vers)} {os.path.getsize(out)/1e6:.1f} MB {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    wait_nas()
    a, b = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2020-01", "2024-12")
    os.makedirs(LOCAL_STAGE_DL, exist_ok=True); os.makedirs(NAS_CT, exist_ok=True)
    y, m = map(int, a.split("-")); ye, me = map(int, b.split("-"))
    while (y, m) <= (ye, me):
        run_month(y, m); m += 1
        if m > 12: y, m = y + 1, 1
    print("완료")
