"""CAMS rung-0 대조장 → NAS/cams_ea.

ADS 카탈로그(2026-09-07 조회):
  cams-global-ghg-reanalysis-egg4         시간범위 2003-01-01 ~ 2020-12-31 (2021년 이후 없음)
  cams-global-greenhouse-gas-inversion    시간범위 1979 ~ 2025, quantity=mean_column, input_observations=surface
    (surface = 지상 in situ만 동화 → OCO-2와 독립; satellite 옵션은 OCO-2 동화라 대조군으로 부적합)
요청:
  (1) EGG4 2020년 12개월: co2_column_mean_molar_fraction, 3시간 step, area 20–50N/100–150E, netcdf_zip
  (2) inversion latest·surface·instantaneous·mean_column 2020–2024 (전지구 1.875°×3.75°, 연 단위)
"""
import os, sys, shutil, zipfile, time
import cdsapi
from no2xco2.config import NAS_CAMS, LOCAL_STAGE_DL, move_to_nas, wait_nas

def client():
    cfg = dict(l.split(":", 1) for l in open(os.path.expanduser("~/.adsapirc")) if ":" in l)
    return cdsapi.Client(url=cfg["url"].strip(), key=cfg["key"].strip(), quiet=True)


def unzip_single(zpath, dst):
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith(".nc")]
        assert len(names) == 1, names
        with z.open(names[0]) as s, open(dst, "wb") as f: shutil.copyfileobj(s, f)
    os.remove(zpath)

def egg4_month(c, y, m):
    out = os.path.join(NAS_CAMS, f"egg4_xco2_{y}{m:02d}.nc")
    if os.path.exists(out): print(f"EGG4 {y}-{m:02d}: 스킵"); return
    import calendar; nd = calendar.monthrange(y, m)[1]
    req = {"variable": ["co2_column_mean_molar_fraction"], "date": [f"{y}-{m:02d}-01/{y}-{m:02d}-{nd}"],
           "step": [str(s) for s in range(0, 24, 3)], "area": [50, 100, 20, 150], "data_format": "netcdf_zip"}
    zp = os.path.join(LOCAL_STAGE_DL, f"egg4_{y}{m:02d}.zip"); t0 = time.time()
    try: c.retrieve("cams-global-ghg-reanalysis-egg4", req, zp)
    except Exception as e:
        print(f"EGG4 {y}-{m:02d}: area 요청 실패({e}) → 전지구 재요청", flush=True); req.pop("area"); c.retrieve("cams-global-ghg-reanalysis-egg4", req, zp)
    loc = zp[:-4] + ".nc"; unzip_single(zp, loc); move_to_nas(loc, out)
    print(f"EGG4 {y}-{m:02d}: {os.path.getsize(out)/1e6:.1f} MB {time.time()-t0:.0f}s", flush=True)

def inversion_year(c, y):
    out = os.path.join(NAS_CAMS, f"cams_inv_surface_meancol_{y}.zip")
    if os.path.exists(out): print(f"INV {y}: 스킵"); return
    req = {"variable": "carbon_dioxide", "quantity": "mean_column", "input_observations": "surface",
           "time_aggregation": "instantaneous", "version": "latest", "year": [str(y)], "month": [f"{m:02d}" for m in range(1, 13)]}
    loc = os.path.join(LOCAL_STAGE_DL, os.path.basename(out)); t0 = time.time()
    if os.path.exists(loc) and not zipfile.is_zipfile(loc):  # 재개 시 판독 검사: 잘린 zip 을 NAS 로 옮기지 않는다 (QA D7). 삭제 대신 .bad
        print(f"INV {y}: 로컬 zip 판독 불가 → .bad 로 보존하고 재요청", flush=True); os.replace(loc, loc + ".bad")
    if not os.path.exists(loc):
        c.retrieve("cams-global-greenhouse-gas-inversion", req, loc + ".part"); os.replace(loc + ".part", loc)
    move_to_nas(loc, out)
    print(f"INV {y}: {os.path.getsize(out)/1e6:.1f} MB {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    wait_nas()
    os.makedirs(LOCAL_STAGE_DL, exist_ok=True); os.makedirs(NAS_CAMS, exist_ok=True)
    c = client(); which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "inv"):
        for y in range(2020, 2025): inversion_year(c, y)
    if which in ("all", "egg4"):
        for m in range(1, 13): egg4_month(c, 2020, m)
    print("완료")
