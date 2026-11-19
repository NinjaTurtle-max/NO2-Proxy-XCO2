"""ERA5 시간별 자료 받기(CDS) → 경계층 바람 · 배경 기상 월 파일.

받는 것 (20–50°N, 100–150°E, 0.25°, 시간별):
  기압면 700–1000 hPa 10층: u, v, 지오퍼텐셜  → <raw>/era5_pl_YYYYMM.nc (era5_syn 입력으로도 쓴다)
  단일면: 경계층 높이, 지표 기압, 지오퍼텐셜(지형), 2 m 기온 → <raw>/era5_sl_YYYYMM.nc
만드는 것: <out>/era5_wind_YYYYMM_z100.nc
  u_pbl, v_pbl : 지표 위 z = max(BLH/2, z_min) 고도의 바람 (기압면 사이 선형 보간, 기본 z_min = 100 m)
  blh, sp, t2m : 단일면 원값
CDS 계정 설정(~/.cdsapirc)이 필요하다.

사용: python scripts/data/fetch_era5.py --range 2020-01 2024-12 [--zmin 100] [--raw data/raw/era5_pl] [--out data/raw/era5_wind]
"""
import argparse
import calendar
import os

import numpy as np
import xarray as xr

from no2xco2.config import ERA5_PL_DIR, ERA5_WIND_DIR

AREA = [50.0, 100.0, 20.0, 150.0]  # N, W, S, E
GRID = [0.25, 0.25]
LEVELS = ["700", "775", "800", "850", "875", "900", "925", "950", "975", "1000"]
DAYS_PER_REQUEST = 5  # CDS 요청당 필드 수 상한 때문에 나눠 받는다
G = 9.80665


def _readable(path: str) -> bool:
    try:
        with xr.open_dataset(path):
            return True
    except Exception:
        return False


def _norm(ds: xr.Dataset) -> xr.Dataset:
    """CDS 판에 따른 좌표 이름 차이를 맞춘다 (valid_time → time, pressure_level → level)."""
    ren = {k: v for k, v in (("valid_time", "time"), ("pressure_level", "level")) if k in ds.coords or k in ds.dims}
    return ds.rename(ren) if ren else ds


def fetch(year: int, month: int, raw: str) -> tuple:
    """기압면 · 단일면 원본을 받아 경로 2개를 돌려준다 (이미 있고 열리면 다시 받지 않음)."""
    import cdsapi
    c = cdsapi.Client(); tag = f"{year}{month:02d}"
    days = [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]; hours = [f"{h:02d}:00" for h in range(24)]
    pl_path = os.path.join(raw, f"era5_pl_{tag}.nc"); sl_path = os.path.join(raw, f"era5_sl_{tag}.nc")
    if not (os.path.exists(pl_path) and _readable(pl_path)):
        parts = []
        for j, i in enumerate(range(0, len(days), DAYS_PER_REQUEST), 1):
            part = os.path.join(raw, f"era5_pl_{tag}_p{j}.nc")
            if not (os.path.exists(part) and _readable(part)):
                c.retrieve("reanalysis-era5-pressure-levels",
                           {"product_type": "reanalysis", "variable": ["u_component_of_wind", "v_component_of_wind", "geopotential"],
                            "pressure_level": LEVELS, "year": str(year), "month": f"{month:02d}", "day": days[i:i + DAYS_PER_REQUEST],
                            "time": hours, "area": AREA, "grid": GRID, "data_format": "netcdf"}, part)
            parts.append(part)
        dss = [xr.open_dataset(p) for p in parts]
        ds = xr.concat([_norm(d) for d in dss], dim="time").sortby("time")
        ds.to_netcdf(pl_path + ".tmp"); ds.close()
        for d in dss:
            d.close()
        os.replace(pl_path + ".tmp", pl_path)
        for p in parts:
            os.remove(p)
    if not (os.path.exists(sl_path) and _readable(sl_path)):
        c.retrieve("reanalysis-era5-single-levels",
                   {"product_type": "reanalysis", "variable": ["boundary_layer_height", "surface_pressure", "geopotential", "2m_temperature"],
                    "year": str(year), "month": f"{month:02d}", "day": days, "time": hours, "area": AREA, "grid": GRID, "data_format": "netcdf"},
                   sl_path + ".part")
        os.replace(sl_path + ".part", sl_path)
    return pl_path, sl_path


def _pbl_wind(u, v, z_agl, blh, z_min=0.0):
    """z = max(BLH/2, z_min) 고도의 바람을 기압면 사이 선형 보간으로 구한다. u, v, z_agl: (lev, lat, lon), blh: (lat, lon).
    z_min: 야간에 BLH 가 얕아져 지면 마찰층 바람을 쓰지 않도록 두는 하한 (m)."""
    order = np.argsort(z_agl, axis=0)
    zs = np.take_along_axis(z_agl, order, axis=0); us = np.take_along_axis(u, order, axis=0); vs = np.take_along_axis(v, order, axis=0)
    target = np.maximum(0.5 * blh, z_min)
    k = np.clip((zs <= target[None]).sum(axis=0) - 1, 0, zs.shape[0] - 2)[None]
    z0 = np.take_along_axis(zs, k, axis=0)[0]; z1 = np.take_along_axis(zs, k + 1, axis=0)[0]
    u0 = np.take_along_axis(us, k, axis=0)[0]; u1 = np.take_along_axis(us, k + 1, axis=0)[0]
    v0 = np.take_along_axis(vs, k, axis=0)[0]; v1 = np.take_along_axis(vs, k + 1, axis=0)[0]
    dz = z1 - z0
    w = np.clip(np.where(np.abs(dz) > 1e-6, (target - z0) / np.where(dz == 0, 1.0, dz), 0.0), 0.0, 1.0)  # 최하층보다 낮으면 외삽하지 않음
    return u0 + w * (u1 - u0), v0 + w * (v1 - v0)


def derive(pl_path: str, sl_path: str, out_path: str, z_min: float) -> None:
    pl = _norm(xr.open_dataset(pl_path)).sortby("level"); sl = _norm(xr.open_dataset(sl_path))
    lat, lon, time = pl["latitude"], pl["longitude"], pl["time"]
    z_sfc = sl["z"].isel(time=0).values / G  # 지형 고도 (m)
    shp = (len(time), len(lat), len(lon)); u_pbl = np.empty(shp, "float32"); v_pbl = np.empty(shp, "float32")
    for s in range(0, len(time), 24):  # 하루씩 읽는다 (압축 청크를 스텝마다 다시 풀지 않도록)
        e = min(s + 24, len(time))
        u_b = pl["u"].isel(time=slice(s, e)).values.astype("float64"); v_b = pl["v"].isel(time=slice(s, e)).values.astype("float64")
        z_b = pl["z"].isel(time=slice(s, e)).values.astype("float64") / G; blh_b = sl["blh"].isel(time=slice(s, e)).values.astype("float64")
        for j in range(e - s):
            up, vp = _pbl_wind(u_b[j], v_b[j], z_b[j] - z_sfc[None], blh_b[j], z_min)
            u_pbl[s + j], v_pbl[s + j] = up.astype("float32"), vp.astype("float32")
    out = xr.Dataset({"u_pbl": (("time", "lat", "lon"), u_pbl), "v_pbl": (("time", "lat", "lon"), v_pbl),
                      "blh": (("time", "lat", "lon"), sl["blh"].values.astype("float32")), "sp": (("time", "lat", "lon"), sl["sp"].values.astype("float32")),
                      "t2m": (("time", "lat", "lon"), sl["t2m"].values.astype("float32"))},
                     coords={"time": time.values, "lat": lat.values, "lon": lon.values},
                     attrs={"def_u_pbl_v_pbl": f"wind at z_agl = max(BLH/2, {z_min} m), linear between pressure levels (m/s)", "z_min_m": float(z_min),
                            "levels_hPa": ",".join(LEVELS), "source": f"{os.path.basename(pl_path)}, {os.path.basename(sl_path)}"})
    out.to_netcdf(out_path + ".tmp", encoding={v: {"zlib": True, "complevel": 4} for v in out.data_vars}); os.replace(out_path + ".tmp", out_path)
    pl.close(); sl.close()


def months_between(a: str, b: str) -> list:
    y, m = int(a[:4]), int(a[5:7]); y2, m2 = int(b[:4]), int(b[5:7]); out = []
    while (y, m) <= (y2, m2):
        out.append((y, m)); m += 1
        if m == 13: y, m = y + 1, 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--range", nargs=2, metavar=("YYYY-MM", "YYYY-MM"), required=True)
    ap.add_argument("--zmin", type=float, default=100.0); ap.add_argument("--raw", default=ERA5_PL_DIR); ap.add_argument("--out", default=ERA5_WIND_DIR)
    a = ap.parse_args(); os.makedirs(a.raw, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    for y, m in months_between(*a.range):
        out_path = os.path.join(a.out, f"era5_wind_{y}{m:02d}_z{int(a.zmin)}.nc")
        if os.path.exists(out_path):
            continue
        pl_path, sl_path = fetch(y, m, a.raw); derive(pl_path, sl_path, out_path, a.zmin)
        with xr.open_dataset(out_path) as ds:
            T = calendar.monthrange(y, m)[1] * 24
            assert ds.sizes["time"] == T and np.isfinite(ds["u_pbl"].values).mean() > 0.5, f"{out_path}: 시간축 또는 값 이상"
        print(f"{y}-{m:02d} → {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
