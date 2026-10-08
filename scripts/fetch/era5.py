"""
ERA5 기압면 바람 → 물질별·고도별 이송장 2종 산출 (요새화 #1).

배경: 이중 감사에서 "XCO2는 연직적분 컬럼량인데 10m 지표풍(u10/v10)으로 이류를
기술하는 것은 물리적 오류"로 지목됨. NO2조차 지표풍이 아니라 PBL 평균풍을 써야
배출 추정 편향(15-60%)을 피한다는 것이 flux-divergence 문헌의 결론.
→ 물질별로 서로 다른 이송장을 만든다:

  - W_col (CO2용): 지표~500hPa 질량가중(Δp가중) 평균풍. XCO2가 컬럼 몰분율이므로
    컬럼의 수평수송은 압력가중 평균풍으로 기술.
  - W_pbl (NO2용): PBL 평균풍. 문헌 근사(PBL 평균 ≈ BLH/2 고도의 바람)를 따라
    z_agl=BLH/2에서 선형보간.

시간해상도는 hourly 유지 — NO2 관측순간 앵커 매칭에 필요(project_no2_time_matching_principle).
용량 대책: 월별로 [다운로드 → 파생장 산출 → 원본 즉시 삭제] 패턴
(00_slice_oco2_east_asia.py와 동일 철학). 기압면 원본은 월 ~3.5GB라 쌓아두면 안 됨.

사용:
    python scripts/fetch/era5.py 2020-01
    python scripts/fetch/era5.py 2020-01 2020-02 2020-03
"""
import os
import sys

import numpy as np
import xarray as xr
import shutil
from no2xco2.config import _same_ends, LOCAL_STAGE_OUT, LOCAL_STAGE_RAW, move_to_nas as _move_to_nas, wait_nas, nas_makedirs, NAS_ERA5_RAW, NAS_ERA5_WIND

RAW_DIR = LOCAL_STAGE_RAW             # 원본 임시(파생 산출 후 삭제) — config 상수 (QA P10)
OUT_DIR = LOCAL_STAGE_OUT             # 파생 이송장(보존)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 동아시아 도메인 — 00/01 스크립트와 동일 (20-50N, 100-150E)
AREA = [50.0, 100.0, 20.0, 150.0]  # N, W, S, E
GRID = [0.25, 0.25]

# 지표~500hPa. CO2 컬럼 이류의 질량가중 적분 구간.
LEVELS_FULL = ["500", "550", "600", "650", "700", "750", "775", "800",
               "825", "850", "875", "900", "925", "950", "975", "1000"]
# 경계층 대역만 — W_pbl(=BLH/2 고도 보간)에 필요한 최소 집합. 고지형(티베트 동단)
# 여유로 700hPa까지 포함. G0는 W_col을 쓰지 않으므로 이쪽이 기본.
LEVELS_PBL = ["700", "775", "800", "850", "875", "900", "925", "950", "975", "1000"]
P_TOP_PA = 50000.0  # 500hPa

# CDS는 요청당 필드 수(레벨×일×시각×변수) 상한이 있어 분할 요청이 필수.
DAYS_PER_REQUEST = 5

G = 9.80665  # 표준중력 — 지오퍼텐셜(m2/s2) → 고도(m)


def _readable(path: str) -> bool:
    """netcdf로 실제 열리는지 — 중단으로 잘린 다운로드를 걸러낸다."""
    try:
        with xr.open_dataset(path):
            return True
    except Exception:
        return False


def _month_days(year: int, month: int) -> list:
    import calendar
    n = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, n + 1)]


def fetch(year: int, month: int, pbl_only: bool = True) -> tuple:
    """기압면(u,v,z)·단일면(blh,sp,z) 원본을 받아 경로 2개 반환.

    기압면은 CDS 필드수 상한 때문에 DAYS_PER_REQUEST 일 단위로 쪼개 받은 뒤 시간축 결합.
    """
    import cdsapi

    c = cdsapi.Client()
    levels = LEVELS_PBL if pbl_only else LEVELS_FULL
    days = _month_days(year, month)
    hours = [f"{h:02d}:00" for h in range(24)]
    tag = f"{year}{month:02d}"

    pl_path = os.path.join(RAW_DIR, f"era5_pl_{tag}.nc")
    sl_path = os.path.join(RAW_DIR, f"era5_sl_{tag}.nc")

    for pth in (pl_path, sl_path):  # 재개 시 판독 검사: 잘린 병합·단일면 파일을 완성본으로 오인하지 않는다 (QA D7)
        if os.path.exists(pth) and not _readable(pth):
            print(f"  {os.path.basename(pth)} 판독 불가 → .bad 로 보존하고 재생성")
            os.replace(pth, pth + ".bad")
    if not os.path.exists(pl_path):
        chunks = [days[i:i + DAYS_PER_REQUEST]
                  for i in range(0, len(days), DAYS_PER_REQUEST)]
        parts = []
        print(f"  기압면 원본 요청 ({len(levels)}레벨×{len(days)}일×24h, "
              f"{len(chunks)}회 분할)...")
        for j, chunk in enumerate(chunks, 1):
            part = os.path.join(RAW_DIR, f"era5_pl_{tag}_p{j}.nc")
            # 존재만 보고 넘기면 **중단으로 잘린 청크를 완성본으로 오인**한다
            # (실측: 15MB짜리 p7이 HDF 오류로 열리지 않음). 열리는지까지 확인한다.
            if os.path.exists(part) and not _readable(part):
                print(f"    [{j}/{len(chunks)}] 손상 청크 폐기 후 재요청")
                os.remove(part)
            if not os.path.exists(part):
                print(f"    [{j}/{len(chunks)}] {chunk[0]}~{chunk[-1]}일")
                c.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": ["u_component_of_wind",
                                     "v_component_of_wind", "geopotential"],
                        "pressure_level": levels,
                        "year": str(year),
                        "month": f"{month:02d}",
                        "day": chunk,
                        "time": hours,
                        "area": AREA,
                        "grid": GRID,
                        "data_format": "netcdf",
                    },
                    part,
                )
            parts.append(part)

        dss = [xr.open_dataset(p) for p in parts]  # 청크 Dataset 을 닫은 뒤 삭제 (Windows 잠금 PermissionError, QA P9)
        ds = xr.concat([_norm(d) for d in dss], dim="time").sortby("time")
        ds.to_netcdf(pl_path + ".tmp")  # 원자적 쓰기: 중단 시 pl_path 가 잘린 채 남지 않게 (QA D7)
        ds.close()
        for d in dss:
            d.close()
        os.replace(pl_path + ".tmp", pl_path)
        for p in parts:
            os.remove(p)
    if not os.path.exists(sl_path):
        print("  단일면 원본 요청 (blh, sp, 지형고도)...")
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["boundary_layer_height", "surface_pressure",
                             "geopotential", "2m_temperature"],
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": hours,
                "area": AREA,
                "grid": GRID,
                "data_format": "netcdf",
            },
            sl_path + ".part",
        )
        os.replace(sl_path + ".part", sl_path)  # 원자적: 중단된 다운로드가 sl_path 로 남지 않게 (QA D7)
    return pl_path, sl_path


def _norm(ds: xr.Dataset) -> xr.Dataset:
    """CDS 버전별 좌표명 차이 흡수(valid_time/time, pressure_level/level)."""
    ren = {}
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        ren["valid_time"] = "time"
    if "pressure_level" in ds.coords or "pressure_level" in ds.dims:
        ren["pressure_level"] = "level"
    return ds.rename(ren) if ren else ds


def _column_mean_wind(u, v, p, sp):
    """지표~500hPa 질량가중(Δp) 평균풍.

    u,v: (lev, lat, lon) / p: (lev,) Pa 오름차순 / sp: (lat, lon) Pa
    각 레벨의 층두께 Δp를 가중치로 쓰되, 지하 레벨(p>sp)은 0,
    최하단 층은 지표기압에서 잘라냄.
    """
    # 레벨 경계(중점) — p는 오름차순(500→1000hPa)
    edges = np.empty(len(p) + 1)
    edges[1:-1] = 0.5 * (p[:-1] + p[1:])
    edges[0] = P_TOP_PA
    edges[-1] = p[-1] + 0.5 * (p[-1] - p[-2])

    top = edges[:-1][:, None, None]                       # (lev,1,1)
    bot = np.minimum(edges[1:][:, None, None], sp[None])  # 지표기압에서 절단
    w = np.clip(bot - top, 0.0, None)                     # (lev,lat,lon)

    wsum = w.sum(axis=0)
    wsum = np.where(wsum > 0, wsum, np.nan)
    return (u * w).sum(axis=0) / wsum, (v * w).sum(axis=0) / wsum


def _pbl_wind(u, v, z_agl, blh, z_min=0.0):
    """PBL 대표풍 — z_agl = max(BLH/2, z_min) 고도의 바람을 선형보간.

    z_min: 표집 고도 하한(m). 야간 안정층에서 BLH가 붕괴하면 BLH/2가 50m 미만이
    되어 지면 마찰층 바람을 뽑게 된다. 실제 오염물질은 그 위 잔류층에 분리되어
    있으므로 수송이 과소평가된다. 하한을 두어 이를 막는다. 0이면 기존 동작.

    u,v,z_agl: (lev, lat, lon) — z_agl은 레벨축 따라 단조. blh: (lat, lon)
    """
    # 고도 오름차순으로 정렬(1000hPa=최하단이 먼저 오도록)
    order = np.argsort(z_agl, axis=0)
    zs = np.take_along_axis(z_agl, order, axis=0)
    us = np.take_along_axis(u, order, axis=0)
    vs = np.take_along_axis(v, order, axis=0)

    target = np.maximum(0.5 * blh, z_min)  # (lat, lon)

    # target 이하인 레벨 개수 → 하단 인덱스 k
    below = zs <= target[None]
    k = below.sum(axis=0) - 1
    k = np.clip(k, 0, zs.shape[0] - 2)[None]  # (1,lat,lon)

    z0 = np.take_along_axis(zs, k, axis=0)[0]
    z1 = np.take_along_axis(zs, k + 1, axis=0)[0]
    u0 = np.take_along_axis(us, k, axis=0)[0]
    u1 = np.take_along_axis(us, k + 1, axis=0)[0]
    v0 = np.take_along_axis(vs, k, axis=0)[0]
    v1 = np.take_along_axis(vs, k + 1, axis=0)[0]

    dz = z1 - z0
    w = np.where(np.abs(dz) > 1e-6, (target - z0) / np.where(dz == 0, 1.0, dz), 0.0)
    w = np.clip(w, 0.0, 1.0)  # 최하단 레벨보다 낮은 BLH/2는 외삽 대신 클립
    return u0 + w * (u1 - u0), v0 + w * (v1 - v0)


def derive(pl_path: str, sl_path: str, out_path: str, pbl_only: bool = True, z_min: float = 0.0) -> None:
    """원본 → 이송장 2종(W_col: CO2용, W_pbl: NO2용) 산출."""
    pl = _norm(xr.open_dataset(pl_path))
    sl = _norm(xr.open_dataset(sl_path))

    # 레벨 오름차순(500→1000hPa) 정렬 후 Pa 변환
    pl = pl.sortby("level")
    p = pl["level"].values.astype("float64") * 100.0

    lat, lon, time = pl["latitude"], pl["longitude"], pl["time"]
    z_sfc = (sl["z"].isel(time=0).values / G)  # 지형고도(m) — 시간불변

    shp = (len(time), len(lat), len(lon))
    u_col = np.empty(shp, dtype="float32")
    v_col = np.empty(shp, dtype="float32")
    u_pbl = np.empty(shp, dtype="float32")
    v_pbl = np.empty(shp, dtype="float32")

    # 하루(24스텝)씩 묶어 읽는다 — 스텝마다 .isel().values를 부르면 netcdf 압축
    # 청크를 매번 다시 푸느라 I/O에 묶인다(실측: 월 하나에 24분+). 블록당 ~70MB라
    # 메모리 절약 목적도 유지된다.
    BLOCK = 24
    for s in range(0, len(time), BLOCK):
        e = min(s + BLOCK, len(time))
        u_b = pl["u"].isel(time=slice(s, e)).values.astype("float64")
        v_b = pl["v"].isel(time=slice(s, e)).values.astype("float64")
        z_b = pl["z"].isel(time=slice(s, e)).values.astype("float64") / G
        sp_b = sl["sp"].isel(time=slice(s, e)).values.astype("float64")
        blh_b = sl["blh"].isel(time=slice(s, e)).values.astype("float64")

        for j in range(e - s):
            i = s + j
            up, vp = _pbl_wind(u_b[j], v_b[j], z_b[j] - z_sfc[None], blh_b[j], z_min)
            u_pbl[i], v_pbl[i] = up.astype("float32"), vp.astype("float32")

            if not pbl_only:  # 경계층 대역만 받은 경우 컬럼 적분이 성립하지 않음
                uc, vc = _column_mean_wind(u_b[j], v_b[j], p, sp_b[j])
                u_col[i], v_col[i] = uc.astype("float32"), vc.astype("float32")

        if (e % 96 == 0) or e == len(time):
            print(f"    {e}/{len(time)} 시간스텝 처리", flush=True)

    dv = {
        "u_pbl": (("time", "lat", "lon"), u_pbl),
        "v_pbl": (("time", "lat", "lon"), v_pbl),
        "blh": (("time", "lat", "lon"), sl["blh"].values.astype("float32")),
        # 배경항 g 입력 — 파생과 무관하게 원본 단일면 그대로 전달
        "sp": (("time", "lat", "lon"), sl["sp"].values.astype("float32")),
        "t2m": (("time", "lat", "lon"), sl["t2m"].values.astype("float32")),
    }
    if not pbl_only:
        dv["u_col"] = (("time", "lat", "lon"), u_col)
        dv["v_col"] = (("time", "lat", "lon"), v_col)

    out = xr.Dataset(
        dv,
        coords={"time": time.values, "lat": lat.values, "lon": lon.values},
        attrs={
            "title": "ERA5 물질별 이송장 — CO2용 컬럼 질량가중풍 / NO2용 PBL풍",
            # ⚠️ 속성 이름에 '/'를 쓰면 NetCDF가 거부한다("Name contains illegal
            # characters"). 744스텝을 다 계산한 뒤 to_netcdf에서 죽어 title만 쓰인
            # 6KB 껍데기가 남는다 — 2026-08-10 산출물이 정확히 이 상태였다.
            "def_u_col_v_col": "지표~500hPa Δp가중 평균풍 (m/s), XCO2 컬럼 이류용",
            "def_u_pbl_v_pbl": f"z_agl=max(BLH/2, {z_min}m) 선형보간 바람 (m/s), NO2 이류용",
            "z_min_m": float(z_min),
            "note": (
                "이중 감사 지적(컬럼량에 지표풍 이류는 물리 오류) 대응. "
                "u10/v10을 두 물질에 공용하던 구 방식을 대체."
            ),
            "levels_hPa": ",".join(LEVELS_PBL if pbl_only else LEVELS_FULL),
            "mode": "pbl_only (G0용, W_col 미산출)" if pbl_only else "full",
            "source": f"{os.path.basename(pl_path)}, {os.path.basename(sl_path)}",
        },
    )
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(out_path + ".tmp", encoding=enc); os.replace(out_path + ".tmp", out_path)  # 원자적: 잘린 파생 파일이 최종 이름으로 남지 않게 (QA P1)
    pl.close()
    sl.close()


def verify_derived(path: str, expect_hours: int) -> tuple:
    """파생 이송장이 실제로 쓸 수 있는 물건인지 확인 — 원본 삭제의 전제조건.

    껍데기 파일(변수 0개), 시간축 누락, 전부 NaN인 바람을 걸러낸다.
    """
    if not os.path.exists(path):
        return False, "파일 없음"
    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        return False, f"열리지 않음: {e}"
    with ds:
        need = {"u_pbl", "v_pbl", "blh"}
        missing = need - set(ds.data_vars)
        if missing:
            return False, f"변수 누락 {sorted(missing)} (있는 것: {list(ds.data_vars)})"
        if ds.sizes.get("time", 0) != expect_hours:
            return False, f"시간축 {ds.sizes.get('time', 0)} ≠ 기대 {expect_hours}"
        u = ds["u_pbl"].values
        finite = np.isfinite(u).mean()
        if finite < 0.5:
            return False, f"u_pbl 유효값 {finite:.1%} — 사실상 비어 있음"
    return True, f"시간축 {expect_hours}, u_pbl 유효 {finite:.1%}"


def _month_range(a: str, b: str) -> list:
    """'2020-01' '2024-12' → ['2020-01', ..., '2024-12']"""
    y, m = int(a[:4]), int(a[5:7]); y2, m2 = int(b[:4]), int(b[5:7]); out = []
    while (y, m) <= (y2, m2):
        out.append(f"{y}-{m:02d}"); m += 1
        if m == 13: y, m = y + 1, 1
    return out


def run_month(tag: str, z_min: float = 0.0, keep_raw: bool = False,
              nas_raw: str = None, nas_out: str = None) -> None:
    """한 달 처리. nas_*가 주어지면 **로컬에서 받고·파생·검증한 뒤 NAS로 이동**한다.

    SMB에 랜덤 I/O(netCDF 청크 읽기)를 하지 않기 위한 구조. NAS에는 순차 쓰기만.
    NAS에 원본이 이미 있으면(재파생 시) CDS 재요청 대신 로컬로 복사해 쓴다.
    """
    year, month = int(tag[:4]), int(tag[5:7])
    suf = "" if z_min == 0 else f"_z{int(z_min)}"
    fname = f"era5_wind_{year}{month:02d}{suf}.nc"
    final_out = os.path.join(nas_out or OUT_DIR, fname)
    if nas_out: wait_nas()
    if os.path.exists(final_out):
        if keep_raw and nas_raw:
            for kind in ("pl", "sl"):
                loc = os.path.join(RAW_DIR, f"era5_{kind}_{year}{month:02d}.nc")
                dst = os.path.join(nas_raw, os.path.basename(loc))
                if not os.path.exists(loc):
                    continue
                if os.path.exists(dst) and _same_ends(loc, dst):  # 크기 + 앞·뒤 4 MB 대조 후에만 로컬 삭제 (QA X8)
                    os.remove(loc)  # NAS 사본 완결 → 로컬 잔여분만 정리
                else:
                    print(f"{tag}: 원본 잔여분 NAS 이동 {os.path.basename(loc)}", flush=True)
                    nas_makedirs(nas_raw); _move_to_nas(loc, dst)
        print(f"{tag}: 이미 존재 → 스킵 ({final_out})")
        return
    out_path = os.path.join(OUT_DIR, fname)          # 로컬 스테이징

    # NAS에 원본이 있으면 로컬로 복사 (CDS 큐 회피)
    if nas_raw:
        for kind in ("pl", "sl"):
            src = os.path.join(nas_raw, f"era5_{kind}_{year}{month:02d}.nc")
            dst = os.path.join(RAW_DIR, f"era5_{kind}_{year}{month:02d}.nc")
            if os.path.exists(src) and not os.path.exists(dst):
                print(f"  NAS 원본 복사: {os.path.basename(src)}", flush=True)
                shutil.copy2(src, dst + ".part"); os.replace(dst + ".part", dst)  # 원자적: 중단된 복사가 dst 로 남아 재사용되지 않게 (QA X9)

    print(f"{tag}: 시작", flush=True)
    pl_path, sl_path = fetch(year, month)
    print("  파생장 산출 중...", flush=True)
    derive(pl_path, sl_path, out_path, z_min=z_min)

    ok, why = verify_derived(out_path, expect_hours=len(_month_days(year, month)) * 24)
    if not ok:
        print(f"{tag}: ⚠️ 파생물 검증 실패 — {why}\n  원본 보존: {pl_path}, {sl_path}")
        return
    size_mb = os.path.getsize(out_path) / 1e6

    if nas_out:
        nas_makedirs(nas_out)
        _move_to_nas(out_path, final_out)
        out_path = final_out
    if keep_raw:
        if nas_raw:
            nas_makedirs(nas_raw)
            for p in (pl_path, sl_path):
                dst = os.path.join(nas_raw, os.path.basename(p))
                if os.path.exists(dst) and _same_ends(p, dst):  # 크기 + 앞·뒤 4 MB 대조 (QA X8)
                    os.remove(p)
                else:
                    _move_to_nas(p, dst)
            print(f"{tag}: 완료 → {out_path} ({size_mb:.0f}MB), 원본 NAS 보존({nas_raw})")
        else:
            print(f"{tag}: 완료 → {out_path} ({size_mb:.0f}MB), 원본 로컬 보존({pl_path})")
        return
    for p in (pl_path, sl_path):
        os.remove(p)
    print(f"{tag}: 완료 → {out_path} ({size_mb:.0f}MB), 원본 삭제됨")


if __name__ == "__main__":
    args = sys.argv[1:]
    z_min, keep_raw, nas, tags = 0.0, False, False, []
    i = 0
    while i < len(args):
        if args[i] == "--zmin":
            z_min = float(args[i + 1]); i += 2
        elif args[i] == "--keep-raw":
            keep_raw = True; i += 1
        elif args[i] == "--nas":            # 최종 목적지를 config.py의 NAS 경로로
            nas = True; i += 1
        elif args[i] == "--range":          # --range 2020-01 2024-12
            tags += _month_range(args[i + 1], args[i + 2]); i += 3
        else:
            tags.append(args[i]); i += 1
    if not tags:
        print(__doc__)
        sys.exit(1)
    nas_raw = NAS_ERA5_RAW if nas else None
    nas_out = NAS_ERA5_WIND if nas else None
    print(f"설정: zmin={z_min} keep_raw={keep_raw} nas={nas} 월 {len(tags)}건 "
          f"({tags[0]}~{tags[-1]})" + (f"\n  raw→{nas_raw}\n  out→{nas_out}" if nas else ""), flush=True)
    for t in tags:
        run_month(t, z_min=z_min, keep_raw=keep_raw, nas_raw=nas_raw, nas_out=nas_out)
