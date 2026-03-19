"""
=============================================================
 TROPOMI L2 NC 파일 관측 시간 검정 스크립트
 - 경로 내 모든 NC 파일의 시간 변수 구조 확인
 - 픽셀별 관측 시각(UTC) 분포 통계
 - 동아시아 도메인(20-50N, 100-150E) 기준 필터링
 - 결과 CSV + 요약 리포트 저장
=============================================================
"""

import numpy as np
import netCDF4 as nc
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Project Root for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
NC_DIR   = config.S5P_DIR
OUT_DIR  = NC_DIR / "time_check"
OUT_DIR.mkdir(exist_ok=True)

# 동아시아 도메인
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0

# 검사할 최대 파일 수 (None = 전체)
MAX_FILES = None


# ─────────────────────────────────────────────
# NC 변수 구조 탐색
# ─────────────────────────────────────────────
def inspect_nc_structure(nc_path: Path) -> dict:
    """
    NC 파일 내부 구조 탐색:
    - 변수 목록 및 차원
    - 시간 관련 변수 식별
    - 위경도 변수 식별
    """
    info = {"file": nc_path.name, "groups": [], "time_vars": [], "geo_vars": [], "all_vars": []}

    try:
        with nc.Dataset(nc_path, "r") as ds:
            # 최상위 변수
            for vname, var in ds.variables.items():
                info["all_vars"].append(f"{vname}{var.dimensions}")
                if any(k in vname.lower() for k in ("time", "delta_time", "utc")):
                    units = getattr(var, "units", "no units")
                    info["time_vars"].append(f"{vname}{var.dimensions} units={units}")
                if any(k in vname.lower() for k in ("lat", "lon", "longitude", "latitude")):
                    info["geo_vars"].append(f"{vname}{var.dimensions}")

            # 그룹 탐색 (TROPOMI L2는 PRODUCT 그룹 안에 데이터가 있음)
            for gname in ds.groups:
                grp = ds.groups[gname]
                info["groups"].append(gname)
                for vname, var in grp.variables.items():
                    info["all_vars"].append(f"{gname}/{vname}{var.dimensions}")
                    if any(k in vname.lower() for k in ("time", "delta_time", "utc")):
                        units = getattr(var, "units", "no units")
                        info["time_vars"].append(f"{gname}/{vname}{var.dimensions} units={units}")
                    if any(k in vname.lower() for k in ("lat", "lon", "longitude", "latitude")):
                        info["geo_vars"].append(f"{gname}/{vname}{var.dimensions}")

                # 서브그룹
                for sgname in grp.groups:
                    sgrp = grp.groups[sgname]
                    for vname, var in sgrp.variables.items():
                        info["all_vars"].append(f"{gname}/{sgname}/{vname}{var.dimensions}")
                        if any(k in vname.lower() for k in ("time", "delta_time", "utc")):
                            units = getattr(var, "units", "no units")
                            info["time_vars"].append(f"{gname}/{sgname}/{vname}{var.dimensions} units={units}")

    except Exception as e:
        info["error"] = str(e)

    return info


# ─────────────────────────────────────────────
# 시간 디코딩 함수
# ─────────────────────────────────────────────
def decode_tropomi_time(ds) -> np.ndarray:
    """
    TROPOMI L2 NC에서 픽셀별 UTC 시각 추출.
    변수 구조:
      PRODUCT/time          : (scanline,)          — 스캔라인 기준 시각 (ms since 2010-01-01)
      PRODUCT/delta_time    : (scanline, pixel)     — 픽셀별 오프셋 (ms)
    → pixel_time = time[scanline] + delta_time[scanline, pixel]
    단위: milliseconds since 2010-01-01 00:00:00 UTC
    """
    epoch = datetime(2010, 1, 1, 0, 0, 0)

    # PRODUCT 그룹 접근
    if "PRODUCT" not in ds.groups:
        return None, "PRODUCT 그룹 없음"

    grp = ds.groups["PRODUCT"]

    # time 변수
    if "time" not in grp.variables:
        return None, "PRODUCT/time 없음"
    time_raw = np.array(grp.variables["time"][:], dtype=float).reshape(-1)  # → 1D (scanline,)

    # delta_time 변수
    if "delta_time" not in grp.variables:
        units = getattr(grp.variables["time"], "units", "")
        return time_raw, units

    delta_raw = np.array(grp.variables["delta_time"][:], dtype=float)
    # (1, scanline, pixel) or (scanline, pixel) → 항상 2D로
    while delta_raw.ndim > 2:
        delta_raw = delta_raw[0]
    if delta_raw.ndim < 2:
        delta_raw = delta_raw.reshape(1, -1)

    # time_raw와 delta_raw 차원 일치 확인 후 pixel_time 계산
    # delta_raw: (scanline, pixel), time_raw: (scanline,) or (1,)
    if len(time_raw) == 1:
        # time이 스칼라 1개인 경우: delta_raw 행 수에 맞게 확장
        time_raw = np.repeat(time_raw, delta_raw.shape[0])

    pixel_time_ms = time_raw[:, np.newaxis] + delta_raw  # (scanline, pixel)

    # 혹시 남은 앞 차원 제거
    while pixel_time_ms.ndim > 2:
        pixel_time_ms = pixel_time_ms[0]

    units = getattr(grp.variables["time"], "units", "ms since 2010-01-01")
    return pixel_time_ms, units


# ─────────────────────────────────────────────
# 단일 파일 시간 분석
# ─────────────────────────────────────────────
def analyze_file_time(nc_path: Path) -> dict:
    """
    단일 NC 파일에서 동아시아 도메인 픽셀의 관측 시각 통계 추출
    """
    result = {
        "file":        nc_path.name,
        "date_str":    "",       # 파일명에서 파싱한 날짜
        "n_pixels_ea": 0,        # 동아시아 도메인 픽셀 수
        "time_min_utc": "",
        "time_max_utc": "",
        "time_mean_utc": "",
        "hour_min":    None,
        "hour_max":    None,
        "hour_mean":   None,
        "has_delta_time": False,
        "time_units":  "",
        "error":       "",
    }

    # 파일명에서 날짜 파싱 (예: S5P_L2__NO2____...20240115...)
    fname = nc_path.stem
    for part in fname.split("_"):
        if len(part) == 8 and part.isdigit():
            result["date_str"] = f"{part[:4]}-{part[4:6]}-{part[6:8]}"
            break

    epoch = datetime(2010, 1, 1, 0, 0, 0)

    try:
        with nc.Dataset(nc_path, "r") as ds:
            if "PRODUCT" not in ds.groups:
                result["error"] = "PRODUCT 그룹 없음"
                return result

            grp = ds.groups["PRODUCT"]

            # 위경도 추출
            lat_key = next((k for k in grp.variables if "lat" in k.lower()), None)
            lon_key = next((k for k in grp.variables if "lon" in k.lower()), None)

            if not lat_key or not lon_key:
                result["error"] = "위경도 변수 없음"
                return result

            lat = np.array(grp.variables[lat_key][:])
            lon = np.array(grp.variables[lon_key][:])

            # FillValue 마스크 처리
            if isinstance(lat, np.ma.MaskedArray):
                lat = lat.filled(np.nan)
            if isinstance(lon, np.ma.MaskedArray):
                lon = lon.filled(np.nan)

            # ★ (1, scanline, pixel) → (scanline, pixel) 2D로 강제 변환
            while lat.ndim > 2:
                lat = lat[0]
            while lon.ndim > 2:
                lon = lon[0]

            # 동아시아 도메인 마스크
            ea_mask = (
                (lat >= LAT_MIN) & (lat <= LAT_MAX) &
                (lon >= LON_MIN) & (lon <= LON_MAX)
            )
            result["n_pixels_ea"] = int(np.sum(ea_mask))

            if result["n_pixels_ea"] == 0:
                result["error"] = "동아시아 픽셀 없음 (궤도 미포함)"
                return result

            # 시간 추출
            pixel_time_ms, units = decode_tropomi_time(ds)
            result["time_units"] = str(units)
            result["has_delta_time"] = "delta_time" in grp.variables

            if pixel_time_ms is None:
                result["error"] = f"시간 추출 실패: {units}"
                return result

            # 동아시아 픽셀의 시간값만 추출
            if pixel_time_ms.ndim == 1 and ea_mask.ndim == 2:
                if pixel_time_ms.shape[0] == ea_mask.shape[0]:
                    pixel_time_ms = np.broadcast_to(pixel_time_ms[:, np.newaxis], ea_mask.shape)
                elif pixel_time_ms.shape[0] == 1:
                    pixel_time_ms = np.full(ea_mask.shape, pixel_time_ms[0])
                else:
                    result["error"] = f"시간 차원 불일치: 시간 {pixel_time_ms.shape}, 마스크 {ea_mask.shape}"
                    return result
            # 만약 2D인데 (1, scanline)이거나 (scanline, 1)인 경우를 대비해서 broadcast 시도
            elif pixel_time_ms.shape != ea_mask.shape:
                if pixel_time_ms.ndim == 2 and pixel_time_ms.shape[0] == 1 and pixel_time_ms.shape[1] == ea_mask.shape[0]:
                    pixel_time_ms = pixel_time_ms.T

                try:
                    pixel_time_ms = np.broadcast_to(pixel_time_ms, ea_mask.shape)
                except ValueError:
                    result["error"] = f"시간 차원 불일치 (불완전): 시간 {pixel_time_ms.shape}, 마스크 {ea_mask.shape}"
                    return result

            ea_times = pixel_time_ms[ea_mask]
            ea_times = ea_times[~np.isnan(ea_times)]

            if len(ea_times) == 0:
                result["error"] = "유효 시간값 없음"
                return result

            # ms → datetime 변환
            def ms_to_dt(ms_val):
                return epoch + timedelta(milliseconds=float(ms_val))

            t_min  = ms_to_dt(ea_times.min())
            t_max  = ms_to_dt(ea_times.max())
            t_mean = ms_to_dt(ea_times.mean())

            result["time_min_utc"]  = t_min.strftime("%Y-%m-%d %H:%M:%S")
            result["time_max_utc"]  = t_max.strftime("%Y-%m-%d %H:%M:%S")
            result["time_mean_utc"] = t_mean.strftime("%Y-%m-%d %H:%M:%S")
            result["hour_min"]  = round(t_min.hour  + t_min.minute / 60, 2)
            result["hour_max"]  = round(t_max.hour  + t_max.minute / 60, 2)
            result["hour_mean"] = round(t_mean.hour + t_mean.minute / 60, 2)

    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    # TROPOMI L2 파일명 패턴: S5P_OFFL_L2__NO2__ 또는 S5P_L2__NO2__
    nc_files = sorted(NC_DIR.glob("S5P_*L2__NO2__*.nc"))
    if not nc_files:
        # 하위 디렉토리도 탐색
        nc_files = sorted(NC_DIR.rglob("S5P_*L2__NO2__*.nc"))

    if not nc_files:
        print(f"❌ NC 파일을 찾을 수 없음: {NC_DIR}")
        return

    if MAX_FILES:
        nc_files = nc_files[:MAX_FILES]

    print(f"총 {len(nc_files)}개 파일 검사")
    print(f"저장 경로: {OUT_DIR}\n")

    # ── Step 1: 첫 번째 파일 구조 탐색 (1회만) ──
    print("=" * 60)
    print("[ NC 파일 내부 구조 탐색 (첫 번째 파일) ]")
    print("=" * 60)
    struct = inspect_nc_structure(nc_files[0])
    print(f"파일: {struct['file']}")
    print(f"그룹: {struct['groups']}")
    print(f"\n시간 관련 변수:")
    for v in struct["time_vars"]:
        print(f"  {v}")
    print(f"\n위경도 변수:")
    for v in struct["geo_vars"]:
        print(f"  {v}")
    print(f"\n전체 변수 수: {len(struct['all_vars'])}")

    # 구조 저장
    struct_txt = OUT_DIR / "nc_structure.txt"
    with open(struct_txt, "w", encoding="utf-8") as f:
        f.write(f"파일: {struct['file']}\n\n")
        f.write("=== 그룹 ===\n")
        f.write("\n".join(struct["groups"]) + "\n\n")
        f.write("=== 시간 관련 변수 ===\n")
        f.write("\n".join(struct["time_vars"]) + "\n\n")
        f.write("=== 위경도 변수 ===\n")
        f.write("\n".join(struct["geo_vars"]) + "\n\n")
        f.write("=== 전체 변수 ===\n")
        f.write("\n".join(struct["all_vars"]) + "\n")
    print(f"\n구조 저장: {struct_txt}")

    # ── Step 2: 전체 파일 시간 분석 ──
    print(f"\n{'=' * 60}")
    print(f"[ 전체 파일 시간 분석 ]")
    print(f"{'=' * 60}")

    results = []
    for i, nc_path in enumerate(nc_files):
        print(f"  [{i+1:4d}/{len(nc_files)}] {nc_path.name[:50]}...", end=" ")
        res = analyze_file_time(nc_path)
        results.append(res)

        if res["error"]:
            print(f"⚠️  {res['error']}")
        else:
            print(f"✅ EA={res['n_pixels_ea']:,}px | "
                  f"{res['time_min_utc'][11:16]}~{res['time_max_utc'][11:16]} UTC "
                  f"(mean {res['time_mean_utc'][11:16]})")

    # ── Step 3: 결과 저장 ──
    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "tropomi_time_check.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 CSV 저장: {csv_path}")

    # ── Step 4: 요약 통계 ──
    valid = df[df["error"] == ""]
    print(f"\n{'=' * 60}")
    print(f"  📊 요약 통계")
    print(f"{'=' * 60}")
    print(f"  전체 파일 수        : {len(df)}")
    print(f"  동아시아 궤도 포함  : {len(valid)}")
    print(f"  오류 / 미포함       : {len(df) - len(valid)}")

    if len(valid) > 0:
        print(f"\n  관측 시각 분포 (UTC, 동아시아 픽셀 기준)")
        print(f"  hour_min  평균: {valid['hour_min'].mean():.2f}h  "
              f"범위: {valid['hour_min'].min():.2f}~{valid['hour_min'].max():.2f}h")
        print(f"  hour_max  평균: {valid['hour_max'].mean():.2f}h  "
              f"범위: {valid['hour_max'].min():.2f}~{valid['hour_max'].max():.2f}h")
        print(f"  hour_mean 평균: {valid['hour_mean'].mean():.2f}h  "
              f"범위: {valid['hour_mean'].min():.2f}~{valid['hour_mean'].max():.2f}h")

        # 요약 텍스트 저장
        summary_txt = OUT_DIR / "time_summary.txt"
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write("TROPOMI L2 관측 시각 검정 요약\n")
            f.write(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"전체 파일 수: {len(df)}\n")
            f.write(f"동아시아 궤도 포함: {len(valid)}\n\n")
            f.write("관측 시각 분포 (UTC)\n")
            f.write(f"  hour_min  평균={valid['hour_min'].mean():.2f}h  "
                    f"범위={valid['hour_min'].min():.2f}~{valid['hour_min'].max():.2f}h\n")
            f.write(f"  hour_max  평균={valid['hour_max'].mean():.2f}h  "
                    f"범위={valid['hour_max'].min():.2f}~{valid['hour_max'].max():.2f}h\n")
            f.write(f"  hour_mean 평균={valid['hour_mean'].mean():.2f}h  "
                    f"범위={valid['hour_mean'].min():.2f}~{valid['hour_mean'].max():.2f}h\n\n")
            f.write("파일별 상세:\n")
            for _, row in valid.iterrows():
                f.write(f"  {row['file'][:50]}  "
                        f"{row['time_min_utc'][11:16]}~{row['time_max_utc'][11:16]} UTC  "
                        f"EA={row['n_pixels_ea']:,}px\n")
        print(f"\n  요약 리포트 저장: {summary_txt}")

    print(f"\n✅ 완료! 결과 경로: {OUT_DIR}")


# ─────────────────────────────────────────────
# 단일 파일 빠른 체크 (테스트용)
# ─────────────────────────────────────────────
def quick_check(nc_path_str: str):
    """
    단일 파일 구조 + 시간 빠른 확인
    사용법: quick_check(r"E:\...\S5P_L2__NO2__....nc")
    """
    nc_path = Path(nc_path_str)
    print(f"=== 구조 탐색: {nc_path.name} ===")
    struct = inspect_nc_structure(nc_path)
    print(f"시간 변수: {struct['time_vars']}")
    print(f"위경도 변수: {struct['geo_vars']}")

    print(f"\n=== 시간 분석 ===")
    res = analyze_file_time(nc_path)
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

    # 단일 파일 테스트 시 아래 주석 해제
    # quick_check(str(config.S5P_DIR / "파일명.nc"))