"""
TROPOMI L2 NO2 데이터 → 동아시아 슬라이싱 → CSV 저장
도메인: 20–50°N, 100–150°E
"""

import os
import glob
import numpy as np
import netCDF4 as nc
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

NC_DIR  = str(config.S5P_DIR)

# 2. 결과 저장 경로 (파일명까지 확실히 지정)
OUT_DIR = str(config.NO2_OUT_DIR)
OUT_CSV = str(config.TROPOMI_SLICED_CSV) # <--- 파일명 추가

# 3. 만약 폴더가 없다면 자동으로 생성하는 코드 (안전장치)
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    print(f"폴더 생성 완료: {OUT_DIR}")
    
# 동아시아 도메인
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0

# QA 필터 임계값
QA_THRESHOLD = 0.75

# ─────────────────────────────────────────
# 추출할 변수 목록 (변수명: 저장 컬럼명)
# ─────────────────────────────────────────
VARIABLES = {
    "nitrogendioxide_tropospheric_column":          "no2_tvcd",          # mol/m2
    "nitrogendioxide_tropospheric_column_precision":"no2_tvcd_precision",
    "qa_value":                                     "qa",
    "surface_pressure":                             "surface_pressure",  # Pa
    "cloud_radiance_fraction_nitrogendioxide_window":"cloud_fraction",
}

# ─────────────────────────────────────────
# 메인 처리
# ─────────────────────────────────────────
def extract_one_file(nc_path: str) -> pd.DataFrame | None:
    """단일 NC 파일에서 동아시아 픽셀 추출 → DataFrame 반환"""
    try:
        ds = nc.Dataset(nc_path)
        grp = ds.groups["PRODUCT"]

        lat = grp.variables["latitude"][0]   # (scanline, pixel)
        lon = grp.variables["longitude"][0]

        # 동아시아 마스크
        mask = (
            (lat >= LAT_MIN) & (lat <= LAT_MAX) &
            (lon >= LON_MIN) & (lon <= LON_MAX)
        )

        # QA 필터 추가
        qa = grp.variables["qa_value"][0]
        mask = mask & (qa >= QA_THRESHOLD)

        if not np.any(mask):
            ds.close()
            return None

        # 날짜·시간·파일명
        fname = os.path.basename(nc_path)
        time_part = fname.split("____")[1]          # 20240102T003958_20240102T0...
        date_str = time_part[:8]                     # YYYYMMDD
        time_start = time_part[:15]                  # YYYYMMDDTHHMMSS (관측 시작)
        time_end   = time_part[16:31]                # YYYYMMDDTHHMMSS (관측 종료)

        rows = {"filename": fname, "date": date_str,
                "time_start": time_start, "time_end": time_end,
                "lat": lat[mask], "lon": lon[mask]}

        for var_key, col_name in VARIABLES.items():
            try:
                data = grp.variables[var_key][0]
                # fill value → NaN
                if hasattr(grp.variables[var_key], "_FillValue"):
                    fv = grp.variables[var_key]._FillValue
                    data = np.where(data == fv, np.nan, data)
                rows[col_name] = data[mask]
            except KeyError:
                rows[col_name] = np.full(np.sum(mask), np.nan)

        ds.close()
        return pd.DataFrame(rows)

    except Exception as e:
        print(f"  [SKIP] {os.path.basename(nc_path)}: {e}")
        return None


def main():
    nc_files = sorted(glob.glob(os.path.join(NC_DIR, "*.nc")))
    print(f"총 {len(nc_files)}개 파일 처리 시작...")

    all_dfs = []
    for i, fp in enumerate(nc_files, 1):
        print(f"  [{i:4d}/{len(nc_files)}] {os.path.basename(fp)}", end=" ... ")
        df = extract_one_file(fp)
        if df is not None:
            all_dfs.append(df)
            print(f"{len(df):,} 픽셀")
        else:
            print("(픽셀 없음 또는 스킵)")

    if not all_dfs:
        print("추출된 데이터가 없습니다.")
        return

    result = pd.concat(all_dfs, ignore_index=True)

    # NO2 단위 변환: mol/m² → μmol/m²
    if "no2_tvcd" in result.columns:
        result["no2_tvcd_umol"] = result["no2_tvcd"] * 1e6

    # CSV 저장
    result.to_csv(OUT_CSV, index=False, float_format="%.6e")
    print(f"\n✅ 저장 완료: {OUT_CSV}")
    print(f"   총 행 수: {len(result):,}")
    print(f"   컬럼:     {list(result.columns)}")
    print(f"   날짜 범위: {result['date'].min()} ~ {result['date'].max()}")


if __name__ == "__main__":
    main()