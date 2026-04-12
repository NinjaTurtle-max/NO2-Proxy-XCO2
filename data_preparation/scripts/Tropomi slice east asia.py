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
# config.py가 Onboard-NO_CO 폴더 내에 있으므로 해당 경로를 최우선으로 추가
grand_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(grand_parent, "Onboard-NO_CO"))
import config

NC_DIR  = r"/mnt/g/S5P_L2__NO2____HiR_2-20260327_051756"

# 2. 결과 저장 경로 (E 드라이브)
OUT_CSV = "/mnt/e/dataset/no2/tropomi_east_asia_sliced.csv"
OUT_DIR = os.path.dirname(OUT_CSV)

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

        df = pd.DataFrame(rows)
        # NO2 단위 변환: mol/m² → μmol/m²
        if "no2_tvcd" in df.columns:
            df["no2_tvcd_umol"] = df["no2_tvcd"] * 1e6

        ds.close()
        return df

    except Exception as e:
        print(f"  [SKIP] {os.path.basename(nc_path)}: {e}")
        return None


def main():
    print(">>> 스크립트 실행 시작...", flush=True)
    
    # 1. 이미 처리된 파일 목록 가져오기 (중복 처리 방지)
    processed_files = set()
    if os.path.exists(OUT_CSV):
        try:
            print(f"기존 파일({os.path.basename(OUT_CSV)})에서 처리 이력을 읽는 중...", flush=True)
            with open(OUT_CSV, 'r', encoding='utf-8') as f:
                next(f) # 헤더 스킵
                for line in f:
                    if line:
                        # 첫 번째 항목인 filename만 추출하여 세트에 추가
                        fname = line.split(',', 1)[0]
                        processed_files.add(fname)
            print(f"이력 확인 완료: 총 {len(processed_files)}개의 파일이 이미 처리됨.", flush=True)
        except Exception as e:
            print(f"기존 파일 읽기 중 오류 발생: {e}", flush=True)

    # 2. 파일 목록 스캔 및 필터링
    all_nc_files = sorted(glob.glob(os.path.join(NC_DIR, "*.nc")))
    nc_files = [f for f in all_nc_files if os.path.basename(f) not in processed_files]
    
    print(f"전체 {len(all_nc_files)}개 중 {len(nc_files)}개 신규 파일 처리 시작...", flush=True)

    for i, fp in enumerate(nc_files, 1):
        print(f"  [{i:4d}/{len(nc_files)}] {os.path.basename(fp)}", end=" ... ", flush=True)
        df = extract_one_file(fp)
        
        if df is not None:
            # CSV에 즉시 추가 (메모리 절약)
            file_exists = os.path.exists(OUT_CSV)
            df.to_csv(OUT_CSV, index=False, header=not file_exists, mode='a', float_format="%.6e")
            print(f"{len(df):,} 픽셀 저장 완료", flush=True)
        else:
            print("(픽셀 없음 또는 스킵)", flush=True)

    print(f"\n✅ 모든 신규 데이터 처리 및 저장 완료: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()