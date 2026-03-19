"""
=============================================================
 TROPOMI L2 NO₂ 파일 무결성 검사
 - 깨진 파일 / 읽기 실패 / 핵심 변수 누락 탐지
 - 결과를 CSV + 콘솔로 출력
=============================================================
"""

import os
import re
import netCDF4 as nc
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

DATA_DIR = config.S5P_DIR
OUTPUT_CSV = config.S5P_DIR / "file_check_result.csv"

# 반드시 존재해야 할 핵심 변수
REQUIRED_VARS = [
    "latitude",
    "longitude",
    "nitrogendioxide_tropospheric_column",
    "qa_value",
]

# ─────────────────────────────────────────────
# 파일 검사 함수
# ─────────────────────────────────────────────
def check_file(fp: Path) -> dict:
    result = {
        "filename":     fp.name,
        "size_MB":      round(fp.stat().st_size / 1024 / 1024, 2),
        "date":         "",
        "status":       "OK",
        "error":        "",
        "n_pixels":     0,
        "qa_valid_pct": 0.0,
        "no2_min":      None,
        "no2_max":      None,
    }

    # 날짜 파싱
    dates = re.findall(r"(\d{8})T", fp.stem)
    result["date"] = dates[0] if dates else "unknown"

    # 파일 크기 0 체크
    if result["size_MB"] == 0:
        result["status"] = "EMPTY"
        result["error"]  = "파일 크기 0"
        return result

    try:
        with nc.Dataset(fp) as ds:
            # PRODUCT 그룹 존재 확인
            if "PRODUCT" not in ds.groups:
                result["status"] = "ERROR"
                result["error"]  = "PRODUCT 그룹 없음"
                return result

            product = ds.groups["PRODUCT"]

            # 핵심 변수 존재 확인
            missing = [v for v in REQUIRED_VARS if v not in product.variables]
            if missing:
                result["status"] = "ERROR"
                result["error"]  = f"변수 누락: {missing}"
                return result

            # 데이터 로드 및 기본 통계
            no2 = product.variables["nitrogendioxide_tropospheric_column"][0]
            qa  = product.variables["qa_value"][0]

            n_total = no2.size
            qa_valid = np.sum(qa >= 0.75)
            no2_vals = no2[qa >= 0.75]

            result["n_pixels"]     = int(n_total)
            result["qa_valid_pct"] = round(float(qa_valid / n_total * 100), 1) if n_total > 0 else 0
            result["no2_min"]      = round(float(np.nanmin(no2_vals)), 6) if no2_vals.size > 0 else None
            result["no2_max"]      = round(float(np.nanmax(no2_vals)), 6) if no2_vals.size > 0 else None

            # 이상값 체크 (음수 과다 또는 비정상적 최대값)
            nan_ratio = float(np.sum(np.isnan(no2_vals)) / no2_vals.size) if no2_vals.size > 0 else 1.0
            if nan_ratio > 0.95:
                result["status"] = "WARN"
                result["error"]  = f"NaN 비율 {nan_ratio*100:.1f}% 초과"
            elif result["no2_max"] and result["no2_max"] > 1.0:
                result["status"] = "WARN"
                result["error"]  = f"NO₂ 최대값 비정상: {result['no2_max']}"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"]  = str(e)

    return result


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────
def main():
    files = sorted(DATA_DIR.glob("S5P_*.nc"))
    print(f"총 {len(files)}개 파일 발견\n")

    if not files:
        print("❌ NC 파일이 없습니다. 경로를 확인하세요.")
        return

    results = []
    for fp in tqdm(files, desc="파일 검사 중"):
        results.append(check_file(fp))

    df = pd.DataFrame(results)

    # ── 요약 출력 ──
    total   = len(df)
    ok      = len(df[df["status"] == "OK"])
    warn    = len(df[df["status"] == "WARN"])
    error   = len(df[df["status"] == "ERROR"])
    empty   = len(df[df["status"] == "EMPTY"])

    print(f"\n{'='*55}")
    print(f"  📊 검사 결과 요약")
    print(f"{'='*55}")
    print(f"  전체 파일 수  : {total}")
    print(f"  ✅ 정상 (OK)  : {ok}")
    print(f"  ⚠️  경고 (WARN): {warn}")
    print(f"  ❌ 오류 (ERROR): {error}")
    print(f"  🈳 빈파일      : {empty}")
    print(f"{'='*55}")

    # ── 문제 파일 상세 출력 ──
    problems = df[df["status"] != "OK"]
    if not problems.empty:
        print(f"\n🚨 문제 파일 목록 ({len(problems)}개):")
        for _, row in problems.iterrows():
            print(f"  [{row['status']}] {row['filename']}")
            print(f"         날짜: {row['date']}  크기: {row['size_MB']} MB")
            print(f"         원인: {row['error']}")
    else:
        print("\n🎉 모든 파일이 정상입니다!")

    # ── 날짜별 현황 ──
    print(f"\n📅 날짜별 파일 수:")
    date_summary = df.groupby("date")["status"].value_counts().unstack(fill_value=0)
    print(date_summary.to_string())

    # ── QA 통계 ──
    ok_df = df[df["status"] == "OK"]
    if not ok_df.empty:
        print(f"\n📈 정상 파일 QA 통계:")
        print(f"  QA≥0.75 유효 픽셀 비율 평균: {ok_df['qa_valid_pct'].mean():.1f}%")
        print(f"  QA≥0.75 유효 픽셀 비율 최소: {ok_df['qa_valid_pct'].min():.1f}%")
        print(f"  NO₂ 전체 범위: {ok_df['no2_min'].min():.4f} ~ {ok_df['no2_max'].max():.4f} mol/m²")

    # ── CSV 저장 ──
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n💾 결과 저장: {OUTPUT_CSV}")
    print("\n✅ 검사 완료!")


if __name__ == "__main__":
    main()