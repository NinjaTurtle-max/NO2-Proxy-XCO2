"""
=============================================================
 OCO-2 / OCO-3 LtCO2 NC4 전수 처리 → 동아시아 슬라이싱 → CSV
 LAT: 20–50°N,  LON: 100–150°E

 입력:
   E:/oco/OCO2_L2_Lite_FP_11.2r-20251109_063042/*.nc4  (2374개)
   E:/oco/OCO3_L2_Lite_FP_11r-20251109_113145/*.nc4   (1576개)

 출력 (센서+연도별 CSV):
   E:/oco/east_asia_csv/oco2_east_asia_2018.csv
   E:/oco/east_asia_csv/oco3_east_asia_2019.csv  ...

 포함 변수:
   - 최상위 1D 변수 전체 (lat, lon, time, xco2, sza 등)
   - date(N,7) -> "date" 컬럼으로 YYYY-MM-DD HH:MM:SS 변환
   - 하위 그룹(Meteorology/Preprocessors/Retrieval/Sounding) 내 1D 변수 전체
     (컬럼명 접두사 예: met_windspeed_u_met, ret_xco2_raw ...)
   - 다차원 배열(pressure_levels 등) 및 크기가 다른 L1b 그룹은 제외

 실행 환경:
   C:/Users/LEMON/anaconda3/python.exe 사용
   (netCDF4, pandas, numpy 포함 환경)
=============================================================
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import time as _time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0

SENSORS = {
    "oco2": config.OCO2_DIR,
    "oco3": config.OCO3_DIR,
}
OUT_DIR = config.OCO_CSV_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 그룹명 → 컬럼 접두사 (L1b는 사운딩 수가 달라 제외)
GROUP_PREFIX = {
    "Auxiliary":    "aux_",
    "Meteorology":  "met_",
    "Preprocessors":"pre_",
    "Retrieval":    "ret_",
    "Sounding":     "snd_",
}


# ─────────────────────────────────────────────
# 파일명에서 (연도, 월, 일) 파싱
#   oco2_LtCO2_YYMMDD_...  →  20YY
# ─────────────────────────────────────────────
def parse_year_from_name(fname: str) -> str:
    parts = fname.split("_")
    for p in parts:
        if len(p) == 6 and p.isdigit():      # YYMMDD
            yy = int(p[:2])
            # OCO-2 2014~, OCO-3 2019~ → 모두 2000년대
            return f"20{p[:2]}"
    return "unknown"


# ─────────────────────────────────────────────
# date 변수 (N, 7) → "YYYY-MM-DD HH:MM:SS" 문자열 배열
# ─────────────────────────────────────────────
def decode_date_var(arr: np.ndarray) -> np.ndarray:
    """arr shape: (N, 7)  [year, month, day, hour, min, sec, microsec]"""
    try:
        yr  = arr[:, 0].astype(int)
        mo  = arr[:, 1].astype(int)
        dy  = arr[:, 2].astype(int)
        hr  = arr[:, 3].astype(int)
        mn  = arr[:, 4].astype(int)
        sc  = arr[:, 5].astype(int)
        result = np.array([
            f"{y:04d}-{m:02d}-{d:02d} {h:02d}:{n:02d}:{s:02d}"
            for y, m, d, h, n, s in zip(yr, mo, dy, hr, mn, sc)
        ])
        return result
    except Exception:
        return np.full(len(arr), "")


# ─────────────────────────────────────────────
# masked array → numpy 배열 (정수형 NaN 안전 처리)
# ─────────────────────────────────────────────
def _safe_fill(arr):
    """masked array를 일반 ndarray로 변환. 정수형은 -9999로 채움."""
    if not hasattr(arr, "filled"):
        return np.array(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.filled(-9999)
    return arr.filled(np.nan)


# ─────────────────────────────────────────────
# 단일 NC4 파일 → 동아시아 슬라이싱 후 DataFrame 반환
# ─────────────────────────────────────────────
def slice_file(nc_path: Path) -> pd.DataFrame | None:
    try:
        with nc.Dataset(nc_path, "r") as ds:
            lat = np.array(ds.variables["latitude"][:], dtype=float)
            lon = np.array(ds.variables["longitude"][:], dtype=float)

            N = len(lat)
            mask = (
                (lat >= LAT_MIN) & (lat <= LAT_MAX) &
                (lon >= LON_MIN) & (lon <= LON_MAX)
            )
            if not np.any(mask):
                return None          # 동아시아 데이터 없음

            row_dict = {}

            # ── 최상위 1D 변수 ──
            for name, var in ds.variables.items():
                if var.ndim == 1 and var.shape[0] == N:
                    raw = var[mask]
                    raw = _safe_fill(raw)
                    row_dict[name] = raw

                elif name == "date" and var.ndim == 2 and var.shape[0] == N:
                    # date (N,7) → datetime 문자열
                    raw = np.array(var[:])
                    if hasattr(raw, "filled"):
                        raw = raw.filled(0)
                    row_dict["date"] = decode_date_var(raw)[mask]

                # 나머지 다차원 변수(pressure_levels 등)는 제외

            # ── 하위 그룹 1D 변수 ──
            for grp_name, prefix in GROUP_PREFIX.items():
                if grp_name not in ds.groups:
                    continue
                grp = ds.groups[grp_name]
                for vname, var in grp.variables.items():
                    if var.ndim == 1 and var.shape[0] == N:
                        raw = var[mask]
                        raw = _safe_fill(raw)
                        row_dict[f"{prefix}{vname}"] = raw

            if not row_dict:
                return None

            df = pd.DataFrame(row_dict)

            # sounding_id가 bytes 타입인 경우 디코딩
            if "sounding_id" in df.columns:
                try:
                    df["sounding_id"] = df["sounding_id"].astype(str)
                except Exception:
                    pass

            return df

    except Exception as e:
        print(f"    [WARN] 읽기 오류: {nc_path.name} -- {e}")
        return None


# ─────────────────────────────────────────────
# 센서별 전수 처리
# ─────────────────────────────────────────────
def process_sensor(sensor: str, nc_dir: Path):
    files = sorted(nc_dir.glob("*.nc4"))
    total = len(files)
    print(f"\n{'='*60}")
    print(f"  {sensor.upper()}  ({total}개 파일)")
    print(f"{'='*60}")

    if total == 0:
        print("  [ERROR] NC4 파일 없음")
        return

    # 연도별 누적 버퍼
    year_buffers: dict[str, list[pd.DataFrame]] = {}
    year_rows:    dict[str, int] = {}
    t0 = _time.time()

    for i, fpath in enumerate(files, 1):
        year = parse_year_from_name(fpath.name)
        df = slice_file(fpath)

        if df is not None and len(df) > 0:
            year_buffers.setdefault(year, []).append(df)
            year_rows[year] = year_rows.get(year, 0) + len(df)

        # 진행 상황 출력 (100파일마다)
        if i % 100 == 0 or i == total:
            elapsed = _time.time() - t0
            ea_pts = sum(year_rows.values())
            pct = i / total * 100
            print(
                f"  [{i:4d}/{total}] {pct:5.1f}%  "
                f"동아시아 누적: {ea_pts:,}점  "
                f"경과: {elapsed:.0f}s"
            )

    # 연도별 CSV 저장
    print(f"\n  [CSV] 저장 중...")
    for year, dfs in sorted(year_buffers.items()):
        out_csv = OUT_DIR / f"{sensor}_east_asia_{year}.csv"
        df_year = pd.concat(dfs, ignore_index=True)

        # 부동소수점 정밀도 최적화
        for col in df_year.select_dtypes(include="float64").columns:
            df_year[col] = df_year[col].astype("float32")

        df_year.to_csv(out_csv, index=False, encoding="utf-8-sig")
        csv_mb = out_csv.stat().st_size / 1024 / 1024
        print(
            f"    [OK] {out_csv.name}  "
            f"{len(df_year):,}행  {df_year.shape[1]}컬럼  {csv_mb:.1f} MB"
        )

    total_pts = sum(year_rows.values())
    print(f"\n  {sensor.upper()} 완료: 동아시아 총 {total_pts:,}점")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  OCO-2/3 동아시아 슬라이싱")
    print(f"  LAT {LAT_MIN}–{LAT_MAX}°N  /  LON {LON_MIN}–{LON_MAX}°E")
    print(f"  출력 디렉터리: {OUT_DIR}")
    print("=" * 60)

    t_start = _time.time()
    for sensor, nc_dir in SENSORS.items():
        process_sensor(sensor, nc_dir)

    elapsed = _time.time() - t_start
    print(f"\n[DONE] 전체 완료  (총 소요: {elapsed/60:.1f}분)")
    print(f"   → {OUT_DIR}")


if __name__ == "__main__":
    main()
