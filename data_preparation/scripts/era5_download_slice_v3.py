"""
=============================================================
 ERA5 기상재분석 다운로드 → 슬라이싱 → CSV 저장 → 원본 삭제  (v3)
 - 입력: tropomi_east_asia_sliced.csv
         컬럼: date, time_start, time_end
         (time 형식: YYYYMMDDTHHmmss  예) 20240102T003958)
 - 다운로드 대상 시각:
     time_start의 정시(hour) ~ time_end의 정시(hour) 사이
     모든 ERA5 정시 슬롯 (관측 창 내 시간대별 기상)
 - 일평균(24h mean) 도 함께 출력
 - 변수: u10, v10, blh
 - 대상 지역: 동아시아 (20–50°N, 100–150°E)
=============================================================
 v1→v2 수정 이력 (era5_download_slice.py 참고)
 v2→v3 변경 사항
   [CHG-1] 입력 CSV 교체
           tropomi_time_check.csv (hour_mean) →
           tropomi_east_asia_sliced.csv (time_start, time_end)
   [CHG-2] 관측 시간 산출 방식 변경
           단일 hour_mean → time_start~time_end 내 모든 정시
   [CHG-3] YEARS 자동 감지 (CSV의 date 컬럼에서 추출)
=============================================================
 사전 준비:
   pip install cdsapi netCDF4 pandas numpy

   ~/.cdsapirc 또는 %USERPROFILE%\.cdsapirc 파일:
   url: https://cds.climate.copernicus.eu/api
   key: YOUR_API_KEY
=============================================================
"""

import time
import numpy as np
import netCDF4 as nc
import pandas as pd
import cdsapi
import warnings
import sys
from pathlib import Path

# Project Root for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
NC_DIR  = config.ERA5_NC_DIR
CSV_DIR = config.ERA5_CSV_V3_DIR

# [CHG-1] 새 입력 CSV
TROPOMI_CSV_PATH = config.TROPOMI_SLICED_CSV

NC_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = [f"{m:02d}" for m in range(1, 13)]

AREA      = [50, 100, 20, 150]   # [North, West, South, East]
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "boundary_layer_height",
]
# 일평균 산출을 위해 24시간 전체 다운로드 (슬라이싱 단계에서 관측 시각만 추출)
TIMES = [f"{h:02d}:00" for h in range(24)]

LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0

MAX_RETRY   = 3
MIN_SIZE_MB = 1.0
RETRY_WAIT  = 60


# ─────────────────────────────────────────────
# [CHG-2] TROPOMI CSV 파싱 (time_start / time_end 기반)
# ─────────────────────────────────────────────
def get_tropomi_dict(csv_path: Path) -> dict:
    """
    date, time_start, time_end 컬럼 파싱.
    time_start ~ time_end 사이 모든 정시(hour) 수집.

    time 형식: YYYYMMDDTHHmmss  (예: 20240102T003958)
      → time_start=00:39, time_end=02:21  →  hours = {0, 1, 2}

    반환: { year: { month: { day_str: set([hours]) } } }
    """
    print("📋 TROPOMI CSV 로딩 중...")
    if not csv_path.exists():
        print(f"❌ 파일 없음: {csv_path}")
        return {}

    # 픽셀마다 같은 granule 정보가 반복 → 고유 granule만 추출
    df = pd.read_csv(csv_path, usecols=["date", "time_start", "time_end"])
    df = df.drop_duplicates(subset=["date", "time_start", "time_end"])
    df = df.dropna(subset=["date", "time_start", "time_end"])

    # [CHG-2] 시간 파싱 (YYYYMMDDTHHmmss)
    df["ts"] = pd.to_datetime(
        df["time_start"].astype(str), format="%Y%m%dT%H%M%S", errors="coerce"
    )
    df["te"] = pd.to_datetime(
        df["time_end"].astype(str), format="%Y%m%dT%H%M%S", errors="coerce"
    )
    n_fail = df["ts"].isna().sum()
    if n_fail > 0:
        print(f"  ⚠️  시간 파싱 실패 {n_fail}건 제외")
    df = df.dropna(subset=["ts", "te"])

    tropomi_dict: dict = {}
    for _, row in df.iterrows():
        date_str = str(int(row["date"]))          # YYYYMMDD (int→str)
        y = date_str[0:4]
        m = date_str[4:6]
        d = date_str[6:8]

        h_start = row["ts"].hour
        h_end   = row["te"].hour
        # 관측 창(time_start ~ time_end) 내 모든 정시 포함
        hours = set(range(h_start, h_end + 1))

        tropomi_dict.setdefault(y, {}).setdefault(m, {}).setdefault(d, set()).update(hours)

    # 연도·일자 수 출력
    for yr, yd in sorted(tropomi_dict.items()):
        n_days   = sum(len(dd) for dd in yd.values())
        n_hours  = sum(len(hs) for md in yd.values() for hs in md.values())
        print(f"  {yr}년: {n_days}일 / 관측 시각 슬롯 총 {n_hours}개")

    return tropomi_dict


# ─────────────────────────────────────────────
# NC 무결성 검사
# ─────────────────────────────────────────────
def verify_netcdf(fpath: Path) -> tuple[bool, str]:
    size_mb = fpath.stat().st_size / 1024 / 1024
    if size_mb < MIN_SIZE_MB:
        return False, f"파일 크기 미달 ({size_mb:.2f} MB)"
    try:
        with nc.Dataset(fpath, "r") as ds:
            missing = [v for v in ["u10", "v10"] if v not in ds.variables]
            if missing:
                return False, f"핵심 변수 누락: {missing}"
            u10 = ds.variables["u10"][:]
            if u10.size == 0:
                return False, "u10 데이터 크기 0"
            nan_ratio = float(
                np.sum(np.isnan(u10.filled(np.nan))) / u10.size
            )
            if nan_ratio > 0.99:
                return False, f"NaN 비율 {nan_ratio*100:.1f}%"
    except Exception as e:
        return False, f"NetCDF 읽기 실패: {e}"
    return True, "OK"


# ─────────────────────────────────────────────
# CSV 무결성 검사
# ─────────────────────────────────────────────
def verify_csv(fpath: Path) -> tuple[bool, str]:
    try:
        df = pd.read_csv(fpath, nrows=5)
        required_cols = {
            "date", "obs_time", "lat", "lon",
            "u10_obs", "v10_obs", "u10_daily", "v10_daily",
        }
        missing = required_cols - set(df.columns)
        if missing:
            return False, f"필수 컬럼 누락: {missing}"
        n_rows = sum(1 for _ in open(fpath, encoding="utf-8")) - 1
        if n_rows < 10:
            return False, f"행 수 너무 적음 ({n_rows}행)"
    except Exception as e:
        return False, f"CSV 읽기 실패: {e}"
    return True, "OK"


# ─────────────────────────────────────────────
# NC → CSV 슬라이싱
#   · 관측 시각별(obs_time) ERA5 순간값 + 일평균(daily) 함께 출력
#   · numpy 벡터화 / meshgrid 방식으로 격자 루프 제거
# ─────────────────────────────────────────────
def slice_nc_to_csv(
    nc_path: Path,
    csv_path: Path,
    month_dict: dict,          # { day_str: set([hours]) }
    year: str,
    month: str,
) -> tuple[bool, str, int]:
    try:
        with nc.Dataset(nc_path, "r") as ds:

            # ── 위경도·시간 변수 탐색 ──
            lat_key = next(
                (k for k in ds.variables if k.lower() in ("latitude",  "lat")), None
            )
            lon_key = next(
                (k for k in ds.variables if k.lower() in ("longitude", "lon")), None
            )
            # ERA5 new CDS API → valid_time, 구버전 → time
            time_key = next(
                (k for k in ds.variables
                 if k.lower() in ("valid_time", "time")), None
            )

            if not all([lat_key, lon_key, time_key]):
                return False, (
                    f"좌표 변수 탐색 실패 "
                    f"(lat={lat_key}, lon={lon_key}, time={time_key})"
                ), 0

            lat   = np.array(ds.variables[lat_key][:])
            lon   = np.array(ds.variables[lon_key][:])
            times = np.array(ds.variables[time_key][:])

            # 시간 단위 → datetime
            time_units = ds.variables[time_key].units
            try:
                time_objs = nc.num2date(
                    times, time_units,
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True,
                )
                time_dts = pd.to_datetime(
                    [d.strftime("%Y-%m-%d %H:%M:%S") for d in time_objs]
                )
            except Exception as e:
                return False, f"시간 변환 실패: {e}", 0

            # ── 도메인 인덱스 ──
            lat_idx = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
            lon_idx = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
            if len(lat_idx) == 0 or len(lon_idx) == 0:
                return False, "도메인 내 격자점 없음", 0

            lat_sub = lat[lat_idx]
            lon_sub = lon[lon_idx]

            # ── 변수 추출 (도메인 슬라이싱 포함) ──
            def extract_sub(var_key):
                if var_key not in ds.variables:
                    return None
                raw = np.array(ds.variables[var_key][:], dtype=float)
                fv  = getattr(ds.variables[var_key], "_FillValue", None)
                if fv is not None:
                    raw[raw == float(fv)] = np.nan
                if raw.ndim == 3:
                    return raw[
                        :,
                        lat_idx[0]:lat_idx[-1] + 1,
                        lon_idx[0]:lon_idx[-1] + 1,
                    ]
                elif raw.ndim == 2:
                    return raw[
                        lat_idx[0]:lat_idx[-1] + 1,
                        lon_idx[0]:lon_idx[-1] + 1,
                    ]
                return raw

            u10_sub = extract_sub("u10")   # shape: (T, n_lat, n_lon)
            v10_sub = extract_sub("v10")
            blh_sub = extract_sub("blh")

            # ── 시간 배열 ──
            dt_days  = time_dts.day.to_numpy()
            dt_hours = time_dts.hour.to_numpy()

            # meshgrid로 격자 좌표 배열 미리 생성 (루프 밖)
            lon_grid, lat_grid = np.meshgrid(lon_sub, lat_sub)
            lat_flat = lat_grid.ravel()
            lon_flat = lon_grid.ravel()

            all_rows = []

            for d_str, hours_set in month_dict.items():
                day_int   = int(d_str)
                day_mask  = (dt_days == day_int)
                day_indices = np.where(day_mask)[0]
                if len(day_indices) == 0:
                    continue

                # ── 일평균 (24h, 벡터화) ──
                u10_daily_2d = np.nanmean(u10_sub[day_indices], axis=0)
                v10_daily_2d = np.nanmean(v10_sub[day_indices], axis=0)
                blh_daily_2d = (
                    np.nanmean(blh_sub[day_indices], axis=0)
                    if blh_sub is not None else None
                )

                u10_d_flat = u10_daily_2d.ravel()
                v10_d_flat = v10_daily_2d.ravel()
                blh_d_flat = (
                    blh_daily_2d.ravel()
                    if blh_daily_2d is not None
                    else np.full(len(u10_d_flat), np.nan)
                )

                ws_daily = np.sqrt(u10_d_flat**2 + v10_d_flat**2)
                wd_daily = (
                    np.degrees(np.arctan2(u10_d_flat, v10_d_flat)) + 360
                ) % 360

                # ── 관측 창 내 각 정시별 순간값 ──
                for obs_h in sorted(hours_set):
                    obs_mask    = day_mask & (dt_hours == obs_h)
                    obs_indices = np.where(obs_mask)[0]
                    if len(obs_indices) == 0:
                        continue
                    ti = obs_indices[0]

                    u10_o_flat = u10_sub[ti].ravel()
                    v10_o_flat = v10_sub[ti].ravel()
                    blh_o_flat = (
                        blh_sub[ti].ravel()
                        if blh_sub is not None
                        else np.full(len(u10_o_flat), np.nan)
                    )

                    n = len(lat_flat)
                    date_col    = np.full(n, f"{year}{month}{d_str}")
                    obstime_col = np.full(n, f"{obs_h:02d}:00")

                    ws_obs = np.sqrt(u10_o_flat**2 + v10_o_flat**2)
                    wd_obs = (
                        np.degrees(np.arctan2(u10_o_flat, v10_o_flat)) + 360
                    ) % 360

                    block = pd.DataFrame({
                        "date":               date_col,
                        "obs_time":           obstime_col,
                        "lat":                np.round(lat_flat, 4),
                        "lon":                np.round(lon_flat, 4),
                        # 관측 순간 (time_start~time_end 창 내 각 정시)
                        "u10_obs":            np.round(u10_o_flat, 6),
                        "v10_obs":            np.round(v10_o_flat, 6),
                        "blh_obs":            np.round(blh_o_flat, 4),
                        "wind_speed_obs":     np.round(ws_obs, 4),
                        "wind_dir_deg_obs":   np.round(wd_obs, 2),
                        # 관측일 일평균
                        "u10_daily":          np.round(u10_d_flat, 6),
                        "v10_daily":          np.round(v10_d_flat, 6),
                        "blh_daily":          np.round(blh_d_flat, 4),
                        "wind_speed_daily":   np.round(ws_daily, 4),
                        "wind_dir_deg_daily": np.round(wd_daily, 2),
                    })
                    all_rows.append(block)

            if not all_rows:
                return False, "슬라이싱 결과 행 없음", 0

            df_out = pd.concat(all_rows, ignore_index=True)
            df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
            return True, "OK", len(df_out)

    except Exception as e:
        return False, f"슬라이싱 오류: {e}", 0


# ─────────────────────────────────────────────
# 단일 월 처리
# ─────────────────────────────────────────────
def process_month(
    client,
    year: str,
    month: str,
    month_dict: dict,   # { day_str: set([hours]) }
) -> dict:
    csv_path = CSV_DIR / f"era5_{year}_{month}.csv"
    nc_path  = NC_DIR  / f"era5_{year}_{month}.nc"
    nc_tmp   = NC_DIR  / f"era5_{year}_{month}_tmp.nc"

    result = {
        "year": year, "month": month,
        "status": "", "csv_rows": 0,
        "nc_mb": 0.0, "csv_kb": 0.0,
        "attempts": 0, "error": "",
    }

    if not month_dict:
        print(f"  ⏭️  스킵 ({year}-{month}: TROPOMI 관측 없음)")
        result["status"] = "SKIP (No Data)"
        return result

    # 이미 처리된 CSV 존재 → 검증 후 스킵
    if csv_path.exists():
        ok, msg = verify_csv(csv_path)
        if ok:
            csv_kb = csv_path.stat().st_size / 1024
            print(f"  ⏭️  스킵 (CSV 존재): {csv_path.name} ({csv_kb:.0f} KB)")
            result.update({"status": "SKIP", "csv_kb": round(csv_kb, 1)})
            for f in [nc_path, nc_tmp]:
                if f.exists():
                    f.unlink()
            return result
        else:
            print(f"  ⚠️  기존 CSV 손상 → 재처리: {msg}")
            csv_path.unlink()

    target_days   = sorted(month_dict.keys())
    n_hour_slots  = sum(len(hs) for hs in month_dict.values())
    print(
        f"\n  {year}-{month}: {len(target_days)}일 / "
        f"관측 시각 슬롯 {n_hour_slots}개"
    )

    for attempt in range(1, MAX_RETRY + 1):
        result["attempts"] = attempt
        print(f"\n  [{attempt}/{MAX_RETRY}] 시도 중...")

        nc_tmp.unlink(missing_ok=True)

        # Step 1: 다운로드 (24h 전체 → 슬라이싱 시 관측 시각 + 일평균 추출)
        print(f"    📥 ERA5 다운로드 중...", end=" ", flush=True)
        try:
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable":     VARIABLES,
                    "year":         [year],
                    "month":        [month],
                    "day":          target_days,
                    "time":         TIMES,          # 00~23시 전체
                    "area":         AREA,
                    "data_format":  "netcdf",
                },
                str(nc_tmp),
            )
            nc_mb = nc_tmp.stat().st_size / 1024 / 1024
            print(f"완료 ({nc_mb:.1f} MB)")
            result["nc_mb"] = round(nc_mb, 2)
        except Exception as e:
            err = str(e).split("\n")[0]
            print(f"❌ 다운로드 실패: {err}")
            result["error"] = f"download: {err}"
            nc_tmp.unlink(missing_ok=True)
            _wait_retry(attempt)
            continue

        # Step 2: NC 무결성
        print(f"    🔍 NC 무결성 검사...", end=" ", flush=True)
        ok, msg = verify_netcdf(nc_tmp)
        if not ok:
            print(f"❌ {msg}")
            result["error"] = f"nc_verify: {msg}"
            nc_tmp.unlink(missing_ok=True)
            _wait_retry(attempt)
            continue
        print("OK")
        nc_tmp.rename(nc_path)

        # Step 3: 슬라이싱
        print(f"    ✂️  슬라이싱 중...", end=" ", flush=True)
        ok, msg, n_rows = slice_nc_to_csv(
            nc_path, csv_path, month_dict, year, month
        )
        if not ok:
            print(f"❌ {msg}")
            result["error"] = f"slice: {msg}"
            _wait_retry(attempt)
            continue
        csv_kb = csv_path.stat().st_size / 1024
        print(f"완료 ({n_rows:,}행, {csv_kb:.0f} KB)")

        # Step 4: CSV 무결성
        print(f"    🔍 CSV 무결성 검사...", end=" ", flush=True)
        ok, msg = verify_csv(csv_path)
        if not ok:
            print(f"❌ {msg}")
            result["error"] = f"csv_verify: {msg}"
            csv_path.unlink(missing_ok=True)
            _wait_retry(attempt)
            continue
        print("OK")

        # Step 5: NC 삭제 (용량 절감)
        nc_path.unlink()
        print(f"    🗑️  NC 삭제 ({result['nc_mb']:.1f} MB 절감)")

        result.update({
            "status": "OK",
            "csv_rows": n_rows,
            "csv_kb": round(csv_kb, 1),
            "error": "",
        })
        return result

    result["status"] = "FAILED"
    if nc_path.exists():
        print(f"  ⚠️  NC 보존 (수동 슬라이싱 가능): {nc_path.name}")
    print(f"  🚨 {year}-{month} 실패")
    return result


def _wait_retry(attempt: int):
    wait = RETRY_WAIT * (2 ** (attempt - 1))
    if attempt < MAX_RETRY:
        print(f"    → {wait}초 후 재시도...")
        time.sleep(wait)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    c = cdsapi.Client()

    tropomi_dict = get_tropomi_dict(TROPOMI_CSV_PATH)
    if not tropomi_dict:
        print("❌ TROPOMI 데이터 없음. 종료.")
        return

    # [CHG-3] 연도 자동 감지
    years = sorted(tropomi_dict.keys())
    print(f"\n▶ ERA5 다운로드 대상 연도: {', '.join(years)}년")
    print(f"  NC 임시: {NC_DIR}")
    print(f"  CSV 저장: {CSV_DIR}\n")

    results = []
    for year in years:
        print(f"\n{'='*55}")
        print(f"  📅 {year}년")
        print(f"{'='*55}")
        year_dict = tropomi_dict.get(year, {})

        for month in MONTHS:
            month_dict = year_dict.get(month, {})
            res = process_month(c, year, month, month_dict)
            results.append(res)
            time.sleep(3)

    # ── 결과 요약 ──
    df = pd.DataFrame(results)
    ok_n     = len(df[df["status"] == "OK"])
    skip_n   = len(df[df["status"].str.startswith("SKIP")])
    failed_n = len(df[df["status"] == "FAILED"])

    nc_saved  = df[df["status"] == "OK"]["nc_mb"].sum()
    csv_total = df[df["status"].isin(["OK", "SKIP"])]["csv_kb"].sum() / 1024

    print(f"\n{'='*55}")
    print(f"  📊 결과 요약")
    print(f"{'='*55}")
    print(f"  ✅ 성공: {ok_n}  ⏭️ 스킵: {skip_n}  ❌ 실패: {failed_n}")
    print(f"  💾 NC 삭제: {nc_saved:.1f} MB  /  CSV 합계: {csv_total:.1f} MB")
    if nc_saved > 0:
        print(f"  📉 절감률: {(1 - csv_total / nc_saved) * 100:.1f}%")

    if failed_n > 0:
        fail_df = df[df["status"] == "FAILED"][
            ["year", "month", "attempts", "nc_mb", "error"]
        ]
        print(f"\n  🚨 실패 목록:\n{fail_df.to_string(index=False)}")
        fail_csv = CSV_DIR / "era5_failed_list.csv"
        fail_df.to_csv(fail_csv, index=False, encoding="utf-8-sig")
        print(f"  실패 목록 저장: {fail_csv}")

    log_csv = CSV_DIR / "era5_process_log.csv"
    df.to_csv(log_csv, index=False, encoding="utf-8-sig")
    print(f"\n  📋 로그: {log_csv}")
    print(f"✅ 완료! → {CSV_DIR}")


# ─────────────────────────────────────────────
# 실패 재시도
# ─────────────────────────────────────────────
def retry_from_csv(fail_csv_path: str):
    c = cdsapi.Client()
    tropomi_dict = get_tropomi_dict(TROPOMI_CSV_PATH)
    df = pd.read_csv(fail_csv_path)
    print(f"재시도 대상: {len(df)}개\n")
    for _, row in df.iterrows():
        y = str(row["year"])
        m = str(row["month"]).zfill(2)
        month_dict = tropomi_dict.get(y, {}).get(m, {})
        res = process_month(c, y, m, month_dict)
        print(f"  → {res['status']}: {y}-{m}")
        time.sleep(5)


if __name__ == "__main__":
    main()

    # 실패 재시도 시 아래 주석 해제
    # retry_from_csv(str(CSV_DIR / "era5_failed_list.csv"))
