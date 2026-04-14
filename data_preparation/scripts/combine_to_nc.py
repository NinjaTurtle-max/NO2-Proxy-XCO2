"""
이 스크립트는 OCO-2/3, TROPOMI NO2, ERA5 기상, GPW 인구밀도, ODIAC 배출량 데이터를
시계열 순으로 병합(Collocation)하여 하나의 NetCDF (.nc) 파일로 생성합니다.

[ERA5 변수 현황]
현재 ERA5 파일(era5_YYYY_MM.csv)에 포함된 변수:
  ✅ U10 (u10_daily)       : 10m 동서 풍속 일평균
  ✅ V10 (v10_daily)       : 10m 남북 풍속 일평균
  ✅ BLH (blh_daily)       : 경계층고도 일평균
  ✅ wind_speed_daily       : 10m 풍속 크기 일평균
  ✅ wind_dir_deg_daily     : 10m 풍향(°) 일평균
  ❌ T2m                   : 현재 ERA5 파일에 없음 → 별도 ERA5 다운로드 필요
  ❌ RH                    : 현재 ERA5 파일에 없음 → 별도 ERA5 다운로드 필요

의존성: pip install polars netCDF4 rasterio scipy pandas pyarrow numpy
"""

import os
import gc
import glob
import multiprocessing
import numpy as np
import pandas as pd
import rasterio
import netCDF4 as nc4
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("[경고] polars 없음 → pandas 대체. pip install polars 권장")

# ─────────────────────────────────────────────
# 1. 경로 설정
# ─────────────────────────────────────────────
BASE_DIR            = "/Volumes/100.118.65.89/dataset/XCO2연구 데이터"
OUT_NC              = os.path.join(BASE_DIR, "integrated_dataset.nc")
TROPOMI_CSV         = os.path.join(BASE_DIR, "tropomi_east_asia_sliced.csv")
TROPOMI_PARQUET_DIR = os.path.join(BASE_DIR, "_tropomi_by_date")
ERA5_DIR            = os.path.join(BASE_DIR, "ERA_5_xco2")
ERA5_PARQUET_DIR    = os.path.join(BASE_DIR, "_era5_by_date")
TEMP_PARQUET_DIR    = os.path.join(BASE_DIR, "_oco_tmp")
GPW_TIF             = os.path.join(BASE_DIR,
    "gpw-v4-population-density-rev11_2020_15_min_tif",
    "gpw_v4_population_density_rev11_2020_15_min_east_asia.tif")

N_WORKERS      = 2        # 메모리 절약: 동시 워커 수 제한 (None → CPU 코어 수 자동)
NC_WRITE_BATCH = 200_000

# ── 보존할 OCO 핵심 과학 컬럼 (100개 → 30개 축소) ──────────────────────
# OCO-2/3 공통 + 연구에 필요한 핵심 변수만 지정
# None 으로 바꾸면 전체 컬럼 유지 (메모리 많이 필요)
KEEP_OCO_COLS: set | None = {
    # 식별 / 시공간
    "sounding_id", "date", "latitude", "longitude", "tai_seconds",
    # 품질 플래그
    "xco2_quality_flag", "xco2_qf_bitflag", "xco2_qf_simple_bitflag",
    # 핵심 산출물
    "xco2", "xco2_x2019", "xco2_uncertainty", "xco2_apriori",
    # 지표·기상
    "ret_psurf", "ret_tcwv", "ret_tcwv_uncertainty",
    "ret_aod_total", "ret_aod_dust", "ret_aod_water", "ret_aod_ice",
    "ret_snow_flag", "ret_surface_type",
    # 관측 기하
    "solar_zenith_angle", "sensor_zenith_angle",
    "snd_airmass", "snd_land_fraction", "snd_glint_angle",
    "snd_operation_mode", "snd_land_water_indicator", "snd_altitude",
    # ERA5 기상 (일평균)
    "era5_u10", "era5_v10", "era5_blh",
    "era5_wind_speed", "era5_wind_dir",
    # 병합 추가 컬럼 (처리 중 생성)
    "tropomi_no2", "population_density", "odiac_emission", "file_source",
}

# ─────────────────────────────────────────────
# 2. 유틸리티
# ─────────────────────────────────────────────
def _normalize_date_str(s: pd.Series) -> pd.Series:
    """날짜 컬럼 정규화: '20190101.0' → '20190101'"""
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """타입 안전 다운캐스팅.

    - object 컬럼: 숫자로 변환 가능한 것만 변환 (문자열 컬럼 보호)
    - uint64 → int64 (NC 호환성)
    - float64 → float32, int64 → int32
    """
    for col in df.select_dtypes("object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # 90% 이상 변환 성공해야만 교체 (snd_target_name 등 보호)
        if converted.notna().mean() > 0.9:
            df[col] = converted
    for col in df.select_dtypes("uint64").columns:
        df[col] = df[col].astype(np.int64)  # NC 호환
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype(np.int32)
    return df


def sample_tif(points: list, tif_path: str) -> np.ndarray:
    if not os.path.exists(tif_path):
        return np.full(len(points), np.nan, dtype=np.float32)
    with rasterio.open(tif_path) as src:
        values = [v[0] for v in src.sample(points)]
    return np.array(values, dtype=np.float32)


def get_odiac_path(year: int, month: int) -> str:
    return os.path.join(BASE_DIR, "odiac_sliced",
        f"odiac2024_1km_excl_intl_{str(year)[-2:]}{str(month).zfill(2)}.tif")


def _resolve_cols(df: pd.DataFrame):
    """컬럼명 해석.

    time 컬럼이 float(TAI초)이므로, 날짜 파싱은 date(문자열) 컬럼 우선.
    """
    # date(문자열) > time(float TAI초) 순서로 datetime 파싱 컬럼 선택
    if "date" in df.columns:
        time_col = "date"
    elif "time" in df.columns:
        time_col = "time"
    else:
        raise ValueError("date 또는 time 컬럼이 없습니다.")

    lat_col = "lat"  if "lat"  in df.columns else "latitude"
    lon_col = "lon"  if "lon"  in df.columns else "longitude"
    return time_col, lat_col, lon_col


# ─────────────────────────────────────────────
# 3. OCO 컬럼 합집합 선계산
# ─────────────────────────────────────────────
def get_oco_column_union(oco_files: list[str]) -> list[str]:
    """모든 OCO 파일 헤더를 nrows=0 으로 읽어 컬럼 합집합 반환."""
    all_cols: set[str] = set()
    for f in oco_files:
        all_cols.update(pd.read_csv(f, nrows=0).columns.tolist())

    # float 'time' 컬럼은 'tai_seconds'로 이름 변경 예정
    if "time" in all_cols:
        all_cols.discard("time")
        all_cols.add("tai_seconds")

    # 처리 중 추가 컬럼 (TROPOMI, ERA5, TIF, 위성 출처)
    all_cols.update(["tropomi_no2", "population_density", "odiac_emission",
                     "file_source"])  # Force: 항상 col_union에 포함
    all_cols.update(ERA5_COL_MAP.values())  # era5_u10, era5_v10, era5_blh, ...
    # 날짜 파싱용 date → 'time' (datetime)으로 교체
    all_cols.discard("date")
    all_cols.add("time")
    # 내부 임시 컬럼 제거
    for tmp in ("_date_str", "_year", "_month", "_min_dist"):
        all_cols.discard(tmp)

    # KEEP_OCO_COLS 필터 (None이면 전체)
    if KEEP_OCO_COLS is not None:
        all_cols &= KEEP_OCO_COLS | {"time"}

    # NC에 쓸 수 없는 순수 문자열 컬럼 제거 (NC float/int 변수만 허용)
    # → object 컬럼은 사전 필터링 단계에서 처리
    return sorted(all_cols)


# ─────────────────────────────────────────────
# 4-A. TROPOMI 사전 분할 (최초 1회)
# ─────────────────────────────────────────────
def preprocess_tropomi() -> None:
    """TROPOMI CSV → 날짜별 Parquet 분할 (최초 1회만 실행)."""
    if os.path.isdir(TROPOMI_PARQUET_DIR) and \
       any(Path(TROPOMI_PARQUET_DIR).glob("*.parquet")):
        n = len(list(Path(TROPOMI_PARQUET_DIR).glob("*.parquet")))
        print(f"[TROPOMI] 기존 Parquet {n}개 → 스킵")
        return

    os.makedirs(TROPOMI_PARQUET_DIR, exist_ok=True)
    print(f"[TROPOMI] 날짜별 Parquet 분할 중...")

    # 91GB 대용량 CSV → 스트리밍 청크 처리 (전체 로드 절대 금지)
    # Polars scan_csv(lazy) 또는 pandas chunksize 방식으로 메모리 일정 유지
    dtype_map_pd = {
        "lat":           np.float32,
        "lon":           np.float32,
        "no2_tvcd_umol": np.float32,
    }
    buf: dict[str, list] = {}
    buf_rows = 0
    FLUSH_ROWS = 5_000_000  # 500만 행 누적마다 Parquet flush

    def _flush_buf(buf: dict) -> dict:
        """누적된 날짜별 버퍼를 Parquet으로 flush하고 초기화."""
        for dv, parts in buf.items():
            out = os.path.join(TROPOMI_PARQUET_DIR, f"{dv}.parquet")
            merged = pd.concat(parts, ignore_index=True)
            if os.path.exists(out):
                merged = pd.concat([pd.read_parquet(out), merged], ignore_index=True)
            merged.to_parquet(out, engine="pyarrow", compression="zstd")
        return {}

    # 91GB 대용량 CSV: pandas chunksize 스트리밍 (메모리 상수 유지)
    # Polars read_csv/collect은 전체를 RAM에 올리므로 사용 불가
    for ci, chunk in enumerate(pd.read_csv(
        TROPOMI_CSV,
        usecols=["date", "lat", "lon", "no2_tvcd_umol"],
        dtype=dtype_map_pd,
        chunksize=1_000_000,
    ), 1):
        chunk["date"] = _normalize_date_str(chunk["date"])
        for date_val, grp in chunk.groupby("date"):
            buf.setdefault(date_val, []).append(grp.drop(columns=["date"]))
        buf_rows += len(chunk)
        if ci % 10 == 0:
            print(f"  {ci * 1_000_000 / 1e6:.0f}M행 처리 중... (버퍼 날짜 수: {len(buf)})")
        if buf_rows >= FLUSH_ROWS:
            buf = _flush_buf(buf)
            buf_rows = 0
            gc.collect()

    if buf:
        _flush_buf(buf)
    del buf
    gc.collect()

    n = len(list(Path(TROPOMI_PARQUET_DIR).glob("*.parquet")))
    print(f"[TROPOMI] 분할 완료: {n}개 날짜 Parquet")


# ─────────────────────────────────────────────
# 4-B. ERA5 사전 분할 (최초 1회)
# ─────────────────────────────────────────────

# ERA5에서 추출할 일평균 컬럼 → 출력 컬럼명 매핑
ERA5_COL_MAP = {
    "u10_daily":          "era5_u10",
    "v10_daily":          "era5_v10",
    "blh_daily":          "era5_blh",
    "wind_speed_daily":   "era5_wind_speed",
    "wind_dir_deg_daily": "era5_wind_dir",
}


def preprocess_era5() -> None:
    """ERA5 월별 CSV → 날짜별 Parquet 분할 저장 (최초 1회만 실행).

    ERA5 파일은 이미 월별로 나뉘어 있으나,
    OCO 날짜 매칭을 위해 YYYYMMDD 단위 Parquet으로 재분할한다.
    저장 컬럼: lat, lon + ERA5_COL_MAP 값들(era5_u10 등)
    """
    if os.path.isdir(ERA5_PARQUET_DIR) and \
       any(Path(ERA5_PARQUET_DIR).glob("*.parquet")):
        n = len(list(Path(ERA5_PARQUET_DIR).glob("*.parquet")))
        print(f"[ERA5] 기존 Parquet {n}개 → 스킵")
        return

    os.makedirs(ERA5_PARQUET_DIR, exist_ok=True)
    era5_files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_20*.csv")))
    if not era5_files:
        print(f"[ERA5] 경고: CSV 파일 없음 → 매칭 스킵 ({ERA5_DIR})")
        return

    print(f"[ERA5] {len(era5_files)}개 월별 CSV → 날짜별 Parquet 분할 중...")

    src_cols  = ["date", "lat", "lon"] + list(ERA5_COL_MAP.keys())
    rename_map = ERA5_COL_MAP  # src → era5_*

    for fpath in era5_files:
        if HAS_POLARS:
            df = pl.read_csv(fpath, columns=src_cols)
            # date: int(20200101) → str "20200101"
            df = df.with_columns(
                pl.col("date").cast(pl.Utf8)
                  .str.replace(r"\.0+$", "", literal=False)
                  .str.strip_chars()
            ).rename(rename_map)
            # 날짜별로 저장 (일평균이므로 obs_time 불필요)
            for date_val in df["date"].unique().to_list():
                sub = df.filter(pl.col("date") == date_val) \
                        .drop("date")
                out = os.path.join(ERA5_PARQUET_DIR, f"{date_val}.parquet")
                # 같은 날짜 Parquet이 이미 있으면 append
                if os.path.exists(out):
                    existing = pl.read_parquet(out)
                    pl.concat([existing, sub]).write_parquet(out, compression="zstd")
                else:
                    sub.write_parquet(out, compression="zstd")
            del df
        else:
            df = pd.read_csv(fpath, usecols=src_cols,
                              dtype={c: np.float32 for c in ERA5_COL_MAP})
            df["date"] = _normalize_date_str(df["date"])
            df.rename(columns=rename_map, inplace=True)
            for date_val, grp in df.groupby("date"):
                grp = grp.drop(columns=["date"])
                out = os.path.join(ERA5_PARQUET_DIR, f"{date_val}.parquet")
                if os.path.exists(out):
                    pd.concat([pd.read_parquet(out), grp], ignore_index=True) \
                      .to_parquet(out, engine="pyarrow", compression="zstd")
                else:
                    grp.to_parquet(out, engine="pyarrow", compression="zstd")
            del df
        gc.collect()

    n = len(list(Path(ERA5_PARQUET_DIR).glob("*.parquet")))
    print(f"[ERA5] 분할 완료: {n}개 날짜 Parquet")


# ─────────────────────────────────────────────
# 5. 단일 OCO 파일 처리 (병렬 워커)
# ─────────────────────────────────────────────
def _match_parquet_by_date(
    coords_df: pd.DataFrame,   # _date_str / lat_col / lon_col 경량 DF
    lat_col: str,
    lon_col: str,
    parquet_dir: str,
    value_cols: list[str],
) -> dict[str, np.ndarray]:
    """날짜별 Parquet에서 여러 변수를 동시에 KDTree 최근접 매칭.

    TROPOMI(1변수)와 ERA5(다변수) 모두 이 함수로 처리.
    workers=-1 로 멀티스레드 KNN 실행.
    반환: {컬럼명: np.ndarray(float32)} 딕셔너리
    """
    n = len(coords_df)
    results = {col: np.full(n, np.nan, dtype=np.float32) for col in value_cols}

    for date_val, idx in coords_df.groupby("_date_str").indices.items():
        pq = os.path.join(parquet_dir, f"{date_val}.parquet")
        if not os.path.exists(pq):
            continue
        grp = pl.read_parquet(pq).to_pandas() if HAS_POLARS \
              else pd.read_parquet(pq)

        # 해당 Parquet에 없는 컬럼은 스킵
        available = [c for c in value_cols if c in grp.columns]
        if not available:
            del grp
            continue

        tree = cKDTree(grp[["lat", "lon"]].values)
        _, ni = tree.query(coords_df.loc[idx, [lat_col, lon_col]].values,
                           k=1, workers=-1)
        for col in available:
            results[col][idx] = grp.iloc[ni][col].values.astype(np.float32)
        del grp
    return results


def process_one_oco_file(args: tuple) -> str:
    """단일 OCO 파일 처리 → 임시 Parquet 경로 반환."""
    oco_path, temp_dir = args
    fname       = os.path.basename(oco_path)
    out_parquet = os.path.join(temp_dir, fname.replace(".csv", ".parquet"))

    try:
        # ── 헤더만 먼저 읽어 usecols 결정 (전체 로드 전에 컬럼 필터) ──
        all_header = pd.read_csv(oco_path, nrows=0).columns.tolist()
        if KEEP_OCO_COLS is not None:
            # 시공간 필수 컬럼 + KEEP_OCO_COLS 교집합만 읽기
            needed = KEEP_OCO_COLS | {"date", "time", "latitude", "longitude",
                                      "lat", "lon", "tai_seconds"}
            usecols = [c for c in all_header if c in needed]
        else:
            usecols = None

        # ── 로드: 필요한 컬럼만 읽어 메모리 절약 ──
        df = pd.read_csv(oco_path, usecols=usecols, low_memory=False)
        df['file_source'] = fname.lower()  # 위성 출처 보존 (OCO-2/3 판별용)
        if KEEP_OCO_COLS is not None:
            KEEP_OCO_COLS.add('file_source')

        # float 'time' 컬럼 → 'tai_seconds'로 보존 (파싱 충돌 방지)
        if "time" in df.columns:
            df.rename(columns={"time": "tai_seconds"}, inplace=True)

        df = downcast_df(df)
        time_col, lat_col, lon_col = _resolve_cols(df)

        # date 문자열 파싱 (형식 자동 추론: "2020-01-01 03:46:43" 등 대응)
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

        # ── 불필요 컬럼 추가 제거 (usecols 이후 남은 것) ──
        if KEEP_OCO_COLS is not None:
            keep = (KEEP_OCO_COLS | {time_col, lat_col, lon_col, "tai_seconds"}
                    ) - {"time"}
            drop_cols = [c for c in df.columns if c not in keep]
            df.drop(columns=drop_cols, inplace=True, errors="ignore")

        # 순수 문자열 object 컬럼 제거 (NC float 변수 호환 불가)
        str_cols = [c for c in df.select_dtypes("object").columns
                    if c not in (time_col, "file_source")]
        df.drop(columns=str_cols, inplace=True, errors="ignore")

        # ── TROPOMI + ERA5 매칭: 경량 coords_df(3컬럼)만 분리 전달 ──
        coords_df = df[[lat_col, lon_col]].copy()
        coords_df["_date_str"] = df[time_col].dt.strftime("%Y%m%d")

        # TROPOMI NO2
        tropo = _match_parquet_by_date(
            coords_df, lat_col, lon_col,
            parquet_dir=TROPOMI_PARQUET_DIR,
            value_cols=["no2_tvcd_umol"],
        )
        df["tropomi_no2"] = tropo["no2_tvcd_umol"]

        # ERA5 기상 (U10, V10, BLH, wind_speed, wind_dir)
        era5_out_cols = list(ERA5_COL_MAP.values())  # era5_u10, era5_v10, ...
        era5 = _match_parquet_by_date(
            coords_df, lat_col, lon_col,
            parquet_dir=ERA5_PARQUET_DIR,
            value_cols=era5_out_cols,
        )
        for col, arr in era5.items():
            df[col] = arr

        del coords_df, tropo, era5
        gc.collect()

        # ── GPW 인구밀도 ──
        pts = list(zip(df[lon_col].values, df[lat_col].values))
        df["population_density"] = sample_tif(pts, GPW_TIF)
        del pts
        gc.collect()

        # ── ODIAC 배출량 ──
        df["_year"]  = df[time_col].dt.year.astype(np.int16)
        df["_month"] = df[time_col].dt.month.astype(np.int8)
        odiac = np.full(len(df), np.nan, dtype=np.float32)
        for (y, m), grp in df.groupby(["_year", "_month"]):
            tif = get_odiac_path(y, m)
            if os.path.exists(tif):
                pts = list(zip(grp[lon_col].values, grp[lat_col].values))
                odiac[grp.index] = sample_tif(pts, tif)
        df["odiac_emission"] = odiac
        df.drop(columns=["_year", "_month"], inplace=True, errors="ignore")
        del odiac
        gc.collect()

        # date 컬럼 → 'time'으로 통일
        if time_col != "time":
            df.rename(columns={time_col: "time"}, inplace=True)

        df.to_parquet(out_parquet, engine="pyarrow", compression="zstd")
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        print(f"  [완료] {fname}: {len(df):,}행 | {mem_mb:.1f}MB")
        return out_parquet

    except Exception as e:
        import traceback
        print(f"  [오류] {fname}: {e}\n{traceback.format_exc()}")
        return ""


# ─────────────────────────────────────────────
# 6. NC 저장 (메인 스레드, append 모드)
# ─────────────────────────────────────────────
def _init_netcdf(nc_path: str, col_union: list[str]) -> None:
    """컬럼 합집합 기반 NC 구조 초기화 (unlimited obs 차원)."""
    with nc4.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.createDimension("obs", None)
        ds.title       = "Integrated XCO2 and NO2 Proxy Dataset"
        ds.description = "OCO-2/3 targeting matched with TROPOMI NO2, GPW, ODIAC."

        tv = ds.createVariable("time", "f8", ("obs",), zlib=True, complevel=4)
        tv.units    = "seconds since 1970-01-01 00:00:00"
        tv.calendar = "proleptic_gregorian"

        for col in col_union:
            if col == "time":
                continue
            if col == "sounding_id":
                # 15자리 정수 ID → int64 보존 (궤도 단위 분할 핵심 키)
                ds.createVariable(col, "i8", ("obs",),
                                   zlib=True, complevel=4,
                                   fill_value=np.int64(-9999))
            elif col == "file_source":
                # 위성 파일명 보존용 문자열 변수 (NetCDF4 지원)
                ds.createVariable(col, str, ("obs",), zlib=True)
            else:
                ds.createVariable(col, "f4", ("obs",),
                                   zlib=True, complevel=4,
                                   fill_value=np.float32(9.969209968386869e+36))


def _append_batch(nc_path: str, batch: pd.DataFrame) -> None:
    with nc4.Dataset(nc_path, "a") as ds:
        start = ds.dimensions["obs"].size
        n     = len(batch)
        ds["time"][start:start + n] = batch["time"].astype("int64").values / 1e9
        for col in batch.columns:
            if col == "time" or col not in ds.variables:
                continue
            vals = batch[col].values
            # 숫자형이 아닌 경우 강제 변환
            if not np.issubdtype(vals.dtype, np.number):
                vals = pd.to_numeric(pd.Series(vals), errors="coerce").values
            # sounding_id: int64 보존 (float32 변환 시 15자리 정밀도 손실)
            if col == "sounding_id":
                ds[col][start:start + n] = vals.astype(np.int64)
            elif col == "file_source":
                ds[col][start:start + n] = vals.astype(object)
            else:
                ds[col][start:start + n] = vals.astype(np.float32)


def write_parquet_to_nc(parquet_path: str, col_union: list[str],
                         is_first: bool) -> None:
    """Parquet → NC_WRITE_BATCH 단위 append. 누락 컬럼은 NaN 패딩."""
    df = pd.read_parquet(parquet_path)
    for col in col_union:
        if col not in df.columns and col != "time":
            # file_source는 문자열 변수 → 빈 문자열로 패딩 (np.nan 금지)
            df[col] = "" if col == "file_source" else np.nan

    if is_first:
        _init_netcdf(OUT_NC, col_union)

    total = len(df)
    for s in range(0, total, NC_WRITE_BATCH):
        _append_batch(OUT_NC, df.iloc[s:s + NC_WRITE_BATCH])
        print(f"    NC: {min(s + NC_WRITE_BATCH, total):,}/{total:,}행 기록")
    del df
    gc.collect()


# ─────────────────────────────────────────────
# 7. 메인
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # WSL/Linux: fork가 기본값 (spawn은 출력 차단 문제 있음)
    multiprocessing.set_start_method("fork", force=True)

    # ── Step 1: TROPOMI + ERA5 사전 분할 ──
    preprocess_tropomi()
    preprocess_era5()

    # ── Step 2: OCO 파일 목록 + 컬럼 합집합 선계산 ──
    oco_files = sorted(
        glob.glob(os.path.join(BASE_DIR, "**", "oco*_east_asia_*.csv"),
                  recursive=True)
    )
    if not oco_files:
        raise FileNotFoundError(f"OCO CSV 없음: {BASE_DIR}")

    print(f"\n총 OCO 파일 수: {len(oco_files)}")
    print("컬럼 합집합 계산 중...")
    col_union = get_oco_column_union(oco_files)
    print(f"  → 총 {len(col_union)}개 컬럼: {col_union}")

    os.makedirs(TEMP_PARQUET_DIR, exist_ok=True)
    if os.path.exists(OUT_NC):
        os.remove(OUT_NC)
        print(f"기존 NC 삭제: {OUT_NC}")

    # ── Step 3: 병렬 OCO 처리 ──
    n_workers = N_WORKERS or os.cpu_count()
    print(f"\n병렬 처리 시작: {n_workers}개 워커")

    completed: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {
            exe.submit(process_one_oco_file, (p, TEMP_PARQUET_DIR)): p
            for p in oco_files
        }
        for f in as_completed(futures):
            oco_path = futures[f]
            result   = f.result()
            if result:
                completed[oco_path] = result

    # ── Step 4: NC 순차 append (원래 정렬 순서 유지) ──
    print("\n[NC 저장] 순서대로 append 중...")
    is_first = True
    for oco_path in oco_files:
        pq = completed.get(oco_path)
        if not pq or not os.path.exists(pq):
            print(f"  [스킵] {os.path.basename(oco_path)}")
            continue
        print(f"  → {os.path.basename(pq)}")
        write_parquet_to_nc(pq, col_union, is_first=is_first)
        is_first = False
        os.remove(pq)

    try:
        os.rmdir(TEMP_PARQUET_DIR)
    except OSError:
        pass

    print(f"\n✅ 완료! → {OUT_NC}")
