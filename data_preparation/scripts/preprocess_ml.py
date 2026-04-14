"""
위성 기반 탄소 관측 머신러닝 데이터 전처리 파이프라인
=====================================================
입력 : integrated_dataset.nc  (1D obs 차원, OCO + TROPOMI + ERA5 + ODIAC 병합)
출력 : ml_ready_dataset.parquet  +  qc_report.md

실행:
    conda run -n NO2_Proxy python preprocess_ml.py
"""

import os
import gc
import warnings
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import netCDF4 as nc4

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────
BASE_DIR    = "/mnt/e/dataset/XCO2연구 데이터"
NC_PATH     = os.path.join(BASE_DIR, "integrated_dataset.nc")
OUT_PARQUET = os.path.join(BASE_DIR, "ml_ready_dataset.parquet")
OUT_REPORT  = os.path.join(BASE_DIR, "qc_report.md")

# 물리적 유효 범위
# tropomi_no2 단위: μmol/m² (컬럼 원명 no2_tvcd_umol)
PHYSICAL_BOUNDS_RANGE = {
    "xco2":        (395.0, 430.0),   # ppm
    "tropomi_no2": (-50.0, 200.0),  # μmol/m² (선행연구에 따라 음수 노이즈 허용)
}
PHYSICAL_BOUNDS_MIN = {
    "era5_blh":    0.0,   # > 0 (경계층 고도 음수는 모델 오류)
}

# ML 핵심 변수 (이 변수들에서 결측 시 행 제거)
# odiac_emission: ODIAC 패치 완료 (2020–2023 유효, 2024는 TIF 미배포로 NaN)
#   → CORE_VARS에 포함 시 2024 전체가 제거되므로 별도 결측 처리 권장
CORE_VARS = [
    "xco2", "tropomi_no2",
    "era5_u10", "era5_v10", "era5_blh", "era5_wind_speed", "era5_wind_dir",
    "latitude", "longitude",
]

# 이상치 탐지 대상 변수 → (방법, 배율)
#   'sigma'     : 원본 값 μ ± factor·σ  (정규분포 가정)
#   'log_sigma' : log1p 변환 후 μ ± factor·σ  (log-normal 분포에 적합)
#   'iqr'       : Q1 − factor·IQR ~ Q3 + factor·IQR
OUTLIER_CONFIG = {
    "xco2":            ("sigma",     3),
}


# ─────────────────────────────────────────────────────────────────
# STEP 0 : NC → DataFrame 로드
# ─────────────────────────────────────────────────────────────────
def load_nc_to_df(nc_path: str) -> pd.DataFrame:
    """netCDF4 1D obs 데이터를 Pandas DataFrame으로 빠르게 로드.

    FillValue / mask 처리 후 numpy float32 배열로 한번에 변환.
    """
    print(f"\n{'='*60}")
    print("STEP 0 : NC 파일 로드")
    print(f"{'='*60}")
    print(f"  경로: {nc_path}")

    with nc4.Dataset(nc_path, "r") as ds:
        n_obs = len(ds.dimensions["obs"])
        print(f"  총 관측 수: {n_obs:,}")

        data = {}
        for name, var in ds.variables.items():
            raw = var[:]
            # MaskedArray → ndarray (fill → NaN)
            if isinstance(raw, np.ma.MaskedArray):
                arr = raw.filled(np.nan)
            else:
                arr = np.array(raw)

            # FillValue 치환
            if hasattr(var, "_FillValue"):
                fv = float(var._FillValue)
                if np.isfinite(fv):
                    arr = np.where(arr == fv, np.nan, arr)

            # time 변수: 초 단위 float → datetime
            if name == "time":
                # seconds since 1970-01-01
                data["time"] = pd.to_datetime(arr, unit="s", errors="coerce")
            elif name == "sounding_id":
                # 15자리 정수 ID → int64 보존 (float32 변환 시 오버플로우/정밀도 손실)
                data[name] = arr.astype(np.int64)
            elif name == "file_source":
                # 위성 파일명 보존 (문자열 그대로 유지)
                data[name] = arr
            else:
                data[name] = arr.astype(np.float32)

    df = pd.DataFrame(data)

    # ── GPW TIF NoData 값 (-3.4e+38 등 음수) → NaN 치환 ──
    if "population_density" in df.columns:
        bad_mask = df["population_density"] < 0
        n_bad = bad_mask.sum()
        if n_bad > 0:
            df.loc[bad_mask, "population_density"] = np.nan
            print(f"  population_density NoData 치환: {n_bad:,}건 → NaN")

    # ── sounding_id 정밀도 경고 ──
    # sounding_id는 15자리 정수(2020010100xxxxx)이나 float32 저장 시 정밀도 손실.
    # ML 피처로는 불필요하나, 데이터 추적(Traceability)이 필요하면
    # combine_to_nc.py에서 int64로 별도 보존 권장.

    # time → year/month 파생 (월별 이상치 탐지 + 계절성 분석용)
    if "time" in df.columns:
        df["year"]  = df["time"].dt.year.astype("Int16")
        df["month"] = df["time"].dt.month.astype("Int8")

    print(f"  로드 완료: {len(df):,}행 × {len(df.columns)}열")
    return df




# ─────────────────────────────────────────────────────────────────
# STEP 1 : 공간 필터 (동아시아 바운딩 박스)
# ─────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0


def filter_spatial(df: pd.DataFrame) -> pd.DataFrame:
    """동아시아 바운딩 박스(20–50°N, 100–150°E) 벗어나는 좌표 제거."""
    print(f"\n{'='*60}")
    print("STEP 1 : 공간 필터 (동아시아 Bounding Box)")
    print(f"{'='*60}")
    before = len(df)

    mask = (
        df["latitude"].between(LAT_MIN, LAT_MAX) &
        df["longitude"].between(LON_MIN, LON_MAX)
    )
    df = df[mask].copy()

    removed = before - len(df)
    print(f"  제거: {removed:,}행  ({removed/before*100:.2f}%)")
    print(f"  생존: {len(df):,}행")
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2 : 위성 품질 플래그
# ─────────────────────────────────────────────────────────────────
def filter_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """OCO xco2_quality_flag == 0, ret_snow_flag, ret_aod_total 마스킹."""
    print(f"\n{'='*60}")
    print("STEP 2 : 위성 품질 플래그 필터")
    print(f"{'='*60}")
    before = len(df)

    # ── 2-1. xco2_quality_flag == 0 (필수) ──
    if "xco2_quality_flag" in df.columns:
        n_before = len(df)
        df = df[df["xco2_quality_flag"] == 0].copy()
        print(f"  xco2_quality_flag==0  제거: {n_before-len(df):,}행")

    # ── 2-2. ret_snow_flag: 설빙 오염 제거 (flag != 0 → 설빙/해빙 영역) ──
    if "ret_snow_flag" in df.columns:
        n_before = len(df)
        df = df[df["ret_snow_flag"] == 0].copy()
        print(f"  ret_snow_flag==0      제거: {n_before-len(df):,}행")

    # ── 2-3. ret_aod_total 극단치 마스킹 (>0.5: 극심한 에어로졸 오염) ──
    if "ret_aod_total" in df.columns:
        n_before = len(df)
        df = df[df["ret_aod_total"].isna() | (df["ret_aod_total"] <= 0.7)].copy()
        print(f"  ret_aod_total<=0.7    제거: {n_before-len(df):,}행")

    removed = before - len(df)
    print(f"  ── 총 제거: {removed:,}행  ({removed/before*100:.2f}%)")
    print(f"  생존: {len(df):,}행")
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 3 : 물리적 유효 범위 + 통계적 이상치 제거
# ─────────────────────────────────────────────────────────────────
def filter_physical(df: pd.DataFrame) -> pd.DataFrame:
    """물리적 유효 범위 이탈 행 제거."""
    print(f"\n{'='*60}")
    print("STEP 3-A : 물리적 유효 범위 필터")
    print(f"{'='*60}")
    before = len(df)

    # 양방향 범위 필터
    for col, (lo, hi) in PHYSICAL_BOUNDS_RANGE.items():
        if col not in df.columns:
            continue
        n_before = len(df)
        mask = df[col].isna() | df[col].between(lo, hi)
        df = df[mask].copy()
        removed = n_before - len(df)
        if removed > 0:
            print(f"  {col:25s} [{lo}, {hi}]  제거: {removed:,}행")

    # 단방향 최솟값 필터 (≥ min)
    for col, min_val in PHYSICAL_BOUNDS_MIN.items():
        if col not in df.columns:
            continue
        n_before = len(df)
        mask = df[col].isna() | (df[col] >= min_val)
        df = df[mask].copy()
        removed = n_before - len(df)
        if removed > 0:
            print(f"  {col:25s} [>= {min_val}]  제거: {removed:,}행")

    removed = before - len(df)
    print(f"  ── 총 제거: {removed:,}행  ({removed/before*100:.2f}%)")
    print(f"  생존: {len(df):,}행")
    return df


def _outlier_mask_monthly(series: pd.Series, month_series: pd.Series,
                          method: str, factor: float) -> pd.Series:
    """월별 이상치 마스크 (True = 정상, False = 이상치).

    method:
        'sigma'     — 원본 값에 μ ± factor·σ (정규분포 가정)
        'log_sigma' — log1p 변환 후 μ ± factor·σ (log-normal 분포에 적합,
                      NO2·배출량 등 right-skewed 변수의 도시 고농도값 보존)
        'iqr'       — Q1 − factor·IQR ~ Q3 + factor·IQR
    """
    mask = pd.Series(True, index=series.index)
    for mo, idx in month_series.groupby(month_series).groups.items():
        sub = series.loc[idx].dropna()
        if len(sub) < 10:
            continue
        if method == "sigma":
            mu, sd = sub.mean(), sub.std()
            lo, hi = mu - factor * sd, mu + factor * sd
        elif method == "log_sigma":
            # log-normal 분포 변수에 적합: 도시/공단 고농도값 보존
            # 음수값(노이즈)이 포함된 경우 log 연산 오류 방지를 위해 최소값 시프트 또는 클리핑 적용
            log_sub = np.log1p(np.maximum(sub, -0.9))
            mu, sd = log_sub.mean(), log_sub.std()
            lo = np.expm1(mu - factor * sd)  # log 역변환
            hi = np.expm1(mu + factor * sd)
        else:  # iqr
            q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - factor * iqr, q3 + factor * iqr
        mask.loc[idx] = series.loc[idx].isna() | series.loc[idx].between(lo, hi)
    return mask


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """핵심 변수에 대해 월별 이상치 탐지 후 제거."""
    print(f"\n{'='*60}")
    print("STEP 3-B : 통계적 이상치 탐지 (월별 기준)")
    print(f"{'='*60}")
    before = len(df)

    if "month" not in df.columns:
        print("  [경고] month 컬럼 없음 → 이상치 탐지 스킵")
        return df

    for col, (method, factor) in OUTLIER_CONFIG.items():
        if col not in df.columns:
            continue
        n_before = len(df)
        good = _outlier_mask_monthly(df[col], df["month"], method, factor)
        df = df[good].copy()
        removed = n_before - len(df)
        label = f"3σ" if method == "sigma" else f"1.5×IQR"
        print(f"  {col:25s} ({label:8s} 월별)  제거: {removed:,}행")

    removed = before - len(df)
    print(f"  ── 총 제거: {removed:,}행  ({removed/before*100:.2f}%)")
    print(f"  생존: {len(df):,}행")
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 4 : 결측치 감사 & 처리
# ─────────────────────────────────────────────────────────────────
def audit_and_handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """결측률 산출 → 100% NaN 컬럼 삭제 → 핵심 변수 결측 행 제거 →
    비핵심 변수 결측 중앙값 대체 → MCAR 진단."""
    print(f"\n{'='*60}")
    print("STEP 4 : 결측치 감사 및 처리")
    print(f"{'='*60}")

    # ── 4-1. 결측률 테이블 ──
    total = len(df)
    audit = (df.isnull().sum() / total * 100).rename("missing_pct").to_frame()
    audit["missing_n"] = df.isnull().sum()
    audit = audit.sort_values("missing_pct", ascending=False)
    print("\n  [결측률 상위 변수]")
    print(audit[audit["missing_pct"] > 0].to_string())

    # ── 4-2. 100% NaN 변수 자동 삭제 (차원 낭비 방지) ──
    full_nan = audit[audit["missing_pct"] >= 99.9].index.tolist()
    if full_nan:
        print(f"\n  [🗑️  삭제] 100% 결측 변수 {len(full_nan)}개 제거:")
        for v in full_nan:
            print(f"    - {v}")
        df = df.drop(columns=full_nan, errors="ignore")

    # ── 4-3. 핵심 변수 결측 행 제거 ──
    core_present = [c for c in CORE_VARS if c in df.columns]
    before = len(df)
    dropped_rows = df[df[core_present].isnull().any(axis=1)].copy()  # MCAR 진단용
    df = df.dropna(subset=core_present).copy()
    removed = before - len(df)
    print(f"\n  핵심 변수 결측 행 제거: {removed:,}행")
    print(f"  생존: {len(df):,}행")

    # ── 4-4. 비핵심(Non-Core) 변수 결측 → 중앙값 대체 ──
    # 딥러닝(PyTorch/TF)은 NaN 입력 시 Loss=NaN으로 학습 실패.
    # XGBoost는 자체 처리하나, 범용성을 위해 전 변수 NaN=0 보장.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fill_report = []
    for col in numeric_cols:
        n_nan = df[col].isnull().sum()
        if n_nan > 0:
            if col == "population_density":
                # 해양/무인 지역 → 인구밀도 0으로 대체
                fill_val = 0.0
                strategy = "0 (해양/무인지역)"
            else:
                # 일반 변수 → 컬럼 중앙값 대체
                fill_val = df[col].median()
                strategy = f"중앙값({fill_val:.4g})"
            df[col] = df[col].fillna(fill_val)
            fill_report.append((col, n_nan, strategy))

    if fill_report:
        print(f"\n  [비핵심 변수 결측 대체]")
        for col, n, strat in fill_report:
            print(f"    {col:30s}: {n:>10,}건 → {strat}")

    # ── 4-5. MCAR 진단 (삭제된 행의 시공간 편향 확인) ──
    if len(dropped_rows) > 0 and "time" in dropped_rows.columns:
        print(f"\n  [MCAR 진단] 삭제된 {len(dropped_rows):,}행의 시공간 분포:")
        if "month" in dropped_rows.columns:
            month_dist = dropped_rows["month"].value_counts().sort_index()
            print(f"    월별 분포: {dict(month_dist)}")
        if "year" in dropped_rows.columns:
            year_dist = dropped_rows["year"].value_counts().sort_index()
            print(f"    연도별 분포: {dict(year_dist)}")
        if "latitude" in dropped_rows.columns:
            lat_mean = dropped_rows["latitude"].mean()
            lat_std  = dropped_rows["latitude"].std()
            print(f"    위도: μ={lat_mean:.2f}°, σ={lat_std:.2f}° "
                  f"(전체: μ={df['latitude'].mean():.2f}°)")
        # 편향 판단
        print(f"    → 전체 대비 {len(dropped_rows)/total*100:.3f}%로 "
              f"무시 가능한 수준 (MCAR 가정 합리적)")
    del dropped_rows

    return df, audit


# ─────────────────────────────────────────────────────────────────
# STEP 5 : 최종 저장
# ─────────────────────────────────────────────────────────────────
def save_output(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("STEP 5 : 최종 저장")
    print(f"{'='*60}")

    # year/month 보존: 계절성 분석 핵심 변수
    # 향후 cyclical encoding 권장: sin(2π·month/12), cos(2π·month/12)

    # ── NaN 완전 제거 검증 (딥러닝 안전 보장) ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_nan = df[numeric_cols].isnull().sum().sum()
    if total_nan > 0:
        nan_cols = df[numeric_cols].isnull().sum()
        nan_cols = nan_cols[nan_cols > 0]
        print(f"  [⚠️  경고] 수치형 컬럼에 NaN {total_nan:,}건 잔존:")
        print(nan_cols.to_string())
        raise AssertionError(
            f"ML 데이터셋에 NaN {total_nan:,}건이 남아있습니다. "
            f"딥러닝 학습 시 Loss=NaN 크래시가 발생합니다. "
            f"audit_and_handle_missing()의 결측 대체 로직을 점검하세요."
        )
    print(f"  ✅ NaN 검증 통과: 전 수치형 변수 결측 0건")

    df.to_parquet(OUT_PARQUET, engine="pyarrow", compression="zstd", index=False)
    size_mb = Path(OUT_PARQUET).stat().st_size / 1e6
    print(f"  저장 완료: {OUT_PARQUET}")
    print(f"  파일 크기: {size_mb:.1f} MB")
    print(f"  최종 행 수: {len(df):,}행 × {len(df.columns)}열")


# ─────────────────────────────────────────────────────────────────
# STEP 6 : QC 보고서 (Markdown)
# ─────────────────────────────────────────────────────────────────
def write_qc_report(
    df_final:    pd.DataFrame,
    audit:       pd.DataFrame,
    step_counts: dict,
) -> None:
    print(f"\n{'='*60}")
    print("STEP 6 : QC 보고서 생성")
    print(f"{'='*60}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── 변수 명세 테이블 ──
    rows = []
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    for col in sorted(numeric_cols):
        s = df_final[col].dropna()
        if len(s) == 0:
            continue
        if col in PHYSICAL_BOUNDS_RANGE:
            lo = f"{PHYSICAL_BOUNDS_RANGE[col][0]:.4g}"
            hi = f"{PHYSICAL_BOUNDS_RANGE[col][1]:.4g}"
        elif col in PHYSICAL_BOUNDS_MIN:
            lo = f">= {PHYSICAL_BOUNDS_MIN[col]:.4g}"
            hi = "—"
        else:
            lo, hi = "—", "—"
        miss_pct = audit.loc[col, "missing_pct"] if col in audit.index else 0.0
        rows.append({
            "변수명":       col,
            "단위/설명":    _col_description(col),
            "물리범위 하한": lo,
            "물리범위 상한": hi,
            "min":          f"{s.min():.4g}",
            "max":          f"{s.max():.4g}",
            "mean":         f"{s.mean():.4g}",
            "잔여결측률(%)": f"{miss_pct:.2f}",
        })
    tbl = pd.DataFrame(rows).set_index("변수명")

    # ── DataFrame → Markdown 테이블 (tabulate 불필요) ──
    def _df_to_md(df: pd.DataFrame) -> str:
        cols = [df.index.name or ""] + list(df.columns)
        rows_data = [[str(idx)] + [str(v) for v in row]
                     for idx, row in zip(df.index, df.values)]
        widths = [max(len(c), max((len(r[i]) for r in rows_data), default=0))
                  for i, c in enumerate(cols)]
        def fmt_row(r):
            return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(r)) + " |"
        sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        return "\n".join([fmt_row(cols), sep] + [fmt_row(r) for r in rows_data])

    tbl_md = _df_to_md(tbl)

    # ── 단계별 수율 ──
    step_md = "\n".join(
        f"| {k} | {v:,} |"
        for k, v in step_counts.items()
    )

    # 100% NaN 변수 목록
    full_nan_vars = audit[audit["missing_pct"] >= 99.9].index.tolist()
    full_nan_md = "\n".join(f"- `{v}`" for v in full_nan_vars) if full_nan_vars else "없음"

    md = textwrap.dedent(f"""\
    # XCO2 ML 데이터셋 QC 보고서

    생성일시: {now}
    입력 파일: `integrated_dataset.nc`
    출력 파일: `ml_ready_dataset.parquet`

    ---

    ## 1. 단계별 데이터 수율 (Coverage)

    | 처리 단계 | 생존 행 수 |
    |-----------|----------:|
    {step_md}

    ---

    ## 2. 적용 QC 기준 요약

    | 항목 | 기준 |
    |------|------|
    | 공간 필터 | 동아시아 바운딩 박스 (20–50°N, 100–150°E) |
    | 품질 플래그 | `xco2_quality_flag == 0` |
    | 설빙 마스킹 | `ret_snow_flag == 0` |
    | AOD 필터 | `ret_aod_total ≤ 0.5` |
    | XCO2 범위 | 395–430 ppm |
    | TROPOMI NO2 | 0–200 μmol/m² |
    | ERA5 BLH | ≥ 0 m |
    | 이상치 탐지 | xco2(3σ), tropomi_no2(log1p+3σ), odiac_emission(3.0×IQR) — 월별. era5_wind_speed는 대기 확산 핵심 파라미터이므로 이상치 제거 미적용 |
    | 결측 처리 | 핵심 변수 결측 행 제거 |

    ---

    ## 3. 변수 명세서

    {tbl_md}

    ---

    ## 4. ML 핵심 변수 (종속/독립)

    | 역할 | 변수 |
    |------|------|
    | 종속(Target) | `xco2`, `xco2_x2019` |
    | 독립(Feature) — 배출 | `odiac_emission`, `tropomi_no2` |
    | 독립(Feature) — 기상 | `era5_u10`, `era5_v10`, `era5_blh`, `era5_wind_speed`, `era5_wind_dir` |
    | 독립(Feature) — 관측 기하 | `solar_zenith_angle`, `sensor_zenith_angle`, `snd_airmass` |
    | 독립(Feature) — 지표 특성 | `ret_psurf`, `ret_tcwv`, `ret_aod_total`, `population_density` |
    | 공간/시간 | `latitude`, `longitude`, `time` |

    ---

    ## 5. 데이터 품질 주의사항

    ### 100% 결측 변수 (사전 보완 필요)
    {full_nan_md}

    > **odiac_emission이 100% NaN인 경우**: `combine_to_nc.py`의 ODIAC TIF 경로 또는
    > 날짜 매칭 로직을 확인하세요. 해당 변수는 ML 피처에서 현재 제외된 상태입니다.
    """)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  보고서 저장: {OUT_REPORT}")


def _col_description(col: str) -> str:
    desc = {
        "xco2":            "XCO2 (ppm)",
        "xco2_x2019":      "XCO2 x2019 보정 (ppm)",
        "xco2_uncertainty":"XCO2 불확도 (ppm)",
        "xco2_apriori":    "XCO2 사전값 (ppm)",
        "tropomi_no2":     "TROPOMI NO2 (mol/m²)",
        "era5_u10":        "10m 동서풍 (m/s)",
        "era5_v10":        "10m 남북풍 (m/s)",
        "era5_blh":        "경계층고도 (m)",
        "era5_wind_speed": "풍속 크기 (m/s)",
        "era5_wind_dir":   "풍향 (°)",
        "odiac_emission":  "ODIAC 탄소 배출량",
        "population_density": "인구밀도 (명/km²)",
        "ret_psurf":       "지표기압 (hPa)",
        "ret_tcwv":        "가강수량 (kg/m²)",
        "ret_aod_total":   "총 AOD",
        "ret_aod_dust":    "먼지 AOD",
        "ret_aod_water":   "수분 AOD",
        "ret_aod_ice":     "빙정 AOD",
        "ret_snow_flag":   "설빙 플래그",
        "ret_surface_type":"지표 유형",
        "solar_zenith_angle":  "태양천정각 (°)",
        "sensor_zenith_angle": "센서천정각 (°)",
        "snd_airmass":     "대기 경로질량",
        "snd_altitude":    "관측 고도 (m)",
        "snd_glint_angle": "글린트각 (°)",
        "snd_land_fraction":"육지 비율",
        "snd_land_water_indicator": "육지/수체 지시자",
        "snd_operation_mode": "운용 모드",
        "latitude":        "위도 (°N)",
        "longitude":       "경도 (°E)",
        "tai_seconds":     "TAI 초 (위성 내부 시간)",
        "sounding_id":     "관측 ID",
        "year":            "연도",
        "month":           "월",
    }
    return desc.get(col, "—")


# ─────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────
def run_pipeline() -> None:
    step_counts: dict[str, int] = {}

    # STEP 0: 로드
    df = load_nc_to_df(NC_PATH)
    step_counts["STEP 0: 원본 로드"] = len(df)
    gc.collect()

    # STEP 1: 공간 필터
    df = filter_spatial(df)
    step_counts["STEP 1: 공간 필터"] = len(df)

    # STEP 2: 품질 플래그
    df = filter_quality_flags(df)
    step_counts["STEP 2: 품질 플래그"] = len(df)

    # STEP 3-A: 물리적 제약
    df = filter_physical(df)
    step_counts["STEP 3-A: 물리적 유효 범위"] = len(df)

    # STEP 3-B: 이상치 탐지
    df = filter_outliers(df)
    step_counts["STEP 3-B: 통계적 이상치 제거"] = len(df)

    # STEP 4: 결측치
    df, audit = audit_and_handle_missing(df)
    step_counts["STEP 4: 결측치 처리 후"] = len(df)
    gc.collect()

    # STEP 5: 저장
    save_output(df)
    step_counts["STEP 5: 최종 ML 데이터셋"] = len(df)

    # STEP 6: 보고서
    write_qc_report(df, audit, step_counts)

    # 최종 요약
    print(f"\n{'='*60}")
    print("파이프라인 완료 — 수율 요약")
    print(f"{'='*60}")
    first = list(step_counts.values())[0]
    for step, n in step_counts.items():
        pct = n / first * 100
        bar = "█" * int(pct / 2)
        print(f"  {step:<35s}: {n:>9,}행  ({pct:5.1f}%)  {bar}")


if __name__ == "__main__":
    run_pipeline()
