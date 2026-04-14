import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────
BASE_DIR         = "/mnt/e/dataset/XCO2연구 데이터"
PARQUET_IN       = os.path.join(BASE_DIR, "01_super_obs_output/super_obs_dataset.parquet")
PARQUET_OCO3_IN  = os.path.join(BASE_DIR, "01_super_obs_output/oco3_super_obs_dataset.parquet")
OUT_DIR          = os.path.join(BASE_DIR, "02_anomaly_standard_output")
os.makedirs(OUT_DIR, exist_ok=True)

ZARR_OUT         = os.path.join(OUT_DIR, "refined_xco2_anom.zarr")
FIG_OUT          = os.path.join(OUT_DIR, "anomaly_statistics.png")
PARQUET_OCO3_ANOM_OUT = os.path.join(OUT_DIR, "oco3_anom_1d.parquet")

# ─── 격자 설정 ───
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0
RESOLUTION       = 0.5
MIN_OBS_CLIM     = 12   # 고유 관측 월(Year-Month) 기준
MIN_OBS_YEAR     = 5   # 연간 최소 관측 일수(Days) 기준

lat_edges = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION)
lon_edges = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION)
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2


def load_and_prepare(parquet_path: str) -> pd.DataFrame:
    print("=" * 60)
    print("STEP 1: 데이터 로드 & 시계열 인덱스 추출")
    print("=" * 60)
    
    df = pd.read_parquet(parquet_path)
    print(f"  [Load] 1D Parquet Data: {len(df):,} 행")
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M')
    
    # 위도대(Lat Band) 추가 (10도 간격)
    df['lat_band'] = (df['latitude'] // 10) * 10
    df['lat_band_str'] = df['lat_band'].astype(int).astype(str) + "-" + (df['lat_band'] + 10).astype(int).astype(str) + "°N"
    
    return df


def compute_strict_climatology(df: pd.DataFrame) -> xr.DataArray:
    """공간 평활화 없이 엄격한 관측 누적 개월 수를 기준으로 Climatology 할당"""
    print("\n" + "=" * 60)
    print("STEP 2: Climatology C_clim(i,j,m) 산출 (No Smoothing)")
    print("=" * 60)
    
    # 격자별 총 고유 '연-월'(Year-Month) 관측 개수 산출
    grid_ym_counts = df.groupby(['lat_idx', 'lon_idx'])['year_month'].nunique()
    valid_grids = grid_ym_counts[grid_ym_counts >= MIN_OBS_CLIM].reset_index()
    
    print(f"  [임계값 검증] 5년 중 {MIN_OBS_CLIM}개월 이상 관측된 유효 격자: {len(valid_grids):,} 개")
    
    # 유효 격자 집합
    valid_grid_keys = set(zip(valid_grids['lat_idx'], valid_grids['lon_idx']))
    
    # 월별 합계 및 카운트
    monthly_agg = df.groupby(['lat_idx', 'lon_idx', 'month']).agg(
        xco2_mean=('xco2', 'mean')
    ).reset_index()
    
    clim = xr.DataArray(
        data=np.full((12, len(lat_centers), len(lon_centers)), np.nan, dtype=np.float32),
        dims=["month", "lat", "lon"],
        coords={"month": np.arange(1, 13), "lat": lat_centers, "lon": lon_centers},
        name="xco2_climatology",
    )
    
    assigned_count = 0
    for _, row in monthly_agg.iterrows():
        li, lo = int(row['lat_idx']), int(row['lon_idx'])
        if (li, lo) in valid_grid_keys:
            m_idx = int(row['month']) - 1
            clim.values[m_idx, li, lo] = row['xco2_mean']
            assigned_count += 1
            
    print(f"  [결과] Baseline 할당 완료 (유효 격자-월 조합): {assigned_count:,} 개")
    return clim, valid_grid_keys


def compute_strict_yearly_deviation(df: pd.DataFrame, valid_grid_keys: set) -> xr.DataArray:
    """공간 평활화 없이 연 단위 편차(잔차 트렌드) 추출"""
    print("\n" + "=" * 60)
    print("STEP 3: Yearly Deviation ΔC_year(i,j,y) 산출 (No Smoothing)")
    print("=" * 60)
    
    years = sorted(df['year'].unique())
    delta = xr.DataArray(
        data=np.full((len(years), len(lat_centers), len(lon_centers)), np.nan, dtype=np.float32),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat_centers, "lon": lon_centers},
        name="delta_c_year",
    )
    
    # 1. 대상 격자의 전체 기간 평균 (Baseline 편차의 0점)
    # 반드시 Climatology가 유효한 격자에 대해서만 산출
    valid_df = df[df.apply(lambda r: (r['lat_idx'], r['lon_idx']) in valid_grid_keys, axis=1)]
    c_all_series = valid_df.groupby(['lat_idx', 'lon_idx'])['xco2'].mean()
    c_all_dict = c_all_series.to_dict()
    
    # 2. 연간 평균 산출 (최소 측정 일수 조건 반영)
    yearly_agg = df.groupby(['lat_idx', 'lon_idx', 'year']).agg(
        xco2_mean=('xco2', 'mean'),
        obs_count=('xco2', 'count')
    ).reset_index()
    
    valid_year_agg = yearly_agg[
        (yearly_agg['obs_count'] >= MIN_OBS_YEAR) & 
        (yearly_agg.apply(lambda r: (r['lat_idx'], r['lon_idx']) in valid_grid_keys, axis=1))
    ]
    
    for _, row in valid_year_agg.iterrows():
        y_idx = years.index(int(row['year']))
        li, lo = int(row['lat_idx']), int(row['lon_idx'])
        c_all = c_all_dict.get((li, lo), np.nan)
        
        if np.isfinite(c_all):
            delta.values[y_idx, li, lo] = row['xco2_mean'] - c_all
            
    print(f"  [결과] ΔC_year 산출 완료. 평가 범위: [{np.nanmin(delta.values):.3f}, {np.nanmax(delta.values):.3f}] ppm")
    return delta


def calculate_anomaly_and_trend(df: pd.DataFrame, clim: xr.DataArray, delta: xr.DataArray) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 4: Anomaly 산출 및 횡단면 트렌드 검증")
    print("=" * 60)
    
    years = list(delta.coords["year"].values)
    clim_vals = clim.values
    delta_vals = delta.values
    
    month_idx = df["month"].values.astype(int) - 1
    lat_i = df["lat_idx"].values
    lon_i = df["lon_idx"].values
    
    year_to_idx = {y: i for i, y in enumerate(years)}
    year_idx = np.array([year_to_idx.get(int(y), -1) for y in df["year"].values])
    
    c_clim_arr = clim_vals[month_idx, lat_i, lon_i]
    
    valid_year = year_idx >= 0
    dc_arr = np.full(len(df), np.nan, dtype=np.float32)
    dc_arr[valid_year] = delta_vals[year_idx[valid_year], lat_i[valid_year], lon_i[valid_year]]
    
    df["xco2_baseline"] = c_clim_arr + dc_arr
    df["xco2_anomaly"] = df["xco2"].values - df["xco2_baseline"].values
    
    # 드랍 (기준에 미달된 격자들은 NaN)
    valid_df = df.dropna(subset=['xco2_anomaly'])
    print(f"  [결과] Baseline 미달 제외, 최종 유효 Anomaly 행: {len(valid_df):,} / {len(df):,}")
    
    # ── 잔류 트렌드 (Residual Trend) 위도대/연도별 표작성 ──
    print("\n  📊 [Residual Trend Analysis] 연도별 / 위도대별 Anomaly 잔차 평균 (ppm)")
    pivot_trend = pd.pivot_table(
        valid_df, 
        values='xco2_anomaly', 
        index='year', 
        columns='lat_band_str', 
        aggfunc='mean'
    ).round(3)
    
    print(pivot_trend.to_string())
    
    # 전체 0.5 ppm 확인
    total_mean = valid_df['xco2_anomaly'].mean()
    print(f"\n  전체 잔차 평균: {total_mean:.4f} ppm {'✅ 성공적 통제' if abs(total_mean) < 0.5 else '⚠️ 바이어스 유입 의심'}")
    
    return valid_df


def plot_qq_statistics(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 5: 통계 및 Extreme Tail 시각화 (Q-Q Plot)")
    print("=" * 60)
    
    anom = df["xco2_anomaly"].dropna().values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(anom, bins=200, density=True, alpha=0.6, color="#2CA02C", edgecolor="none", label="Raw Anomaly")
    mu, sigma = float(np.mean(anom)), float(np.std(anom))
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label=f"Normal Fit (μ={mu:.2f}, σ={sigma:.2f})")
    ax1.set_xlabel("XCO2 Anomaly (ppm)")
    ax1.set_title("Anomaly Distribution (Smoothing Removed)")
    ax1.legend()
    ax1.set_xlim(mu - 4*sigma, mu + 4*sigma)
    
    # Q-Q Plot
    ax2 = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(anom, dist="norm")
    ax2.scatter(osm, osr, s=1, alpha=0.3, color="#2CA02C")
    qq_line = slope * osm + intercept
    ax2.plot(osm, qq_line, "r-", lw=2, label=f"R² = {r**2:.4f}")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles (ppm)")
    ax2.set_title("Q-Q Plot (Identifying Heavy Tail)")
    ax2.legend()
    
    kurt = stats.kurtosis(anom, fisher=True)
    skew = stats.skew(anom)
    print(f"  [Stats] Skewness: {skew:.4f}")
    print(f"  [Stats] Kurtosis: {kurt:.4f} {'(Heavy-tail)' if kurt > 1 else '(Normal-like)'}")
    
    fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [저장] Q-Q 꼬리분포 시각화 리포트: {FIG_OUT}")


def export_to_zarr(df: pd.DataFrame, clim: xr.DataArray, delta: xr.DataArray):
    print("\n" + "=" * 60)
    print("STEP 6: 최종 Anomaly 큐브 Zarr Export")
    print("=" * 60)
    
    years = sorted(df["year"].unique())
    months = np.arange(1, 13)
    shape = (len(years), 12, len(lat_centers), len(lon_centers))
    
    anom_grid     = np.full(shape, np.nan, dtype=np.float32)
    obs_grid      = np.full(shape, np.nan, dtype=np.float32)
    baseline_grid = np.full(shape, np.nan, dtype=np.float32)
    
    # 일별 -> 월별 격자 집계로 압축 (Zarr의 보관/활용 목적)
    grp = df.groupby(["year", "month", "lat_idx", "lon_idx"]).agg(
        anom_mean=("xco2_anomaly", "mean"),
        obs_mean=("xco2", "mean"),
        baseline_mean=("xco2_baseline", "mean")
    ).reset_index()
    
    for _, row in grp.iterrows():
        yi = years.index(int(row["year"]))
        mi = int(row["month"]) - 1
        li, lo = int(row["lat_idx"]), int(row["lon_idx"])
        
        anom_grid[yi, mi, li, lo]     = row["anom_mean"]
        obs_grid[yi, mi, li, lo]      = row["obs_mean"]
        baseline_grid[yi, mi, li, lo] = row["baseline_mean"]
        
    ds = xr.Dataset(
        {
            "xco2_anomaly":  (["year", "month", "lat", "lon"], anom_grid),
            "xco2_obs":      (["year", "month", "lat", "lon"], obs_grid),
            "xco2_baseline": (["year", "month", "lat", "lon"], baseline_grid),
            "climatology":   clim,
            "delta_c_year":  delta,
        },
        coords={
            "year":  years,
            "month": months,
            "lat":   lat_centers,
            "lon":   lon_centers,
        },
        attrs={
            "title": "Refined XCO2 Anomaly (No Smoothing, Strict Threshold)",
            "min_obs_climatology_months": MIN_OBS_CLIM,
            "min_obs_yearly_days": MIN_OBS_YEAR,
        },
    )
    
    import shutil
    if os.path.exists(ZARR_OUT):
        shutil.rmtree(ZARR_OUT)
        
    ds.to_zarr(ZARR_OUT, mode="w")
    print(f"  [성공] Zarr Export 완료: {ZARR_OUT} (Size: {ds.nbytes / 1e6:.1f} MB)")


def apply_oco2_climatology_to_oco3(clim: xr.DataArray, delta: xr.DataArray) -> None:
    """OCO-2 기반 climatology/delta를 OCO-3 super-obs에 적용하여 anomaly 산출.

    OCO-3는 독립 검증 세트로 사용되므로, OCO-2 파이프라인이 학습한
    climatology를 그대로 적용합니다 (cross-satellite validation).
    """
    print("\n" + "=" * 60)
    print("OCO-3 독립 검증: OCO-2 Climatology 적용")
    print("=" * 60)

    if not os.path.exists(PARQUET_OCO3_IN):
        print(f"  ⚠️ OCO-3 super-obs 파일 없음 ({PARQUET_OCO3_IN}) — 건너뜀")
        return

    df3 = pd.read_parquet(PARQUET_OCO3_IN)
    print(f"  [Load] OCO-3 Super-obs: {len(df3):,} 행")

    df3['date'] = pd.to_datetime(df3['date'])
    df3['year'] = df3['date'].dt.year
    df3['month'] = df3['date'].dt.month
    df3['year_month'] = df3['date'].dt.to_period('M')
    df3['lat_band'] = (df3['latitude'] // 10) * 10
    df3['lat_band_str'] = (df3['lat_band'].astype(int).astype(str) + "-" +
                           (df3['lat_band'] + 10).astype(int).astype(str) + "°N")

    # OCO-2 climatology/delta 적용 (동일 격자+월 매핑)
    years = list(delta.coords["year"].values)
    year_to_idx = {int(y): i for i, y in enumerate(years)}
    clim_vals  = clim.values
    delta_vals = delta.values

    lat_i    = df3["lat_idx"].values
    lon_i    = df3["lon_idx"].values
    month_idx = df3["month"].values.astype(int) - 1
    year_idx  = np.array([year_to_idx.get(int(y), -1) for y in df3["year"].values])

    c_clim_arr = clim_vals[month_idx, lat_i, lon_i]

    dc_arr = np.full(len(df3), np.nan, dtype=np.float32)
    valid_year = year_idx >= 0
    dc_arr[valid_year] = delta_vals[year_idx[valid_year], lat_i[valid_year], lon_i[valid_year]]

    df3["xco2_baseline"] = c_clim_arr + dc_arr
    df3["xco2_anomaly"]  = df3["xco2"].values - df3["xco2_baseline"].values

    df3_valid = df3.dropna(subset=["xco2_anomaly"])
    print(f"  [결과] OCO-3 유효 Anomaly: {len(df3_valid):,} / {len(df3):,} 행")

    if len(df3_valid) > 0:
        anom = df3_valid["xco2_anomaly"].values
        from scipy import stats as _stats
        kurt = _stats.kurtosis(anom, fisher=True)
        print(f"  OCO-3 Anomaly 분포: μ={anom.mean():.4f}, σ={anom.std():.4f}, Kurt={kurt:.3f}")

    save_cols = [c for c in [
        'date', 'lat_idx', 'lon_idx', 'xco2', 'xco2_anomaly', 'xco2_baseline',
        'tropomi_no2', 'year', 'month', 'year_month',
        'latitude', 'longitude', 'lat_band', 'lat_band_str',
        'era5_wind_speed', 'era5_blh', 'era5_u10', 'era5_v10',
        'population_density', 'odiac_emission', 'n_soundings'
    ] if c in df3_valid.columns]
    df3_valid[save_cols].to_parquet(PARQUET_OCO3_ANOM_OUT, index=False)
    print(f"  [저장] OCO-3 Anomaly 1D Parquet: {PARQUET_OCO3_ANOM_OUT}")
    print(f"  Shape: {df3_valid[save_cols].shape}")


if __name__ == "__main__":
    df_raw = load_and_prepare(PARQUET_IN)
    clim, valid_keys = compute_strict_climatology(df_raw)
    delta = compute_strict_yearly_deviation(df_raw, valid_keys)
    df_anom = calculate_anomaly_and_trend(df_raw, clim, delta)
    plot_qq_statistics(df_anom)

    # 최종 결과물은 검증된 Anomaly에 한정하여 Zarr화
    export_to_zarr(df_anom, clim, delta)

    # 03 스크립트용 1D Anomaly Parquet 저장
    anom_1d_path = os.path.join(OUT_DIR, "anom_1d.parquet")
    save_cols = [c for c in [
        'date', 'lat_idx', 'lon_idx', 'xco2', 'xco2_anomaly', 'xco2_baseline',
        'tropomi_no2', 'year', 'month', 'year_month',
        'latitude', 'longitude', 'lat_band', 'lat_band_str',
        'era5_wind_speed', 'era5_blh', 'era5_u10', 'era5_v10',
        'population_density', 'odiac_emission', 'n_soundings'
    ] if c in df_anom.columns]
    df_anom[save_cols].to_parquet(anom_1d_path, index=False)
    print(f"\n  [저장] Anomaly 1D Parquet (03 스크립트 입력용): {anom_1d_path}")
    print(f"  Shape: {df_anom[save_cols].shape}")

    # OCO-3 독립 검증 세트 생성 (OCO-2 climatology 기반)
    apply_oco2_climatology_to_oco3(clim, delta)

    print("\n✅ 모든 Anomaly 고도화 산출 및 평가 완료!")
