import os
import pandas as pd
import numpy as np
import scipy.stats
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ─────────────────────────────────────────────────────────────────
# 환경 및 경로 설정 (0.25도 고해상도용)
# ─────────────────────────────────────────────────────────────────
BASE_DIR = "/Volumes/100.118.65.89/dataset/XCO2연구 데이터"
PARQUET_IN = os.path.join(BASE_DIR, "ml_ready_dataset.parquet")
OUT_DIR = os.path.join(BASE_DIR, "01_super_obs_output_025")
PARQUET_OUT     = os.path.join(OUT_DIR, "super_obs_dataset.parquet")
PARQUET_OCO3_OUT = os.path.join(OUT_DIR, "oco3_super_obs_dataset.parquet")
FIG_OUT = os.path.join(OUT_DIR, "n_map_super_ops_025.png")

os.makedirs(OUT_DIR, exist_ok=True)

# 격자 설정 (동아시아 0.25도 고해상도)
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0
RESOLUTION = 0.25

lat_edges = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION)
lon_edges = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION)
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

# ─────────────────────────────────────────────────────────────────
# Vectorized Aggregation for Groupby
# ─────────────────────────────────────────────────────────────────
def perform_super_observation(df: pd.DataFrame) -> pd.DataFrame:
    print(f"  [Aggregating] Grouping data by Date and {RESOLUTION} degree grid...")
    
    # 1. 대상 변수 정의
    median_cols = ['xco2', 'tropomi_no2']
    mean_cols = ['era5_wind_speed', 'era5_blh', 'era5_u10', 'era5_v10', 'latitude', 'longitude']
    
    if 'population_density' in df.columns:
        mean_cols.append('population_density')
    if 'odiac_emission' in df.columns:
        mean_cols.append('odiac_emission')
        
    # Groupby 객체 생성
    grouped = df.groupby(['date', 'lat_idx', 'lon_idx'])
    
    # 그룹별 기본 통계량 연산
    print("  [Aggregating] Computing Medians and Means...")
    agg_funcs = {col: 'median' for col in median_cols}
    agg_funcs.update({col: 'mean' for col in mean_cols})
    agg_funcs['xco2'] = ['median', 'count']
    
    out_df = grouped.agg(agg_funcs)
    
    # 컬럼 레벨 평탄화
    out_df.columns = [
        f"{col[0]}_median" if col[1] == 'median' and col[0] in median_cols else
        f"{col[0]}" if col[1] == 'median' and col[0] == 'xco2' else
        f"n_soundings" if col[1] == 'count' else
        col[0] for col in out_df.columns
    ]
    rename_dict = {'xco2_median': 'xco2', 'tropomi_no2_median': 'tropomi_no2'}
    out_df.rename(columns=rename_dict, inplace=True)
    out_df.reset_index(inplace=True)
    
    # 2. Bootstrap 불확실성 연산
    print("  [Aggregating] Computing Bootstrap Uncertainties...")
    from scipy.stats import bootstrap

    def safe_boot_std(vals):
        vals = np.asarray(vals, dtype=np.float32)
        n = len(vals)
        if n >= 3:
            try:
                res = bootstrap((vals,), np.median, n_resamples=100, 
                                random_state=42, method='basic')
                return res.standard_error
            except Exception:
                return np.std(vals, ddof=1) / np.sqrt(n)
        elif n == 2:
            return np.std(vals, ddof=1)
        else:
            return np.nan

    boot_std = grouped['xco2'].apply(safe_boot_std).reset_index(name='xco2_bootstrap_std')
    out_df = out_df.merge(boot_std, on=['date', 'lat_idx', 'lon_idx'], how='left')
    
    return out_df

def detect_satellite(df: pd.DataFrame) -> pd.DataFrame:
    satellite_id = pd.Series(-1, index=df.index, dtype=np.int8)
    if 'satellite' in df.columns:
        sat_str = df['satellite'].astype(str).str.lower()
        is_oco3 = sat_str.str.contains(r'oco.?3', regex=True, na=False)
        is_oco2 = sat_str.str.contains(r'oco.?2', regex=True, na=False)
        satellite_id[is_oco3] = 1
        satellite_id[is_oco2 & ~is_oco3] = 0
    elif 'file_source' in df.columns:
        src_str = df['file_source'].astype(str).str.lower()
        is_oco3 = src_str.str.contains('oco3', na=False)
        is_oco2 = src_str.str.contains('oco2', na=False)
        satellite_id[is_oco3] = 1
        satellite_id[is_oco2 & ~is_oco3] = 0
    df['satellite_id'] = satellite_id
    return df

def run_super_observation():
    print("=" * 60)
    print(f"STEP 1: Data Loading & Spatial Indexing (Res: {RESOLUTION})")
    print("=" * 60)
    
    df = pd.read_parquet(PARQUET_IN)
    df['date'] = df['time'].dt.date
    
    # 격자 인덱스 부여
    df["lat_idx"] = np.searchsorted(lat_edges, df["latitude"].values, side="right") - 1
    df["lon_idx"] = np.searchsorted(lon_edges, df["longitude"].values, side="right") - 1
    
    valid_spatial = (
        (df["lat_idx"] >= 0) & (df["lat_idx"] < len(lat_centers)) &
        (df["lon_idx"] >= 0) & (df["lon_idx"] < len(lon_centers))
    )
    df = df[valid_spatial].copy()
    
    # QF 및 AOD 필터
    AOD_THRESHOLD = 0.7
    mask_aod = (df['xco2_quality_flag'] == 0) & (df['ret_aod_total'] <= AOD_THRESHOLD)
    df = df[mask_aod].copy()

    # 위성 분리
    df = detect_satellite(df)
    df_oco3 = df[df['satellite_id'] == 1].drop(columns=['satellite_id']).copy()
    df_oco2 = df[df['satellite_id'] == 0].drop(columns=['satellite_id']).copy()

    # OCO-2 Aggregation
    print("\n" + "=" * 60)
    print("STEP 2: OCO-2 Super-observation Aggregation (1D)")
    print("=" * 60)
    agg_df = perform_super_observation(df_oco2)
    agg_df.to_parquet(PARQUET_OUT, index=False)
    
    # OCO-3 Aggregation
    if len(df_oco3) > 0:
        print("\n" + "=" * 60)
        print("STEP 3: OCO-3 Super-observation Aggregation (Validation)")
        print("=" * 60)
        agg_oco3 = perform_super_observation(df_oco3)
        agg_oco3.to_parquet(PARQUET_OCO3_OUT, index=False)

    print("\n✅ Super-observation (0.25 deg) 완료")

if __name__ == "__main__":
    run_super_observation()
