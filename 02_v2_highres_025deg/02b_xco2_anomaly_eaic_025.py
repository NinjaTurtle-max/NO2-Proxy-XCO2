import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────
# 경로 (0.25도 고해상도용)
# ──────────────────────────────────────────────────────────────
BASE_DIR   = "/Volumes/100.118.65.89/dataset/XCO2연구 데이터"
PARQUET_IN = os.path.join(BASE_DIR, "01_super_obs_output_025/super_obs_dataset.parquet")
PARQUET_OCO3_IN = os.path.join(BASE_DIR, "01_super_obs_output_025/oco3_super_obs_dataset.parquet")
OUT_DIR    = os.path.join(BASE_DIR, "02_anomaly_eaic_output_025")
os.makedirs(OUT_DIR, exist_ok=True)

PARQUET_OUT = os.path.join(OUT_DIR, "anom_1d_eaic.parquet")
ZARR_OUT   = os.path.join(OUT_DIR, "refined_xco2_anom_eaic.zarr")
FIG_OUT    = os.path.join(OUT_DIR, "anomaly_statistics_eaic_025.png")
PARQUET_OCO3_ANOM_OUT = os.path.join(OUT_DIR, "oco3_anom_1d_eaic.parquet")

# EAIC Sub-region 정의
EAIC_REGIONS = {
    "NCP": {"lat": (34.0, 41.0), "lon": (113.0, 122.0)},
    "YRD": {"lat": (28.5, 33.0), "lon": (118.0, 123.0)},
    "PRD": {"lat": (21.5, 24.5), "lon": (112.0, 115.5)},
    "KCR": {"lat": (35.0, 38.5), "lon": (125.0, 129.0)},
    "JKT": {"lat": (34.5, 37.0), "lon": (138.5, 141.0)},
}

RESOLUTION = 0.25
lat_edges = np.arange(20.0, 50.0 + RESOLUTION, RESOLUTION)
lon_edges = np.arange(100.0, 150.0 + RESOLUTION, RESOLUTION)
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

def assign_eaic_region(df):
    df['eaic_region'] = 'OUT'
    for code, spec in EAIC_REGIONS.items():
        mask = (df['latitude'] >= spec['lat'][0]) & (df['latitude'] < spec['lat'][1]) & \
               (df['longitude'] >= spec['lon'][0]) & (df['longitude'] < spec['lon'][1])
        df.loc[mask, 'eaic_region'] = code
    return df

def calculate_daily_zonal_anomaly(df, label="Train/Test"):
    print(f"\nSTEP: 일일 위도대 청정 픽셀 배경농도 산출 ({label})")
    
    def get_clean_median(group):
        if 'odiac_emission' not in group.columns or group['odiac_emission'].nunique() <= 1:
            return group['xco2'].median()
        clean_thresh = group['odiac_emission'].quantile(0.10)
        clean_pixels = group[group['odiac_emission'] <= clean_thresh]
        return clean_pixels['xco2'].median() if len(clean_pixels) >= 5 else group['xco2'].median()

    daily_bg = df.groupby(['date', 'lat_idx']).apply(get_clean_median, include_groups=False).reset_index(name='xco2_background')
    df = df.merge(daily_bg, on=['date', 'lat_idx'], how='left')
    df['xco2_anomaly'] = df['xco2'] - df['xco2_background']
    return df.dropna(subset=['xco2_anomaly']).reset_index(drop=True)

if __name__ == "__main__":
    print(f"Loading 0.25 deg super-obs: {PARQUET_IN}")
    df = pd.read_parquet(PARQUET_IN)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df = assign_eaic_region(df)
    df = df[df['eaic_region'] != 'OUT'].copy()
    
    df_anom = calculate_daily_zonal_anomaly(df, label="OCO-2 0.25deg")
    df_anom.to_parquet(PARQUET_OUT, index=False)
    
    if os.path.exists(PARQUET_OCO3_IN):
        df3 = pd.read_parquet(PARQUET_OCO3_IN)
        df3['date'] = pd.to_datetime(df3['date'])
        df3 = assign_eaic_region(df3)
        df3 = df3[df3['eaic_region'] != 'OUT'].copy()
        df3_anom = calculate_daily_zonal_anomaly(df3, label="OCO-3 0.25deg")
        df3_anom.to_parquet(PARQUET_OCO3_ANOM_OUT, index=False)

    print("\n✅ EAIC Anomaly (0.25 deg) 추출 완료")
