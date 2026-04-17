import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
import joblib

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────
# 경로 및 상수 (0.25도 고해상도용)
# ─────────────────────────────────────────────────────────────────
BASE_DIR        = "/Volumes/100.118.65.89/dataset/XCO2연구 데이터"
PARQUET_IN      = os.path.join(BASE_DIR, "02_anomaly_eaic_output_025/anom_1d_eaic.parquet")
OUT_DIR         = os.path.join(BASE_DIR, "03_split_output_025")
os.makedirs(OUT_DIR, exist_ok=True)

FIG1_PATH         = os.path.join(OUT_DIR, "Figure_1_Final_025.png")
SPLIT_PATH        = os.path.join(OUT_DIR, "split_indices_v2_025.json")
SCALER_PATH       = os.path.join(OUT_DIR, "scalers_v2_025.joblib")
PARQUET_BALANCED_OUT = os.path.join(OUT_DIR, "anom_1d_balanced_025.parquet")

RESOLUTION = 0.25
lat_edges = np.arange(20.0, 50.0 + RESOLUTION, RESOLUTION)
lon_edges = np.arange(100.0, 150.0 + RESOLUTION, RESOLUTION)
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

# 상관 분석 최소 관측수 (0.25도에서는 데이터 밀도가 낮아지므로 5로 하향 조정 고려 가능하나 일단 10 유지)
MIN_OBS_CORR = 10

def load_data():
    df = pd.read_parquet(PARQUET_IN)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    
    # 0.25도 인덱스 갱신
    df["lat_idx"] = np.searchsorted(lat_edges, df["latitude"].values, side="right") - 1
    df["lon_idx"] = np.searchsorted(lon_edges, df["longitude"].values, side="right") - 1
    
    mask = (df["lat_idx"] >= 0) & (df["lat_idx"] < len(lat_centers)) & \
           (df["lon_idx"] >= 0) & (df["lon_idx"] < len(lon_centers))
    return df[mask].copy()

def compute_spatial_correlation(df):
    grouped = df.groupby(["lat_idx", "lon_idx"])
    r_map = np.full((len(lat_centers), len(lon_centers)), np.nan)
    n_map = np.zeros((len(lat_centers), len(lon_centers)))
    
    for (li, lo), grp in grouped:
        n = len(grp)
        n_map[li, lo] = n
        if n >= MIN_OBS_CORR:
            r, _ = pearsonr(grp["xco2_anomaly"], grp["tropomi_no2"])
            r_map[li, lo] = r
    return r_map, n_map

if __name__ == "__main__":
    print(f"Loading 0.25 deg anomaly: {PARQUET_IN}")
    df = load_data()
    
    # 1. Stratified Split (단순 연도 기반)
    all_years = sorted(df["year"].unique())
    df["split"] = "train"
    if len(all_years) >= 3:
        df.loc[df["year"] == all_years[-1], "split"] = "test"
        df.loc[df["year"] == all_years[-2], "split"] = "val"
    
    # split_indices_v2_025.json 저장
    split_info = {
        "train_indices": df[df["split"] == "train"].index.tolist(),
        "test_indices":  df[df["split"] == "test"].index.tolist(),
        "val_indices":   df[df["split"] == "val"].index.tolist(),
    }
    with open(SPLIT_PATH, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved split indices: {SPLIT_PATH}")

    # 2. Feature Scalling (StandardScaler - Exp A)
    from sklearn.preprocessing import StandardScaler
    feature_cols = ["tropomi_no2", "era5_wind_speed", "era5_blh", "population_density", "odiac_emission"]
    X_train = df.loc[df["split"] == "train", feature_cols].values
    scaler = StandardScaler().fit(X_train)
    scaler_bundle = {"scaler_exp_a": scaler, "feature_columns": feature_cols}
    joblib.dump(scaler_bundle, SCALER_PATH)
    print(f"  Saved scaler bundle: {SCALER_PATH}")
    
    # 상관관계 분석
    r_map, n_map = compute_spatial_correlation(df)
    
    # 결과 요약
    valid_r = r_map[np.isfinite(r_map)]
    print(f"\n[Statistics 0.25 deg]")
    if len(valid_r) > 0:
        print(f"  Overall r (Mean) = {np.mean(valid_r):.4f}")
        print(f"  Max r = {np.max(valid_r):.4f}")
        print(f"  Grids with |r| >= 0.3: {np.sum(np.abs(valid_r) >= 0.3)}")
    else:
        print("  No grids with sufficient data for correlation.")
        
    # 저장
    df.to_parquet(PARQUET_BALANCED_OUT, index=False)
    print(f"  Saved balanced dataset: {PARQUET_BALANCED_OUT}")
    print("\n✅ Correlation & Split (0.25 deg) 완료")
