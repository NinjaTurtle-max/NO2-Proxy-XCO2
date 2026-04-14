import os
import numpy as np
import pandas as pd
import netCDF4 as nc4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# 1. 경로 및 설정
# ─────────────────────────────────────────────────────────────────
BASE_DIR = "/Volumes/100.118.65.89/dataset/XCO2연구 데이터"
NC_PATH  = os.path.join(BASE_DIR, "integrated_dataset.nc")
OUT_FIG  = "odiac_removed_points_map.png"

# preprocess_ml.py와 동일한 설정 (IQR 3.0)
IQR_FACTOR = 3.0

print(f"Loading data from {NC_PATH}...")

def load_data():
    with nc4.Dataset(NC_PATH, "r") as ds:
        # 필요한 변수만 로드 (메모리 절약)
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]
        time_raw = ds.variables["time"][:]
        
        if "odiac_emission" in ds.variables:
            odiac = ds.variables["odiac_emission"][:]
        else:
            v_list = ds.variables.keys()
            print(f"Available vars: {v_list}")
            raise KeyError("odiac_emission variable not found in NC file.")

        # masked array 처리
        if isinstance(odiac, np.ma.MaskedArray): odiac = odiac.filled(np.nan)
        if isinstance(lat, np.ma.MaskedArray): lat = lat.filled(np.nan)
        if isinstance(lon, np.ma.MaskedArray): lon = lon.filled(np.nan)

    df = pd.DataFrame({
        "latitude": lat,
        "longitude": lon,
        "odiac_emission": odiac,
        "time": pd.to_datetime(time_raw, unit='s', origin='1970-01-01')
    })
    df['month'] = df['time'].dt.month
    return df

# ─────────────────────────────────────────────────────────────────
# 2. 이상치 판별 (Removed rows only)
# ─────────────────────────────────────────────────────────────────
def get_removed_rows(df):
    print("Identifying outliers using Monthly IQR 3.0...")
    all_outliers = []
    
    for m in range(1, 13):
        sub = df[df['month'] == m].copy()
        vals = sub['odiac_emission'].dropna()
        if len(vals) == 0: continue
        
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        
        upper = q3 + IQR_FACTOR * iqr
        lower = q1 - IQR_FACTOR * iqr
        
        mask_removed = (sub['odiac_emission'] > upper) | (sub['odiac_emission'] < lower)
        outliers = sub[mask_removed]
        all_outliers.append(outliers)
        
        print(f"  Month {m:2d}: Outliers = {len(outliers):,} (Cut-off={upper:.4f})")
        
    return pd.concat(all_outliers)

# ─────────────────────────────────────────────────────────────────
# 3. 단순 시각화 (Cartopy 없이 실행 가능)
# ─────────────────────────────────────────────────────────────────
def plot_simple(removed_df):
    print(f"Generating map with {len(removed_df):,} points...")
    
    plt.figure(figsize=(10, 8), facecolor='white')
    
    # 삭제된 포인트 플로팅 (매우 작고 투명하게)
    plt.scatter(removed_df['longitude'], removed_df['latitude'], 
                s=0.05, c='red', alpha=0.3)
    
    # 주요 도시 표시
    cities = {
        "Seoul": (126.97, 37.56), "Beijing": (116.40, 39.90),
        "Shanghai": (121.47, 31.23), "Tokyo": (Tokyo_lon := 139.69, Tokyo_lat := 35.68)
    }
    for name, (lon, lat) in cities.items():
        plt.plot(lon, lat, 'kx', markersize=8)
        plt.text(lon+0.3, lat+0.3, name, fontsize=10, fontweight='bold')

    plt.title(f"Spatial Distribution of Removed ODIAC Outliers (N={len(removed_df):,})", fontsize=13)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(100, 150)
    plt.ylim(20, 50)
    
    plt.savefig(OUT_FIG, dpi=200, bbox_inches='tight')
    print(f"Map saved to {OUT_FIG}")

if __name__ == "__main__":
    try:
        data = load_data()
        removed = get_removed_rows(data)
        if len(removed) > 0:
            plot_simple(removed)
            print("\n✅ Verification complete. Check the generated png file.")
        else:
            print("No outliers detected.")
    except Exception as e:
        print(f"Error during execution: {e}")
