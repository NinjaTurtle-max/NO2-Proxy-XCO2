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
OUT_FIG  = "no2_removed_points_map.png"

# 로그상에 찍힌 1.5를 기준으로 삭제 위치 추적
IQR_FACTOR = 1.5

print(f"Loading data from {NC_PATH} for TROPOMI NO2 diagnostic...")

def load_data():
    with nc4.Dataset(NC_PATH, "r") as ds:
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]
        time_raw = ds.variables["time"][:]
        
        # TROPOMI NO2 컬럼명 확인 (no2_tvcd_umol -> tropomi_no2로 로드됨)
        if "no2_tvcd_umol" in ds.variables:
            no2 = ds.variables["no2_tvcd_umol"][:]
        elif "tropomi_no2" in ds.variables:
            no2 = ds.variables["tropomi_no2"][:]
        else:
            v_list = ds.variables.keys()
            print(f"Available vars: {v_list}")
            raise KeyError("TROPOMI NO2 variable not found.")

        # masked array 처리
        if isinstance(no2, np.ma.MaskedArray): no2 = no2.filled(np.nan)
        if isinstance(lat, np.ma.MaskedArray): lat = lat.filled(np.nan)
        if isinstance(lon, np.ma.MaskedArray): lon = lon.filled(np.nan)

    df = pd.DataFrame({
        "latitude": lat,
        "longitude": lon,
        "no2": no2,
        "time": pd.to_datetime(time_raw, unit='s', origin='1970-01-01')
    })
    df['month'] = df['time'].dt.month
    return df

def get_removed_rows(df):
    print(f"Identifying NO2 outliers using Monthly IQR {IQR_FACTOR}...")
    all_outliers = []
    
    for m in range(1, 13):
        sub = df[df['month'] == m].copy()
        vals = sub['no2'].dropna()
        if len(vals) == 0: continue
        
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        
        upper = q3 + IQR_FACTOR * iqr
        lower = q1 - IQR_FACTOR * iqr
        
        mask_removed = (sub['no2'] > upper) | (sub['no2'] < lower)
        outliers = sub[mask_removed]
        all_outliers.append(outliers)
        
        print(f"  Month {m:2d}: Outliers = {len(outliers):,} (Upper Boundary={upper:.4f})")
        
    return pd.concat(all_outliers)

def plot_simple(removed_df):
    print(f"Generating NO2 outlier map with {len(removed_df):,} points...")
    
    plt.figure(figsize=(10, 8), facecolor='white')
    
    # 삭제된 포인트 (파란색 계열로 표시하여 ODIAC과 차별화)
    plt.scatter(removed_df['longitude'], removed_df['latitude'], 
                s=0.05, c='blue', alpha=0.3)
    
    cities = {
        "Seoul": (126.97, 37.56), "Beijing": (116.40, 39.90),
        "Shanghai": (121.47, 31.23), "Tokyo": (139.69, 35.68)
    }
    for name, (lon, lat) in cities.items():
        plt.plot(lon, lat, 'rx', markersize=8)
        plt.text(lon+0.3, lat+0.3, name, fontsize=10, fontweight='bold')

    plt.title(f"Spatial Distribution of Removed TROPOMI NO2 Outliers (N={len(removed_df):,})", fontsize=13)
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
            print("\n✅ NO2 Outlier Diagnostic Map generated.")
        else:
            print("No NO2 outliers detected.")
    except Exception as e:
        print(f"Error: {e}")
