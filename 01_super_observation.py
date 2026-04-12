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
# 환경 및 경로 설정
# ─────────────────────────────────────────────────────────────────
BASE_DIR = "/mnt/e/dataset/XCO2연구 데이터"
PARQUET_IN = os.path.join(BASE_DIR, "ml_ready_dataset.parquet")
OUT_DIR = os.path.join(BASE_DIR, "anomaly_output")
PARQUET_OUT = os.path.join(OUT_DIR, "super_obs_dataset.parquet")
FIG_OUT = os.path.join(OUT_DIR, "n_map_super_ops.png")

os.makedirs(OUT_DIR, exist_ok=True)

# 격자 설정 (동아시아 0.1도)
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0
RESOLUTION = 0.1

lat_edges = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION)
lon_edges = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION)
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

# ─────────────────────────────────────────────────────────────────
# Vectorized Aggregation for Groupby
# ─────────────────────────────────────────────────────────────────
def perform_super_observation(df: pd.DataFrame) -> pd.DataFrame:
    print("  [Aggregating] Grouping data by Date and 0.1 degree grid...")
    
    # 1. 대상 변수 정의 (xco2_uncertainty는 별도 재산출하므로 제외)
    median_cols = ['xco2', 'tropomi_no2']
    mean_cols = ['era5_wind_speed', 'era5_blh', 'era5_u10', 'era5_v10', 'latitude', 'longitude']
    
    if 'population_density' in df.columns:
        mean_cols.append('population_density')
    if 'odiac_emission' in df.columns:
        mean_cols.append('odiac_emission')
        
    # Groupby 객체 생성
    grouped = df.groupby(['date', 'lat_idx', 'lon_idx'])
    
    # 그룹별 기본 통계량 연산 (초고속 벡터화)
    print("  [Aggregating] Computing Medians and Means...")
    agg_funcs = {col: 'median' for col in median_cols}
    agg_funcs.update({col: 'mean' for col in mean_cols})
    
    # 그룹별 관측치 갯수 (N)
    agg_funcs['xco2'] = ['median', 'count'] # xco2에 대해서는 카운트도 획득
    
    # Aggregation 실행
    out_df = grouped.agg(agg_funcs)
    
    # 컬럼 레벨 평탄화 (Flat columns)
    out_df.columns = [
        f"{col[0]}_median" if col[1] == 'median' and col[0] in median_cols else
        f"{col[0]}" if col[1] == 'median' and col[0] == 'xco2' else
        f"n_soundings" if col[1] == 'count' else
        col[0] for col in out_df.columns
    ]
    # xco2_median -> xco2, tropomi_no2_median -> tropomi_no2 원복
    rename_dict = {'xco2_median': 'xco2', 'tropomi_no2_median': 'tropomi_no2'}
    out_df.rename(columns=rename_dict, inplace=True)
    out_df.reset_index(inplace=True)
    
    # 2. Bootstrap 불확실성 연산 (N>=3 조건)
    print("  [Aggregating] Computing Bootstrap Uncertainties (This may take a few minutes)...")
    
    # xco2 배열들의 리스트 추출 (groupby.apply로 list를 반환받아 처리)
    # 속도를 위해 Groupby apply 대신 데이터 행을 iteration하는 방식에서, xco2 시리즈를 그룹별로 획득
    xco2_groups = grouped['xco2'].apply(list).reset_index(name='xco2_list')
    
    # N을 벡터 형태로 확인
    n_counts = out_df['n_soundings'].values
    
    uncertainties = np.zeros(len(out_df), dtype=np.float32)
    n_boots = 100
    
    # tqdm 기반 수동 Loop (pandas apply보다 numpy 레벨에서 훨씬 빠름)
    xco2_lists = xco2_groups['xco2_list'].values
    
    for i in tqdm(range(len(out_df)), desc="  [Bootstrap]"):
        n = n_counts[i]
        vals = np.array(xco2_lists[i], dtype=np.float32)
        if n >= 3:
            # Numpy Vectorized Bootstrap 
            boot_samples = np.random.choice(vals, size=(n_boots, n), replace=True)
            boot_medians = np.median(boot_samples, axis=1)
            uncertainties[i] = np.std(boot_medians)
        elif n == 2:
            uncertainties[i] = np.std(vals, ddof=1)
        else:
            # 단일값일 경우 자체 Uncertainty 보존 불가 시 0으로 세팅 혹은 보수적 접근
            uncertainties[i] = np.nan 
            
    out_df['xco2_bootstrap_std'] = uncertainties
    
    return out_df

# ─────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────
def run_super_observation():
    print("=" * 60)
    print("STEP 1: Data Loading & Spatial Indexing")
    print("=" * 60)
    
    # 읽어올 컬럼 목록 (필요 최소한 명시 + operation 모드 추가)
    cols = ['time', 'latitude', 'longitude', 'xco2', 'tropomi_no2',
            'era5_wind_speed', 'era5_blh', 'era5_u10', 'era5_v10',
            'xco2_quality_flag', 'ret_aod_total', 
            'snd_operation_mode', 'snd_land_water_indicator']
    
    # 존재 유무를 확인하며 추가 컬럼 로드 (populate variables)
    df = pd.read_parquet(PARQUET_IN)
    # 실제 존재하는 컬럼만 남기기
    load_cols = [c for c in cols if c in df.columns]
    if 'population_density' in df.columns: load_cols.append('population_density')
    if 'odiac_emission' in df.columns: load_cols.append('odiac_emission')
    if 'xco2_uncertainty' in df.columns: load_cols.append('xco2_uncertainty')
        
    df = df[load_cols].copy()
    print(f"  [Load] 원본 Parquet 로드 완료: {len(df):,} rows")
    
    df['date'] = df['time'].dt.date
    
    # 0.1도 격자 인덱스 부여
    df["lat_idx"] = np.searchsorted(lat_edges, df["latitude"].values, side="right") - 1
    df["lon_idx"] = np.searchsorted(lon_edges, df["longitude"].values, side="right") - 1
    
    valid_spatial = (
        (df["lat_idx"] >= 0) & (df["lat_idx"] < len(lat_centers)) &
        (df["lon_idx"] >= 0) & (df["lon_idx"] < len(lon_centers))
    )
    df = df[valid_spatial].copy()
    
    print("\n" + "=" * 60)
    print("STEP 2: QC Filtering & Operation Mode Analysis")
    print("=" * 60)
    
    # 1. 이중 필터 비교 실험
    mask_qf = (df['xco2_quality_flag'] == 0)
    n_qf = mask_qf.sum()
    
    mask_aod = mask_qf & (df['ret_aod_total'] <= 0.5)
    n_aod = mask_aod.sum()
    
    print(f"  [전체 데이터 보유량]: {len(df):,} rows")
    print(f"  [조건 A] (QF==0): {n_qf:,} rows 보유율 ({(n_qf/len(df)*100):.1f}%)")
    print(f"  [조건 B] (QF==0 & AOD<=0.5): {n_aod:,} rows 보유율 ({(n_aod/len(df)*100):.1f}%)")
    print(f"  → AOD 기반 강화 필터링으로 인한 추가 데이터 손실: {(1 - (n_aod / n_qf))*100 if n_qf > 0 else 0:.1f}%")
    
    # 강화된 B필터를 기본으로 모델링 수행
    df = df[mask_aod].copy()
    print("  ✅ [적용] 엄격한 광학 필터(조건 B)가 걸러낸 가장 신뢰도 높은 데이터를 집계에 사용합니다.")
    
    # 2. 운영 모드 (Operation Mode) 및 육상/해양 비중 검토
    if 'snd_operation_mode' in df.columns and 'snd_land_water_indicator' in df.columns:
        print("\n  [운영 모드 및 Land Fraction 점검]")
        # 0: Nadir, 1: Glint, 2: Target 등 
        # OCO-2 LWI: 0=Land, 1=Water, 2=Mixed 등 
        mode_counts = df.groupby(['snd_operation_mode', 'snd_land_water_indicator']).size().reset_index(name='count')
        print("  - Mode별 / 환경별 관측 건수 분포:")
        for _, row in mode_counts.iterrows():
            mode = row['snd_operation_mode']
            env = row['snd_land_water_indicator']
            cnt = row['count']
            
            # 모드 이름 매핑 (대략적인 OCO 기준, 실제 문자열일수도 있음)
            mode_name = "Nadir" if mode == 0 else "Glint" if mode == 1 else "Target/Other" if mode == 2 else str(mode)
            env_name = "Land" if env == 0 else "Ocean" if env == 1 else "Coast/Mixed" if env == 2 else str(env)
            
            # Glint + Land 에 대한 과도성 경고 로직
            warning = " ⚠️ [주의: 육상 픽셀 내 Glint 관측]" if (mode == 1 or mode == '1') and (env == 0 or env == '0') else ""
            print(f"    Mode: {mode_name:10s} | Env: {env_name:10s} | Count: {cnt:,} {warning}")
            
    print("\n" + "=" * 60)
    print("STEP 3: Super-observation Aggregation (1D Tabular)")
    print("=" * 60)
    
    # Aggregation 실행
    agg_df = perform_super_observation(df)
    
    print(f"  [Result] Super-observation 단위 병합 완료: 총 {len(agg_df):,} 개의 시공간 1D 행(Row) 산출")
    
    print("\n" + "=" * 60)
    print("STEP 4: Visualization & Memory-Efficient Export")
    print("=" * 60)
    
    # 1. n_map.png 생성
    print("  [Visualization] 관측 밀도(N_map) 렌더링 중...")
    n_map = np.zeros((len(lat_centers), len(lon_centers)))
    n_sum = agg_df.groupby(['lat_idx', 'lon_idx'])['n_soundings'].sum().reset_index()
    for _, row in n_sum.iterrows():
        li, lo = int(row['lat_idx']), int(row['lon_idx'])
        n_map[li, lo] = row['n_soundings']
        
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':', edgecolor='gray')
    ax.add_feature(cfeature.OCEAN, facecolor='#EAEAEA')
    
    n_map_plot = np.where(n_map > 0, n_map, np.nan)
    mesh = ax.pcolormesh(lon_edges, lat_edges, n_map_plot, transform=ccrs.PlateCarree(),
                         cmap="viridis", shading='flat')
                         
    cbar = plt.colorbar(mesh, ax=ax, aspect=40, shrink=0.8, pad=0.04)
    cbar.set_label("Total Soundings (N) Aggregated", fontsize=13)
    ax.set_title("Total Observation Density Map (1D Super-obs Strategy)", fontsize=14, fontweight='bold', pad=15)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Success] N-Map 저장 완료: {FIG_OUT}")
    
    # 2. 1D DataFrame 그대로 Parquet 고효율 저장
    # 불필요한 Index Drop 후 결측 처리 (xco2, no2가 모두 존재하는 행만)
    drop_subset = ['xco2']
    if 'tropomi_no2_median' in agg_df.columns: drop_subset.append('tropomi_no2_median')
    elif 'tropomi_no2' in agg_df.columns: drop_subset.append('tropomi_no2')
    
    if 'era5_wind_speed' in agg_df.columns: drop_subset.append('era5_wind_speed')
        
    agg_df = agg_df.dropna(subset=drop_subset)
    agg_df = agg_df.reset_index(drop=True)
    agg_df = agg_df.reset_index(drop=True)
    
    print("  [Export] Zarr 3D 공간을 회피하고 고밀도 1D Parquet 포맷으로 직렬화 중...")
    agg_df.to_parquet(PARQUET_OUT, index=False)
    
    mb_size = os.path.getsize(PARQUET_OUT) / (1024 * 1024)
    print(f"  [Success] 머신러닝 학습용 Tabular 저장 완료!")
    print(f"  📍 파일 경로: {PARQUET_OUT}")
    print(f"  📦 용량: {mb_size:.1f} MB (극초소형 고효율 포맷)")
    print(f"  📊 최종 Shape: {agg_df.shape}")

if __name__ == "__main__":
    run_super_observation()
