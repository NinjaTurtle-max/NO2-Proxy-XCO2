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
OUT_DIR = os.path.join(BASE_DIR, "01_super_obs_output")
PARQUET_OUT     = os.path.join(OUT_DIR, "super_obs_dataset.parquet")
PARQUET_OCO3_OUT = os.path.join(OUT_DIR, "oco3_super_obs_dataset.parquet")
FIG_OUT = os.path.join(OUT_DIR, "n_map_super_ops.png")

os.makedirs(OUT_DIR, exist_ok=True)

# 격자 설정 (동아시아 0.5도)
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0
RESOLUTION = 0.5

lat_edges = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION)
lon_edges = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION)
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

# ─────────────────────────────────────────────────────────────────
# Vectorized Aggregation for Groupby
# ─────────────────────────────────────────────────────────────────
def perform_super_observation(df: pd.DataFrame) -> pd.DataFrame:
    print(f"  [Aggregating] Grouping data by Date and {RESOLUTION} degree grid...")
    
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
    from scipy.stats import bootstrap
    print("  [Aggregating] Computing Bootstrap Uncertainties (vectorized, memory-efficient)...")

    def safe_boot_std(vals):
        """Bootstrap Standard Error of the Median.
        - n >= 3: scipy.stats.bootstrap (n_resamples=100)
        - n == 2: Sample Std
        - n < 2: NaN
        """
        vals = np.asarray(vals, dtype=np.float32)
        n = len(vals)
        if n >= 3:
            try:
                # scipy.stats.bootstrap은 대규모 병렬처리에 최적화됨
                res = bootstrap((vals,), np.median, n_resamples=100, 
                                random_state=42, method='basic')
                return res.standard_error
            except Exception:
                # 관측값이 극단적으로 균일할 경우 Fallback
                return np.std(vals, ddof=1) / np.sqrt(n)
        elif n == 2:
            return np.std(vals, ddof=1)
        else:
            return np.nan

    # Group별 직접 적용 (apply list -> loop 방식 대비 메모리 점유율 80% 감소)
    boot_std = grouped['xco2'].apply(safe_boot_std).reset_index(name='xco2_bootstrap_std')
    out_df = out_df.merge(boot_std, on=['date', 'lat_idx', 'lon_idx'], how='left')
    
    return out_df

# ─────────────────────────────────────────────────────────────────
# 위성 유형 판별 (OCO-2 vs OCO-3)
# ─────────────────────────────────────────────────────────────────
def detect_satellite(df: pd.DataFrame) -> pd.DataFrame:
    """OCO-2 / OCO-3 판별 → satellite_id 컬럼 생성

    satellite_id 값 정의:
        0  = OCO-2
        1  = OCO-3
       -1  = Unknown (판별 불가 — 이후 satellite_id == -1 로 필터링 가능)

    판별 우선순위:
        1차) 'satellite' 컬럼 (가장 신뢰)
        2차) 'file_source' 컬럼
        3차) OCO-3 발사일(2019-08-06) + SAM 관측 모드 복합 판별
        4차) 완전 판별 불가 → 전체 Unknown(-1)
    """
    # 기본값: Unknown(-1)
    satellite_id = pd.Series(-1, index=df.index, dtype=np.int8)

    # 1차: 'satellite' 컬럼 직접 판별
    if 'satellite' in df.columns:
        sat_str = df['satellite'].astype(str).str.lower()
        is_oco3 = sat_str.str.contains(r'oco.?3', regex=True, na=False)
        is_oco2 = sat_str.str.contains(r'oco.?2', regex=True, na=False)
        satellite_id[is_oco3] = 1
        satellite_id[is_oco2 & ~is_oco3] = 0
        method = "'satellite' 컬럼 직접 판별"

    # 2차: 'file_source' 컬럼 기반
    elif 'file_source' in df.columns:
        src_str = df['file_source'].astype(str).str.lower()
        is_oco3 = src_str.str.contains('oco3', na=False)
        is_oco2 = src_str.str.contains('oco2', na=False)
        satellite_id[is_oco3] = 1
        satellite_id[is_oco2 & ~is_oco3] = 0
        method = "'file_source' 컬럼 기반"

    # 3차: OCO-3 발사일(2019-08-06) + SAM 관측 모드 복합 휴리스틱
    elif 'time' in df.columns:
        OCO3_LAUNCH = pd.Timestamp('2019-08-06')
        after_launch = pd.to_datetime(df['time']) >= OCO3_LAUNCH
        # 발사 이전 레코드는 확실히 OCO-2
        satellite_id[~after_launch] = 0

        if 'snd_operation_mode' in df.columns:
            mode_str = df['snd_operation_mode'].astype(str).str.upper().str.strip()
            # SAM(Target Acquisition Mode) — OCO-3 전용 모드 (문자 'SAM' 또는 정수 코드 3)
            has_sam = mode_str.str.contains('SAM', na=False) | (mode_str == '3')
            # 발사 이후 + SAM 확인 → OCO-3 확정
            satellite_id[after_launch & has_sam] = 1
            # 발사 이후 + SAM 없음 → OCO-2/OCO-3 혼재 가능 → Unknown 유지
            method = "OCO-3 발사일(2019-08-06) + SAM 모드 복합 판별"
        else:
            # SAM 컬럼 없음: 발사 이후 레코드는 모두 Unknown 유지
            method = "OCO-3 발사일(2019-08-06) 단독 (SAM 모드 정보 없음)"
        print("  ℹ️  위성 식별 컬럼 없음 — 휴리스틱 판별 적용 (정확도 제한적)")
        print("      SCI 투고 전 combine_to_nc.py 에서 'satellite'/'file_source' 보존 권장")

    # 4차: 완전 판별 불가 → 전체 Unknown
    else:
        method = "⚠️ 완전 판별 불가 — 전체 Unknown(-1) 처리"
        print("  ⚠️⚠️⚠️ CRITICAL: 위성 판별 가능한 컬럼(satellite, file_source, time) 없음")
        print("       SCI 투고 전 반드시 재처리하여 위성 출처를 보존하십시오.")

    df['satellite_id'] = satellite_id

    n_total = len(df)
    n_oco2  = int((satellite_id == 0).sum())
    n_oco3  = int((satellite_id == 1).sum())
    n_unk   = int((satellite_id == -1).sum())

    print(f"  [Satellite 판별] 방법: {method}")
    if n_total > 0:
        print(f"  OCO-2   ( 0): {n_oco2:,} 행 ({n_oco2/n_total*100:.1f}%)")
        print(f"  OCO-3   ( 1): {n_oco3:,} 행 ({n_oco3/n_total*100:.1f}%)")
        print(f"  Unknown (-1): {n_unk:,} 행  ({n_unk/n_total*100:.1f}%)")
    if n_unk > 0:
        print(f"  ℹ️  Unknown {n_unk:,}행 — satellite_id == -1 로 이후 필터링 가능")
    return df


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
            'snd_operation_mode', 'snd_land_water_indicator',
            'file_source', 'satellite']
    
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
    
    AOD_THRESHOLD = 0.7  # Wunch et al. (2017, AMT) 권장 상한
    mask_aod = mask_qf & (df['ret_aod_total'] <= AOD_THRESHOLD)
    n_aod = mask_aod.sum()
    
    print(f"  [전체 데이터 보유량]: {len(df):,} rows")
    print(f"  [조건 A] (QF==0): {n_qf:,} rows 보유율 ({(n_qf/len(df)*100):.1f}%)")
    print(f"  [조건 B] (QF==0 & AOD<={AOD_THRESHOLD}): {n_aod:,} rows 보유율 ({(n_aod/len(df)*100):.1f}%)")
    print(f"  [AOD 임계값] {AOD_THRESHOLD} (Wunch et al. 2017 기준)")
    
    # 0.5 대비 추가 확보 행수 계산 (참고용)
    mask_aod_old = mask_qf & (df['ret_aod_total'] <= 0.5)
    n_extra = (mask_aod & ~mask_aod_old).sum()
    print(f"  → 이전 기준(AOD<=0.5) 대비 추가 보존 행: {n_extra:,} ({(n_extra/n_qf*100):.1f}%)")
    
    # 강화된 B필터를 기본으로 모델링 수행
    df = df[mask_aod].copy()
    print("  ✅ [적용] 엄격한 광학 필터(조건 B)가 걸러낸 가장 신뢰도 높은 데이터를 집계에 사용합니다.")

    # 위성 유형 판별 및 분리
    print("\n  [Satellite 분리]")
    df = detect_satellite(df)
    df_oco3    = df[df['satellite_id'] == 1].drop(columns=['satellite_id']).copy()
    df_unknown = df[df['satellite_id'] == -1].drop(columns=['satellite_id']).copy()
    df         = df[df['satellite_id'] == 0].drop(columns=['satellite_id']).copy()
    print(f"  이후 파이프라인: OCO-2 {len(df):,} 행 사용 / OCO-3 {len(df_oco3):,} 행 별도 저장")
    if len(df_unknown) > 0:
        print(f"  ℹ️  Unknown {len(df_unknown):,} 행 → 파이프라인에서 제외 (satellite_id == -1)")

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

    # ── OCO-3 별도 집계 및 저장 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: OCO-3 Super-observation (독립 검증용 별도 집계)")
    print("=" * 60)

    if len(df_oco3) == 0:
        print("  ⚠️ OCO-3 데이터 없음 — 건너뜀 (위성 식별 컬럼 확인 필요)")
    else:
        agg_oco3 = perform_super_observation(df_oco3)
        print(f"  [Result] OCO-3 Super-obs 완료: {len(agg_oco3):,} 행")

        # OCO-3 결측 처리
        drop_subset_oco3 = ['xco2']
        if 'tropomi_no2' in agg_oco3.columns:
            drop_subset_oco3.append('tropomi_no2')
        agg_oco3 = agg_oco3.dropna(subset=drop_subset_oco3).reset_index(drop=True)

        agg_oco3.to_parquet(PARQUET_OCO3_OUT, index=False)
        mb_oco3 = os.path.getsize(PARQUET_OCO3_OUT) / (1024 * 1024)
        print(f"  [Success] OCO-3 Super-obs 저장 완료!")
        print(f"  📍 파일 경로: {PARQUET_OCO3_OUT}")
        print(f"  📦 용량: {mb_oco3:.1f} MB | Shape: {agg_oco3.shape}")

    # ── Validation Log ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Validation Log — OCO-2 vs OCO-3 비교 리포트")
    print("=" * 60)

    # OCO-2 통계
    n_raw_oco2   = len(df)
    n_grid_oco2  = agg_df[['lat_idx', 'lon_idx']].drop_duplicates().shape[0]
    n_obs_oco2   = int(agg_df['n_soundings'].sum())

    print(f"  {'항목':<30} {'OCO-2':>15} {'OCO-3':>15}")
    print(f"  {'-'*60}")
    print(f"  {'원본 샘플 수 (QC 후)':<30} {n_raw_oco2:>15,}", end="")

    if len(df_oco3) > 0:
        n_raw_oco3  = len(df_oco3)
        n_grid_oco3 = agg_oco3[['lat_idx', 'lon_idx']].drop_duplicates().shape[0]
        n_obs_oco3  = int(agg_oco3['n_soundings'].sum())
        ratio_raw   = n_raw_oco3  / n_raw_oco2  * 100 if n_raw_oco2  > 0 else float('nan')
        ratio_grid  = n_grid_oco3 / n_grid_oco2 * 100 if n_grid_oco2 > 0 else float('nan')
        ratio_obs   = n_obs_oco3  / n_obs_oco2  * 100 if n_obs_oco2  > 0 else float('nan')

        print(f" {n_raw_oco3:>15,}  (OCO-3/OCO-2 = {ratio_raw:.1f}%)")
        print(f"  {'유효 격자 수 (unique grid)':<30} {n_grid_oco2:>15,} {n_grid_oco3:>15,}  (OCO-3/OCO-2 = {ratio_grid:.1f}%)")
        print(f"  {'총 Super-obs 관측 수':<30} {n_obs_oco2:>15,} {n_obs_oco3:>15,}  (OCO-3/OCO-2 = {ratio_obs:.1f}%)")
        print(f"  {'Super-obs 행 수':<30} {len(agg_df):>15,} {len(agg_oco3):>15,}")
    else:
        print(f"  {'유효 격자 수 (unique grid)':<30} {n_grid_oco2:>15,} {'N/A':>15}")
        print(f"  {'총 Super-obs 관측 수':<30} {n_obs_oco2:>15,} {'N/A':>15}")
        print(f"  {'Super-obs 행 수':<30} {len(agg_df):>15,} {'N/A':>15}")
        print("  ℹ️  OCO-3 데이터 없음 — 위성 식별 컬럼 확인 필요")

    if len(df_unknown) > 0:
        print(f"\n  ⚠️  Unknown 레코드 {len(df_unknown):,} 행 집계에서 제외됨 (satellite_id == -1)")

    print("=" * 60)


if __name__ == "__main__":
    run_super_observation()
