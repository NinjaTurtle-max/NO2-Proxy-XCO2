"""
02b_xco2_anomaly_eaic.py
──────────────────────────────────────────────────────────────────
EAIC (East Asian Industrial Corridor) 집중형 Anomaly 추출

Stage 2: Per-sub-region adaptive baseline with fallback hierarchy

설계 원칙:
  - 5개 sub-region (NCP, YRD, PRD, KCR, JKT) 정의
  - 격자 단위 baseline 시도 → 부족 시 sub-region 평균 fallback
  - sub-region 외부 데이터는 ROI 외 (out-of-scope) 로 폐기
  - MIN_OBS 차등: 도시 밀집 sub-region은 12, 외곽은 6
──────────────────────────────────────────────────────────────────
"""

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
# 경로
# ──────────────────────────────────────────────────────────────
BASE_DIR   = "/Volumes/100.118.65.89/dataset/XCO2연구 데이터"
PARQUET_IN = os.path.join(BASE_DIR, "anomaly_output/super_obs_dataset.parquet")
OUT_DIR    = os.path.join(BASE_DIR, "anomaly_output")
PARQUET_OUT = os.path.join(OUT_DIR, "anom_1d_eaic.parquet")
ZARR_OUT   = os.path.join(OUT_DIR, "refined_xco2_anom_eaic.zarr")
FIG_OUT    = os.path.join(OUT_DIR, "anomaly_statistics_eaic.png")

# ──────────────────────────────────────────────────────────────
# EAIC Sub-region 정의 (학술적 근거 인용 가능한 좌표)
# ──────────────────────────────────────────────────────────────
EAIC_REGIONS = {
    "NCP": {  # North China Plain — Beijing-Tianjin-Hebei + Shandong
        "lat": (34.0, 41.0),
        "lon": (113.0, 122.0),
        "min_obs_clim": 12,
        "min_obs_year": 6,
        "description": "North China Plain (Zheng 2018, Nature)",
    },
    "YRD": {  # Yangtze River Delta — Shanghai 광역
        "lat": (28.5, 33.0),
        "lon": (118.0, 123.0),
        "min_obs_clim": 12,
        "min_obs_year": 6,
        "description": "Yangtze River Delta (Hakkarainen 2016, GRL)",
    },
    "PRD": {  # Pearl River Delta — Hong Kong/Guangzhou
        "lat": (21.5, 24.5),
        "lon": (112.0, 115.5),
        "min_obs_clim": 8,   # 위도 낮아 관측 적음, 임계값 완화
        "min_obs_year": 4,
        "description": "Pearl River Delta",
    },
    "KCR": {  # Korean Capital Region — Seoul 광역
        "lat": (36.5, 38.5),
        "lon": (126.0, 128.0),
        "min_obs_clim": 6,   # 한반도 관측 매우 sparse
        "min_obs_year": 3,
        "description": "Korean Capital Region (Seoul Metropolitan)",
    },
    "JKT": {  # Japan Kanto — Tokyo 광역
        "lat": (34.5, 37.0),
        "lon": (138.5, 141.0),
        "min_obs_clim": 8,
        "min_obs_year": 4,
        "description": "Japan Kanto (Tokyo Metropolitan)",
    },
}

# 격자 (전체 보존, ROI 마스킹은 별도)
LAT_MIN_GRID, LAT_MAX_GRID = 20.0, 50.0
LON_MIN_GRID, LON_MAX_GRID = 100.0, 150.0
RESOLUTION = 0.5

lat_edges = np.arange(LAT_MIN_GRID, LAT_MAX_GRID + RESOLUTION, RESOLUTION)
lon_edges = np.arange(LON_MIN_GRID, LON_MAX_GRID + RESOLUTION, RESOLUTION)
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2


# ──────────────────────────────────────────────────────────────
# Sub-region assignment (vectorized)
# ──────────────────────────────────────────────────────────────
def assign_eaic_region(df: pd.DataFrame) -> pd.DataFrame:
    """각 행에 EAIC sub-region 라벨 할당. 비EAIC는 'OUT'."""
    df['eaic_region'] = 'OUT'
    
    for code, spec in EAIC_REGIONS.items():
        lat_lo, lat_hi = spec['lat']
        lon_lo, lon_hi = spec['lon']
        mask = (
            (df['latitude'] >= lat_lo) & (df['latitude'] < lat_hi) &
            (df['longitude'] >= lon_lo) & (df['longitude'] < lon_hi)
        )
        df.loc[mask, 'eaic_region'] = code
    
    print("  [EAIC 할당 결과]")
    counts = df['eaic_region'].value_counts()
    total = len(df)
    for code in list(EAIC_REGIONS.keys()) + ['OUT']:
        n = counts.get(code, 0)
        pct = n / total * 100
        bar = '█' * int(pct / 2)
        print(f"    {code:5s}: {n:>8,} 행 ({pct:5.1f}%) {bar}")
    return df


def load_and_prepare(parquet_path: str) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 1: 데이터 로드 & EAIC 영역 할당")
    print("=" * 70)
    
    df = pd.read_parquet(parquet_path)
    print(f"  [Load] Super-obs Parquet: {len(df):,} 행")
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M')
    
    df = assign_eaic_region(df)
    
    # ROI 외부 폐기
    n_before = len(df)
    df = df[df['eaic_region'] != 'OUT'].copy().reset_index(drop=True)
    print(f"\n  [ROI 필터] EAIC 내부만 보존: {n_before:,} → {len(df):,} ({len(df)/n_before*100:.1f}%)")
    
    return df


# ──────────────────────────────────────────────────────────────
# Sub-region별 climatology — local + fallback hierarchy
# ──────────────────────────────────────────────────────────────
def compute_hierarchical_climatology(df: pd.DataFrame):
    """
    3-tier baseline hierarchy:
      Tier 1: 격자별 climatology (격자가 충분 관측을 보유)
      Tier 2: sub-region 평균 climatology (격자 부족 시)
      Tier 3: 전역 EAIC 평균 (sub-region조차 부족 시 — 거의 발생 X)
    """
    print("\n" + "=" * 70)
    print("STEP 2: 3-Tier Hierarchical Climatology 산출")
    print("=" * 70)
    
    # Tier 1: 격자별 — sub-region별 차등 임계값 적용
    grid_clim = {}  # (lat_idx, lon_idx, month) → mean xco2
    valid_grid_keys = set()
    
    grid_ym = df.groupby(['eaic_region', 'lat_idx', 'lon_idx'])['year_month'].nunique().reset_index()
    
    for region, spec in EAIC_REGIONS.items():
        thr = spec['min_obs_clim']
        sub = grid_ym[(grid_ym['eaic_region'] == region) & (grid_ym['year_month'] >= thr)]
        for _, row in sub.iterrows():
            valid_grid_keys.add((int(row['lat_idx']), int(row['lon_idx'])))
        print(f"  [{region}] 임계값 {thr}개월 — Tier 1 유효 격자: {len(sub):,}개")
    
    # Tier 1 climatology 산출
    grid_monthly = df.groupby(['lat_idx', 'lon_idx', 'month'])['xco2'].mean().reset_index()
    for _, row in grid_monthly.iterrows():
        key = (int(row['lat_idx']), int(row['lon_idx']))
        if key in valid_grid_keys:
            grid_clim[(int(row['lat_idx']), int(row['lon_idx']), int(row['month']))] = row['xco2']
    
    # Tier 2: sub-region 평균 climatology (모든 sub-region에 대해 산출)
    region_clim = {}  # (region, month) → mean xco2
    region_monthly = df.groupby(['eaic_region', 'month'])['xco2'].mean().reset_index()
    for _, row in region_monthly.iterrows():
        region_clim[(row['eaic_region'], int(row['month']))] = row['xco2']
    
    print(f"\n  [Tier 2] sub-region × month climatology: {len(region_clim)}개 조합")
    
    # Tier 3: EAIC 전역 평균 (월별)
    global_clim = df.groupby('month')['xco2'].mean().to_dict()
    print(f"  [Tier 3] EAIC 전역 월평균 fallback 준비 완료")
    
    return grid_clim, region_clim, global_clim, valid_grid_keys


def compute_yearly_deviation_hierarchical(df: pd.DataFrame, valid_grid_keys: set):
    """
    Yearly deviation도 계층 구조 적용:
      Tier 1: 격자-연도 (충분 관측)
      Tier 2: sub-region-연도 평균
    """
    print("\n" + "=" * 70)
    print("STEP 3: 3-Tier Yearly Deviation 산출")
    print("=" * 70)
    
    # 격자별 전체 평균 (Tier 1 기준점)
    c_all_grid = df.groupby(['lat_idx', 'lon_idx'])['xco2'].mean()
    c_all_dict = c_all_grid.to_dict()
    
    # 격자-연도 평균 (충분 관측 격자만)
    grid_yearly = df.groupby(['eaic_region', 'lat_idx', 'lon_idx', 'year']).agg(
        xco2_mean=('xco2', 'mean'),
        n_obs=('xco2', 'count')
    ).reset_index()
    
    grid_delta = {}
    for _, row in grid_yearly.iterrows():
        region = row['eaic_region']
        spec = EAIC_REGIONS[region]
        thr = spec['min_obs_year']
        key = (int(row['lat_idx']), int(row['lon_idx']))
        if row['n_obs'] >= thr and key in valid_grid_keys:
            c_all = c_all_dict.get(key, np.nan)
            if np.isfinite(c_all):
                grid_delta[(int(row['lat_idx']), int(row['lon_idx']), int(row['year']))] = \
                    row['xco2_mean'] - c_all
    
    # sub-region-연도 평균 (Tier 2)
    region_all = df.groupby('eaic_region')['xco2'].mean().to_dict()
    region_yearly = df.groupby(['eaic_region', 'year'])['xco2'].mean().reset_index()
    region_delta = {}
    for _, row in region_yearly.iterrows():
        region_delta[(row['eaic_region'], int(row['year']))] = row['xco2'] - region_all[row['eaic_region']]
    
    print(f"  [Tier 1] 격자-연도 ΔC: {len(grid_delta):,}개 조합")
    print(f"  [Tier 2] sub-region-연도 ΔC: {len(region_delta)}개 조합")
    
    return grid_delta, region_delta


def calculate_anomaly_hierarchical(df, grid_clim, region_clim, global_clim,
                                    grid_delta, region_delta):
    """벡터화된 계층적 anomaly 계산"""
    print("\n" + "=" * 70)
    print("STEP 4: 계층적 Anomaly 계산 (Tier 1 → 2 → 3)")
    print("=" * 70)
    
    n = len(df)
    baseline_arr = np.full(n, np.nan, dtype=np.float32)
    tier_used = np.full(n, 0, dtype=np.int8)  # 어느 tier가 사용되었는지 추적
    
    lat_i = df['lat_idx'].values.astype(int)
    lon_i = df['lon_idx'].values.astype(int)
    months = df['month'].values.astype(int)
    years = df['year'].values.astype(int)
    regions = df['eaic_region'].values
    
    for i in tqdm(range(n), desc="  [Processing Anomaly]"):
        # Climatology 부분
        key_grid = (lat_i[i], lon_i[i], months[i])
        c_clim = grid_clim.get(key_grid, np.nan)
        tier_c = 1 if np.isfinite(c_clim) else 0
        
        if not np.isfinite(c_clim):
            c_clim = region_clim.get((regions[i], months[i]), np.nan)
            tier_c = 2 if np.isfinite(c_clim) else 0
        
        if not np.isfinite(c_clim):
            c_clim = global_clim.get(months[i], np.nan)
            tier_c = 3 if np.isfinite(c_clim) else 0
        
        # Yearly deviation 부분
        key_gyear = (lat_i[i], lon_i[i], years[i])
        dc = grid_delta.get(key_gyear, np.nan)
        tier_d = 1 if np.isfinite(dc) else 0
        
        if not np.isfinite(dc):
            dc = region_delta.get((regions[i], years[i]), 0.0)  # 없으면 0
            tier_d = 2
        
        baseline_arr[i] = c_clim + dc
        tier_used[i] = tier_c * 10 + tier_d  # 11=both T1, 21=clim T2 dev T1 등
    
    df['xco2_baseline'] = baseline_arr
    df['xco2_anomaly'] = df['xco2'].values - baseline_arr
    df['baseline_tier'] = tier_used
    
    valid_df = df.dropna(subset=['xco2_anomaly']).reset_index(drop=True)
    print(f"  [결과] 유효 anomaly: {len(valid_df):,} / {len(df):,} ({len(valid_df)/len(df)*100:.1f}%)")
    
    # Tier 사용 분포
    print("\n  [Baseline Tier 분포]")
    tier_counts = pd.Series(valid_df['baseline_tier']).value_counts().sort_index()
    tier_labels = {11: 'Grid-Grid (최고품질)', 12: 'Grid-Region', 21: 'Region-Grid', 
                   22: 'Region-Region', 31: 'Global-Grid', 32: 'Global-Region'}
    for t, n in tier_counts.items():
        label = tier_labels.get(t, f'Tier {t}')
        pct = n / len(valid_df) * 100
        print(f"    {label:25s}: {n:>7,} ({pct:5.1f}%)")
    
    # 잔차 진단 (sub-region × year)
    print("\n  [Residual Trend 진단 — sub-region × year]")
    pivot = pd.pivot_table(valid_df, values='xco2_anomaly',
                            index='year', columns='eaic_region', aggfunc='mean').round(3)
    print(pivot.to_string())
    
    total_mean = valid_df['xco2_anomaly'].mean()
    flag = '✅' if abs(total_mean) < 0.3 else '⚠️'
    print(f"\n  전체 잔차 평균: {total_mean:+.4f} ppm {flag}")
    
    return valid_df


def plot_qq(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 5: 분포 진단 시각화")
    print("=" * 70)
    
    anom = df['xco2_anomaly'].dropna().values
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor='white')
    
    # 전체 분포
    ax = axes[0, 0]
    ax.hist(anom, bins=200, density=True, alpha=0.6, color='#2CA02C', edgecolor='none')
    mu, sigma = float(np.mean(anom)), float(np.std(anom))
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
            label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
    ax.set_xlabel('XCO2 Anomaly (ppm)')
    ax.set_title('Overall Anomaly Distribution (EAIC)')
    ax.legend()
    
    # Q-Q
    ax = axes[0, 1]
    (osm, osr), (slope, intercept, r) = stats.probplot(anom, dist='norm')
    ax.scatter(osm, osr, s=1, alpha=0.3, color='#2CA02C')
    ax.plot(osm, slope*osm + intercept, 'r-', lw=2, label=f'R² = {r**2:.4f}')
    ax.set_title('Q-Q Plot (Normal)')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles (ppm)')
    ax.legend()
    
    # Sub-region별 분포
    ax = axes[1, 0]
    for region in EAIC_REGIONS.keys():
        sub = df[df['eaic_region'] == region]['xco2_anomaly']
        if len(sub) > 50:
            ax.hist(sub, bins=80, density=True, alpha=0.4, label=f'{region} (n={len(sub):,})')
    ax.set_xlabel('XCO2 Anomaly (ppm)')
    ax.set_title('Sub-region Distributions')
    ax.legend()
    ax.set_xlim(-8, 8)
    
    # Tier 사용 비율
    ax = axes[1, 1]
    tier_counts = df['baseline_tier'].value_counts().sort_index()
    tier_labels_short = {11: 'G-G', 12: 'G-R', 21: 'R-G', 22: 'R-R', 31: 'Gl-G', 32: 'Gl-R'}
    labels = [tier_labels_short.get(t, str(t)) for t in tier_counts.index]
    ax.bar(labels, tier_counts.values, color='steelblue', edgecolor='black')
    ax.set_title('Baseline Tier Usage')
    ax.set_ylabel('# observations')
    
    fig.suptitle(f'EAIC Anomaly Diagnostics  (Total N = {len(df):,})', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    kurt = stats.kurtosis(anom, fisher=True)
    skew = stats.skew(anom)
    print(f"  [통계] μ={mu:.4f}, σ={sigma:.4f}, Skew={skew:.3f}, Kurt={kurt:.3f}")
    print(f"  [저장] {FIG_OUT}")


def export(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 6: Export — Parquet (1D)")
    print("=" * 70)
    
    df.to_parquet(PARQUET_OUT, index=False)
    mb = os.path.getsize(PARQUET_OUT) / (1024 * 1024)
    print(f"  [저장] {PARQUET_OUT} ({mb:.1f} MB, {df.shape})")


if __name__ == "__main__":
    df = load_and_prepare(PARQUET_IN)
    grid_clim, region_clim, global_clim, valid_keys = compute_hierarchical_climatology(df)
    grid_delta, region_delta = compute_yearly_deviation_hierarchical(df, valid_keys)
    df_anom = calculate_anomaly_hierarchical(df, grid_clim, region_clim, global_clim,
                                              grid_delta, region_delta)
    plot_qq(df_anom)
    export(df_anom)
    
    print("\n✅ EAIC 집중형 Anomaly 추출 완료")
