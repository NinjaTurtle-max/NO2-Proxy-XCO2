import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────
# 경로 및 상수
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = "/Volumes/100.118.65.89/dataset/XCO2연구 데이터"
PARQUET_IN = os.path.join(BASE_DIR, "anomaly_output/super_obs_dataset.parquet")

# 분석 임계값 (MIN_OBS_YEAR)
THRESHOLDS = [12, 5, 3]
MIN_OBS_CLIM = 12  # 고정 (Climatology 무결성 기준)

# 클러스터 정의
CLUSTERS = {
    "C1_NC_Plain":  {"lat": (32, 42), "lon": (112, 122)},
    "C2_South_China": {"lat": (22, 32), "lon": (105, 120)},
    "C3_Korea":      {"lat": (33, 40), "lon": (124, 131)},
    "C4_Japan":      {"lat": (30, 46), "lon": (130, 146)},
    "C6_Cont_North": {"lat": (42, 50), "lon": (100, 135)},
}

def analyze_retention():
    if not os.path.exists(PARQUET_IN):
        print(f"❌ 파일을 찾을 수 없습니다: {PARQUET_IN}")
        return

    print(f"📊 RAW 데이터 로드 중: {PARQUET_IN}")
    df = pd.read_parquet(PARQUET_IN)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M')

    # 1. 클러스터 할당
    df['cluster'] = "C5_Marine_Other"
    for name, bounds in CLUSTERS.items():
        lat_min, lat_max = bounds["lat"]
        lon_min, lon_max = bounds["lon"]
        mask = (df['latitude'] >= lat_min) & (df['latitude'] < lat_max) & \
               (df['longitude'] >= lon_min) & (df['longitude'] < lon_max)
        df.loc[mask, 'cluster'] = name

    # 2. 격자별 통계 사전 계산
    # (lat_idx, lon_idx) 별 고유 월수 및 연간 관측수
    grid_stats = df.groupby(['lat_idx', 'lon_idx', 'cluster']).agg(
        total_rows=('xco2', 'count'),
        unique_months=('year_month', 'nunique'),
    ).reset_index()

    # 연간 관측수는 (격자, 연도)별로 뽑아야 함
    yearly_stats = df.groupby(['lat_idx', 'lon_idx', 'year']).size().reset_index(name='n_obs_year')

    results = []

    for t in THRESHOLDS:
        print(f"\n🔍 분석 중: MIN_OBS_YEAR = {t} (MIN_OBS_CLIM = 12)")
        
        # 필터 1: 고유 월수 >= 12
        valid_clim_grids = grid_stats[grid_stats['unique_months'] >= MIN_OBS_CLIM][['lat_idx', 'lon_idx']]
        
        # 필터 2: 특정 연도에 n_obs >= T 만족하는 (격자, 연도) 쌍
        valid_year_grids = yearly_stats[yearly_stats['n_obs_year'] >= t]
        
        # 합치기: (lat_idx, lon_idx, year)가 최종 Anomaly 행이 됨
        # (먼저 월수 조건을 만족하는 격자만 남김)
        merged = valid_year_grids.merge(valid_clim_grids, on=['lat_idx', 'lon_idx'])
        
        # 클러스터별 집계
        # merged와 grid_stats를 매칭하여 클러스터 정보 가져오기
        final_with_cluster = merged.merge(grid_stats[['lat_idx', 'lon_idx', 'cluster']], on=['lat_idx', 'lon_idx'])
        
        summary = final_with_cluster.groupby('cluster').agg(
            n_grids=('lat_idx', 'nunique'),
            n_rows=('n_obs_year', 'sum')
        ).reset_index()
        
        summary['threshold'] = t
        results.append(summary)

    # 3. 결과 출력
    full_res = pd.concat(results)
    
    # 가독성을 위한 피봇
    pivot_grids = full_res.pivot(index='cluster', columns='threshold', values='n_grids').fillna(0).astype(int)
    pivot_rows = full_res.pivot(index='cluster', columns='threshold', values='n_rows').fillna(0).astype(int)

    print("\n" + "="*80)
    print(f"📊 클러스터별 유효 격수(Unique Grids) 비교 (MIN_OBS_CLIM=12 고정)")
    print("="*80)
    print(pivot_grids[[12, 5, 3]])
    
    print("\n" + "="*80)
    print(f"📊 클러스터별 최종 Anomaly 행수(Total Rows) 비교")
    print("="*80)
    print(pivot_rows[[12, 5, 3]])

    # 4. 결론 도출을 위한 증감률 확인
    print("\n" + "="*80)
    print("💡 분석 인사이트")
    print("="*80)
    for cluster in pivot_grids.index:
        g12 = pivot_grids.loc[cluster, 12]
        g3 = pivot_grids.loc[cluster, 3]
        gain = (g3 / g12 * 100) if g12 > 0 else float('inf')
        print(f"  - {cluster:<15}: T=12 -> T=3 변경 시 격자 수 {gain:.1f}% 확보 가능")

if __name__ == "__main__":
    analyze_retention()
