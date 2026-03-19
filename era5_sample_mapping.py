import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# 1. 파일 경로 설정
csv_path = config.ERA5_CSV_V3_DIR / "era5_2024_01.csv"
output_image_path = config.DATA_PROCESS_DIR / "era5_sample_mapping.png"

print(f"데이터를 불러오는 중: {csv_path}")
# 필요한 컬럼만 추출하여 메모리 절약
df = pd.read_csv(csv_path, usecols=['date', 'obs_time', 'lat', 'lon', 'wind_speed_obs', 'blh_obs'])

# 2. 특정 관측일, 관측시간대 추출 (데이터셋의 첫 번째 관측 시간 기준)
sample_date = df['date'].iloc[0]
sample_time = df['obs_time'].iloc[0]

print(f"-> 추출된 관측일: {sample_date}, 관측시간대: {sample_time}")

# 3. 해당 날짜와 시간대로 데이터 필터링
df_slice = df[(df['date'] == sample_date) & (df['obs_time'] == sample_time)].copy()
print(f"-> 해당 시간대({sample_date} {sample_time}) 픽셀 데이터 개수: {len(df_slice):,}개\n")

# ==============================================================================
# [방법 A] 데이터 맵핑 (Data Mapping) - TROPOMI 등과 비교하기 위한 시간형식 변환
# ==============================================================================
df_slice['datetime_str'] = df_slice['date'].astype(str) + " " + df_slice['obs_time']
df_slice['datetime_mapped'] = pd.to_datetime(df_slice['datetime_str'], format='%Y%m%d %H:%M')

print("[방법 A] 데이터 시간 맵핑 (Datetime 매칭키 생성)")
print(df_slice[['date', 'obs_time', 'datetime_mapped']].head())
print("-" * 60)

# ==============================================================================
# [방법 B] 공간 맵핑 (Spatial Mapping) - 위도/경도를 이용해 공간에 시각화
# ==============================================================================
print("[방법 B] 공간 시각화 맵핑 진행 중...")
plt.figure(figsize=(10, 8))

# 산점도로 위경도 상에 매개변수(풍속) 맵핑
sc = plt.scatter(df_slice['lon'], df_slice['lat'], 
                 c=df_slice['wind_speed_obs'], 
                 cmap='jet', 
                 s=5, 
                 alpha=0.8)

plt.colorbar(sc, label='Wind Speed (m/s)')
plt.title(f'ERA5 Spatial Mapping ({sample_date} {sample_time})', fontsize=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle='--', alpha=0.5)

# 맵핑 이미지 저장
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"공간 맵핑(이미지)이 성공적으로 저장되었습니다:\n  -> {output_image_path}")
