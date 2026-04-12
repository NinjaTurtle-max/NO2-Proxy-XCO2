#!/usr/bin/env python3
"""
ODIAC 데이터를 동아시아 영역 (20-50°N, 100-150°E)으로 슬라이싱
SW: lat=20, lon=100
NE: lat=50, lon=150
"""

import rasterio
from rasterio.windows import from_bounds
import os
from pathlib import Path
import numpy as np

# 경로 설정
input_folder = '/mnt/e/extracted_odiac_2020_2023'
output_folder = '/home/lemon/win_desktop/2026 상반기_ 연구/NO2 Proxy XCO2/odiac_sliced'

# 슬라이싱 영역 정의 (20-50°N, 100-150°E)
# SW: (20, 100), NE: (50, 150)
lon_min, lon_max = 100, 150
lat_min, lat_max = 20, 50

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 입력 파일 목록
input_files = sorted(Path(input_folder).glob('odiac2024_1km_excl_intl_*.tif'))

print(f"총 {len(input_files)}개 파일 처리 시작...")
print(f"슬라이싱 영역: SW(lat={lat_min}, lon={lon_min}), NE(lat={lat_max}, lon={lon_max})")
print(f"출력 폴더: {output_folder}\n")

# 각 파일 처리
for idx, input_file in enumerate(input_files, 1):
    filename = input_file.name
    output_file = os.path.join(output_folder, filename)

    # 이미 처리된 파일은 건너뛰기
    if os.path.exists(output_file):
        print(f"[{idx}/{len(input_files)}] 건너뜀: {filename} (이미 존재)")
        continue

    try:
        with rasterio.open(input_file) as src:
            # 슬라이싱할 윈도우 계산
            window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)

            # 데이터 읽기
            data = src.read(1, window=window)

            # 새로운 transform 계산
            transform = src.window_transform(window)

            # 메타데이터 업데이트
            out_meta = src.meta.copy()
            out_meta.update({
                'height': window.height,
                'width': window.width,
                'transform': transform
            })

            # 슬라이싱된 데이터 저장
            with rasterio.open(output_file, 'w', **out_meta) as dst:
                dst.write(data, 1)

        print(f"[{idx}/{len(input_files)}] 완료: {filename} -> Shape: {data.shape}")

    except Exception as e:
        print(f"[{idx}/{len(input_files)}] 오류: {filename} - {str(e)}")
        continue

print("\n슬라이싱 완료!")
print(f"출력 위치: {output_folder}")

# 결과 확인
output_files = list(Path(output_folder).glob('*.tif'))
print(f"\n생성된 파일 개수: {len(output_files)}")

# 첫 번째 파일 정보 출력
if output_files:
    with rasterio.open(output_files[0]) as f:
        print(f"\n샘플 파일 정보: {output_files[0].name}")
        print(f"  Shape: {f.shape}")
        print(f"  Bounds: {f.bounds}")
        print(f"  CRS: {f.crs}")
