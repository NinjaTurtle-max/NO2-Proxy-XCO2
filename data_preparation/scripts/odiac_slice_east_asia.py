"""
ODIAC 1km GeoTIFF → 동아시아 슬라이싱 → GeoTIFF 저장
도메인: 20–50°N, 100–150°E
입력:  /mnt/e/extracted_odiac_2020_2023/*.tif  (전구 43200×21600, ~3.7GB/file)
출력:  /mnt/e/dataset/odiac/EA_odiac_YYMM.tif  (동아시아 영역만, ~수 MB/file)
"""

import os
import glob
import time
import numpy as np
import rasterio
from rasterio.windows import from_bounds

# ─────────────────────────────────────────
# 1. 설정
# ─────────────────────────────────────────
INPUT_DIR  = '/mnt/e/extracted_odiac_2020_2023'
OUTPUT_DIR = '/mnt/e/dataset/odiac'

# 동아시아 도메인 (프로젝트 공통)
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 2. 파일 목록
# ─────────────────────────────────────────
tif_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.tif')))
print(f"총 {len(tif_files)}개 파일 발견\n")

if not tif_files:
    print("⚠️ 파일이 없습니다. 경로를 확인해주세요.")
    exit(1)

# ─────────────────────────────────────────
# 3. 슬라이싱 루프
# ─────────────────────────────────────────
success, skip = 0, 0
t_start = time.time()

for i, fpath in enumerate(tif_files, 1):
    fname = os.path.basename(fpath)
    # 파일명에서 YYMM 추출: odiac2024_1km_excl_intl_2208.tif → 2208
    yymm = fname.replace('.tif', '').split('_')[-1]
    out_name = f"EA_odiac_{yymm}.tif"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    # 이미 처리된 파일 스킵
    if os.path.exists(out_path):
        print(f"  [{i:2d}/{len(tif_files)}] {fname} → 이미 존재, 스킵")
        skip += 1
        continue

    try:
        t0 = time.time()
        with rasterio.open(fpath) as src:
            # 윈도우 계산 (픽셀 좌표로 변환)
            window = from_bounds(
                LON_MIN, LAT_MIN, LON_MAX, LAT_MAX,
                transform=src.transform
            )

            # 윈도우 영역만 읽기 (메모리 효율적)
            data = src.read(1, window=window)

            # 슬라이싱된 영역의 Transform 계산
            win_transform = src.window_transform(window)

            # 새 GeoTIFF 저장
            profile = src.profile.copy()
            profile.update(
                width=data.shape[1],
                height=data.shape[0],
                transform=win_transform,
                compress='deflate',       # 압축 적용 → 파일 크기 대폭 감소
                predictor=2,              # 수치 데이터 압축 효율 향상
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(data, 1)

        elapsed = time.time() - t0
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  [{i:2d}/{len(tif_files)}] {fname} → {out_name}  "
              f"({data.shape[1]}×{data.shape[0]} px, {size_mb:.1f} MB, {elapsed:.1f}s)")
        success += 1

    except Exception as e:
        print(f"  [{i:2d}/{len(tif_files)}] {fname} → ❌ 오류: {e}")

# ─────────────────────────────────────────
# 4. 완료 요약
# ─────────────────────────────────────────
total_time = time.time() - t_start
print(f"\n{'='*60}")
print(f"✅ 처리 완료")
print(f"   성공: {success}개  |  스킵: {skip}개  |  총 소요: {total_time:.1f}s")
print(f"   출력 경로: {OUTPUT_DIR}")

# 결과 파일 목록
out_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'EA_odiac_*.tif')))
total_size_mb = sum(os.path.getsize(f) for f in out_files) / 1024 / 1024
print(f"   총 파일 수: {len(out_files)}개  |  총 크기: {total_size_mb:.1f} MB")

# 샘플 파일 검증
if out_files:
    sample = out_files[0]
    with rasterio.open(sample) as src:
        d = src.read(1)
        print(f"\n📋 샘플 검증 ({os.path.basename(sample)}):")
        print(f"   Shape: {d.shape}")
        print(f"   CRS:   {src.crs}")
        print(f"   Bounds: {src.bounds}")
        print(f"   Min={d.min():.4f}, Max={d.max():.4f}, Mean={d.mean():.4f}")
