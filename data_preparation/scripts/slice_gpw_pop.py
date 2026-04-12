import os
import rasterio
from rasterio.windows import from_bounds

# ─────────────────────────────────────────────
# 1. 설정
# ─────────────────────────────────────────────
# 입력 원본 TIF 파일 경로
IN_TIF_PATH = "/home/lemon/win_desktop/2026 상반기_ 연구/gpw-v4-population-density-rev11_2020_15_min_tif/gpw_v4_population_density_rev11_2020_15_min.tif"

# 출력될 잘린 TIF 파일 경로 (동일 폴더 내에 이름만 변경)
OUT_TIF_PATH = "/home/lemon/win_desktop/2026 상반기_ 연구/gpw-v4-population-density-rev11_2020_15_min_tif/gpw_v4_population_density_rev11_2020_15_min_east_asia.tif"

# 슬라이싱 영역 (동아시아 도메인)
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0

def slice_geotiff():
    if not os.path.exists(IN_TIF_PATH):
        print(f"❌ 원본 파일을 찾을 수 없습니다: {IN_TIF_PATH}")
        return

    print("✂️  GPW 인구 밀도 GeoTIFF 슬라이싱 시작...")
    print(f"  • 대상 영역: 위도 {LAT_MIN}~{LAT_MAX}, 경도 {LON_MIN}~{LON_MAX}")

    with rasterio.open(IN_TIF_PATH) as src:
        # 우리가 원하는 위경도 좌표(Bounds)를 바탕으로 픽셀 창(Window)을 계산
        # from_bounds 매개변수 순서: (left, bottom, right, top, transform) -> (min_lon, min_lat, max_lon, max_lat, ...)
        window = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform)

        # 행/열 값이 소수점이 나올 수 있으므로 정수형(올림/내림) 픽셀 범위로 반올림 처리
        window = window.round_lengths().round_offsets()

        # 계산된 창 크기만큼 원본에서 데이터 읽기
        data = src.read(window=window)

        # 잘라낸 영역에 맞는 새로운 지리적 변환 공식(Transform) 생성
        win_transform = src.window_transform(window)

        # 새 파일 저장을 위한 메타데이터 갱신 (높이, 너비, 변환공식)
        kwargs = src.meta.copy()
        kwargs.update({
            'height': window.height,
            'width': window.width,
            'transform': win_transform
        })

        # 결과 TIF 파일로 저장
        with rasterio.open(OUT_TIF_PATH, 'w', **kwargs) as dst:
            dst.write(data)

    print(f"\n✅ 슬라이싱 완료!")
    print(f"  • 저장 위치: {OUT_TIF_PATH}")
    
    # 원본 파일 크기와 생성된 파일 크기 비교
    in_mb = os.path.getsize(IN_TIF_PATH) / (1024 * 1024)
    out_mb = os.path.getsize(OUT_TIF_PATH) / (1024 * 1024)
    print(f"  • 용량 변화: {in_mb:.1f} MB  →  {out_mb:.1f} MB")

if __name__ == "__main__":
    slice_geotiff()
