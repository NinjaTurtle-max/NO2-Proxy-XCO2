"""격자 상수와 자료 경로 (저장소 루트 기준 상대 경로). 모든 스크립트는 인자로 경로를 바꿀 수 있다."""

GRID_LAT0, GRID_LON0, GRID_D = 20.0, 100.0, 0.25   # 잠재 격자 = ERA5 격자 (121 × 201, 20–50°N · 100–150°E)

OCO_NC = "data/raw/integrated_dataset.nc"           # OCO-2/3 사운딩 원천 (netCDF)
OCO_NODES_DIR = "data/raw/oco_nodes"                # 사운딩 표 (year_month 파티션 parquet)
TROPOMI_DIR = "data/raw/tropomi"                    # TROPOMI NO₂ 화소 (granule 별 parquet)
ERA5_PL_DIR = "data/raw/era5_pl"                    # ERA5 기압면 · 단일면 원본 (월 netCDF)
ERA5_WIND_DIR = "data/raw/era5_wind"                # 경계층 바람 · t2m · blh · sp (era5_wind_YYYYMM_z100.nc)
ERA5_SYN_DIR = "data/raw/era5_syn"                  # 850 hPa 고도 · 700–1000 hPa 두께 (era5_syn_YYYYMM.nc)
INDEX_DIR = "data/processed/index"                  # 모델 입력 인덱스 (build_index 산출)
