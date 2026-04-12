# NO2-Proxy XCO2 Research Pipeline

이 저장소는 OCO-2 XCO2 데이터의 궤적 아티팩트를 제거하고, TROPOMI NO2를 대리 변수(Proxy)로 활용하여 고해상도 XCO2 지도를 복원하는 연구를 위한 파이프라인입니다.

## 📂 디렉토리 구조

### 1. Main Pipeline (Root)
연구의 핵심 흐름을 담당하는 스크립트들입니다. (순서대로 실행 권장)
- `01_super_observation.py`: 위성 sounding 데이터의 격자화(0.1°) 및 Bootstrap 불확실성 산출.
- `02_xco2_anomaly_extraction.py`: 5년 치 통계를 활용하여 기후적 배경 농도(Baseline)를 제거하고 순수 이상치(Anomaly) 추출. (Strict Threshold 적용)
- `03_correlation_and_data_split.py`: NO2-XCO2 공간 상관계수 산출(BH-Y FDR 보정) 및 학습/검증/테스트 데이터셋 분할(Gap Months 반영).

### 2. `data_preparation/scripts/`
원천 데이터(Raw Data)를 지역별로 슬라이싱하거나 전처리하여 학습용 Parquet 파일을 생성하는 유틸리티입니다.
- `preprocess_ml.py`: 여러 소스의 데이터를 병합하여 `ml_ready_dataset.parquet` 생성.
- `Tropomi slice east asia.py`, `oco_slice_east_asia.py`, `era5_download_slice_v3.py` 등.

### 3. `archive/`
연구 과정에서 생성된 이전 버전의 스크립트나 임시 분석용 파일들을 보관합니다.

### 4. `DATA_process/`
데이터 처리 과정에서 생성되는 중간 파일들이 보관되는 디렉토리입니다.

---
## 🚀 실행 가이드
1. `data_preparation/scripts/` 내 스크립트들을 통해 동아시아 영역 데이터를 슬라이싱합니다.
2. `preprocess_ml.py`를 실행하여 통합 데이터셋을 생성합니다.
3. Root의 `01` -> `02` -> `03` 스크립트를 순차적으로 실행하여 모델링 준비를 완료합니다.
