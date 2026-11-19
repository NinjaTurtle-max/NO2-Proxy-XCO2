# NO₂-Proxy XCO₂

TROPOMI NO₂(매일)를 ERA5 경계층 바람으로 수송하는 0.25° 잠재 격자에 주입하고, 듬성한 OCO-2/3 XCO₂ 사운딩 위치에서 디코드하는 물리 주입 그래프 모델. 영역 20–50°N · 100–150°E, 기간 2020–2024 (60개월).

모델 한 줄: 관측(NO₂ 화소 · XCO₂ 사운딩)은 원 위치에 두고 최근접 4점 이중선형 엣지로 0.25° 격자 노드 24,321개(121 × 201, 8채널)와 잇는다 → 격자 위에서 1시간마다 이류(ERA5 바람, z = max(BLH/2, 100 m)) · 확산(학습 D) · 감쇠(학습 k) → 사운딩 자리에서 디코드 ŷ = μ_train + g(배경 11변수) + β·NO₂_latent + r(z). 학습 파라미터 3,414개.

## 주요 결과

같은 시험 사운딩 1,204,908개(공간 분할 시험 5° 칸 12개, 2020–2024)에서의 비교. 학습은 seed 0 한 번, 분할은 공간 폴드 0 하나다.

| 모델 | RMSE (ppm) | R² | 편향 (ppm) | ΔRMSE − GNN 켬 [95 % 구간] |
|---|---|---|---|---|
| GNN, 수송 켬 | 1.4203 | 0.8935 | −0.0724 | — |
| GNN, 수송 끔 | 1.5005 | 0.8812 | −0.0324 | +0.0802 [+0.0527, +0.1083] |
| 크리깅 (시공간) | 1.6532 | 0.8557 | −0.0076 | +0.2329 [+0.1861, +0.2852] |
| LightGBM (같은 입력) | 1.3877 | 0.8983 | +0.0073 | −0.0326 [−0.0684, +0.0037] |

- ΔRMSE 구간: UTC 일 블록 부트스트랩 1,000회 (seed 0). 생성: `scripts/evaluate/compare_models.py` → `runs/tables/compare_models.csv`.
- 시험 칸 12개별 ΔRMSE(끔 − 켬): 12개 모두 양수(+0.008 – +0.185 ppm), 구간이 0 을 포함하지 않는 칸 6개 → `runs/tables/cells_transport.csv`.
- 학습된 값: 수송 켬 β 0.560 · 1/k 1,460.8 h (선택 에폭 46 / 57), 수송 끔 β 0.412 · 1/k 67.3 h (선택 에폭 14 / 25).

## 재현

```bash
pip install -e .          # Python 3.10, 판은 pyproject.toml
make all                  # 저장된 모델 파일로 평가 · 지도 · 그림 (자료 준비 뒤)
make train baselines      # 모델을 처음부터 다시 학습
```

순서와 명령은 `Makefile` 에 있다. 단계별:

| 단계 | 명령 | 입력 → 출력 |
|---|---|---|
| 1 자료 | `make data` | 원천 자료 → `data/raw/era5_*` · `data/processed/index` |
| 2 NO₂ 피처 | `make features` | 인덱스 → `runs/no2_features/no2_features.parquet` |
| 3 GNN 학습 | `make train` | 인덱스 · ERA5 → `runs/gnn_m60_{on,off}/`, `runs/gnn_m12_on/` (state · pred · summary) |
| 4 기준선 | `make baselines` | 인덱스 · NO₂ 피처 · ERA5 → `runs/kriging/`, `runs/lgbm_no2_wind/` |
| 5 평가 | `make evaluate` | 네 모델 예측 → `runs/tables/` |
| 6 지도 | `make maps` | 모델 파일 · 인덱스 → `runs/maps/` (저장된 예측을 재현해야 저장) |
| 7 그림 | `make figures` | → `figures/out/` |

`make all` 은 3·4 단계 산출물(예측 parquet, 합계 약 0.6 GB)이 있어야 돈다. 예측은 크기 때문에 저장소에 없다. 저장된 모델 파일에서 60개월 예측 전체를 다시 만드는 명령은 없으므로(지도 단계는 2020-01 사운딩만 다시 계산해 저장 예측과 대조한다) 3·4 단계를 다시 돌린다.

## 그림 · 표와 프로그램

| 그림 · 표 | 프로그램 | 출력 |
|---|---|---|
| Fig. 1 연구 영역 · 시험 칸 | `figures/region_points.py` → `figures/fig1_region.py` | `figures/out/fig1_region.png` |
| Fig. 2 모델 구조 · 비교 설계 (그림 재료) | `figures/fig2/thumbs.py` · `thumbs2.py` · `thumb_target_train.py` · `tiles_final.py` | `figures/out/fig2/*.png` |
| Fig. 3 사례일 지도 · 산점도 | `scripts/maps/grid_map_gnn.py` · `grid_map_baselines.py` → `figures/fig3_maps.py` | `figures/out/fig_maps_scatter.png` |
| Table 2 네 모델 비교 | `scripts/evaluate/compare_models.py` | `runs/tables/compare_models.csv` |
| 시험 칸별 수송 효과 | `scripts/evaluate/compare_models.py` | `runs/tables/cells_transport.csv` |

## 학습된 모델

| 파일 | 내용 |
|---|---|
| `runs/gnn_m60_on/state_space0_s0_phys.pt` | GNN 수송 켬, 60개월 (선택된 에폭의 에폭 평균 파라미터, 24 KB) |
| `runs/gnn_m60_off/state_space0_s0_nophys.pt` | GNN 수송 끔, 60개월 |
| `runs/gnn_m12_on/state_space0_s0_phys.pt` | GNN 수송 켬, 2020 12개월 (Fig. 2 예측 썸네일) |
| `runs/lgbm_no2_wind/model_lgbm_no2wind.txt` | LightGBM (num_leaves 31, 761회) |
| `runs/kriging/variogram.csv` | 크리깅 경험 변동함수 (모형은 이것을 다시 맞춰 얻는다) |

읽기: `torch.load(path)["params"]` → `no2xco2.model.Model(8, device, n_bg=11).load_state_dict(params, strict=False)`. 수송 끔은 `model.phys.advect`, `model.phys.diffuse` 를 항등 함수로 바꾼다 (`scripts/maps/grid_map_gnn.py`).

## 방법 요약

| 항목 | 설정 |
|---|---|
| 타깃 | OCO-2/3 XCO₂, quality flag 0 · 380–450 ppm · 영역 안 (17,415,604 사운딩 중 10,000,122) |
| 입력 | TROPOMI 대류권 NO₂ (qa ≥ 0.75, 전 기간 μ · σ 로 표준화), ERA5 시간별 경계층 바람 · t2m · blh · sp · 850 hPa 고도 · 700–1000 hPa 두께 |
| 분할 | 5° 칸 60개를 5개 폴드로, 시험 = 폴드 0 (12칸), 공간 버퍼 1°. val = 시간 폴드 1 (7일 블록, ±2일 버퍼). train 5,555,414 · val 1,624,172 · test 1,204,908 |
| 학습 | 손실 = 훈련 사운딩 MSE, 48 h 절단 역전파, Adam 3e-3, 기울기 자르기 1.0, 화소 15 % 부표본, 달 순서 무작위 + 달 시작 상태 캐시, 에폭 평균 파라미터, 깨끗한 val 롤아웃으로 선택, patience 10 · 상한 200 에폭, seed 0 |
| 크리깅 | OLS 추세(배경 11변수) + 시공간 지수 변동함수(대원 거리, 너겟 포함) + 국소 단순 크리깅. 모형 · k 는 val 로 선택 → 단일 모형 k = 128 |
| LightGBM | 배경 11변수 + no2_24h + 경계층 u · v, 목표 xco2 − μ_train, 학습률 0.05, 조기 종료 100, num_leaves {31, 63, 255} 중 val 로 선택 |

## 자료

원천 자료는 공개 자료이고, 저장소에는 넣지 않는다. 경로는 `src/no2xco2/config.py`.

| 자료 | 판 · 조건 | 받는 곳 | 저장소 안 처리 |
|---|---|---|---|
| OCO-2 · OCO-3 Level 2 Lite XCO₂ | v11 (B11210Ar), 동아시아 2020–2024 사운딩을 한 파일로 합친 `integrated_dataset.nc` (17,415,604행) | NASA GES DISC | `no2xco2.data.oco_nodes` → `data/raw/oco_nodes` |
| Sentinel-5P TROPOMI 대류권 NO₂ L2 | v02.04–02.08, granule 별 parquet (열 lat · lon · qa · obs_time · no2_tvcd · orbit, qa ≥ 0.66) | NASA GES DISC (S5P_L2__NO2____HiR) | `no2xco2.data.build_index` → `trop_hi_*.parquet` |
| ERA5 시간별 기압면 · 단일면 | 0.25°, 700–1000 hPa 10층 u · v · z, blh · sp · z · t2m | Copernicus CDS | `scripts/data/fetch_era5.py`, `no2xco2.data.era5_syn` |

- 두 위성 자료를 동아시아로 잘라 위 형식으로 만드는 스크립트는 이 저장소에 없다. 위 표의 형식(열 이름)을 맞추면 이후 단계는 그대로 돈다.
- 크기 (2020년 12개월, 측정): `data/raw/era5_wind` 2.2 GB · `data/raw/era5_syn` 1.3 GB · `data/processed/encoder_index` (2020-01) 376 MB. `data/processed/index` 는 사운딩 60개월 + NO₂ 화소 2020년 12개월이 4.2 GB.

## 계산 환경

| 항목 | 값 |
|---|---|
| 소프트웨어 | Python 3.10.20 · numpy 1.24.3 · pandas 2.3.3 · pyarrow 23.0.1 · torch 2.11.0 (CPU) · xarray 2025.6.1 · netCDF4 1.7.4 · h5py 3.16.0 · scipy 1.13.1 · lightgbm 4.7.0 · matplotlib 3.10.8 · cartopy 0.25.0 · cdsapi 0.7.7 |
| 글꼴 | 그림을 같은 바이트로 다시 만들려면 Pretendard(Fig. 2 · 3)와 Apple SD Gothic Neo(Fig. 1 · 2 썸네일)가 필요하다 |
| 난수 | 학습 seed 0 (torch · numpy), 달 순서 default_rng(seed·1000 + 에폭), 부트스트랩 · val 부표본 seed 0 |
| GNN 학습 (60개월) | AWS c6a.4xlarge (16 vCPU), CPU. 수송 켬 942 s/에폭 × 57 에폭, 수송 끔 682 s/에폭 × 25 에폭. 12개월 켬 187 s/에폭 × 54 에폭 |
| 기준선 (60개월) | 같은 서버, 16 스레드. 크리깅 1,533 s · 최대 RSS 7.2 GB, LightGBM 107 s · 최대 RSS 11.3 GB |
| 평가 · 지도 · 그림 | Apple M2 · 8 GB. `compare_models.py` 58 s · 0.9 GB, `grid_map_gnn.py` 124 s · 1.0 GB, `grid_map_baselines.py krige` 249 s · 1.9 GB, `fig3_maps.py` 29 s · 0.8 GB |
| 같은 코드 반복 차이 | 2020-01 · 2 에폭 학습을 두 번 돌리면 예측이 최대 7e-7 ppm 다르다 (CPU 다중 스레드 float32 합산 순서). 요약 RMSE 는 소수 여섯째 자리까지 같다 |

## 코드 구성

```
src/no2xco2/         패키지 (pip install -e .)
  config.py            격자 상수 · 자료 경로
  model.py             물리 연산자(이류 · 확산 · 감쇠) · 엣지 어텐션 · 주입 · 디코드 · 월 롤아웃
  train.py             분할 마스크 · UTC 일 블록 · 짝 부트스트랩
  train5.py            여러 달 학습 · 평가 (scripts/train.py 진입점)
  data/                사운딩 표 · ERA5 읽기 · 종관 지표 · 격자 인덱스 · 분할 · 모델 입력
scripts/
  data/                ERA5 받기 · 인덱스 · NO₂ 피처
  train.py             GNN 학습
  baselines/           크리깅 · LightGBM
  evaluate/            모델 비교 표
  maps/                사례일 격자 지도
figures/             Fig. 1–3
runs/                학습된 모델 파일 · 결과 표 (예측 · 지도 배열은 로컬)
```

## 한계

- 학습은 seed 하나, 공간 분할은 폴드 하나로 돌렸다. seed · 폴드에 따른 변동은 측정하지 않았다.
- LightGBM 과의 차이 구간(−0.068, +0.004)은 0 을 포함한다.
- 표의 R² · RMSE 는 시험 사운딩 전체(5년 · 12칸)에서 계산했으므로 계절 · 연 변화와 공간 차이가 함께 들어 있다.
- TCCON 같은 독립 지상 관측으로는 검증하지 않았다.

## 참고문헌

- Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. doi:10.1002/qj.3803
- Tadić, J. M., Qiu, X., Miller, S., Michalak, A. M. (2017). Spatio-temporal approach to moving window block kriging of satellite data v1.0. *Geoscientific Model Development*, 10, 709–720. doi:10.5194/gmd-10-709-2017
- NASA JPL. OCO-2 / OCO-3 Level 2 Lite XCO₂, v11 (B11210Ar). NASA GES DISC.
- KNMI / ESA. Sentinel-5P TROPOMI Tropospheric NO₂ Level 2 (v02.04–02.08). NASA GES DISC.
