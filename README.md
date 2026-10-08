# NO2-Proxy-XCO2

TROPOMI NO₂(매일, 3.5×5.5 km)를 프록시로 써서 OCO-2/3 XCO₂(16일 반복, 듬성듬성)의 빈 곳을 채우는 물리 주입 시공간 그래프 모델. 동아시아 20–50°N, 100–150°E, 2020–2024.

- 설계 문서(결정 1–14·임계값·측정 결과): https://claude.ai/code/artifact/7a98dcda-53bc-4d07-9912-56c21c67f107
- 잠재 노드 층 설명: https://claude.ai/code/artifact/959ae3dd-2498-456c-8c67-87d00ad128e3
- 참고문헌 대장: https://claude.ai/code/artifact/44c6f4c9-f284-4419-9d9f-a581650278e5 (원본 `docs/references.md`)
- 보고 규칙: `CLAUDE.md`

## 모델 한 줄

관측(NO₂ 화소·XCO₂ 사운딩)은 원 위치 유지 → 최근접 4점 이중선형 엣지로 0.25° 고정 잠재 노드 24,321개에 주입 → 잠재 층에서 ERA5 바람 이류 + 확산(D 학습) + 감쇠(k 학습), Δt = 1 h, 48스텝 → 사운딩 자리에서 디코드: XCO₂ = μ + g(위치·시각·기상) + β·NO₂_latent + 잔차항.

## 디렉토리

```
src/no2xco2/                 import 되는 코드 (pip install -e .)
  config.py · nas.py           경로(NAS_ROOT, NO2_NAS_ROOT)·격자 상수 · NAS 대기/이동
  data/encoder_index.py        관측 → 잠재 4점 인덱스·가중·시간 스텝 (월 6 s)
  data/splits.py               7일 시간 블록 · 5° 공간 블록(버퍼 1°) · 연도 폴드
  data/oco_nodes.py            integrated_dataset.nc → 월별 parquet, label
  physics.py                   물리 전용 롤아웃 (이류·확산, 학습 없음)
  model.py                     Physics · EdgeAttention · Model · prep · run
  train.py                     nested_masks · fit_one · block_bootstrap (시드 실험·B1)
  baselines/kriging.py         rung 1 회귀 시공간 크리깅 + 거리 층
  baselines/obs_gat.py         rung 3 관측=노드 엣지 어텐션 (물리 없음)
  baselines/reanalysis.py      rung 0 CAMS EGG4 · CarbonTracker → 사운딩 보간
  eval/tccon.py                TCCON 5년 공동배치·B3·evaluate()
scripts/                     진입점 (얇음, 인자 → 모듈 __main__)
  fetch/{era5,slice_tropomi,tccon,odiac2025,carbontracker,cams}.py
  build_encoder_index · build_splits · build_oco_nodes · physics_rollout
  run_pilot · run_rung1 · run_rung3 · run_rung0 · run_tccon · build_refs_artifact
configs/    pilot_202001.yaml (실행 설정 기록; yaml 로더 미연결)
experiments/ pilot2_seeds · rung0 · rung1 · rung3 · tccon   (실행별 산출)
data/raw/   pilot_202001/ · era5_wind/ · tropomi_qa0/ · integrated_dataset.nc
data/processed/  encoder_index · splits · physics_fields · model_pilot · tropomi_qa_scan.csv
data/logs/  조달·실험 로그
docs/       references.md · results/ · nas_inventory · nas_cleanup 스크립트 · eda_scan_*
archive/    이전 설계 스크립트·정정 전 산출물 (git 제외)
```

## 자료 위치 (NAS `dataset/NO2-Proxy-XCO2/` 공유 — 경로는 환경변수 `NO2_NAS_ROOT` 또는 `configs/nas_local.txt`, 견본 `configs/nas_local.example.txt`)

| 디렉토리 | 내용 |
|---|---|
| `era5_wind_z100/`, `era5_pl_raw/` | ERA5 파생 바람·원본 (월별 nc) |
| `carbontracker_ea/` | CT2022 / CT-NRT.v2025-1 동아시아 부분집합 60개월 |
| `cams_ea/` | CAMS inversion 2020–24 · EGG4 2020 |
| `tccon_ggg2020/` | TCCON 5 사이트 |
| `oco_nodes/` | OCO 사운딩 노드 테이블 (year_month 파티션, 17,415,604행) |
| `odiac2025/` | ODIAC2025 2024년 |
| `../XCO2연구 데이터/_tropomi_ea_v2/` | TROPOMI NO₂ granule parquet 8,011개 (본 실험 원천) |

## 실행 (파일럿 2020-01)

```
pip install -e .
python scripts/build_encoder_index.py --month 202001
python scripts/build_splits.py --month 202001
python scripts/physics_rollout.py --mode continuous --save-fields
python scripts/run_pilot.py --configs time:2,space:0 --seeds 0,1,2
python scripts/run_rung1.py ; python scripts/run_rung3.py ; python scripts/run_rung0.py ; python scripts/run_tccon.py
```

환경: `/opt/miniconda3/envs/NO2_Proxy_XCO2/bin/python` (torch 2.11, xarray, pyarrow, scipy). NAS 조달은 NAS PC에서 `nas_fetch_win.zip`으로 실행.

## 현재 상태 (2026-09-22)

- 자료: 60개월 인덱스·ERA5 z100·종관 파생(era5_syn)·CT 전부 로컬/NAS 완비. TROPOSIF 슬라이스는 PC 진행 중
- 정정: ERA5 위도 반전(09-22) → 파일럿 2차·rung 1·3 수치 무효. 유효 측정·승인 이력은 설계 문서 08b절과 `docs/_Log.md`
- 2020년 12개월 (seed 0, 30 ep): time2 물리 1.457 (R² 0.680, 배경 9변수) · space0 물리 1.571 = 절제 1.571 (B1 미통과) · CT 편향제거 대조 1.296 / 1.361
- 세션 체계: AI/Algorithm · Data/GIS·Viz · QA · 연구책임자(도메인) — 인수 기록 `docs/handover_*_2026-09-22.md`, 시각화 분석 `docs/viz_analysis_2026-09-22.md`, 계획 `docs/plan_2026-09-22.md`
- 미구현: 5년 학습(계획 B) · rung 2 ConvLSTM+U-Net · β(계절·지역) · SIF 입력(계획 C)
