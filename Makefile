# 재현 순서. 저장소 루트에서 실행한다. 자료(data/)를 먼저 준비한다 (README "자료").
# make all        : 학습된 모델 파일로 평가 · 지도 · 그림 (학습 없이 결과 재현)
# make train      : 60개월 GNN 수송 켬 · 끔 다시 학습 (CPU 16 vCPU 기준 켬 약 15 h · 끔 약 5 h)
PY ?= python
export PYTHONPATH := src

MONTHS ?= 202001-202412

.PHONY: all data features train baselines evaluate maps figures check

all: evaluate maps figures

data:
	$(PY) scripts/data/fetch_era5.py --range 2020-01 2024-12 --zmin 100
	$(PY) -m no2xco2.data.era5_syn
	$(PY) scripts/data/build_oco_nodes.py data/raw/oco_nodes
	$(PY) scripts/data/build_index.py
	$(PY) scripts/data/build_encoder_index.py --month 202001

features:
	$(PY) scripts/data/build_no2_features.py --months $(MONTHS) --out runs/no2_features

train:
	$(PY) scripts/train.py --months $(MONTHS) --configs space:0 --seeds 0 --out runs/gnn_m60_on --ckpt-dir runs/gnn_m60_on/ckpt --resume
	$(PY) scripts/train.py --months $(MONTHS) --configs space:0 --seeds 0 --nophys-only --out runs/gnn_m60_off --ckpt-dir runs/gnn_m60_off/ckpt --resume
	$(PY) scripts/train.py --months 202001-202012 --configs space:0 --seeds 0 --out runs/gnn_m12_on

baselines: features
	$(PY) scripts/baselines/run_baselines.py --months $(MONTHS) --out runs

evaluate:
	$(PY) scripts/evaluate/compare_models.py --runs runs --out runs/tables

maps:
	$(PY) scripts/maps/grid_map_gnn.py
	$(PY) scripts/maps/grid_map_baselines.py krige
	$(PY) scripts/maps/grid_map_baselines.py feats
	$(PY) scripts/maps/grid_map_baselines.py lgbm

figures:
	$(PY) figures/region_points.py
	mkdir -p figures/out
	$(PY) figures/fig1_region.py runs/region_pts.npz figures/out/fig1_region.png
	$(PY) figures/fig2/thumbs.py
	$(PY) figures/fig2/thumbs2.py
	$(PY) figures/fig2/thumb_target_train.py
	$(PY) figures/fig2/tiles_final.py
	$(PY) figures/fig3_maps.py

check:
	$(PY) scripts/baselines/kriging.py
	$(PY) scripts/data/build_no2_features.py --selftest
