"""학습 · 평가 진입점 (no2xco2.train5). 예: python scripts/train.py --months 202001-202412 --configs space:0 --seeds 0 --out runs/gnn_m60_on"""
import runpy

runpy.run_module("no2xco2.train5", run_name="__main__", alter_sys=True)
