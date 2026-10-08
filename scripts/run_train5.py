"""5년 학습 루프 진입점. 예: python scripts/run_train5.py --months 202001-202412 --configs time:2 --seeds 0 --ablation"""
import runpy

runpy.run_module("no2xco2.train5", run_name="__main__", alter_sys=True)
