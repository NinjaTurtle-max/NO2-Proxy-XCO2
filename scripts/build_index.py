"""5년 전처리 인덱스 진입점 (D1–D4). 예: python scripts/build_index.py --months 202001 --tropomi-dir data/raw/pilot_202001/tropomi --oco-dir data/raw/pilot_202001 --out data/processed/index_test"""
import runpy

runpy.run_module("no2xco2.data.build_index", run_name="__main__", alter_sys=True)
