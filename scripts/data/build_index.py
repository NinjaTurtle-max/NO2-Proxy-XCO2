"""인덱스 진입점 (no2xco2.data.build_index). 예: python scripts/data/build_index.py --months 202001 202002"""
import runpy

runpy.run_module("no2xco2.data.build_index", run_name="__main__", alter_sys=True)
