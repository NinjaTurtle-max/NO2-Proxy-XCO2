"""한 달 화소 · 사운딩 격자 인덱스 진입점 (no2xco2.data.encoder_index). 예: python scripts/data/build_encoder_index.py --month 202001"""
import runpy

runpy.run_module("no2xco2.data.encoder_index", run_name="__main__", alter_sys=True)
