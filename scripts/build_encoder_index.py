"""진입점: python scripts/build_encoder_index.py [인자]  →  no2xco2.data.encoder_index 의 __main__ 실행."""
import runpy
runpy.run_module("no2xco2.data.encoder_index", run_name="__main__", alter_sys=True)
