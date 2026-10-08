"""진입점: python scripts/run_rung3.py [인자]  →  no2xco2.baselines.obs_gat 의 __main__ 실행."""
import runpy
runpy.run_module("no2xco2.baselines.obs_gat", run_name="__main__", alter_sys=True)
