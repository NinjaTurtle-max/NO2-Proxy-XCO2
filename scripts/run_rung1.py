"""진입점: python scripts/run_rung1.py [인자]  →  no2xco2.baselines.kriging 의 __main__ 실행."""
import runpy
runpy.run_module("no2xco2.baselines.kriging", run_name="__main__", alter_sys=True)
