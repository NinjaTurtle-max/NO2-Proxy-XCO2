"""진입점: python scripts/run_pilot.py [인자]  →  no2xco2.train 의 __main__ 실행."""
import runpy
runpy.run_module("no2xco2.train", run_name="__main__", alter_sys=True)
