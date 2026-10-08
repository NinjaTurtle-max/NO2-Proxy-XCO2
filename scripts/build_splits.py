"""진입점: python scripts/build_splits.py [인자]  →  no2xco2.data.splits 의 __main__ 실행."""
import runpy
runpy.run_module("no2xco2.data.splits", run_name="__main__", alter_sys=True)
