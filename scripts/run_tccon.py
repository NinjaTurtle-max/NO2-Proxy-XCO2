"""진입점: python scripts/run_tccon.py [인자]  →  no2xco2.eval.tccon 의 __main__ 실행."""
import runpy
runpy.run_module("no2xco2.eval.tccon", run_name="__main__", alter_sys=True)
