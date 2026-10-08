"""진입점: python scripts/build_oco_nodes.py [인자]  →  no2xco2.data.oco_nodes 의 __main__ 실행."""
import runpy
runpy.run_module("no2xco2.data.oco_nodes", run_name="__main__", alter_sys=True)
