"""사운딩 표 진입점 (no2xco2.data.oco_nodes). 예: python scripts/data/build_oco_nodes.py data/raw/oco_nodes"""
import runpy

runpy.run_module("no2xco2.data.oco_nodes", run_name="__main__", alter_sys=True)
