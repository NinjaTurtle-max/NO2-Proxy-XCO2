"""진입점: python scripts/physics_rollout.py [인자]  →  no2xco2.physics 의 __main__ 실행."""
import runpy
runpy.run_module("no2xco2.physics", run_name="__main__", alter_sys=True)
