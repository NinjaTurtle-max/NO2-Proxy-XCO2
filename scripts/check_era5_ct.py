"""ERA5 z100 · CarbonTracker 60개월 전수 검증 진입점 (항목 8). 실행: python scripts/check_era5_ct.py [--skip-era5] [--skip-ct]"""
import runpy

runpy.run_module("no2xco2.data.check_forcing", run_name="__main__", alter_sys=True)
