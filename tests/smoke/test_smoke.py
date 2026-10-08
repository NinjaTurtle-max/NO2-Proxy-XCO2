"""스모크 테스트: 2020-01 로컬 자산으로 '측정된 수치가 바뀌지 않았는가'를 고정한다.
전제 파일: data/raw/pilot_202001/oco_nodes, data/raw/era5_wind/era5_wind_202001_z100.nc,
          data/processed/index/{trop_hi,oco}_202001.parquet + no2_stats.json (없으면 해당 테스트 skip). 인덱스 카운트는 test_index.py.
실행: pytest -q tests/smoke   (전체) · Stop hook 은 -m "not slow" (빠른 스모크만).
slow 마커 = rung0(외부 자료) · train5 1에폭 학습(D-6 결정 (c), 사용자 승인 2026-09-24 02:3x — 학습 런과의 자원 경합 방지, QA 가 수동 1회 실행: docs/tests.json S-4).
기준값은 2026-09-14 측정치 (docs/results/, 설계 문서 잔여 표).
"""
import glob, os, subprocess, sys
import numpy as np, pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
os.chdir(ROOT)
need = lambda *p: pytest.mark.skipif(not all(os.path.exists(os.path.join(ROOT, x)) for x in p), reason="파일럿 자산 없음")


@need("data/raw/pilot_202001/oco_nodes")
def test_splits_counts():
    """13 분할: 2020-01 사운딩 101,620 · L_train 55,376 · time fold 2 test 12,103."""
    import pyarrow.parquet as pq
    import no2xco2.data.splits as S
    df = pq.read_table(sorted(glob.glob("data/raw/pilot_202001/oco_nodes/*.parquet")), columns=["row_idx", "latitude", "longitude", "time", "label"]).to_pandas()
    df = S.assign(df); lab = (df.label == "L_train").to_numpy()
    assert len(df) == 101_620 and lab.sum() == 55_376
    assert int(((df.fold_time == 2).to_numpy() & lab).sum()) == 12_103


@need("data/processed/index/trop_hi_202001.parquet", "data/raw/era5_wind/era5_wind_202001_z100.nc")
def test_physics_freerun_coverage(tmp_path):
    """14 물리 롤아웃 자유 전파 (1/11 03 UTC, D=5000): 48 h 커버리지 0.477 · 질량비 1.419 (2026-09-22 위도 오름차순 정정 판, 승인됨; 구 기준 0.404/0.730 은 남북 뒤집힌 바람)."""
    import pandas as pd
    r = subprocess.run([PY, "scripts/physics_rollout.py", "--mode", "free-run", "--start", "240", "--out", str(tmp_path)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    df = pd.read_csv(tmp_path / "physics_freerun_202001_s243.csv"); r48 = df[df.lag == 48].iloc[0]; base = df[df.lag == 0].iloc[0]
    assert abs(r48.coverage - 0.477) < 0.002 and abs(r48.mass / base.mass - 1.419) < 0.002


@pytest.mark.slow
@need("experiments/rung1/rung1_time2.parquet", "data/raw/pilot_202001/egg4_xco2_202001.nc")
def test_rung0_rmse(tmp_path):
    """20 rung 0: EGG4 vs OCO L_train RMSE 1.330 · CT 1.346 (2020-01)."""
    import json
    r = subprocess.run([PY, "scripts/run_rung0.py", "--out", str(tmp_path)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    s = json.load(open(tmp_path / "summary.json"))
    assert abs(s["EGG4"]["rmse"] - 1.330) < 0.002 and abs(s["CT"]["rmse"] - 1.346) < 0.002


@pytest.mark.slow  # D-6 (c): hook 스모크에서 제외, 수동 실행 `pytest -q -m slow tests/smoke/test_smoke.py::test_train5_one_epoch_runs`
@need("data/processed/index/trop_hi_202001.parquet", "data/processed/index/oco_202001.parquet", "data/processed/index/no2_stats.json", "data/raw/era5_wind/era5_wind_202001_z100.nc")
def test_train5_one_epoch_runs(tmp_path):
    """train5 (D2–D4 로더·warm start) 1 에폭: 실행 완료 · 분할 25,450/9,090/12,103 · test RMSE 유한 (수치 고정은 시드 실험에서)."""
    import pandas as pd
    r = subprocess.run([PY, "scripts/run_train5.py", "--months", "202001", "--configs", "time:2", "--seeds", "0", "--epochs", "1", "--out", str(tmp_path)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    assert "train 25,450 / val 9,090 / test 12,103" in r.stdout
    s = pd.read_csv(tmp_path / "summary.csv"); row = s.iloc[0]
    assert np.isfinite(row.test_clean) and 1.0 < row.test_clean < 3.0


_T5_NEED = ("data/processed/index/trop_hi_202001.parquet", "data/processed/index/oco_202001.parquet", "data/processed/index/no2_stats.json", "data/raw/era5_wind/era5_wind_202001_z100.nc")


@pytest.mark.slow  # 결정 7-a 옵션 경로 (연구책임자 조건 2026-09-24): 1에폭 실행 · 월별 β 열 12개 · 훈련 월(1월) β 유한 · 나머지 달 NaN
@need(*_T5_NEED)
def test_train5_beta_month_runs(tmp_path):
    import pandas as pd
    r = subprocess.run([PY, "scripts/run_train5.py", "--months", "202001", "--configs", "time:2", "--seeds", "0", "--epochs", "1", "--beta-mode", "month", "--out", str(tmp_path)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    row = pd.read_csv(tmp_path / "summary.csv").iloc[0]
    assert row.beta_mode == "month" and all(f"beta_m{i:02d}" in row.index for i in range(1, 13))
    assert np.isfinite(row.beta_m01) and np.isfinite(row.beta) and np.isfinite(row.test_clean) and 1.0 < row.test_clean < 3.0
    assert all(np.isnan(row[f"beta_m{i:02d}"]) for i in range(2, 13))  # 훈련 행 없는 달 = NaN (QA X4, 미학습 초기값을 값으로 기록하지 않음)
    assert abs(row.beta - row.beta_m01) < 1e-12  # 훈련 행 있는 달만의 가중 평균 = 1월 값


@pytest.mark.slow  # 결정 13-a 옵션 경로 (연구책임자 조건 2026-09-24): 1에폭 실행 · weight 열 = scene · test RMSE 유한
@need(*_T5_NEED)
def test_train5_weight_scene_runs(tmp_path):
    import pandas as pd
    r = subprocess.run([PY, "scripts/run_train5.py", "--months", "202001", "--configs", "time:2", "--seeds", "0", "--epochs", "1", "--weight", "scene", "--out", str(tmp_path)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    row = pd.read_csv(tmp_path / "summary.csv").iloc[0]
    assert row.weight == "scene" and np.isfinite(row.test_clean) and 1.0 < row.test_clean < 3.0
