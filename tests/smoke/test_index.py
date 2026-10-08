"""5년 인덱스(D1–D4) 스모크: 2020-01 산출물이 기존 인코더 인덱스·분할 측정치와 일치하는가.
전제: data/processed/index/{trop_hi,oco}_202001.parquet (없으면 index_test 로 대체; 둘 다 없으면 skip)."""
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)


def _idx_dir():
    for d in ("data/processed/index", "data/processed/index_test"):
        if os.path.exists(f"{d}/trop_hi_202001.parquet") and os.path.exists(f"{d}/oco_202001.parquet"):
            return d
    return None


@pytest.mark.skipif(_idx_dir() is None, reason="인덱스 없음")
def test_index_202001_counts():
    """D1: qa≥0.75 화소 6,145,013 · 주입 스텝 127 · D3: 2020-01 은 step_g == step_h · D4: 사운딩 101,620 / L_train 55,376 / time fold 2 test 12,103 / space fold 0 test 9,642."""
    import pyarrow.parquet as pq
    d = _idx_dir()
    hi = pq.read_table(f"{d}/trop_hi_202001.parquet", columns=["step_g", "qa"]).to_pandas()
    assert len(hi) == 6_145_013 and float(hi.qa.min()) >= 0.75 and hi.step_g.nunique() == 127
    bg = ["bg_t2m", "bg_blh", "bg_sp"] + [c for c in ("bg_z850", "bg_thk") if c in pq.read_schema(f"{d}/oco_202001.parquet").names]  # 종관 지표 열(승인 2026-09-22)은 있으면 검사
    oc = pq.read_table(f"{d}/oco_202001.parquet", columns=["label", "step_g", "step_h", "fold_time", "fold_space", *bg]).to_pandas()
    lab = (oc.label == "L_train").to_numpy()
    assert len(oc) == 101_620 and lab.sum() == 55_376 and (oc.step_g == oc.step_h).all()
    assert int(((oc.fold_time == 2).to_numpy() & lab).sum()) == 12_103 and int(((oc.fold_space == 0).to_numpy() & lab).sum()) == 9_642
    assert np.isfinite(oc[bg].to_numpy()).all()
