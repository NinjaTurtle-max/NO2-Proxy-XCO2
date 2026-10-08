"""시각화 스모크 (2026-09-22 설계): 3그림 생성 + 수용 기준.
- strata: 2020-01 3시드 층 계산의 '전체' RMSE 가 train5 summary.csv 의 test_clean 과 일치 (행 정렬·분할 검증)
- monthly / lc: 그림 CSV 사본 = 원본 값 (변환 없음)
전제: experiments/{ct_compare_2020, train5_202001_fixed*, lc2020_*} 산출물. 없으면 skip."""
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)  # 세션 전체 cwd 를 바꾸지 않음
from no2xco2.viz import runs as R  # noqa: E402  (경로 등록부 — plot_results 와 같은 경로, QA B8)

from no2xco2.config import INDEX_DIR  # noqa: E402

J = os.path.join  # 절대 경로 NO2_EXP_DIR 도 그대로 살도록 f"{ROOT}/…" 대신 join (QA N2)
IDX = J(ROOT, INDEX_DIR); CT = J(ROOT, R.CT_DIR); CT12 = J(ROOT, R.CT_DIR_LC12); SEED_DIRS = [J(ROOT, d) for d in R.SEED_DIRS]  # CT12: lc2020_12m 결과와 대조할 ct_compare
LC = {k: J(ROOT, d) for k, d in R.LC_DIRS.items()}


def _have(*paths):
    return all(os.path.exists(p) for p in paths)


@pytest.mark.skipif(not _have(f"{IDX}/oco_202001.parquet", f"{CT}/strata_space0.csv", *[f"{d}/summary.csv" for d in SEED_DIRS],
                             *[J(ROOT, d, f"pred_space0_s{s}_{m}.parquet") for s, d in R.SEED_DIR.items() for m in ("phys", "nophys")]), reason="입력 없음")
def test_strata_matches_summary(tmp_path):
    from no2xco2.viz import results2020 as V
    sd = V.strata_202001_3seed(IDX, SEED_DIRS)
    summ = pd.concat([pd.read_csv(f"{d}/summary.csv") for d in SEED_DIRS if os.path.exists(f"{d}/summary.csv")])
    summ = summ[(summ.scheme == "space") & (summ.fold == 0) & summ.physics.astype(str).isin(["True", "False"])]
    tot = sd[sd["층 km"] == "전체"].set_index(["model", "seed"]).rmse
    n = 0
    for _, r in summ.iterrows():
        key = ("ours_phys" if str(r.physics) == "True" else "ours_nophys", int(r.seed))
        if key in tot.index:
            assert abs(tot[key] - r.test_clean) < 5e-4, f"{key}: 층 합산 {tot[key]:.4f} ≠ summary {r.test_clean:.4f}"; n += 1
    assert n == 6  # 3시드 × 물리/절제
    png, csv = V.plot_strata(f"{CT}/strata_space0.csv", sd, "test", str(tmp_path))
    assert os.path.getsize(png) > 10_000 and os.path.exists(csv)


@pytest.mark.skipif(not _have(f"{CT}/monthly_time2.csv"), reason="입력 없음")
def test_monthly_csv_identity(tmp_path):
    from no2xco2.viz import results2020 as V
    png, csv = V.plot_monthly(f"{CT}/monthly_time2.csv", "test", str(tmp_path))
    a = pd.read_csv(f"{CT}/monthly_time2.csv"); b = pd.read_csv(csv)
    assert len(a) == 12 and np.allclose(a.to_numpy(float), b[a.columns].to_numpy(float)) and os.path.getsize(png) > 10_000


@pytest.mark.skipif(not _have(*[J(ROOT, x) for x in R.inputs("lc", INDEX_DIR)]), reason="입력 없음")  # plot_results 와 같은 입력 목록 (QA Y5·Y7)
def test_learning_curve_values(tmp_path):
    from no2xco2.viz import results2020 as V
    png, csv = V.plot_learning_curve(LC, "test", str(tmp_path)); t = pd.read_csv(csv); assert (t.n_test == 124_943).all()
    for n_m, d in LC.items():
        s = pd.read_csv(f"{d}/summary.csv"); s = s[(s.physics.astype(str) == "True") & (s.scheme == "time") & (s.fold == 2)]; assert len(s) == 1; s = s.iloc[0]; r = t[t.train_months == n_m].iloc[0]
        assert abs(r.test_clean - s.test_clean) < 1e-9 and abs(r.beta - s.beta) < 1e-9
    assert os.path.getsize(png) > 10_000


T2_NEW = J(ROOT, R.TIME2_NEW); T2_OLD = J(ROOT, R.TIME2_OLD)
S0 = {k: J(ROOT, v) for k, v in R.SPACE0.items()}
MONTHS12 = [f"2020{m:02d}" for m in range(1, 13)]


def _summary(d, physics, scheme, fold):
    s = pd.read_csv(J(ROOT, R.exp(d), "summary.csv")); s = s[(s.physics.astype(str) == str(physics)) & (s.scheme == scheme) & (s.fold == fold)]
    assert len(s) == 1, f"{d}: {scheme}:{fold} physics={physics} 행 {len(s)}개"  # QA B10: 폴드 필터 없는 .iloc[0] 금지
    return s.iloc[0]


@pytest.mark.skipif(not _have(T2_NEW, S0["phys"], S0["nophys"], f"{CT12}/summary.csv", J(ROOT, R.exp("lc2020_12m"), "summary.csv"), J(ROOT, R.exp("train5_2020_space0"), "summary.csv"),
                             *[f"{IDX}/oco_{m}.parquet" for m in MONTHS12]), reason="입력 없음")
def test_residual_map_matches_summary(tmp_path):
    """수용 기준: 조인 후 테스트 행 RMSE = summary test_clean_all · 셀 n 합 + 격자 밖 = 테스트 수 · 셀 평균의 n 가중 평균 = 전체 편향.
    전체 편향의 대조는 ct_compare summary(ours_phys, time) — lc2020_12m 의 test_bias 는 --eval-months(10–12월) 구간 값이라 폴드 전체가 아님."""
    from no2xco2.viz import results2020 as V
    g, meta = V.residual_grid(V.load_test_rows(T2_NEW, IDX, "time", 2)); s = _summary("lc2020_12m", True, "time", 2)
    ct = V.ct_summary_row(pd.read_csv(f"{CT12}/summary.csv"), "ours_phys", "time", 2)  # 폴드·표본 필터 + 1행 단언 (QA N10·Y1)
    assert abs(meta["rmse"] - s.test_clean_all) < 5e-4 and abs(meta["bias"] - ct.bias) < 5e-4 and meta["n_test"] == 437_972 and meta["n_test"] == ct.n
    assert g.n.sum() + meta["n_outside"] + meta["n_nan"] == meta["n_test"] and abs(meta["bias_grid"] - meta["bias"]) < 1e-6  # 격자 밖 0 이므로 격자 가중 평균 = 전체 편향
    gp, mp = V.residual_grid(V.load_test_rows(S0["phys"], IDX, "space", 0)); gn, mn = V.residual_grid(V.load_test_rows(S0["nophys"], IDX, "space", 0))
    assert abs(mp["rmse"] - _summary("train5_2020_space0", True, "space", 0).test_clean_all) < 5e-4 and abs(mn["rmse"] - _summary("train5_2020_space0", False, "space", 0).test_clean_all) < 5e-4
    d = V.rmse_diff(gp, gn); assert len(d) == len(gp)
    png, csv = V.plot_residual_map(gp, mp, "t", "resmap_test", "test", str(tmp_path), diff=d, stat_run="물리 런"); assert os.path.getsize(png) > 10_000 and os.path.exists(csv)


@pytest.mark.skipif(not _have(T2_NEW, T2_OLD, f"{CT12}/summary.csv", J(ROOT, R.exp("lc2020_12m"), "summary.csv"), J(ROOT, R.exp("train5_2020"), "summary.csv"),
                             *[f"{IDX}/oco_{m}.parquet" for m in MONTHS12]), reason="입력 없음")
def test_monthly_r2_matches_summary(tmp_path):
    """수용 기준: 전월 합산 RMSE = 각 summary test_clean(구 1.976·신 1.457) · 신 전체 R² = ct_compare summary ours_phys r2 · 구 전체 R² 0.411 (작업 대장 2026-09-22)."""
    from no2xco2.viz import results2020 as V
    t = V.monthly_compare({"old": T2_OLD, "new": T2_NEW}, IDX); tot = t[t["월"] == 0].set_index("run")
    assert abs(tot.loc["old", "rmse"] - _summary("train5_2020", True, "time", 2).test_clean) < 5e-4 and abs(tot.loc["new", "rmse"] - _summary("lc2020_12m", True, "time", 2).test_clean_all) < 5e-4
    r2_ct = float(V.ct_summary_row(pd.read_csv(f"{CT12}/summary.csv"), "ours_phys", "time", 2).r2)  # QA N10·Y1
    assert abs(tot.loc["new", "r2"] - r2_ct) < 5e-4 and abs(tot.loc["old", "r2"] - 0.411) < 5e-4
    assert sorted(t[t["월"] > 0]["월"].unique().tolist()) == list(range(1, 13))
    png, csv = V.plot_monthly_r2(t, {"old": "구", "new": "신"}, {"old": "#a09f98", "new": "#2a78d6"}, "t", "r2_test", "test", str(tmp_path)); assert os.path.getsize(png) > 10_000


@pytest.mark.skipif(not _have(*[J(ROOT, d, "summary.csv") for _, d, *_ in R.B1_ROWS]), reason="입력 없음")
def test_b1_forest_matches_summary(tmp_path):
    """V-C 수용 기준: 행 8 · 값 = summary 무변환 · pass_B1 == (CI 하한 > 0) 8/8."""
    from no2xco2.viz import results2020 as V
    rows = [(lb, J(ROOT, d), sc, f, nb) for lb, d, sc, f, nb in R.B1_ROWS]
    t = V.b1_table(rows); assert len(t) == 10 and (t.pass_B1 == t.pass_def).all()  # 8 + A-2 (09-26) + A-3 (09-28)
    for r in t.itertuples():
        s = pd.read_csv(J(ROOT, R.exp(r.run), "summary.csv")); s = s[(s.scheme == r.scheme) & (s.fold == r.fold) & (s.seed == r.seed)]
        b = s[s.physics.astype(str) == "B1"].iloc[0]; p = s[s.physics.astype(str) == "True"].iloc[0]
        assert (r.delta, r.ci_lo, r.ci_hi, r.rmse_phys) == (b.delta, b.ci_lo, b.ci_hi, p.test_clean)
    png, csv = V.plot_b1_forest(t, "test", str(tmp_path)); assert os.path.getsize(png) > 10_000 and len(pd.read_csv(csv)) == 10


@pytest.mark.skipif(not _have(J(ROOT, R.FREERUN_DIR, f"physics_freerun_202001_s{R.FREERUN_REF}.csv")), reason="입력 없음")
def test_freerun_matches_reference(tmp_path):
    """V-D 수용 기준: 기준 시작(s243) lag 48 커버리지 0.477 ± 0.002 · 질량비 1.419 ± 0.002 (test_smoke 물리 기준값과 동일)."""
    from no2xco2.viz import results2020 as V
    t = V.freerun_table(J(ROOT, R.FREERUN_DIR)); r = t[(t.start == R.FREERUN_REF) & (t.lag == 48)].iloc[0]
    assert abs(r.coverage - 0.477) < 0.002 and abs(r.mass_ratio - 1.419) < 0.002
    png, csv = V.plot_freerun(t, R.FREERUN_REF, "test", str(tmp_path)); assert os.path.getsize(png) > 10_000


def test_grid_constants_and_no_overwrite(tmp_path):
    """W9: viz 격자 상수 = era5·encoder_index 값. W1: save() 는 기존 파일을 덮어쓰지 않는다 (N-2)."""
    import matplotlib.pyplot as plt
    from no2xco2.data.encoder_index import NLON
    from no2xco2.data.era5 import NLAT
    from no2xco2.viz import results2020 as V
    from no2xco2.viz import save
    assert (V.NLAT, V.NLON) == (NLAT, NLON)
    save(plt.figure(), pd.DataFrame({"a": [1]}), "x", "d", str(tmp_path))
    with pytest.raises(FileExistsError):
        save(plt.figure(), pd.DataFrame({"a": [1]}), "x", "d", str(tmp_path))


@pytest.mark.skipif(not _have(f"{CT}/monthly_time2.csv", f"{CT}/strata_space0.csv"), reason="입력 없음")
def test_ct_d3_format_reads(tmp_path):
    """QA Y1: ct_compare 두 형식을 다 읽는다 — 공통 열을 뺀 구 형식('old')과 공통 열이 있는 D-3 판('d3', 09-24 02:5x 합의) 각각 그림 생성."""
    from no2xco2.viz import results2020 as V
    mt = pd.read_csv(f"{CT}/monthly_time2.csv"); mt = mt[[c for c in mt.columns if not c.endswith("_공통")]]; assert V.ct_format(mt) == "old"  # 공통 열을 빼면 구 형식
    g0 = tmp_path / "old.csv"; mt.to_csv(g0, index=False); png, _ = V.plot_monthly(str(g0), "old", str(tmp_path)); assert os.path.getsize(png) > 10_000
    mt["n_공통"] = mt.n - 1; mt["SD_공통"] = mt.SD; mt["ours_phys_rmse_공통"] = mt.ours_phys_rmse + 0.01; mt["ours_phys_bias_공통"] = mt.ours_phys_bias
    f = tmp_path / "m.csv"; mt.to_csv(f, index=False); assert V.ct_format(mt) == "d3"
    png, csv = V.plot_monthly(str(f), "test", str(tmp_path)); assert os.path.getsize(png) > 10_000 and "ours_phys_rmse_공통" in pd.read_csv(csv).columns
    st = pd.read_csv(f"{CT}/strata_space0.csv"); st = st[[c for c in st.columns if not c.endswith("_공통")]]; st["n_공통"] = st.n; st["ours_phys_공통"] = st.ours_phys
    if "ours_nophys" in st.columns:
        st["ours_nophys_공통"] = st.ours_nophys
    g = tmp_path / "s.csv"; st.to_csv(g, index=False)
    sd = pd.DataFrame([{"층 km": "67.5–250", "n": 10, "model": m, "seed": s, "rmse": 1.0} for m in ("ours_phys", "ours_nophys") for s in (0, 1)])
    png, _ = V.plot_strata(str(g), sd, "test", str(tmp_path)); assert os.path.getsize(png) > 10_000


@pytest.mark.skipif(not _have(J(ROOT, R.CT_DIR, "monthly_time2.csv")), reason="입력 없음")
def test_r2_vs_ct_values(tmp_path):
    """E 수용: 공통 표본 정의(RMSE·SD 모두 공통)의 연구책임자 세션 재계산값 — 우리 3월 −0.7771 · 4월 −1.8016 · 7월 +0.7401 / CT 3월 −0.1462 · 9월 −0.6974.
    09-28 정정: 처음 전달된 소수 2자리 값(3월 −0.77)은 우리 쪽을 전체 표본으로 계산한 값이라 정의가 달랐음 (연구책임자 세션 확인). 정의는 E 설계 그대로."""
    from no2xco2.viz import results2020 as V
    t = V.r2_vs_ct_table(J(ROOT, R.CT_DIR, "monthly_time2.csv")).set_index("월")
    for m, col, v in ((3, "r2_ours", -0.7771), (4, "r2_ours", -1.8016), (7, "r2_ours", 0.7401), (3, "r2_ct", -0.1462), (9, "r2_ct", -0.6974)):
        assert abs(t.loc[m, col] - v) < 0.001, f"{m}월 {col} {t.loc[m, col]:.4f} ≠ {v}"
    png, _ = V.plot_r2_vs_ct(t.reset_index(), "t", "test", str(tmp_path)); assert os.path.getsize(png) > 10_000


@pytest.mark.skipif(not _have(*[J(ROOT, x) for x in R.inputs("runs", INDEX_DIR)]), reason="입력 없음")
def test_runs_a_totals_match_summary():
    """F 수용: 런별 전체 RMSE = 각 summary test_clean (같은 테스트 행)."""
    from no2xco2.viz import results2020 as V
    t = V.monthly_compare({k: J(ROOT, d, "pred_time2_s0_phys.parquet") for k, d in R.RUNS_A.items()}, IDX); tot = t[t["월"] == 0].set_index("run")
    for k, d in R.RUNS_A.items():
        s = pd.read_csv(J(ROOT, d, "summary.csv")); s = s[s.physics.astype(str) == "True"].iloc[0]
        assert abs(tot.loc[k, "rmse"] - s.test_clean) < 5e-4, f"{k}: {tot.loc[k, 'rmse']:.4f} ≠ {s.test_clean:.4f}"
