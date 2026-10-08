"""2020년 결과 그림 진입점. 예: python scripts/plot_results.py --fig all --date 2026-09-22
--fig: strata(거리 층) · monthly(월별 vs CT) · lc(학습 곡선) · resmap(잔차 지도 3장) · r2(결정 1 전후) · syn(종관 11변수 런 판 3장) · b1(B1 forest) · r2ct(월별 R² vs CT) · runs(12개월 런 비교 A-2·7a·13a·A-3) · freerun(물리 자유전파; 입력은 physics_rollout.py --mode free-run --out experiments/physics_freerun_202001) · all.
파일명 날짜 = 생성일 (N-2, 사용자 승인 2026-09-24): --date 로 과거 날짜를 주지 말 것 — 구 파일을 덮어쓴다.
입력 경로는 no2xco2.viz.runs (NO2_EXP_DIR 로 루트 변경). 입력 파일이 없는 그림은 건너뛰고 이유를 출력한다."""
import argparse
import datetime as dt
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # 헤드리스 (viz 모듈은 백엔드를 건드리지 않음)

from no2xco2.config import INDEX_DIR
import no2xco2.viz as VZ
from no2xco2.viz import RESULTS_DIR
from no2xco2.viz import results2020 as V
from no2xco2.viz.runs import RUNS_A, CT_LABEL, OUTPUTS, inputs, FREERUN_D, B1_ROWS, FREERUN_DIR, FREERUN_REF, CT_DIR, CT_SYN, LC_DIRS, SEED_DIRS, SPACE0, TIME2_NEW, TIME2_OLD, TIME2_SYN

FIGS = ["strata", "monthly", "lc", "resmap", "r2", "syn", "b1", "freerun", "r2ct", "runs"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fig", default="all", help=f"{'|'.join(FIGS)}|all (쉼표 가능)")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="파일명 날짜 = 생성일(N-2). 오늘이 아니면 --allow-past-date 와 함께만"); ap.add_argument("--allow-past-date", action="store_true", help="오늘이 아닌 --date 허용 (덮어쓰기는 별도 --overwrite, QA Y2)"); ap.add_argument("--overwrite", action="store_true", help="기존 결과 파일 덮어쓰기 허용 (N-2 예외, 명시할 때만)"); ap.add_argument("--suffix", default="", help="같은 날 재생성 파일명 접미사 (예: _r2)"); ap.add_argument("--idx-dir", default=INDEX_DIR); ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--ct-dir", default=CT_DIR); ap.add_argument("--ct-syn-dir", default=CT_SYN, help="syn 그림의 ct_compare 산출 (QA N3)")
    ALL = set(FIGS); a = ap.parse_args(); figs = ALL if a.fig == "all" else set(a.fig.split(","))
    if not figs <= ALL:
        ap.error(f"--fig 알 수 없는 항목: {sorted(figs - ALL)} (가능: {FIGS}, all)")
    if a.date != dt.date.today().isoformat() and not a.allow_past_date:
        ap.error(f"--date {a.date} ≠ 오늘 — N-2 생성일 규약. 의도했으면 --allow-past-date")
    for f in [f for f in FIGS if f in figs]:
        miss = [x for x in inputs(f, a.idx_dir, a.ct_dir, a.ct_syn_dir) if not os.path.exists(x)]
        if miss:
            print(f"[건너뜀] {f}: 입력 없음 {miss}"); figs.discard(f)
    with VZ.save_options(overwrite=a.overwrite, suffix=a.suffix):  # QA W1·Y6
        if not a.overwrite:  # 그리기 전에 출력 충돌을 전부 검사 — 중간에 멈춰 일부만 저장되는 일 방지 (QA Y3)
            clash = [p for f in FIGS if f in figs for n in OUTPUTS[f] for p in VZ.out_paths(n, a.date, a.out) if os.path.exists(p)]
            if clash:
                ap.error(f"출력 파일이 이미 있음 {clash} — --suffix _r2 등 또는 의도했으면 --overwrite")
        run(a, figs)


def run(a, figs):
    if "strata" in figs:
        sd = V.strata_202001_3seed(a.idx_dir, SEED_DIRS); print(sd.pivot_table(index="층 km", columns=["model", "seed"], values="rmse").round(3).to_string())  # 파일은 정확히 1곳 (중복 시 예외)
        print(V.plot_strata(f"{a.ct_dir}/strata_space0.csv", sd, a.date, a.out))
    if "monthly" in figs:
        ttl = f"월별 테스트 RMSE·편향 — 2020 · time fold 2 · seed 0 (우리 = {CT_LABEL['time']})" if a.ct_dir == CT_DIR else None  # 다른 --ct-dir 이면 기본 제목
        print(V.plot_monthly(f"{a.ct_dir}/monthly_time2.csv", a.date, a.out, title=ttl))
    if "resmap" in figs:  # ① time2 물리 ② space0 물리 ③ space0 절제 − 물리 RMSE 차
        g, meta = V.residual_grid(V.load_test_rows(TIME2_NEW, a.idx_dir, "time", 2)); print("time2", meta)
        print(V.plot_residual_map(g, meta, "잔차 지도 — 2020 · time fold 2 · seed 0 · 물리 (lc2020_12m, 결정 1 배경)", "resmap_time2_phys", a.date, a.out))
        gp, mp = V.residual_grid(V.load_test_rows(SPACE0["phys"], a.idx_dir, "space", 0)); print("space0 phys", mp)
        print(V.plot_residual_map(gp, mp, "잔차 지도 — 2020 · space fold 0 · seed 0 · 물리 (train5_2020_space0)", "resmap_space0_phys", a.date, a.out))
        gn, mn = V.residual_grid(V.load_test_rows(SPACE0["nophys"], a.idx_dir, "space", 0)); print("space0 nophys", mn)
        d = V.rmse_diff(gp, gn); print(f"셀별 RMSE 차 (절제−물리): 중앙값 {d['diff'].median():+.3f}, + 셀 비율 {(d['diff'] > 0).mean():.1%} (n {len(d):,})")
        print(V.plot_residual_map(gp, mp, f"셀별 RMSE 차 — 2020 · space fold 0 · seed 0 · 절제 {mn['rmse']:.3f} vs 물리 {mp['rmse']:.3f}", "resmap_space0_diff", a.date, a.out, diff=d, stat_run="물리 런"))
    if "r2" in figs:
        t = V.monthly_compare({"old": TIME2_OLD, "new": TIME2_NEW}, a.idx_dir); print(t.round(3).to_string(index=False))
        print(V.plot_monthly_r2(t, {"old": "구 배경항 (7입력 · NO₂ μσ 13개월)", "new": "결정 1 (9입력 · NO₂ μσ 60개월)"}, {"old": V.C_MUTED, "new": V.C_OURS},
                                "결정 1 전후 월별 R²·RMSE·편향 — 2020 · time fold 2 · seed 0 · 30 ep · 같은 테스트 사운딩 (두 런은 NO₂ 표준화 기준도 다름)", "monthly_r2_decision1", a.date, a.out))
    if "syn" in figs:  # 종관 2열(11변수) 배경항 런 — time2 만. ct_compare_2020_syn 의 space 행·strata 는 9변수 런 예측이라 층 그림은 만들지 않는다 (QA 관찰 2026-09-22)
        print(V.plot_monthly(f"{a.ct_syn_dir}/monthly_time2.csv", a.date, a.out, name="monthly_time2_syn",
                             title="월별 테스트 RMSE·편향 — 2020 · time fold 2 · seed 0 (우리 = train5_2020_syn 물리, 배경항 11변수: 결정 1 + 종관 z850·thk)",
                             note="space 블록 미갱신: ct_compare_2020_syn 의 space 행·거리 층 CSV 는 9변수 런(train5_2020_space0) 예측이므로 종관 판 층 그림은 만들지 않음"))
        t = V.monthly_compare({"base": TIME2_NEW, "syn": TIME2_SYN}, a.idx_dir); print(t.round(3).to_string(index=False))
        print(V.plot_monthly_r2(t, {"base": "배경항 9변수 (결정 1)", "syn": "배경항 11변수 (+종관 z850·thk)"}, {"base": V.C_MUTED, "syn": V.C_OURS},
                                "배경항 9 → 11변수 월별 R²·RMSE·편향 — 2020 · time fold 2 · seed 0 · 30 ep · 같은 테스트 사운딩 (단일 시드)", "monthly_r2_syn_vs_base", a.date, a.out))
        g, meta = V.residual_grid(V.load_test_rows(TIME2_SYN, a.idx_dir, "time", 2)); print("time2 syn", meta)
        print(V.plot_residual_map(g, meta, "잔차 지도 — 2020 · time fold 2 · seed 0 · 물리 (train5_2020_syn, 배경항 11변수)", "resmap_time2_syn", a.date, a.out))
    if "lc" in figs:
        t = V.learning_curve_table(LC_DIRS); print(t.round(3).to_string(index=False)); print(V.plot_learning_curve(LC_DIRS, a.date, a.out, table=t))  # 1회 계산 (QA N9)
    if "b1" in figs:
        t = V.b1_table(B1_ROWS); print(t.round(4).to_string(index=False)); print(V.plot_b1_forest(t, a.date, a.out))
    if "r2ct" in figs:
        t = V.r2_vs_ct_table(f"{a.ct_dir}/monthly_time2.csv"); print(t.round(3).to_string(index=False)); print(V.plot_r2_vs_ct(t, CT_LABEL["time"], a.date, a.out))
    if "runs" in figs:  # 같은 테스트 행(437,972)에서 월별 R²·RMSE·편향. 모두 물리 런, 배경 11변수, seed 0
        t = V.monthly_compare({k: f"{d}/pred_time2_s0_phys.parquet" for k, d in RUNS_A.items()}, a.idx_dir); print(t[t["월"] == 0].round(4).to_string(index=False))
        pal = dict(zip(RUNS_A, [V.C_OURS, "#eda100", "#e87ba4", "#4a3aa7"]))  # 범주 slot 1·4·5·7 (2 주황 = CT, 3 청록 = 절제 의미라 제외)
        print(V.plot_monthly_r2(t, {k: k for k in RUNS_A}, pal, "12개월 time2 물리 런 비교 — 2020 · time fold 2 · seed 0 · 배경 11변수 · 최대 100 ep + patience 10 · 같은 테스트 사운딩 (단일 시드)", "monthly_runs_a", a.date, a.out))
    if "freerun" in figs:
        t = V.freerun_table(FREERUN_DIR); print(t[t.lag.isin([0, 24, 48])][["start", "lag", "coverage", "mass_ratio", "info_ratio"]].round(4).to_string(index=False))
        print(V.plot_freerun(t, FREERUN_REF, a.date, a.out, D=FREERUN_D))


if __name__ == "__main__":
    main()
