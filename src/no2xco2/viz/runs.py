"""시각화 입력 실험 경로 등록부 (QA B8, 2026-09-24): plot_results·test_viz 가 같은 경로를 쓰도록 한 곳에 둔다.
루트는 환경변수 NO2_EXP_DIR (기본 experiments) 로 바꿀 수 있다."""
import os

EXP = os.environ.get("NO2_EXP_DIR", "experiments")


def exp(*parts: str) -> str:
    return os.path.join(EXP, *parts)


CT_DIR = exp("ct_compare_2020_20260926")  # D-3 산출 (AI 09-26 00:0x; time2 = A-2 train5_2020_3a 물리, space0 = train5_2020_space0 9변수)
CT_LABEL = {"time": "train5_2020_3a 물리 (A-2, 배경 11변수, 100 ep)", "space": "train5_2020_space0 (배경 9변수, 30 ep)"}  # CT_DIR 의 우리 예측 출처 — 그림 제목용
CT_DIR_LC12 = exp("ct_compare_2020_20260924")  # time2 우리 = lc2020_12m (9변수). lc2020_12m 기반 그림·테스트의 대조용
CT_DIR_OLD = exp("ct_compare_2020"); CT_SYN = exp("ct_compare_2020_syn")  # 구 형식 (CT_SYN 은 D-3 판 없음)
SEED_DIRS = [exp("train5_202001_fixed"), exp("train5_202001_fixed_s12")]
SEED_DIR = {0: SEED_DIRS[0], 1: SEED_DIRS[1], 2: SEED_DIRS[1]}  # 2020-01 3시드 pred 위치 (QA W7: 한 곳에서만)
LC_DIRS = {3: exp("lc2020_3m"), 6: exp("lc2020_6m"), 12: exp("lc2020_12m")}
TIME2_NEW = exp("lc2020_12m", "pred_time2_s0_phys.parquet")      # 결정 1 (배경항 9변수)
TIME2_OLD = exp("train5_2020", "pred_time2_s0_phys.parquet")     # 구 배경항 (7변수)
TIME2_SYN = exp("train5_2020_syn", "pred_time2_s0_phys.parquet")  # 종관 (배경항 11변수)
# 12개월 time2 물리 런 비교 (F, 09-28): 모두 배경 11변수, seed 0, 최대 100 ep + patience 10
RUNS_A = {"A-2": exp("train5_2020_3a"), "7a 월별 β": exp("train5_2020_7a"), "13a 장면 가중": exp("train5_2020_13a"), "A-3 chunk 96": exp("train5_2020_3b")}
SPACE0 = {k: exp("train5_2020_space0", f"pred_space0_s0_{k}.parquet") for k in ("phys", "nophys")}  # 9변수 런

# B1 forest (V-C): (라벨, summary 디렉토리, scheme, fold, 배경항 입력 수) — 행 순서 = 그림 위→아래
B1_ROWS = [("2020-01 · time2", exp("train5_202001_fixed"), "time", 2, 7), ("2020-01 · time2", exp("train5_202001_fixed_s12"), "time", 2, 7),
           ("2020-01 · space0", exp("train5_202001_fixed"), "space", 0, 7), ("2020-01 · space0", exp("train5_202001_fixed_s12"), "space", 0, 7),
           ("2020 12개월 · time2", exp("train5_2020"), "time", 2, 7), ("2020 12개월 · space0", exp("train5_2020_space0"), "space", 0, 9),
           ("2020 12개월 · time2 · A-2 100 ep", exp("train5_2020_3a"), "time", 2, 11),
           ("2020 12개월 · time2 · A-3 chunk 96", exp("train5_2020_3b"), "time", 2, 11)]

FREERUN_DIR = exp("physics_freerun_202001")  # V-D: physics_rollout.py --mode free-run --start 0/240/480 → 실제 시작 스텝 5/243/484 (각 창의 하루 중 화소 최다 스텝), D 5,000
FREERUN_REF = 243                            # 스모크 기준 시작 스텝 (1/11 03 UTC) — lag 48 커버리지 0.477 · 질량비 1.419
FREERUN_D = 5000.0                           # V-D 롤아웃 확산 계수 (physics_rollout --D 기본값 그대로 실행)


def inputs(fig: str, idx_dir: str, ct_dir: str = CT_DIR, ct_syn_dir: str = CT_SYN) -> list[str]:
    """그림별 입력 파일 전부 — plot_results(건너뛰기)와 test_viz(skipif)가 같은 목록을 쓴다 (QA Y7·Y5)."""
    oco12 = [os.path.join(idx_dir, f"oco_2020{m:02d}.parquet") for m in range(1, 13)]
    seed_preds = [os.path.join(d, f"pred_space0_s{s}_{m}.parquet") for s, d in SEED_DIR.items() for m in ("phys", "nophys")]
    return {"strata": [f"{ct_dir}/strata_space0.csv", os.path.join(idx_dir, "oco_202001.parquet"), *seed_preds],
            "monthly": [f"{ct_dir}/monthly_time2.csv"],
            "lc": [p for d in LC_DIRS.values() for p in (f"{d}/summary.csv", f"{d}/pred_time2_s0_phys.parquet")],
            "resmap": [TIME2_NEW, *SPACE0.values(), *oco12],
            "r2": [TIME2_OLD, TIME2_NEW, *oco12],
            "syn": [f"{ct_syn_dir}/monthly_time2.csv", TIME2_NEW, TIME2_SYN, *oco12],
            "b1": [f"{d}/summary.csv" for _, d, *_ in B1_ROWS],
            "freerun": [os.path.join(FREERUN_DIR, f"physics_freerun_202001_s{FREERUN_REF}.csv")],
            "r2ct": [f"{ct_dir}/monthly_time2.csv"],
            "runs": [p for d in RUNS_A.values() for p in (f"{d}/summary.csv", f"{d}/pred_time2_s0_phys.parquet")] + oco12}[fig]


# 그림별 출력 이름 (진입점의 사전 충돌 검사용, QA Y3)
OUTPUTS = {"strata": ["strata_space0"], "monthly": ["monthly_time2"], "lc": ["learning_curve_2020"],
           "resmap": ["resmap_time2_phys", "resmap_space0_phys", "resmap_space0_diff"], "r2": ["monthly_r2_decision1"],
           "syn": ["monthly_time2_syn", "monthly_r2_syn_vs_base", "resmap_time2_syn"], "b1": ["b1_forest"], "freerun": ["physics_freerun"],
           "r2ct": ["monthly_r2_vs_ct"], "runs": ["monthly_runs_a"]}
