"""
pysr_kepler_train.py
====================
Dr. Julian R. Kepler's Symbolic Regression Pipeline
for XCO2 Anomaly Prediction in East Asian Hotspots.

철학: Occam's Razor — 가장 단순한 방정식이 가장 위대한 물리 법칙이다.
      연구자 편향 파생변수 투입 엄금. PySR이 대기분산 역학을 자율 도출.

실행: python pysr_kepler_train.py
"""

import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import RobustScaler

# ─────────────────────────────────────────────
# 0. 경로 및 출력 디렉토리 설정
# ─────────────────────────────────────────────
# 실제 데이터가 위치한 외부 볼륨 경로
DATA_ROOT = Path("/Volumes/100.118.65.89/dataset/XCO2연구 데이터/03_split_output")

PARQUET_PATH = DATA_ROOT / "anom_1d_balanced.parquet"
SPLIT_IDX_PATH = DATA_ROOT / "split_indices_v2.json"
SCALER_PATH = DATA_ROOT / "scalers_v2.joblib"

# 출력물(로그, 방정식 등)은 현재 스크립트 실행 위치 하위의 kepler_outputs에 저장
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "kepler_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

EQUATION_FILE = OUTPUT_DIR / "hall_of_fame.csv"
PARETO_SNAPSHOT_DIR = OUTPUT_DIR / "pareto_snapshots"
PARETO_SNAPSHOT_DIR.mkdir(exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ─────────────────────────────────────────────
# 1. 로거 설정
# ─────────────────────────────────────────────
log_path = OUTPUT_DIR / f"kepler_run_{RUN_TIMESTAMP}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("kepler")

# ─────────────────────────────────────────────
# 2. 특성 컬럼 정의 (Tier 분류)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # Tier 1 — Proxy (배출 신호)
    "tropomi_no2",
    "odiac_emission",
    # Tier 2 — Transport (대기 분산 역학)
    "WS_eff",
    "BLH_eff",
    # Tier 3 — Confounder (사회경제적 혼재변수)
    "population_density",
    # Tier 4 — Spatiotemporal (공간·계절 정보)
    "latitude",
    "doy_sin",
    "doy_cos",
]

TARGET_COL = "xco2_anomaly"

# One-hot 변수 집합 (현재 사용 가능한 원핫 변수가 없음)
ONE_HOT_COLS = set()

# 대기 분산 역학 변수 — 분모에 등장해야 물리적으로 타당
DILUTION_VARS = {"WS_eff", "BLH_eff"}


# ─────────────────────────────────────────────
# 3. 데이터 로드 및 전처리
# ─────────────────────────────────────────────
def load_and_preprocess():
    """
    Parquet 로드 → 물리 피처 생성 → 분할 인덱스 적용 → 결측치 처리 → Step 03 Scaler 적용.
    """
    logger.info("=== 데이터 로드 및 전처리 시작 ===")

    # 3-1. Parquet 읽기
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {PARQUET_PATH}")
    
    df = pd.read_parquet(PARQUET_PATH)
    logger.info(f"원본 데이터 로드 완료: {df.shape}")

    # 3-2. [Kepler Dynamic Feature Engineering]
    # Step 03 결과물인 anom_1d_balanced.parquet에는 아직 없는 물리 피처들을 생성합니다.
    logger.info("물리 피처 및 지역 원핫 변수 생성 중...")
    
    # (A) 대기 희석 변수 (WS_eff, BLH_eff)
    df['WS_eff']  = np.maximum(df['era5_wind_speed'], 0.1)
    df['BLH_eff'] = np.maximum(df['era5_blh'], 10.0)
    
    # (B) 지역 원핫 (eaic_region 컬럼 기반)
    if 'eaic_region' in df.columns:
        for reg in ['NCP', 'YRD', 'PRD', 'KCR', 'JKT']:
            df[f'is_{reg}'] = (df['eaic_region'] == reg).astype(np.float64)
    else:
        logger.warning("eaic_region 컬럼이 없어 지역 원핫 변수를 생성하지 못했습니다.")

    # (C) 시계열 주기성 (Date 기반 doy_sin/cos)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['doy'] = df['date'].dt.dayofyear
        df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)

    # 3-3. 훈련 인덱스 적용
    with open(SPLIT_IDX_PATH, "r") as f:
        split_info = json.load(f)
    
    # 키 이름 후보 순서대로 탐색 (train_indices / train / train_idx)
    train_idx = split_info.get(
        "train_indices",
        split_info.get("train", split_info.get("train_idx")),
    )
    if train_idx is None:
        raise KeyError(
            f"split_indices_v2.json에서 훈련 인덱스 키를 찾을 수 없습니다. "
            f"존재하는 키: {list(split_info.keys())}"
        )
    train_idx = [int(i) for i in train_idx]  # JSON 파싱 결과가 str일 경우 대비
    df_train = df.iloc[train_idx].copy()
    logger.info(f"훈련 세트 분리 완료 (Train N={len(df_train)})")

    # 3-4. 결측치 처리 (중앙값 대치)
    df_train[FEATURE_COLS] = df_train[FEATURE_COLS].fillna(df_train[FEATURE_COLS].median())
    y_train = df_train[TARGET_COL].values.astype(np.float64)

    # 3-5. 스케일러 적용 (Step 03의 StandardScaler 활용)
    if SCALER_PATH.exists():
        bundle = joblib.load(SCALER_PATH)
        # Step 03의 scaler_bundle 구조 확인 (StandardScaler)
        if isinstance(bundle, dict) and "scaler_exp_a" in bundle:
            scaler = bundle["scaler_exp_a"]
            logger.info("Step 03에서 생성된 StandardScaler(Exp A)를 로드했습니다.")
        else:
            logger.warning("알 수 없는 스케일러 포맷입니다. RobustScaler를 새로 생성합니다.")
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(df_train[FEATURE_COLS])
    else:
        logger.warning("스케일러 파일이 없습니다. RobustScaler를 새로 생성합니다.")
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler().fit(df_train[FEATURE_COLS])

    # [Dr. Sterling의 처방] 기호회귀는 물리량 보존이 생명입니다. 스케일링 엄금.
    # (단, 딥러닝용 파이프라인 유지를 위해 scaler 객체 자체는 남겨두되 적용만 하지 않음)
    X_train = df_train[FEATURE_COLS].values.astype(np.float64)

    logger.info(f"최종 PySR 입력 준비 완료: X {X_train.shape}, y {y_train.shape}")
    return X_train, y_train, FEATURE_COLS


# ─────────────────────────────────────────────
# 4. Pareto Front 모니터링 스레드
# ─────────────────────────────────────────────
class ParetoMonitor(threading.Thread):
    """
    30분마다 equation_file을 읽어 Pareto Front 스냅샷을 저장하는 백그라운드 스레드.
    PySR의 --equation_file 자동저장 기능과 연동.
    """

    INTERVAL_SEC = 30 * 60  # 30분

    def __init__(self, equation_file: Path, snapshot_dir: Path):
        super().__init__(daemon=True)
        self.equation_file = equation_file
        self.snapshot_dir = snapshot_dir
        self._stop_event = threading.Event()
        self.snapshot_count = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        logger.info("Pareto 모니터 스레드 시작 (30분 간격 스냅샷)")
        while not self._stop_event.wait(self.INTERVAL_SEC):
            self._take_snapshot()

    def _take_snapshot(self):
        if not self.equation_file.exists():
            logger.warning("equation_file 미존재 — 스냅샷 건너뜀")
            return

        try:
            df_eq = pd.read_csv(self.equation_file)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.snapshot_dir / f"pareto_snapshot_{ts}.csv"
            df_eq.to_csv(out_path, index=False)
            self.snapshot_count += 1
            logger.info(
                f"[스냅샷 #{self.snapshot_count}] Pareto Front 저장: {out_path} "
                f"(방정식 수: {len(df_eq)})"
            )
        except Exception as e:
            logger.error(f"스냅샷 실패: {e}")


# ─────────────────────────────────────────────
# 5. PySR 모델 초기화 (Dr. Kepler 하이퍼파라미터)
# ─────────────────────────────────────────────
def build_model(feature_names: list) -> PySRRegressor:
    """
    Dr. Kepler's exact hyperparameter specification을 PySRRegressor에 매핑.

    핵심 설계 결정:
    - 커스텀 Huber Loss (Julia 문법): 이상치에 견고, MSE보다 물리적으로 타당
    - log1p/sqrt 허용, exp/sin/cos 엄금: 물리적 해석 가능성 유지
    - constraints: 분수/곱셈의 깊이 제한으로 차원 붕괴 방지
    - procs=0, multithreading=True: Julia 멀티스레딩 활용, 별도 프로세스 없음
    """
    # 커스텀 Huber Loss (Julia 문법, delta=0.5)
    # |r| < 0.5 → 0.5*r^2  (MSE 영역)
    # |r| ≥ 0.5 → 0.5*|r| - 0.125  (MAE 영역)
    huber_loss = (
        "loss(prediction, target) = "
        "abs(prediction - target) < 0.5 ? "
        "0.5 * (prediction - target)^2 : "
        "0.5 * abs(prediction - target) - 0.125"
    )

    model = PySRRegressor(
        # ── 탐색 공간 (방정식 구조·복잡도 제어) ──────────────────
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log1p", "sqrt", "square"],
        maxsize=22,
        constraints={"/": (4, 4), "*": (4, 4), "square": 2},
        nested_constraints={
            "log1p": {"log1p": 0, "sqrt": 0},
            "sqrt": {"sqrt": 0},
        },
        # ── 진화 제어 (EA 최적화) ────────────────────────────────
        niterations=1000,
        populations=30,
        population_size=33,
        tournament_selection_n=12,
        parsimony=0.001,
        # ── 유전 확률 (유전 연산 비율) ───────────────────────────
        fraction_replaced=0.066,
        weight_mutate_constant=0.7,
        # ── 최적화·성능 (손실함수·컴퓨팅) ───────────────────────
        elementwise_loss=huber_loss,
        optimizer_iterations=15,
        procs=0,
        multithreading=True,
        timeout_in_seconds=28800,  # 8시간 제한
        # ── 출력·로깅 ────────────────────────────────────────────
        temp_equation_file=str(EQUATION_FILE),
        verbosity=1,
        variable_names=feature_names,
        # 재현성 (선택적)
        # random_state=42,
    )

    logger.info("PySRRegressor 초기화 완료")
    logger.info(f"  maxsize={model.maxsize}, niterations={model.niterations}")
    logger.info(f"  populations={model.populations}, population_size={model.population_size}")
    logger.info(f"  timeout={model.timeout_in_seconds}s (8시간)")
    return model


# ─────────────────────────────────────────────
# 6. Pareto Front 추출 및 엘보우 포인트 선정
# ─────────────────────────────────────────────
def extract_pareto_front(model: PySRRegressor) -> pd.DataFrame:
    """
    model.equations_에서 Pareto Front를 추출하고 정렬.
    complexity 5~15 범위에서 Top 3 엘보우 포인트 선정.
    """
    if model.equations_ is None or len(model.equations_) == 0:
        raise RuntimeError("학습된 방정식이 없습니다. 학습이 정상 완료되었는지 확인하세요.")

    df_eq = model.equations_.copy()

    # 컬럼명 정규화 (PySR 버전에 따라 다를 수 있음)
    col_map = {}
    for col in df_eq.columns:
        if "loss" in col.lower():
            col_map[col] = "loss"
        if "complex" in col.lower():
            col_map[col] = "complexity"
        if "equation" in col.lower() or "sympy" in col.lower():
            col_map[col] = "equation"
    df_eq = df_eq.rename(columns=col_map)

    df_eq = df_eq.sort_values("complexity").reset_index(drop=True)

    # 엘보우 범위 필터
    elbow_df = df_eq[
        (df_eq["complexity"] >= 5) & (df_eq["complexity"] <= 15)
    ].copy()

    if elbow_df.empty:
        logger.warning("복잡도 5~15 방정식 없음 — 전체에서 Top 3 선정")
        elbow_df = df_eq.copy()

    # 손실 기준 오름차순 Top 3
    top3 = elbow_df.nsmallest(3, "loss").reset_index(drop=True)

    logger.info(f"전체 Pareto Front 방정식 수: {len(df_eq)}")
    logger.info(f"엘보우 포인트 Top 3 (complexity 5~15):")
    for i, row in top3.iterrows():
        logger.info(
            f"  [{i+1}] complexity={row.get('complexity', 'N/A'):.0f}, "
            f"loss={row.get('loss', 'N/A'):.6f}\n"
            f"      equation: {row.get('equation', row.get('sympy_format', ''))}"
        )

    return df_eq, top3


# ─────────────────────────────────────────────
# 7. Dr. Kepler's Physical Audit (물리 타당성 감사)
# ─────────────────────────────────────────────
def kepler_audit(top3: pd.DataFrame) -> pd.DataFrame:
    """
    선정된 Top 3 방정식에 대해 물리 타당성을 검증:

    [검증 기준]
    1. 희석 효과 (Dilution Effect):
       WS_eff 또는 BLH_eff가 분모에 등장해야 물리적으로 타당.
       (바람이 강할수록, 혼합층이 높을수록 XCO2 희석)

    2. One-Hot 지배 방정식 제거:
       is_NCP, is_YRD, is_KCR, is_JKT 만으로 구성된 방정식은
       지역 효과에 과적합된 것으로 제외.

    Returns
    -------
    audited_df : 감사 결과 컬럼이 추가된 DataFrame
    """
    audit_results = []

    for _, row in top3.iterrows():
        eq_str = str(row.get("equation", row.get("sympy_format", "")))

        # ── 검증 1: 희석 변수 분모 등장 여부 ──────────────────
        # 분모 패턴: "/WS_eff", "/ WS_eff", "(WS_eff)", etc.
        # 간단한 휴리스틱: 분자/분모 분리 어려우므로 분모 위치 근사
        dilution_in_denominator = False
        for var in DILUTION_VARS:
            # '/ var' 또는 '/(... var ...)' 패턴 탐지
            if re.search(rf"/\s*{var}|/\s*\(.*?{var}.*?\)", eq_str):
                dilution_in_denominator = True
                break

        # ── 검증 2: One-Hot 지배 여부 ──────────────────────────
        # 방정식에 등장하는 변수 중 One-Hot이 아닌 것이 있는지 확인
        non_onehot_present = any(
            var in eq_str
            for var in FEATURE_COLS
            if var not in ONE_HOT_COLS
        )
        onehot_dominated = not non_onehot_present

        # ── 물리적 유효성 종합 판정 ────────────────────────────
        physically_valid = dilution_in_denominator and not onehot_dominated

        audit_results.append(
            {
                "dilution_in_denominator": dilution_in_denominator,
                "onehot_dominated": onehot_dominated,
                "physically_valid": physically_valid,
                "audit_note": _audit_note(
                    dilution_in_denominator, onehot_dominated
                ),
            }
        )

        logger.info(
            f"\n[Kepler Audit] equation: {eq_str}\n"
            f"  희석변수 분모 등장: {dilution_in_denominator}\n"
            f"  One-Hot 지배:       {onehot_dominated}\n"
            f"  물리적 유효:        {physically_valid}"
        )

    audit_df = pd.DataFrame(audit_results)
    audited_top3 = pd.concat(
        [top3.reset_index(drop=True), audit_df], axis=1
    )

    # One-Hot 지배 방정식 경고
    dominated = audited_top3[audited_top3["onehot_dominated"]]
    if not dominated.empty:
        logger.warning(
            f"One-Hot 지배 방정식 {len(dominated)}개 제외 권고:\n"
            f"{dominated[['complexity', 'loss', 'equation']].to_string()}"
        )

    return audited_top3


def _audit_note(dilution_ok: bool, onehot_dom: bool) -> str:
    notes = []
    if dilution_ok:
        notes.append("✓ 대기희석 (WS/BLH 분모) 확인됨")
    else:
        notes.append("✗ 희석변수 분모 미검출 — 물리 해석 재확인 필요")
    if onehot_dom:
        notes.append("✗ One-Hot 지배 — 지역 과적합 의심, 제외 권고")
    else:
        notes.append("✓ 물리 변수 존재 확인됨")
    return " | ".join(notes)


# ─────────────────────────────────────────────
# 8. 성능 평가
# ─────────────────────────────────────────────
def evaluate(model: PySRRegressor, X_train: np.ndarray, y_train: np.ndarray):
    """R2 및 RMSE 계산 후 로깅."""
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    rmse = root_mean_squared_error(y_train, y_pred)

    logger.info(f"\n{'='*50}")
    logger.info("=== 훈련 데이터 성능 평가 ===")
    logger.info(f"  R²   : {r2:.6f}")
    logger.info(f"  RMSE : {rmse:.6f} ppm")
    logger.info(f"{'='*50}")

    return {"R2": r2, "RMSE": rmse}


# ─────────────────────────────────────────────
# 9. 결과 저장 (CSV + Markdown)
# ─────────────────────────────────────────────
def save_results(
    df_eq: pd.DataFrame,
    audited_top3: pd.DataFrame,
    metrics: dict,
):
    """Pareto Front 전체와 Top3 감사 결과를 CSV·Markdown으로 저장."""

    # 9-1. Pareto Front 전체 CSV
    pareto_csv = OUTPUT_DIR / f"pareto_front_{RUN_TIMESTAMP}.csv"
    df_eq.to_csv(pareto_csv, index=False)
    logger.info(f"Pareto Front 저장: {pareto_csv}")

    # 9-2. Top 3 감사 결과 CSV
    top3_csv = OUTPUT_DIR / f"top3_audited_{RUN_TIMESTAMP}.csv"
    audited_top3.to_csv(top3_csv, index=False)
    logger.info(f"Top3 감사 결과 저장: {top3_csv}")

    # 9-3. 요약 Markdown 보고서
    md_path = OUTPUT_DIR / f"kepler_report_{RUN_TIMESTAMP}.md"
    _write_markdown(md_path, df_eq, audited_top3, metrics)
    logger.info(f"Markdown 보고서 저장: {md_path}")


def _write_markdown(
    path: Path,
    df_eq: pd.DataFrame,
    audited_top3: pd.DataFrame,
    metrics: dict,
):
    lines = [
        "# Dr. Kepler's Symbolic Regression Report",
        f"\n**실행 시각**: {RUN_TIMESTAMP}",
        f"**타겟**: {TARGET_COL}",
        f"**특성 수**: {len(FEATURE_COLS)}",
        "",
        "## 학습 성능 (훈련 데이터)",
        f"| 지표 | 값 |",
        f"|------|-----|",
        f"| R²   | {metrics['R2']:.6f} |",
        f"| RMSE | {metrics['RMSE']:.6f} ppm |",
        "",
        "## Pareto Front 요약",
        f"- 총 방정식 수: **{len(df_eq)}**",
        f"- 복잡도 범위: {df_eq['complexity'].min():.0f} ~ {df_eq['complexity'].max():.0f}",
        "",
        "## Top 3 엘보우 포인트 (complexity 5~15)",
        "",
    ]

    for i, row in audited_top3.iterrows():
        eq = row.get("equation", row.get("sympy_format", "N/A"))
        lines += [
            f"### [{i+1}] Complexity={row.get('complexity', 'N/A'):.0f}  |  Loss={row.get('loss', 'N/A'):.6f}",
            f"```",
            f"{eq}",
            f"```",
            f"- **희석변수 분모**: {'✓' if row.get('dilution_in_denominator') else '✗'}",
            f"- **One-Hot 지배**: {'✗ 제외 권고' if row.get('onehot_dominated') else '✓ 정상'}",
            f"- **물리 유효성**: {'✓ PASS' if row.get('physically_valid') else '✗ 검토 필요'}",
            f"- **메모**: {row.get('audit_note', '')}",
            "",
        ]

    lines += [
        "## 핵심 철학 (Occam's Razor)",
        "> 가장 단순한 방정식이 가장 위대한 물리 법칙이다.",
        "> WS_eff, BLH_eff의 분모 등장 = 대기 희석 역학의 자율 도출.",
        "",
        "---",
        "*Generated by pysr_kepler_train.py*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────
# 10. 메인 파이프라인
# ─────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("  Dr. Kepler's Symbolic Regression Pipeline 시작")
    logger.info(f"  실행 시각: {RUN_TIMESTAMP}")
    logger.info("=" * 60)

    # Step 1: 데이터 로드 및 전처리
    X_train, y_train, feature_names = load_and_preprocess()

    # Step 2: 모델 초기화
    model = build_model(feature_names)

    # Step 3: Pareto 모니터 스레드 시작
    monitor = ParetoMonitor(EQUATION_FILE, PARETO_SNAPSHOT_DIR)
    monitor.start()
    logger.info("Pareto 모니터 스레드 가동 완료")

    # Step 4: PySR 학습 실행
    logger.info("\n>>> PySR 학습 시작 (최대 8시간) <<<\n")
    t0 = time.time()

    try:
        model.fit(X_train, y_train)
    except KeyboardInterrupt:
        logger.warning("사용자 중단 (Ctrl+C) — 현재까지 결과 저장 시도")
    finally:
        monitor.stop()

    elapsed = time.time() - t0
    logger.info(f"\n학습 완료. 소요 시간: {elapsed/3600:.2f}시간")

    # 마지막 스냅샷
    monitor._take_snapshot()

    # Step 5: Pareto Front 추출
    df_eq, top3 = extract_pareto_front(model)

    # Step 6: Dr. Kepler Physical Audit
    audited_top3 = kepler_audit(top3)

    # Step 7: 성능 평가
    metrics = evaluate(model, X_train, y_train)

    # Step 8: 결과 저장
    save_results(df_eq, audited_top3, metrics)

    logger.info("\n=== 파이프라인 완료 ===")
    logger.info(f"결과 디렉토리: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
