"""
pysr_kepler_train_025.py
========================
High-Resolution (0.25 deg) Symbolic Regression Pipeline.
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

# ─────────────────────────────────────────────
# 0. 경로 및 출력 디렉토리 설정 (0.25도용)
# ─────────────────────────────────────────────
DATA_ROOT = Path("/Volumes/100.118.65.89/dataset/XCO2연구 데이터/03_split_output_025")
PARQUET_PATH = DATA_ROOT / "anom_1d_balanced_025.parquet"
SPLIT_IDX_PATH = DATA_ROOT / "split_indices_v2_025.json"
SCALER_PATH = DATA_ROOT / "scalers_v2_025.joblib"

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "kepler_outputs_025"
OUTPUT_DIR.mkdir(exist_ok=True)

EQUATION_FILE = OUTPUT_DIR / "hall_of_fame_025.csv"
PARETO_SNAPSHOT_DIR = OUTPUT_DIR / "pareto_snapshots"
PARETO_SNAPSHOT_DIR.mkdir(exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. 로거 설정
log_path = OUTPUT_DIR / f"kepler_run_025_{RUN_TIMESTAMP}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("kepler_025")

# 2. 특성 컬럼 정의
FEATURE_COLS = [
    "tropomi_no2", "odiac_emission", 
    "WS_eff", "BLH_eff", 
    "population_density", "latitude", "doy_sin", "doy_cos"
]
TARGET_COL = "xco2_anomaly"
DILUTION_VARS = {"WS_eff", "BLH_eff"}

# 3. 데이터 로드 및 전처리
def load_and_preprocess():
    logger.info("=== 0.25도 데이터 로드 및 전처리 시작 ===")
    df = pd.read_parquet(PARQUET_PATH)
    
    # 물리 피처 생성
    df['WS_eff']  = np.maximum(df['era5_wind_speed'], 0.1)
    df['BLH_eff'] = np.maximum(df['era5_blh'], 10.0)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['doy'] = df['date'].dt.dayofyear
        df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)

    # 훈련 인덱스 적용
    with open(SPLIT_IDX_PATH, "r") as f:
        split_info = json.load(f)
    train_idx = split_info["train_indices"]
    df_train = df.iloc[train_idx].copy()
    
    # 결측치 처리
    df_train[FEATURE_COLS] = df_train[FEATURE_COLS].fillna(df_train[FEATURE_COLS].median())
    y_train = df_train[TARGET_COL].values.astype(np.float64)
    X_train = df_train[FEATURE_COLS].values.astype(np.float64)

    # [NEW] Sample Weights 도입 (배출 신호 핫스팟에 10배 집중)
    # NO2 농도가 상위 10%인 샘플에 가중치 5.0 부여 (평균은 1.0)
    no2_threshold = df_train['tropomi_no2'].quantile(0.90)
    weights = np.where(df_train['tropomi_no2'] >= no2_threshold, 5.0, 1.0)

    logger.info(f"0.25도 PySR 입력 준비 완료: X {X_train.shape}, y {y_train.shape}")
    logger.info(f"핫스팟 가중치 적용 완료 (상위 10% NO2: {no2_threshold:.2f} -> weight=5.0)")
    return X_train, y_train, weights, FEATURE_COLS

# PySR 모델 초기화
def build_model(feature_names):
    # Dr. Kepler's Weighted Huber Loss (Julia syntax)
    # 0.25도 고해상도 데이터의 핫스팟 신호에 가중치를 부여합니다.
    huber_loss = (
        "loss(prediction, target, weight) = weight * ("
        "abs(prediction - target) < 0.5 ? "
        "0.5 * (prediction - target)^2 : "
        "0.5 * abs(prediction - target) - 0.125"
        ")"
    )
    
    return PySRRegressor(
        # Search Space (Equation Structure & Complexity Control)
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log1p", "sqrt", "square", "inv(x) = 1/x"], # inv(x) 추가로 희석 효과 유도
        maxsize=25, # 복잡도 상향 (비선형 항 구체화)
        constraints={'/': (1, 9), '*': (1, 9), 'square': 2}, # 제약 완화 (물리적 결합 자유도)
        nested_constraints={"log1p": {"log1p": 0, "sqrt": 0}, "sqrt": {"sqrt": 0}, "inv": {"inv": 0}},
        
        # Evolutionary Control
        niterations=1000,
        populations=40, # 탐색 능력 강화를 위해 상향
        population_size=40,
        tournament_selection_n=12,
        parsimony=0.0005, # 복잡해진 만큼 parsimony 소폭 하향

        # Optimization & Performance
        elementwise_loss=huber_loss,
        optimizer_iterations=20,
        procs=0, 
        multithreading=True,
        timeout_in_seconds=7200, # 2시간 짧고 굵은 Refinement
        
        # File management
        temp_equation_file=str(EQUATION_FILE), # PySR 버전에 따라 temp_equation_file 또는 equation_file 사용
        variable_names=feature_names,
        extra_sympy_mappings={'inv': lambda x: 1/x},
    )

def main():
    logger.info("Dr. Kepler's 0.25 deg Deep Search Pipeline 시작")
    X_train, y_train, weights, feature_names = load_and_preprocess()
    model = build_model(feature_names)
    
    logger.info("\n>>> PySR 0.25deg 고강도 학습 시작 (10,000 iters) <<<\n")
    model.fit(X_train, y_train, weights=weights)
    
    # 결과 요약
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    logger.info(f"\n훈련 데이터 성능: R² = {r2:.6f}")
    logger.info(f"결과 저장 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
