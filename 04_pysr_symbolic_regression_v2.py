#!/usr/bin/env python3
"""
04_pysr_symbolic_regression_v2.py
──────────────────────────────────────────────────────────────────
Phase 2 · Step 5: PySR 기호 회귀 — 성능 극대화 버전 (R² 0.7 목표)

개선 사항:
  1. 데이터 대량 복구: 02번의 엄격한 필터링 대신 전면 데이터 로드 및 위도대별 베이스라인 제거
  2. 물리적 힌트(Hints) 제공: NO2/WS, NO2/(WS*BLH) 등 물리 파생 피처 투입
  3. 초고강도 탐색: niterations=500, populations=100 등 파라미터 대폭 상향
  4. 지수 연산(exp) 복구: 물리적 감쇄 효과 포착
──────────────────────────────────────────────────────────────────
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────
# 0. 경로 및 상수
# ─────────────────────────────────────────────────────────────────
BASE_DIR    = "/Volumes/100.118.65.89/dataset/XCO2연구 데이터"
PARQUET_IN  = os.path.join(BASE_DIR, "ml_ready_dataset.parquet")
OUT_DIR     = os.path.join(BASE_DIR, "anomaly_output_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# PySR 학습에 사용할 샘플 수 (데이터 복구로 대폭 상향)
MAX_TRAIN_SAMPLES = 200_000
MAX_TEST_SAMPLES  = 50_000

# 물리 상수
U_EFF_MIN = 0.1
H_EFF_MIN = 10.0

# ─────────────────────────────────────────────────────────────────
# 1. 데이터 로드 및 유연한 Anomaly 생성 (데이터 복구)
# ─────────────────────────────────────────────────────────────────
def load_and_recover_data():
    print("=" * 70)
    print("STEP 1: 데이터 로드 및 Latitudinal Anomaly 생성 (데이터 복구)")
    print("=" * 70)
    
    df = pd.read_parquet(PARQUET_IN)
    print(f"  [Load] ml_ready_dataset: {len(df):,} rows")

    # NaN 제거 (필수 피처 위주)
    essential_cols = ['xco2', 'tropomi_no2', 'era5_u10', 'era5_v10', 'era5_blh', 'latitude']
    df = df.dropna(subset=essential_cols).reset_index(drop=True)
    print(f"  [QC] 필수 결측치 제거 후: {len(df):,} rows")

    # ── 유연한 Anomaly 산출: 위도대(1도) + 월별 평균 제거 ──
    df['lat_bin'] = df['latitude'].round(0)
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    
    print("  [Calc] 위도대별 월평균 Baseline 산출 중...")
    baseline = df.groupby(['lat_bin', 'month'])['xco2'].transform('mean')
    df['xco2_anomaly'] = df['xco2'] - baseline
    
    # Anomaly 검증
    anom_mean = df['xco2_anomaly'].mean()
    anom_std  = df['xco2_anomaly'].std()
    print(f"  [Anomaly] μ={anom_mean:.4f}, σ={anom_std:.4f} ppm (|μ| < 0.01이면 성공)")

    return df

# ─────────────────────────────────────────────────────────────────
# 2. Physics-Informed Feature Engineering (힌트 제공)
# ─────────────────────────────────────────────────────────────────
def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 2: 물리 힌트(Hints) 및 파생 피처 생성")
    print("=" * 70)
    
    # 풍속 합성
    df['WS'] = np.sqrt(df['era5_u10']**2 + df['era5_v10']**2)
    df['WS_eff']  = np.maximum(df['WS'], U_EFF_MIN)
    df['BLH_eff'] = np.maximum(df['era5_blh'], H_EFF_MIN)
    
    # ── 핵심 물리 힌트 ──
    df['proxy_simple'] = df['tropomi_no2'] / df['WS_eff']  # 단순 확산
    df['proxy_volume'] = df['tropomi_no2'] / (df['WS_eff'] * df['BLH_eff'])  # 체적 확산
    df['proxy_log_index'] = np.log1p(df['tropomi_no2'] / df['WS_eff'])  # 로그 확산
    
    # 단위 풍향 및 시간 주기성
    df['u10_unit'] = df['era5_u10'] / df['WS_eff']
    df['v10_unit'] = df['era5_v10'] / df['WS_eff']
    
    df['doy'] = df['time'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)
    
    print("  [Done] 물리 힌트 피처 생성 완료.")
    return df

# ─────────────────────────────────────────────────────────────────
# 3. Train/Test 분할
# ─────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 3: Temporal Train/Test Split (8:2)")
    print("=" * 70)
    
    # 시간 순 정렬
    df = df.sort_values('time').reset_index(drop=True)
    
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()
    
    # PySR 서브샘플링 (성능과 시간의 타협)
    if len(df_train) > MAX_TRAIN_SAMPLES:
        df_train_sr = df_train.sample(n=MAX_TRAIN_SAMPLES, random_state=42)
    else:
        df_train_sr = df_train
        
    if len(df_test) > MAX_TEST_SAMPLES:
        df_test_sr = df_test.sample(n=MAX_TEST_SAMPLES, random_state=42)
    else:
        df_test_sr = df_test
        
    print(f"  Total Data: {len(df):,} rows")
    print(f"  PySR Train Sample: {len(df_train_sr):,} rows")
    print(f"  PySR Test Sample:  {len(df_test_sr):,} rows")
    
    return df_train_sr, df_test_sr

# ─────────────────────────────────────────────────────────────────
# 4. PySR 초고강도 실행
# ─────────────────────────────────────────────────────────────────
def run_pysr_extreme(df_train: pd.DataFrame, df_test: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 4: PySR Symbolic Regression — 초고강도 탐색 시작")
    print("=" * 70)
    
    from pysr import PySRRegressor
    
    feature_cols = [
        'tropomi_no2', 'WS_eff', 'BLH_eff', 'u10_unit', 'v10_unit',
        'proxy_simple', 'proxy_volume', 'proxy_log_index',
        'population_density', 'odiac_emission', 'latitude',
        'doy_sin', 'doy_cos'
    ]
    
    X_train_raw = df_train[feature_cols].values.astype(np.float64)
    y_train_raw = df_train['xco2_anomaly'].values.astype(np.float64)
    X_test_raw  = df_test[feature_cols].values.astype(np.float64)
    y_test_raw  = df_test['xco2_anomaly'].values.astype(np.float64)

    # 스케일링
    feat_scaler = StandardScaler()
    X_train = feat_scaler.fit_transform(X_train_raw)
    X_test  = feat_scaler.transform(X_test_raw)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_test  = y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "square", "sqrt", "log1p"],
        
        # 초고강도 설정
        niterations=500,
        populations=100,
        ncycles_per_iteration=1000,
        
        maxsize=25,
        nested_constraints={"exp": {"exp": 0}, "log1p": {"log1p": 0}},
        
        output_directory=OUT_DIR,
        random_state=42,
        model_selection="best",
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        warm_start=False,
        verbosity=1,
    )
    
    print(f"  [Config] Features: {feature_cols}")
    print(f"  [Start] 탐색을 시작합니다. (R² 0.7 목표)")
    
    model.fit(X_train, y_train, variable_names=feature_cols)
    
    return model, feature_cols, X_train, y_train, X_test, y_test, feat_scaler, y_scaler

# ─────────────────────────────────────────────────────────────────
# 5. 결과 분석 및 보고 (역변환 포함)
# ─────────────────────────────────────────────────────────────────
def finalize_results(model, feature_cols, X_test, y_test, y_scaler):
    print("\n" + "=" * 70)
    print("STEP 5: 최종 성능 평가 (역변환 ppm 기준)")
    print("=" * 70)
    
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"  ★ Final Test R²: {r2:.4f}")
    print(f"  ★ Final Test RMSE: {rmse:.4f} ppm")
    print(f"  ★ Best Equation: {model.get_best()['equation']}")
    
    # 시각화 및 리포트 저장 로직 (이전과 동일하되 경로만 v2로)
    # ... (생략 가능하나 이전 코드의 핵심 시각화 로직 포함 권장)
    return r2, rmse

if __name__ == "__main__":
    df = load_and_recover_data()
    df = engineer_physics_features(df)
    train_sr, test_sr = split_data(df)
    
    model, f_cols, X_tr, y_tr, X_te, y_te, f_scale, y_scale = run_pysr_extreme(train_sr, test_sr)
    finalize_results(model, f_cols, X_te, y_te, y_scale)
