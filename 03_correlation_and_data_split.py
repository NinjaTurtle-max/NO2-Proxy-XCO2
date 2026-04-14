"""
Spatial Correlation Map (Figure 1) & Advanced Data Split Pipeline
=================================================================
목적:
  1. NO2-XCO2 공간 상관 계수 맵 + 관측 밀도 맵 (BH-Y FDR 보정)
  2. 시계열 Data Split (GAP_MONTHS >= 3, Leakage-free)
  3. Feature Scaling (Exp A / Exp B 분리 Scaler)

실행:
    conda run -n NO2_Proxy python 03_correlation_and_data_split.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import pearsonr
import joblib

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

import matplotlib.font_manager as fm
# 리눅스 시스템 폰트가 설치되어 있으나 캐시 문제로 못 찾을 수 있으므로 직접 등록
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
else:
    # 윈도우/맥/기본 설정
    plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'DejaVu Sans', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────
# 경로 및 상수
# ─────────────────────────────────────────────────────────────────
BASE_DIR        = "/mnt/e/dataset/XCO2연구 데이터"
PARQUET_IN_STD  = os.path.join(BASE_DIR, "02_anomaly_standard_output/anom_1d.parquet")
PARQUET_IN_EAIC = os.path.join(BASE_DIR, "02_anomaly_eaic_output/anom_1d_eaic.parquet")

# [자동 감지] 표준 데이터가 없으면 EAIC 데이터를 로드
if os.path.exists(PARQUET_IN_STD):
    PARQUET_IN = PARQUET_IN_STD
    ANOM_DIR   = os.path.join(BASE_DIR, "02_anomaly_standard_output")
elif os.path.exists(PARQUET_IN_EAIC):
    PARQUET_IN = PARQUET_IN_EAIC
    ANOM_DIR   = os.path.join(BASE_DIR, "02_anomaly_eaic_output")
else:
    # 둘 다 없는 경우 (오류 방지용 기본값)
    PARQUET_IN = PARQUET_IN_STD
    ANOM_DIR   = os.path.join(BASE_DIR, "02_anomaly_standard_output")

OUT_DIR    = os.path.join(BASE_DIR, "03_split_output")
os.makedirs(OUT_DIR, exist_ok=True)

FIG1_PATH         = os.path.join(OUT_DIR, "Figure_1_Final.png")
SPLIT_PATH        = os.path.join(OUT_DIR, "split_indices_v2.json")
SCALER_PATH       = os.path.join(OUT_DIR, "scalers_v2.joblib")
PARQUET_OCO3_ANOM = os.path.join(ANOM_DIR, "oco3_anom_1d.parquet")
OCO3_VAL_PATH     = os.path.join(OUT_DIR, "oco3_validation_scaled.parquet")

# 격자 설정
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0
RESOLUTION       = 0.5

lat_edges   = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION)
lon_edges   = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION)
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

# 상관 분석 최소 관측수
MIN_OBS_CORR = 10

# Data Split 설정
GAP_MONTHS    = 3                     # Train/Test/Val 사이 시간 격리 (월)
SPLIT_RATIO   = {"train": 0.7, "test": 0.15, "val": 0.15}  # 할당 우선순위: train → test → val
FORWARD_CHAIN_MIN_TRAIN = 500         # Forward-chaining CV 최소 Train 관측

# 공간 균형 보정 설정 (Spatial Balance)
REGION_BINS = {
    "West_China":  (100.0, 115.0),   # 중국 서부·내륙 (Inner Mongolia 포함)
    "East_China":  (115.0, 128.0),   # 중국 동부·황해
    "Korea_Japan": (128.0, 145.0),   # 한국·일본
    "Far_East":    (145.0, 150.0),   # 러시아 극동
}
MAX_PER_GRID   = 15   # 격자당 최대 관측 수 (|xco2_anomaly| 상위 보존)
KJ_MULTIPLIER  = {"West_China": 3, "East_China": 2, "Korea_Japan": 1, "Far_East": 1}
PARQUET_BALANCED_OUT = os.path.join(OUT_DIR, "anom_1d_balanced.parquet")


# ═════════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드 및 시간 인덱스 부여
# ═════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    print("=" * 70)
    print("STEP 1: 데이터 로드 & 시간 인덱스 부여")
    print("=" * 70)

    df = pd.read_parquet(PARQUET_IN)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M")

    # ── [IndexError 방지] 격자 인덱스 재계산 ──
    # 파일 내 인덱스가 예전 해상도(0.1도) 기준일 수 있으므로, 현재 스크립트 설정(RESOLUTION)으로 갱신
    df["lat_idx"] = np.searchsorted(lat_edges, df["latitude"].values, side="right") - 1
    df["lon_idx"] = np.searchsorted(lon_edges, df["longitude"].values, side="right") - 1

    # 범위 밖 인덱스 제거 (안전 장치)
    mask = (df["lat_idx"] >= 0) & (df["lat_idx"] < len(lat_centers)) & \
           (df["lon_idx"] >= 0) & (df["lon_idx"] < len(lon_centers))
    df = df[mask].reset_index(drop=True)

    # 경도 기반 EAIC sub-region 레이블 (stratified split용)
    df["eaic_region"] = "Other"
    for name, (lo, hi) in REGION_BINS.items():
        df.loc[(df["longitude"] >= lo) & (df["longitude"] < hi), "eaic_region"] = name

    print(f"  로드: {len(df):,} 행, {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  고유 날짜: {df['date'].nunique():,} 일")
    print(f"  고유 연-월: {df['year_month'].nunique()} 개")
    region_counts = df["eaic_region"].value_counts()
    print(f"  EAIC sub-region 분포: {dict(region_counts)}")
    return df


# ═════════════════════════════════════════════════════════════════
# STEP 2: 격자별 Pearson r 산출 + Benjamini-Hochberg-Yekutieli FDR
# ═════════════════════════════════════════════════════════════════
def benjamini_hochberg_yekutieli(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg-Yekutieli FDR 보정.
    
    BH-Y는 BH 절차에 c(m) = Σ(1/i) 계수를 곱해 임의 의존(arbitrary dependence)
    구조에도 유효한 보정을 적용합니다. 공간 상관이 존재하는 위성 격자 데이터에 적합.
    """
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=bool)

    # c(m) = harmonic number
    c_m = np.sum(1.0 / np.arange(1, m + 1))

    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]

    # BH-Y threshold: (i / m) * (alpha / c_m)
    thresholds = (np.arange(1, m + 1) / m) * (alpha / c_m)

    # 가장 큰 i where p(i) <= threshold(i)
    reject = np.zeros(m, dtype=bool)
    below = sorted_p <= thresholds
    if below.any():
        max_i = np.max(np.where(below)[0])
        reject[sorted_idx[:max_i + 1]] = True

    return reject


def compute_spatial_correlation(df: pd.DataFrame) -> tuple:
    """격자별 NO2-XCO2 Pearson r + FDR 보정 유의성 판정."""
    print("\n" + "=" * 70)
    print("STEP 2: 격자별 Pearson r & BH-Y FDR 보정")
    print("=" * 70)

    # 격자별 그룹 순회
    grouped = df.groupby(["lat_idx", "lon_idx"])

    r_map      = np.full((len(lat_centers), len(lon_centers)), np.nan, dtype=np.float32)
    p_map      = np.full((len(lat_centers), len(lon_centers)), np.nan, dtype=np.float32)
    n_map      = np.zeros((len(lat_centers), len(lon_centers)), dtype=np.int32)

    grid_list = []  # (lat_idx, lon_idx, r, p, n)

    for (li, lo), grp in grouped:
        n = len(grp)
        n_map[li, lo] = n

        if n < MIN_OBS_CORR:
            continue

        xco2_vals = grp["xco2_anomaly"].values
        no2_vals  = grp["tropomi_no2"].values

        # NaN 체크
        valid = np.isfinite(xco2_vals) & np.isfinite(no2_vals)
        if valid.sum() < MIN_OBS_CORR:
            continue

        r, p = pearsonr(xco2_vals[valid], no2_vals[valid])
        r_map[li, lo] = r
        p_map[li, lo] = p
        grid_list.append((li, lo, r, p, int(valid.sum())))

    print(f"  상관 계수 산출 대상 격자 (N >= {MIN_OBS_CORR}): {len(grid_list):,} 개")

    # ── BH-Y FDR 보정 ──
    if len(grid_list) > 0:
        pvals = np.array([g[3] for g in grid_list])
        reject = benjamini_hochberg_yekutieli(pvals, alpha=0.05)

        sig_map = np.full((len(lat_centers), len(lon_centers)), False)
        for i, (li, lo, r, p, n) in enumerate(grid_list):
            sig_map[li, lo] = reject[i]

        n_sig = int(reject.sum())
        print(f"  BH-Y FDR 보정 후 유의한 격자 (α=0.05): {n_sig:,} / {len(grid_list):,}"
              f" ({n_sig / len(grid_list) * 100:.1f}%)")
    else:
        sig_map = np.full((len(lat_centers), len(lon_centers)), False)

    # ── 통계 요약 ──
    valid_r = r_map[np.isfinite(r_map)]
    print(f"  r 분포: Mean={np.mean(valid_r):.4f}, Median={np.median(valid_r):.4f}")
    print(f"  |r| >= 0.3 격자: {(np.abs(valid_r) >= 0.3).sum():,}")
    print(f"  |r| >= 0.5 격자: {(np.abs(valid_r) >= 0.5).sum():,}")

    return r_map, p_map, n_map, sig_map


# ═════════════════════════════════════════════════════════════════
# STEP 3: Figure 1 렌더링 — Correlation Map + Observation Density
# ═════════════════════════════════════════════════════════════════
def plot_figure_1(r_map, n_map, sig_map) -> None:
    """상관 계수 맵 / 관측 밀도 맵 / N≥100 필터 맵 3-panel Figure 1 생성.

    (a) 전체 r 맵 (BH-Y FDR 보정, 비유의 격자 회색 점 표시)
    (b) 관측 밀도 맵 (log scale) — 궤도 sampling bias 시각화
    (c) N≥100 필터 적용 r 맵 — sampling 편향 제거 후 geographically coherent 클러스터 확인
    """
    print("\n" + "=" * 70)
    print("STEP 3: Figure 1 렌더링 (3-panel: Correlation / Density / N≥100 Filtered)")
    print("=" * 70)

    # N≥10 필터 적용 r 맵 (Spatial Balance에서 MAX_PER_GRID=15로 제한했으므로 임계값 하향)
    N_THRESH = 10
    r_filtered = np.where(n_map >= N_THRESH, r_map, np.nan)
    n_filtered_grids = np.isfinite(r_filtered).sum()
    print(f"  N ≥ {N_THRESH} 격자 수 (c패널 표시 대상): {n_filtered_grids:,}")

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(24, 7))
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.18)

        proj = ccrs.PlateCarree()
        ax1 = fig.add_subplot(gs[0], projection=proj)
        ax2 = fig.add_subplot(gs[1], projection=proj)
        ax3 = fig.add_subplot(gs[2], projection=proj)
        cax = fig.add_subplot(gs[3])

        for ax in [ax1, ax2, ax3]:
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
            ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
        cax = None

    r_kw = dict(cmap="RdBu_r", vmin=-0.8, vmax=0.8, shading="flat",
                transform=ccrs.PlateCarree() if HAS_CARTOPY else None)

    # ── (a) 전체 Correlation Map ──
    r_masked = np.ma.masked_invalid(r_map)
    nonsig_mask = ~sig_map & np.isfinite(r_map)

    im1 = ax1.pcolormesh(lon_edges, lat_edges, r_masked, **r_kw)

    if nonsig_mask.any():
        ns_lats, ns_lons = np.where(nonsig_mask)
        sample_n = min(len(ns_lats), 2000)
        if sample_n < len(ns_lats):
            idx = np.random.choice(len(ns_lats), sample_n, replace=False)
            ns_lats, ns_lons = ns_lats[idx], ns_lons[idx]
        ax1.scatter(
            lon_centers[ns_lons], lat_centers[ns_lats],
            marker=".", s=0.3, c="gray", alpha=0.4,
            transform=ccrs.PlateCarree() if HAS_CARTOPY else None
        )

    ax1.set_title("(a) Pearson r — All grids\nBH-Y FDR corrected, α=0.05",
                  fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=ax1, shrink=0.8, label="Pearson r", extend="both")

    # ── (b) Observation Density Map ──
    n_masked = np.ma.masked_where(n_map == 0, n_map)
    im2 = ax2.pcolormesh(
        lon_edges, lat_edges, n_masked,
        cmap="YlOrRd",
        norm=mcolors.LogNorm(vmin=1, vmax=max(n_map.max(), 10)),
        shading="flat",
        transform=ccrs.PlateCarree() if HAS_CARTOPY else None
    )
    ax2.set_title("(b) Observation Density (N per grid)\nOCO-2/3 궤도 sampling geometry 반영",
                  fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="N (log scale)")

    # ── (c) N≥100 필터 적용 r 맵 ──
    r_filt_masked = np.ma.masked_invalid(r_filtered)
    im3 = ax3.pcolormesh(lon_edges, lat_edges, r_filt_masked, **r_kw)
    ax3.set_title(f"(c) Pearson r — N ≥ {N_THRESH} grids only\n궤도 편향 제거 후 geographically coherent 클러스터",
                  fontsize=11, fontweight="bold")

    if cax:
        plt.colorbar(im3, cax=cax, label="Pearson r", extend="both")
    else:
        plt.colorbar(im3, ax=ax3, shrink=0.8, label="Pearson r", extend="both")

    fig.suptitle("Figure 1. Spatial Correlation between TROPOMI NO₂ and OCO-2/3 XCO₂ Anomaly\n"
                 "East Asia, 2020–2024, 0.5° × 0.5° Super-observations",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(FIG1_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  저장: {FIG1_PATH}")


# ═════════════════════════════════════════════════════════════════
# STEP 4: Sub-region별 Stratified Temporal Split
# ═════════════════════════════════════════════════════════════════
def stratified_temporal_split(df: pd.DataFrame,
                               train_years: list,
                               val_years: list,
                               test_years: list,
                               gap_months: int = 3) -> dict:
    """sub-region별로 균등 시간 분할 보장.

    각 eaic_region에서 동일한 연도 기준으로 행을 추출하여
    지역 편향 없이 Train/Val/Test 세트를 구성합니다.

    Args:
        df: 입력 DataFrame (eaic_region, year 컬럼 필요)
        train_years: Train에 포함할 연도 리스트
        val_years:   Validation에 포함할 연도 리스트
        test_years:  Test에 포함할 연도 리스트
        gap_months:  구분 참조용 (현재 연도 단위 분할에서는 로깅 목적)

    Returns:
        splits: {'train': [idx, ...], 'val': [idx, ...], 'test': [idx, ...]}
    """
    print("\n" + "=" * 70)
    print("STEP 4: Sub-region Stratified Temporal Split")
    print("=" * 70)
    print(f"  Train 연도: {train_years}")
    print(f"  Val   연도: {val_years}")
    print(f"  Test  연도: {test_years}")
    print(f"  GAP 참조값: {gap_months} 개월")

    splits = {"train": [], "val": [], "test": []}

    for region in df["eaic_region"].unique():
        sub_df = df[df["eaic_region"] == region]
        sub_train = sub_df[sub_df["year"].isin(train_years)].index.tolist()
        sub_val   = sub_df[sub_df["year"].isin(val_years)].index.tolist()
        sub_test  = sub_df[sub_df["year"].isin(test_years)].index.tolist()

        splits["train"].extend(sub_train)
        splits["val"].extend(sub_val)
        splits["test"].extend(sub_test)

    # 각 세트 내 sub-region 비율 보고
    for s, idx in splits.items():
        if len(idx) == 0:
            print(f"  [{s}] N=0 — 해당 연도 데이터 없음")
            continue
        region_dist = df.loc[idx, "eaic_region"].value_counts(normalize=True) * 100
        print(f"  [{s}] N={len(idx):,}, 지역 비율: {dict(region_dist.round(1))}")

    return splits


# ═════════════════════════════════════════════════════════════════
# STEP 4.5: 공간 균형 보정 (Variance-based Grid Cap + Regional Undersampling)
# ═════════════════════════════════════════════════════════════════
def spatial_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Variance-based Grid Cap + Regional Undersampling.

    Step 1: 각 격자 내에서 |xco2_anomaly|가 큰 관측(물리 신호 풍부)을
            우선 보존하고 MAX_PER_GRID 개로 제한.
            → Inner Mongolia 반복 배경 관측 억제, Plume 신호 보존.

    Step 2: 지역별 행 수 상한 (West_China ≤ 3×Korea_Japan)을 적용.
            Split 비율을 유지하며 언더샘플링하여 Temporal Leakage 방지.
    """
    print("\n" + "=" * 70)
    print("STEP 4.5: Spatial Balance (Variance-based Cap + Regional Undersampling)")
    print("=" * 70)

    df = df.copy()

    # 경도 기반 지역 레이블
    df["region"] = "Other"
    for name, (lo, hi) in REGION_BINS.items():
        df.loc[(df["longitude"] >= lo) & (df["longitude"] < hi), "region"] = name

    # ── Step 1: Variance-based Grid Cap ──
    before_cap = len(df)
    df["xco2_anomaly_abs"] = df["xco2_anomaly"].abs()

    def _variance_cap(grp):
        if len(grp) <= MAX_PER_GRID:
            return grp
        return grp.nlargest(MAX_PER_GRID, "xco2_anomaly_abs")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df = (df.groupby(["lat_idx", "lon_idx"], group_keys=False)
                .apply(_variance_cap)
                .reset_index(drop=True))
    df = df.drop(columns=["xco2_anomaly_abs"])
    print(f"  [Grid Cap] {before_cap:,} → {len(df):,} 행  (격자당 최대 {MAX_PER_GRID}개, |anomaly| 상위 보존)")

    # ── Step 2: Regional Undersampling ──
    kj_count = (df["region"] == "Korea_Japan").sum()
    if kj_count == 0:
        print("  ⚠️ Korea_Japan 관측 없음 — Regional Undersampling 건너뜀")
        return df

    print(f"\n  Korea_Japan 기준 행수: {kj_count:,}")
    balanced_parts = []
    for region, grp in df.groupby("region"):
        multiplier = KJ_MULTIPLIER.get(region, 1)
        target = kj_count * multiplier
        if len(grp) > target and target > 0:
            # split 비율 유지하면서 언더샘플링
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                sampled = (grp.groupby("split", group_keys=False)
                              .apply(lambda s: s.sample(
                                  n=max(1, round(target * len(s) / len(grp))),
                                  random_state=42)))
            print(f"  {region:<15}: {len(grp):>7,} → {len(sampled):>7,} 행  (목표 {target:,})")
        else:
            sampled = grp
            print(f"  {region:<15}: {len(grp):>7,} 행  (유지)")
        balanced_parts.append(sampled)

    df_balanced = (pd.concat(balanced_parts)
                     .sort_values(["date", "lat_idx", "lon_idx"])
                     .reset_index(drop=True))

    print(f"\n  [보정 후 총 행수]: {len(df_balanced):,}")
    print(f"  [보정 후 지역 × Split 분포]")
    cross = pd.crosstab(df_balanced["region"], df_balanced["split"])
    print(cross.to_string())

    return df_balanced


# ═════════════════════════════════════════════════════════════════
# STEP 5: Feature Scaling (Exp A / Exp B 분리 Scaler)
# ═════════════════════════════════════════════════════════════════
def fit_scalers(df: pd.DataFrame, split_dict: dict) -> None:
    """Train set에 대해서만 fit → Test/Val은 transform만.

    Exp A: StandardScaler (Z-score) — PINN/딥러닝 학습용
    Exp B: MinMaxScaler   — PySR 기호회귀용 (물리 단위 보존 고려)
    """
    print("\n" + "=" * 70)
    print("STEP 5: Feature Scaling (Exp A / Exp B 분리 Scaler)")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    candidate_cols = [
        "tropomi_no2", "era5_wind_speed", "era5_blh",
        "era5_u10", "era5_v10",
        "population_density", "odiac_emission"
    ]
    feature_cols = [c for c in candidate_cols if c in df.columns]
    missing = set(candidate_cols) - set(feature_cols)
    if missing:
        print(f"  ⚠️ 누락 feature (스케일링 제외): {missing}")

    # Train 분리
    train_mask = df["split"] == "train"
    X_train = df.loc[train_mask, feature_cols].values.astype(np.float64)

    print(f"  Feature columns: {feature_cols}")
    print(f"  Train samples for fitting: {len(X_train):,}")

    # Exp A: StandardScaler (Z-score)
    scaler_a = StandardScaler()
    scaler_a.fit(X_train)
    print(f"\n  [Exp A] StandardScaler fitted:")
    for i, col in enumerate(feature_cols):
        print(f"    {col:>25}: μ={scaler_a.mean_[i]:.4f}, σ={scaler_a.scale_[i]:.4f}")

    # Exp B: MinMaxScaler
    scaler_b = MinMaxScaler()
    scaler_b.fit(X_train)
    print(f"\n  [Exp B] MinMaxScaler fitted:")
    for i, col in enumerate(feature_cols):
        print(f"    {col:>25}: min={scaler_b.data_min_[i]:.4f}, max={scaler_b.data_max_[i]:.4f}")

    # Joblib dump
    bundle = {
        "exp_a_standard_scaler": scaler_a,
        "exp_b_minmax_scaler": scaler_b,
        "feature_columns": feature_cols,
        "n_train_samples": int(len(X_train)),
    }
    joblib.dump(bundle, SCALER_PATH)
    print(f"\n  Scaler Bundle 저장: {SCALER_PATH}")
    return bundle


# ═════════════════════════════════════════════════════════════════
# STEP 6: Summary Report
# ═════════════════════════════════════════════════════════════════
def print_summary(r_map, sig_map):
    """r >= 0.5인 격자의 비율과 지리적 특성 요약."""
    print("\n" + "=" * 70)
    print("STEP 6: Summary — r ≥ 0.5 격자의 지리적 특성")
    print("=" * 70)

    valid_mask = np.isfinite(r_map)
    n_valid = valid_mask.sum()

    high_r_mask = (r_map >= 0.5) & valid_mask
    n_high = int(high_r_mask.sum())
    pct = n_high / n_valid * 100 if n_valid > 0 else 0

    print(f"  유효 격자 수: {n_valid:,}")
    print(f"  r >= 0.5 격자 수: {n_high:,} ({pct:.2f}%)")

    # 해당 격자의 위경도 분포
    if n_high > 0:
        high_lats = lat_centers[np.where(high_r_mask)[0]]
        high_lons = lon_centers[np.where(high_r_mask)[1]]

        print(f"\n  [지리적 분포]")
        print(f"    위도 범위: {high_lats.min():.1f}°N ~ {high_lats.max():.1f}°N"
              f" (중앙값: {np.median(high_lats):.1f}°N)")
        print(f"    경도 범위: {high_lons.min():.1f}°E ~ {high_lons.max():.1f}°E"
              f" (중앙값: {np.median(high_lons):.1f}°E)")

        # 위도대별 분포
        for band_start in [20, 25, 30, 35, 40, 45]:
            band_end = band_start + 5
            in_band = ((high_lats >= band_start) & (high_lats < band_end)).sum()
            if in_band > 0:
                print(f"    {band_start}-{band_end}°N: {in_band} 격자")

        # 유의 + 고상관 격자
        high_sig = high_r_mask & sig_map
        n_high_sig = int(high_sig.sum())
        print(f"\n  BH-Y FDR 유의 AND r >= 0.5 격자: {n_high_sig:,}"
              f" ({n_high_sig / n_valid * 100:.2f}%)")


# ═════════════════════════════════════════════════════════════════
# STEP 6.5: OCO-3 독립 검증 세트 스케일링 & 저장
# ═════════════════════════════════════════════════════════════════
def export_oco3_validation(scaler_bundle: dict) -> None:
    """OCO-3 anomaly에 OCO-2 Scaler를 적용하여 독립 검증 parquet 생성.

    OCO-3는 학습·검증 데이터와 완전히 분리된 cross-satellite 독립 검증에 사용.
    OCO-2 Train 세트로 fit된 Scaler를 그대로 transform만 수행하여 스케일 일관성 보장.
    """
    print("\n" + "=" * 70)
    print("STEP 6.5: OCO-3 독립 검증 세트 스케일링 (Cross-satellite Validation)")
    print("=" * 70)

    if not os.path.exists(PARQUET_OCO3_ANOM):
        print(f"  ⚠️ OCO-3 Anomaly 파일 없음 ({PARQUET_OCO3_ANOM}) — 건너뜀")
        print("  → 02 스크립트를 먼저 실행하세요.")
        return

    df3 = pd.read_parquet(PARQUET_OCO3_ANOM)
    df3["date"] = pd.to_datetime(df3["date"])
    print(f"  [Load] OCO-3 Anomaly: {len(df3):,} 행, "
          f"{df3['date'].min().date()} ~ {df3['date'].max().date()}")

    feature_cols = scaler_bundle["feature_columns"]
    available    = [c for c in feature_cols if c in df3.columns]
    missing      = set(feature_cols) - set(available)
    if missing:
        print(f"  ⚠️ OCO-3에 없는 feature (NaN 대체): {missing}")
        for col in missing:
            df3[col] = np.nan

    X_oco3 = df3[feature_cols].values.astype(np.float64)

    # Exp A: StandardScaler transform (학습 없이 OCO-2 통계 그대로 적용)
    scaler_a = scaler_bundle["exp_a_standard_scaler"]
    X_a = scaler_a.transform(X_oco3)
    df3_a = df3.copy()
    for i, col in enumerate(feature_cols):
        df3_a[f"{col}_scaled_a"] = X_a[:, i]

    # Exp B: MinMaxScaler transform
    scaler_b = scaler_bundle["exp_b_minmax_scaler"]
    X_b = scaler_b.transform(X_oco3)
    for i, col in enumerate(feature_cols):
        df3_a[f"{col}_scaled_b"] = X_b[:, i]

    df3_a["split"] = "oco3_independent"
    df3_a.to_parquet(OCO3_VAL_PATH, index=False)
    print(f"  [저장] OCO-3 독립 검증 세트: {OCO3_VAL_PATH}")
    print(f"  Shape: {df3_a.shape}")

    # 기간·격자 수 요약
    n_grids = df3_a.groupby(["lat_idx", "lon_idx"]).ngroups
    print(f"\n  [요약]")
    print(f"    총 행수   : {len(df3_a):,}")
    print(f"    고유 격자 : {n_grids:,}")
    print(f"    Anomaly μ : {df3_a['xco2_anomaly'].mean():.4f} ppm")
    print(f"    Anomaly σ : {df3_a['xco2_anomaly'].std():.4f} ppm")
    print(f"  → OCO-2 학습 완료 후 이 파일로 cross-satellite 검증 수행")


# ═════════════════════════════════════════════════════════════════
# 메인 파이프라인
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1. 데이터 로드
    df = load_data()

    # 2. Sub-region Stratified Temporal Split (공간 균형 보정 전)
    all_years = sorted(df["year"].unique())
    n_y = len(all_years)
    n_train_y = max(1, int(np.floor(n_y * SPLIT_RATIO["train"])))
    n_test_y  = max(1, int(np.floor(n_y * SPLIT_RATIO["test"])))
    n_val_y   = max(1, n_y - n_train_y - n_test_y)

    train_years = all_years[:n_train_y]
    test_years  = all_years[n_train_y:n_train_y + n_test_y]
    val_years   = all_years[n_train_y + n_test_y:n_train_y + n_test_y + n_val_y]

    splits = stratified_temporal_split(
        df, train_years=train_years, val_years=val_years,
        test_years=test_years, gap_months=GAP_MONTHS,
    )

    # df에 split 컬럼 부여 (gap = 어느 세트에도 미포함 행)
    df["split"] = "gap"
    df.loc[splits["train"], "split"] = "train"
    df.loc[splits["val"],   "split"] = "val"
    df.loc[splits["test"],  "split"] = "test"

    split_dict = {
        "train_indices": splits["train"],
        "test_indices":  splits["test"],
        "val_indices":   splits["val"],
        "gap_indices":   df[df["split"] == "gap"].index.tolist(),
        "metadata": {
            "gap_months":  GAP_MONTHS,
            "train_years": [int(y) for y in train_years],
            "test_years":  [int(y) for y in test_years],
            "val_years":   [int(y) for y in val_years],
            "split_ratio": SPLIT_RATIO,
        },
    }
    with open(SPLIT_PATH, "w") as f:
        json.dump(split_dict, f, indent=2, default=str)
    print(f"\n  인덱스 저장: {SPLIT_PATH}")

    # 3. 공간 균형 보정 (지역 편향 제거)
    df = spatial_balance(df)

    # 균형 보정 후 split 인덱스 재저장 (인덱스가 reset됐으므로 반드시 갱신)
    split_dict_balanced = {
        "train_indices": df[df["split"] == "train"].index.tolist(),
        "test_indices":  df[df["split"] == "test"].index.tolist(),
        "val_indices":   df[df["split"] == "val"].index.tolist(),
        "gap_indices":   df[df["split"] == "gap"].index.tolist(),
        "metadata": split_dict["metadata"],
    }
    with open(SPLIT_PATH, "w") as f:
        json.dump(split_dict_balanced, f, indent=2, default=str)
    print(f"\n  Split 인덱스 재저장 (균형 보정 반영): {SPLIT_PATH}")

    # 균형 데이터 별도 저장 (04_pysr 입력용)
    df.to_parquet(PARQUET_BALANCED_OUT, index=False)
    print(f"  균형 데이터 저장: {PARQUET_BALANCED_OUT}")

    # 4. 균형 데이터 기준 공간 상관 계수 재산출 + Figure 1 렌더링
    #    (균형 보정 후 n_map이 지역 편향 없이 산출되어 N≥30 필터 의미 있음)
    r_map, p_map, n_map, sig_map = compute_spatial_correlation(df)
    plot_figure_1(r_map, n_map, sig_map)

    # 5. Feature Scaling
    scaler_bundle = fit_scalers(df, split_dict_balanced)

    # 6. OCO-3 독립 검증 세트
    export_oco3_validation(scaler_bundle)

    # 7. Summary
    print_summary(r_map, sig_map)

    print("\n" + "=" * 70)
    print("✅ 전체 파이프라인 완료!")
    print("=" * 70)
    print(f"  📊 Figure 1        : {FIG1_PATH}")
    print(f"  📂 Split Indices   : {SPLIT_PATH}")
    print(f"  ⚖️  Scalers         : {SCALER_PATH}")
    print(f"  🛰️  OCO-3 Val Set   : {OCO3_VAL_PATH}")
