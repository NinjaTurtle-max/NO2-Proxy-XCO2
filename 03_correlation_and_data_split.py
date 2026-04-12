"""
Spatial Correlation Map (Figure 1) & Advanced Data Split Pipeline
=================================================================
목적:
  1. NO2-XCO2 공간 상관 계수 맵 + 관측 밀도 맵 (BH-Y FDR 보정)
  2. 시계열 Data Split (GAP_MONTHS >= 3, Leakage-free)
  3. Feature Scaling (Exp A / Exp B 분리 Scaler)

실행:
    conda run -n NO2_Proxy python correlation_and_split.py
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

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────
# 경로 및 상수
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = "/mnt/e/dataset/XCO2연구 데이터"
PARQUET_IN = os.path.join(BASE_DIR, "anomaly_output/super_obs_dataset.parquet")
OUT_DIR    = os.path.join(BASE_DIR, "anomaly_output")
os.makedirs(OUT_DIR, exist_ok=True)

FIG1_PATH    = os.path.join(OUT_DIR, "Figure_1_Final.png")
SPLIT_PATH   = os.path.join(OUT_DIR, "split_indices_v2.json")
SCALER_PATH  = os.path.join(OUT_DIR, "scalers_v2.joblib")

# 격자 설정
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0
RESOLUTION       = 0.1

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

    print(f"  로드: {len(df):,} 행, {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  고유 날짜: {df['date'].nunique():,} 일")
    print(f"  고유 연-월: {df['year_month'].nunique()} 개")
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

        xco2_vals = grp["xco2"].values
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
    """상관 계수 맵과 관측 밀도 맵을 병기한 Figure 1 생성."""
    print("\n" + "=" * 70)
    print("STEP 3: Figure 1 렌더링 (Correlation + Density)")
    print("=" * 70)

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(18, 7))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)

        proj = ccrs.PlateCarree()
        ax1 = fig.add_subplot(gs[0], projection=proj)
        ax2 = fig.add_subplot(gs[1], projection=proj)
        cax = fig.add_subplot(gs[2])

        for ax in [ax1, ax2]:
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
            ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        cax = None

    # ── (a) Correlation Map ──
    r_masked = np.ma.masked_invalid(r_map)

    # 유의하지 않은 격자에 해칭(반투명 마스크)
    nonsig_mask = ~sig_map & np.isfinite(r_map)

    im1 = ax1.pcolormesh(
        lon_edges, lat_edges, r_masked,
        cmap="RdBu_r", vmin=-0.8, vmax=0.8,
        shading="flat",
        transform=ccrs.PlateCarree() if HAS_CARTOPY else None
    )

    # 비유의 격자에 X 마크 오버레이 (sparse sampling)
    if nonsig_mask.any():
        ns_lats, ns_lons = np.where(nonsig_mask)
        # 너무 많을 수 있으므로 최대 2000개만 표시
        sample_n = min(len(ns_lats), 2000)
        if sample_n < len(ns_lats):
            idx = np.random.choice(len(ns_lats), sample_n, replace=False)
            ns_lats, ns_lons = ns_lats[idx], ns_lons[idx]
        ax1.scatter(
            lon_centers[ns_lons], lat_centers[ns_lats],
            marker=".", s=0.3, c="gray", alpha=0.4,
            transform=ccrs.PlateCarree() if HAS_CARTOPY else None
        )

    ax1.set_title("(a) Pearson r (NO₂ vs XCO₂)\nBH-Y FDR corrected, α=0.05",
                   fontsize=12, fontweight="bold")

    if cax:
        plt.colorbar(im1, cax=cax, label="Pearson r", extend="both")
    else:
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
    ax2.set_title("(b) Observation Density (N per Grid Cell)",
                   fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="N (log scale)")

    fig.suptitle("Figure 1. Spatial Correlation between TROPOMI NO₂ and OCO-2 XCO₂\n"
                 "East Asia, 2020–2024, 0.1° × 0.1° Super-observations",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(FIG1_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  저장: {FIG1_PATH}")


# ═════════════════════════════════════════════════════════════════
# STEP 4: 시간 기반 Data Split (GAP_MONTHS >= 3, Leakage-free)
# ═════════════════════════════════════════════════════════════════
def temporal_split(df: pd.DataFrame) -> dict:
    """시계열 기반 Train/Test/Val 분할 (GAP >= 3개월).

    할당 우선순위: Train → Test → Val 순서로 블록을 확보합니다.
    이를 통해 Train이 항상 가장 많은 데이터를 확보하며,
    Test와 Val은 미래 시점에서 추출됩니다.

    시계열 구조:
        [=== TRAIN ===] ... GAP ... [= TEST =] ... GAP ... [= VAL =]
    """
    print("\n" + "=" * 70)
    print("STEP 4: Temporal Data Split (GAP_MONTHS >= 3)")
    print("=" * 70)

    # 연월 단위 정렬
    all_ym = sorted(df["year_month"].unique())
    n_months = len(all_ym)
    print(f"  총 연-월 블록: {n_months} 개 ({all_ym[0]} ~ {all_ym[-1]})")

    # 유효 연산 월(갭 제외): n_months - 2*GAP_MONTHS
    usable = n_months - 2 * GAP_MONTHS
    if usable <= 0:
        raise ValueError(f"데이터 기간({n_months}개월)이 GAP({GAP_MONTHS}*2)보다 짧습니다.")

    n_train = int(np.floor(usable * SPLIT_RATIO["train"]))
    n_test  = int(np.floor(usable * SPLIT_RATIO["test"]))
    n_val   = usable - n_train - n_test  # 나머지는 Val에 흡수

    # 블록 할당
    train_months = all_ym[:n_train]
    test_start   = n_train + GAP_MONTHS
    test_months  = all_ym[test_start:test_start + n_test]
    val_start    = test_start + n_test + GAP_MONTHS
    val_months   = all_ym[val_start:val_start + n_val]
    gap1_months  = all_ym[n_train:test_start]
    gap2_months  = all_ym[test_start + n_test:val_start]

    print(f"\n  ┌─ Train : {train_months[0]} ~ {train_months[-1]} ({len(train_months)} 개월)")
    print(f"  │  GAP 1 : {gap1_months[0]} ~ {gap1_months[-1]} ({len(gap1_months)} 개월)")
    print(f"  ├─ Test  : {test_months[0]} ~ {test_months[-1]} ({len(test_months)} 개월)")
    print(f"  │  GAP 2 : {gap2_months[0]} ~ {gap2_months[-1]} ({len(gap2_months)} 개월)")
    print(f"  └─ Val   : {val_months[0]} ~ {val_months[-1]} ({len(val_months)} 개월)")

    # DataFrame 분할
    train_set = set(train_months)
    test_set  = set(test_months)
    val_set   = set(val_months)

    df["split"] = "gap"
    df.loc[df["year_month"].isin(train_set), "split"] = "train"
    df.loc[df["year_month"].isin(test_set),  "split"] = "test"
    df.loc[df["year_month"].isin(val_set),   "split"] = "val"

    # 행 수 보고
    counts = df["split"].value_counts()
    for s in ["train", "test", "val", "gap"]:
        c = counts.get(s, 0)
        pct = c / len(df) * 100
        print(f"  {s:>5}: {c:>8,} 행 ({pct:5.1f}%)")

    # Forward-chaining CV 제안
    train_count = counts.get("train", 0)
    if train_count < FORWARD_CHAIN_MIN_TRAIN:
        print(f"\n  ⚠️ Train set ({train_count}) < {FORWARD_CHAIN_MIN_TRAIN} — "
              "Forward-chaining CV를 권장합니다.")
        print("     → 시간순으로 확장하는 Expanding Window 방식을 채택하십시오.")
    else:
        print(f"\n  ✅ Train set ({train_count:,}) 충분 — 단일 Hold-out Split 유효")

    # JSON 저장 (인덱스 기반)
    split_dict = {
        "train_indices": df[df["split"] == "train"].index.tolist(),
        "test_indices":  df[df["split"] == "test"].index.tolist(),
        "val_indices":   df[df["split"] == "val"].index.tolist(),
        "gap_indices":   df[df["split"] == "gap"].index.tolist(),
        "metadata": {
            "gap_months": GAP_MONTHS,
            "train_period": f"{train_months[0]} ~ {train_months[-1]}",
            "test_period":  f"{test_months[0]} ~ {test_months[-1]}",
            "val_period":   f"{val_months[0]} ~ {val_months[-1]}",
            "split_ratio":  SPLIT_RATIO,
        },
    }

    with open(SPLIT_PATH, "w") as f:
        json.dump(split_dict, f, indent=2, default=str)
    print(f"\n  인덱스 저장: {SPLIT_PATH}")

    return df, split_dict


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

    feature_cols = [
        "tropomi_no2", "era5_wind_speed", "era5_blh",
        "era5_u10", "era5_v10",
        "population_density", "odiac_emission"
    ]

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
# 메인 파이프라인
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1. 데이터 로드
    df = load_data()

    # 2. 공간 상관 계수 + FDR 보정
    r_map, p_map, n_map, sig_map = compute_spatial_correlation(df)

    # 3. Figure 1 렌더링
    plot_figure_1(r_map, n_map, sig_map)

    # 4. 시계열 Data Split
    df, split_dict = temporal_split(df)

    # 5. Feature Scaling
    fit_scalers(df, split_dict)

    # 6. Summary
    print_summary(r_map, sig_map)

    print("\n" + "=" * 70)
    print("✅ 전체 파이프라인 완료!")
    print("=" * 70)
    print(f"  📊 Figure 1     : {FIG1_PATH}")
    print(f"  📂 Split Indices: {SPLIT_PATH}")
    print(f"  ⚖️  Scalers      : {SCALER_PATH}")
