"""
지역 편향 진단 스크립트
========================
anom_1d.parquet의 지역별 관측 분포를 빠르게 확인합니다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 경로 ──
BASE_DIR   = "/mnt/e/dataset/XCO2연구 데이터"
PARQUET_IN = os.path.join(BASE_DIR, "anomaly_output/anom_1d.parquet")
OUT_DIR    = os.path.join(BASE_DIR, "anomaly_output")

# ── 지역 구분 (경도 기준) ──
REGION_BINS = {
    "West_China":  (100.0, 115.0),   # 중국 서부·내륙
    "East_China":  (115.0, 128.0),   # 중국 동부·황해
    "Korea_Japan": (128.0, 145.0),   # 한국·일본
    "Far_East":    (145.0, 150.0),   # 러시아 극동 등
}

# ── 위도 구간 ──
LAT_BANDS = [(20, 30), (30, 40), (40, 50)]

# ══════════════════════════════════════════════════════════════════
print("=" * 65)
print("진단: anom_1d.parquet 지역 편향 분석")
print("=" * 65)

df = pd.read_parquet(PARQUET_IN)
print(f"\n  총 행수  : {len(df):,}")
print(f"  컬럼 목록: {list(df.columns)}")

# 위경도 컬럼 자동 탐지
lat_col = next((c for c in df.columns if c in ["latitude", "lat"]), None)
lon_col = next((c for c in df.columns if c in ["longitude", "lon"]), None)
print(f"  위경도 컬럼: lat={lat_col}, lon={lon_col}")

if lat_col is None or lon_col is None:
    print("  ⚠️  위경도 컬럼을 찾지 못했습니다. lat_idx/lon_idx로 대체합니다.")
    # lat_idx → 위도 중앙값으로 복원 (0.1° 해상도)
    df["latitude"]  = 20.05 + df["lat_idx"] * 0.1
    df["longitude"] = 100.05 + df["lon_idx"] * 0.1
    lat_col, lon_col = "latitude", "longitude"

# ── 1. 지역 레이블 부여 ──
df["region"] = "Other"
for name, (lo, hi) in REGION_BINS.items():
    df.loc[(df[lon_col] >= lo) & (df[lon_col] < hi), "region"] = name

# ── 2. 지역별 관측 수 ──
print("\n" + "─" * 65)
print("【1】 지역별 관측 수")
region_counts = df["region"].value_counts().sort_index()
total = len(df)
for region, cnt in region_counts.items():
    bar = "█" * int(cnt / total * 50)
    print(f"  {region:<15}: {cnt:>8,} 행  ({cnt/total*100:5.1f}%)  {bar}")

# ── 3. 위도대 × 지역 교차 분포 ──
print("\n" + "─" * 65)
print("【2】 위도대 × 지역 교차 분포")
df["lat_band"] = pd.cut(df[lat_col], bins=[20, 30, 40, 50],
                         labels=["20-30°N", "30-40°N", "40-50°N"])
cross = pd.crosstab(df["lat_band"], df["region"])
print(cross.to_string())
print(f"\n  행 합계: {cross.sum(axis=1).to_dict()}")

# ── 4. 격자당 관측 수 분포 ──
print("\n" + "─" * 65)
print("【3】 격자당 관측 수 분포 (지역별)")
grid_counts = df.groupby(["lat_idx", "lon_idx", "region"]).size().reset_index(name="n_obs")
for region in REGION_BINS:
    sub = grid_counts[grid_counts["region"] == region]["n_obs"]
    if len(sub) == 0:
        continue
    print(f"\n  {region}:")
    print(f"    격자 수 : {len(sub):,}")
    print(f"    관측/격자: mean={sub.mean():.1f}, median={sub.median():.0f}, "
          f"max={sub.max()}, p95={sub.quantile(0.95):.0f}")

# ── 5. 상위 과밀 격자 ──
print("\n" + "─" * 65)
print("【4】 관측 수 상위 20개 격자")
top20 = grid_counts.nlargest(20, "n_obs")[["lat_idx", "lon_idx", "region", "n_obs"]]
top20["lat"] = (20.05 + top20["lat_idx"] * 0.1).round(2)
top20["lon"] = (100.05 + top20["lon_idx"] * 0.1).round(2)
print(top20[["lat", "lon", "region", "n_obs"]].to_string(index=False))

# ── 6. 균형 지표: Gini 계수 (지역별 관측 비중) ──
def gini(arr):
    arr = np.sort(arr.astype(float))
    n = len(arr)
    return (2 * np.sum(np.arange(1, n+1) * arr) / (n * arr.sum()) - (n+1)/n)

region_shares = region_counts.values / total
g = gini(region_shares)
print(f"\n" + "─" * 65)
print(f"【5】 지역 불균형 지표")
print(f"  Gini 계수 (지역별 비중): {g:.4f}  "
      f"(0=완전균등, 1=극단편향)")
print(f"  권장 격자당 최대 관측 수 (p95 기준): "
      f"{int(grid_counts['n_obs'].quantile(0.95))}")

# ── 7. 시각화 저장 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 지역별 파이 차트
axes[0].pie(region_counts.values, labels=region_counts.index,
            autopct="%1.1f%%", startangle=90)
axes[0].set_title("지역별 관측 비중")

# (b) 격자당 관측 수 히스토그램
for region in REGION_BINS:
    sub = grid_counts[grid_counts["region"] == region]["n_obs"]
    if len(sub):
        axes[1].hist(sub, bins=50, alpha=0.6, label=region)
axes[1].set_xlabel("격자당 관측 수")
axes[1].set_ylabel("격자 수")
axes[1].set_title("격자당 관측 수 분포 (지역별)")
axes[1].legend(fontsize=8)
axes[1].set_yscale("log")

plt.tight_layout()
out_fig = os.path.join(OUT_DIR, "diagnose_regional_bias.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  그래프 저장: {out_fig}")

print("\n" + "=" * 65)
print("진단 완료")
print("=" * 65)
