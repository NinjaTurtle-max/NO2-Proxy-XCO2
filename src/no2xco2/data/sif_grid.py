"""TROPOSIF 화소 → 0.25° 달력 월 평균 격자 (A-6·C-1, 사용자 승인 2026-10-08; docs/decision_a6_sif_gate_2026-10-08.md D3–D7).

변수: sif743_dc (일주기 보정 SIF 743 nm, mW m-2 sr-1 nm-1).
격자: ERA5 와 같은 0.25° 121×201 (lat 20→50 오름차순, lon 100→150). 화소 중심을 가장 가까운 격자점(±0.125°)에 배정.
품질 조합(D4): 한 번 읽을 때 아래 조합의 합·개수를 모두 누적 → 조합 결정이 바뀌어도 NAS 를 다시 읽지 않는다.
  qa 하한 {0.5 (승인안), 1.0 (Guanter et al. 2021 §2.5 권고 — 사용자 결정 10-08 채택)} × cloud_fraction {없음, < 0.2 (§3.2 "accurate SIF mean")}
해양(D7): 그 해 전체에서 lc_mask ≠ 0 화소가 하나도 배정되지 않은 격자점 → 0. (lc_mask 0 = 수역 또는 처리 제외 마스크 화소 — 슬라이서가 마스크를 0 으로 채움)
육지 결측(D7): 앞뒤 달(m−1, m+1) 중 유한한 값의 평균. 1월·12월은 한쪽만. 그래도 결측이면 0 (개수는 상태 코드로 보고).
"""
import warnings

import numpy as np

from no2xco2.config import GRID_D, GRID_LAT0, GRID_LON0
from no2xco2.data.era5 import NLAT, NLON

UNITS = "mW m-2 sr-1 nm-1"
COLS = ["lat", "lon", "sif743_dc", "qa743", "cloud_fraction", "lc_mask"]  # 파일에서 읽는 열 (메모리 절약)
VARIANTS = {"qa05": (0.5, None), "qa10": (1.0, None), "qa05_cf02": (0.5, 0.2), "qa10_cf02": (1.0, 0.2)}
DEFAULT_VARIANT = "qa10"  # D4 사용자 결정 10-08: qa743 = 1.0, cloud_fraction 추가 거르기 없음 (설계표 §9)
OCEAN, OBS, FILLED, ZEROED = 0, 1, 2, 3  # 격자점 상태 코드


def grid_coords():
    lat = GRID_LAT0 + GRID_D * np.arange(NLAT); lon = GRID_LON0 + GRID_D * np.arange(NLON)
    return lat, lon


def node_index(lat, lon):
    """화소 → (i, j, 격자 안 여부). 가장 가까운 격자점."""
    i = np.rint((np.asarray(lat, np.float64) - GRID_LAT0) / GRID_D).astype(np.int64)
    j = np.rint((np.asarray(lon, np.float64) - GRID_LON0) / GRID_D).astype(np.int64)
    ok = (i >= 0) & (i < NLAT) & (j >= 0) & (j < NLON)
    return i, j, ok


def new_accum():
    z = lambda: np.zeros((12, NLAT, NLON), np.float64)  # noqa: E731
    acc = {f"sum_{v}": z() for v in VARIANTS} | {f"cnt_{v}": z() for v in VARIANTS}
    acc["n_land"] = z(); acc["n_all"] = z()  # 품질 무관 화소 수 (해양 판정용)
    return acc


def accumulate(acc, df, month):
    """한 파일(DataFrame, COLS 포함)을 month(1–12) 누적에 더한다."""
    i, j, ok = node_index(df["lat"].to_numpy(), df["lon"].to_numpy())
    m = month - 1
    flat = i * NLON + j
    def add(key, sel, w=None):
        acc[key][m] += np.bincount(flat[sel], weights=None if w is None else w[sel], minlength=NLAT * NLON).reshape(NLAT, NLON)
    lc = df["lc_mask"].to_numpy()
    add("n_all", ok); add("n_land", ok & (lc != 0))
    s = df["sif743_dc"].to_numpy(np.float64); qa = df["qa743"].to_numpy(); cf = df["cloud_fraction"].to_numpy()
    base = ok & np.isfinite(s) & np.isfinite(qa)
    for v, (qmin, cfmax) in VARIANTS.items():
        sel = base & (qa >= qmin)
        if cfmax is not None:
            sel &= np.isfinite(cf) & (cf < cfmax)
        add(f"sum_{v}", sel, s); add(f"cnt_{v}", sel)


def finalize(acc, variant=DEFAULT_VARIANT):
    """누적 → (grid[12,lat,lon] float32 NaN 없음, state[12,lat,lon] uint8, mean_raw[12,...] 채움 전)."""
    cnt = acc[f"cnt_{variant}"]
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(cnt > 0, acc[f"sum_{variant}"] / np.maximum(cnt, 1), np.nan)
    land = acc["n_land"].sum(axis=0) > 0  # 그 해 lc_mask ≠ 0 화소가 1개라도 있으면 육지
    grid = raw.copy(); state = np.full(raw.shape, OBS, np.uint8)
    for m in range(12):
        nb = [raw[k] for k in (m - 1, m + 1) if 0 <= k < 12]
        with warnings.catch_warnings():  # 두 달 모두 NaN 인 격자점의 "Mean of empty slice" 경고
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = np.nanmean(np.stack(nb), axis=0)
        miss = land & ~np.isfinite(raw[m])
        canfill = miss & np.isfinite(fill)
        grid[m][canfill] = fill[canfill]; state[m][canfill] = FILLED
        rest = miss & ~np.isfinite(fill)
        grid[m][rest] = 0.0; state[m][rest] = ZEROED
        grid[m][~land] = 0.0; state[m][~land] = OCEAN
    assert np.isfinite(grid).all()
    return grid.astype(np.float32), state, raw


def month_summary(acc, state, variant=DEFAULT_VARIANT):
    """월별 결측 보고 행 (D7 결측 비율)."""
    land = acc["n_land"].sum(axis=0) > 0  # 해안 혼합 판정은 하지 않음: lc_mask 0 에는 처리 제외(마스크) 육지 화소도 섞임 (10-08 측정: 육지 격자점 화소 중 lc≠0 중앙 61 %)
    rows = []
    for m in range(12):
        st = state[m]; n_land = int(land.sum())
        rows.append({"월": m + 1, "variant": variant, "육지 격자점": n_land,
                     "관측": int((st[land] == OBS).sum()), "관측 비율": float((st[land] == OBS).mean()),
                     "앞뒤 달 채움": int((st == FILLED).sum()), "0 처리": int((st == ZEROED).sum()),
                     "화소(품질 통과)": int(acc[f"cnt_{variant}"][m].sum()), "화소(전체)": int(acc["n_all"][m].sum())}
                    | {f"화소_{v}": int(acc[f"cnt_{v}"][m].sum()) for v in VARIANTS})  # 4조합 비교 자료
    return rows
