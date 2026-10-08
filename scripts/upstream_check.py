"""결정 11 ① 타당성 측정 (학습 없음): 문제 사건의 상류에 최근 훈련 사운딩이 있는가.

대상 3건 (연구책임자 세션 지정, 경계 = docs/viz_analysis_2026-09-22.md §1.1 표):
  7/9 궤도  time fold 2 test, 2020-07-09, 40.5–43°N 134–135°E (n 997, 우리 잔차 +5.89 / CT +0.17)
  6/18 셀   space fold 0 test, 2020-06-18, 39.5–40.5°N 139.25–140°E (n 278, 우리 +4.18 / CT +1.50)
  8/11 셀   space fold 0 test, 2020-08-11, 42–43.25°N 147.25–148°E (n 316, 우리 −4.89 / CT +0.12)
  09-24 정정: 09-22 판은 6/18·8/11 경계가 None 이라 그날 test 전체(n 711 / 958)를 대상으로 잡았다
  (결과 docs/results/upstream_check_2026-09-22.csv 는 그 정의의 값으로 보존).
방식 2종 × 창 3종(48·96·192 h):
  (A) 반경: 대상 중심에서 300·600 km 이내, 창 안의 L_train 사운딩 (해당 폴드 test·버퍼 제외 = nested_masks 의 tr)
  (B) 역궤적: 대상 중심을 ERA5 z100 바람(하층·경계층 — XCO₂ 컬럼 수송 대표성 없음, 보완 i 참조)으로 1 h 씩 거슬러 올라가며
      각 시각의 궤적점 반경 300 km 안의 tr 사운딩. 창 = 궤적점 시각 ±3 h, 단 상한은 대상 시각 t_c (반경 방식과 같은 인과 기준, QA D9)
출력: docs/results/upstream_check_<생성일>.csv (--out 기본값, 있으면 중단 — N-2 생성일 규약) + 표준출력 표.
"""
import argparse
import datetime
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from no2xco2.data import loader as L
from no2xco2.data.era5 import open_wind
from no2xco2.train import nested_masks

R = 6371.0
KM_DEG = 111.2
TARGETS = [
    # name, scheme, fold, date, lat0, lat1, lon0, lon1
    ("7/9 궤도", "time", 2, "2020-07-09", 40.5, 43.0, 134.0, 135.0),
    ("6/18 셀", "space", 0, "2020-06-18", 39.5, 40.5, 139.25, 140.0),
    ("8/11 셀", "space", 0, "2020-08-11", 42.0, 43.25, 147.25, 148.0),
]
WINDOWS_H = [48, 96, 192]
RADII_KM = [300, 600]


def load_months(yms, idx_dir):
    cols = ["row_idx", "latitude", "longitude", "label", "xco2", "time", "step_g", "fold_time", "fold_space", "space_block", "year"]
    return pd.concat([L.load_oco(m, idx_dir, columns=cols) for m in yms], ignore_index=True)


def back_trajectory(lat0, lon0, t_end, hours, era5_dir):
    """대상 지점에서 hours 시간 거슬러 올라간 궤적 [(time, lat, lon)]. ERA5 z100(경계층) 바람 — 하층 기준."""
    pts = []; lat, lon = lat0, lon0; t = pd.Timestamp(t_end)
    cache = {}
    for _ in range(hours):
        ym = f"{t.year}{t.month:02d}"
        if ym not in cache:
            ds = open_wind(os.path.join(era5_dir, f"era5_wind_{ym}_z100.nc"), ym)
            cache[ym] = (ds["u_pbl"].values, ds["v_pbl"].values, ds.lat.values, ds.lon.values); ds.close()
        U, V, la, lo = cache[ym]
        h = int((t - pd.Timestamp(f"{t.year}-{t.month:02d}-01")) / pd.Timedelta(hours=1))
        h = min(max(h, 0), U.shape[0] - 1)
        i = int(np.clip(round((lat - la[0]) / 0.25), 0, len(la) - 1)); j = int(np.clip(round((lon - lo[0]) / 0.25), 0, len(lo) - 1))
        u, v = float(U[h, i, j]), float(V[h, i, j])
        lat -= v * 3600 / (KM_DEG * 1000); lon -= u * 3600 / (KM_DEG * 1000 * np.cos(np.deg2rad(lat)))
        lat = float(np.clip(lat, 20, 50)); lon = float(np.clip(lon, 100, 150)); t -= pd.Timedelta(hours=1)
        pts.append((t, lat, lon))
    return pts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--idx-dir", default=L.DEFAULT_IDX); ap.add_argument("--era5-dir", default=L.DEFAULT_ERA5)
    ap.add_argument("--out", default=f"docs/results/upstream_check_{datetime.date.today():%Y-%m-%d}.csv")
    a = ap.parse_args()
    if os.path.exists(a.out):  # N-2: 결과 파일 덮어쓰기 없음
        raise FileExistsError(f"{a.out} 존재 — 다른 --out 을 지정")
    rows = []
    for name, scheme, fold, date, la0, la1, lo0, lo1 in TARGETS:
        d0 = pd.Timestamp(date); yms = sorted({f"{(d0 - pd.Timedelta(days=k)).year}{(d0 - pd.Timedelta(days=k)).month:02d}" for k in (0, 8, 10)})
        big = load_months(yms, a.idx_dir); lab = (big.label == "L_train").to_numpy()
        tr, _, te = nested_masks(big, scheme, fold, lab, 1)
        t = pd.to_datetime(big.time); same_day = (t.dt.date == d0.date()).to_numpy()
        lat_a, lon_a = big.latitude.to_numpy(), big.longitude.to_numpy()
        sel = te & same_day & (lat_a >= la0) & (lat_a <= la1) & (lon_a >= lo0) & (lon_a <= lo1)
        if sel.sum() == 0:
            print(f"{name}: 대상 사운딩 0 — 건너뜀"); continue
        lat_c = float(big.latitude.to_numpy()[sel].mean()); lon_c = float(big.longitude.to_numpy()[sel].mean())
        t_c = pd.Timestamp(t[sel].mean()); y_c = float(big.xco2.to_numpy()[sel].mean())
        print(f"\n=== {name}: n {int(sel.sum())} · 중심 {lat_c:.2f}°N {lon_c:.2f}°E · {t_c} · XCO₂ 평균 {y_c:.2f}")
        trd = big[tr]; ttr = pd.to_datetime(trd.time)
        tree = BallTree(np.deg2rad(trd[["latitude", "longitude"]].to_numpy()), metric="haversine")
        pts_all = back_trajectory(lat_c, lon_c, t_c, max(WINDOWS_H), a.era5_dir)  # 최장 창 궤적 1회 → 창별 pts_all[:W] 절단 (QA D10)
        for W in WINDOWS_H:
            in_win = ((ttr <= t_c) & (ttr >= t_c - pd.Timedelta(hours=W))).to_numpy()
            # (A) 반경
            for rad in RADII_KM:
                idx = tree.query_radius(np.deg2rad([[lat_c, lon_c]]), rad / R)[0]
                m = np.zeros(len(trd), bool); m[idx] = True; m &= in_win
                n = int(m.sum()); mu = float(trd.xco2.to_numpy()[m].mean()) if n else np.nan
                rows.append(dict(target=name, method=f"반경 {rad} km", window_h=W, n=n, xco2_mean=mu, diff=mu - y_c if n else np.nan))
            # (B) 역궤적 (하층 바람)
            hit = np.zeros(len(trd), bool)
            for tp, la, lo in pts_all[:W][::3]:  # 3 h 간격 샘플
                idx = tree.query_radius(np.deg2rad([[la, lo]]), 300 / R)[0]
                if len(idx):
                    win = ((ttr >= tp - pd.Timedelta(hours=3)) & (ttr <= min(tp + pd.Timedelta(hours=3), t_c))).to_numpy()
                    m = np.zeros(len(trd), bool); m[idx] = True; hit |= (m & win)
            n = int(hit.sum()); mu = float(trd.xco2.to_numpy()[hit].mean()) if n else np.nan
            rows.append(dict(target=name, method="역궤적 300 km (하층풍)", window_h=W, n=n, xco2_mean=mu, diff=mu - y_c if n else np.nan))
        rows.append(dict(target=name, method="대상", window_h=0, n=int(sel.sum()), xco2_mean=y_c, diff=0.0))
    df = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out), exist_ok=True); df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200); print("\n" + df.round(2).to_string(index=False)); print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
