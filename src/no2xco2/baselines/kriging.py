"""rung 1 대조군: 회귀 시공간 크리깅 (결정 11), 2020-01, 파일럿 2차와 동일 분할(17.nested_masks, vfold=1).

정보 계약: 훈련 사운딩의 XCO2 만 사용 (NO2 없음, 테스트 사운딩 없음).
  추세: OLS on 결정 8 허용 변수 (lat, lon, DOY, hour, t2m, blh, sp) — tr 로 적합.  → 'trend' 행 (자체 기준선)
  잔차 ST 변동함수: tr 부표본(4,000)에서 경험 변동함수 (거리 ≤1,000 km × 시차 ≤10 d) → 지수 metric 모형
       γ(h,u) = c0 + c1·(1 − exp(−sqrt(h² + (a·u)²)/r))  [단일] · c0 + c1·(…r1) + c2·(…r2) [중첩 2범위]
  예측: 국소 크리깅 — 스케일 좌표 (x km, y km, a·t d) cKDTree k-최근접 훈련 사운딩, k×k 단순 크리깅(잔차) + 추세.
       k ∈ {16, 32, 64, 128}, 모형 {단일, 중첩} 을 val 에서 선택.
거리 층: 테스트 사운딩 → 최근접 훈련 사운딩 haversine d_min (월 전체), 층 경계 0–20–67.5–250–500–1000–∞ km (설계 문서).
출력: <out>/rung1_{scheme}{fold}.parquet (row_idx, pred_trend, pred_krig, d_min, dt_min), strata_{scheme}{fold}.csv, 변동함수 파라미터.
Zeng 2014 / Sheng 2022 의 공분산 모형은 원문 미확인 — 여기서는 지수 metric·중첩 모형.
"""
import argparse, glob, os, time, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
import no2xco2.train as M17; import no2xco2.data.splits as splits_mod

NLON = 201; R_EARTH = 6371.0
STRATA = [0, 20, 67.5, 250, 500, 1000, np.inf]


def load(month="202001"):
    oco = pq.read_table(sorted(glob.glob(f"data/raw/pilot_{month}/oco_nodes/*.parquet")), columns=["row_idx", "latitude", "longitude", "time", "label", "xco2"]).to_pandas()
    eo = pq.read_table(f"data/processed/encoder_index/enc_oco_{month}.parquet", columns=["row_idx", "step_h", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3"]).to_pandas()
    df = oco.merge(eo, on="row_idx"); df = splits_mod.assign(df)
    from no2xco2.data.era5 import open_wind
    ds = open_wind(f"data/raw/era5_wind/era5_wind_{month}_z100.nc", month); T = ds.sizes["time"]  # 위도 오름차순 (2026-09-22 정정)
    t = np.clip(df.step_h.to_numpy(), 0, T - 1)
    for v in ("t2m", "blh", "sp"):
        A = ds[v].values; acc = np.zeros(len(df))
        for k in range(4):
            n = df[f"n{k}"].to_numpy(); acc += df[f"w{k}"].to_numpy() * A[t, n // NLON, n % NLON]
        df[v] = acc
    tt = pd.to_datetime(df.time); df["doy"] = tt.dt.dayofyear + tt.dt.hour / 24; df["hour"] = tt.dt.hour + tt.dt.minute / 60
    df["tdays"] = (tt - pd.Timestamp("2020-01-01")).dt.total_seconds() / 86400
    lat0 = np.deg2rad(35.0); df["x_km"] = np.deg2rad(df.longitude) * R_EARTH * np.cos(lat0); df["y_km"] = np.deg2rad(df.latitude) * R_EARTH
    return df


def trend_fit(df, tr):
    X = np.column_stack([np.ones(len(df)), df.latitude, df.longitude, df.doy, df.hour, df.t2m, df.blh, df.sp])
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9; sd[0] = 1; mu[0] = 0; Xs = (X - mu) / sd
    beta, *_ = np.linalg.lstsq(Xs[tr], df.xco2.to_numpy()[tr], rcond=None)
    return Xs @ beta


def empirical_variogram(P, res, tdays, n_sub=4000, hmax=1000, umax=10, seed=0):
    rng = np.random.default_rng(seed); i = rng.choice(len(P), min(n_sub, len(P)), replace=False)
    P, res, tdays = P[i], res[i], tdays[i]
    dx = P[:, None, 0] - P[None, :, 0]; dy = P[:, None, 1] - P[None, :, 1]; h = np.sqrt(dx ** 2 + dy ** 2); u = np.abs(tdays[:, None] - tdays[None, :])
    g = 0.5 * (res[:, None] - res[None, :]) ** 2; iu = np.triu_indices(len(P), 1); h, u, g = h[iu], u[iu], g[iu]
    hb = np.array([0, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 700, 1000]); ub = np.array([0, 0.5, 1, 2, 3, 5, 7, 10])
    rows = []
    for a, b in zip(hb[:-1], hb[1:]):
        for c, d in zip(ub[:-1], ub[1:]):
            m = (h >= a) & (h < b) & (u >= c) & (u < d)
            if m.sum() >= 30: rows.append(((a + b) / 2, (c + d) / 2, g[m].mean(), int(m.sum())))
    return pd.DataFrame(rows, columns=["h", "u", "gamma", "n"])


def gamma_model(p, h, u, nested):
    if nested:
        c0, c1, r1, c2, r2, a = p; d = np.sqrt(h ** 2 + (a * u) ** 2)
        return c0 + c1 * (1 - np.exp(-d / r1)) + c2 * (1 - np.exp(-d / r2))
    c0, c1, r, a = p; d = np.sqrt(h ** 2 + (a * u) ** 2)
    return c0 + c1 * (1 - np.exp(-d / r))


def fit_variogram(ev, nested):
    w = np.sqrt(ev.n.to_numpy()); s = ev.gamma.max()
    if nested: p0 = [0.1 * s, 0.4 * s, 30, 0.5 * s, 400, 50]; lb = [0, 0, 1, 0, 50, 1]; ub = [s, 2 * s, 200, 2 * s, 5000, 1000]
    else: p0 = [0.1 * s, 0.9 * s, 100, 50]; lb = [0, 0, 1, 1]; ub = [s, 2 * s, 5000, 1000]
    f = lambda p: w * (gamma_model(p, ev.h.to_numpy(), ev.u.to_numpy(), nested) - ev.gamma.to_numpy())
    r = least_squares(f, p0, bounds=(lb, ub)); return r.x


def cov_from_gamma(p, h, u, nested):
    sill = p[0] + p[1] + (p[3] if nested else 0)
    return sill - gamma_model(p, h, u, nested)


def local_kriging(Ptr, ttr, rtr, Pq, tq, p, nested, k):
    a = p[-1]; tree = cKDTree(np.column_stack([Ptr, a * ttr])); Q = np.column_stack([Pq, a * tq])
    dist, idx = tree.query(Q, k=k); out = np.zeros(len(Pq)); sill = p[0] + p[1] + (p[3] if nested else 0)
    for i in range(len(Pq)):
        nb = idx[i]; P = Ptr[nb]; t = ttr[nb]
        hh = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)); uu = np.abs(t[:, None] - t[None, :])
        K = cov_from_gamma(p, hh, uu, nested) + 1e-6 * sill * np.eye(k)
        h0 = np.sqrt(((P - Pq[i]) ** 2).sum(-1)); u0 = np.abs(t - tq[i]); k0 = cov_from_gamma(p, h0, u0, nested)
        try: w = np.linalg.solve(K, k0)
        except np.linalg.LinAlgError: w = np.full(k, 1 / k)
        out[i] = w @ rtr[nb]  # 단순 크리깅 (잔차 평균 0 가정; 추세는 OLS)
    return out


def haversine_min(lat_q, lon_q, lat_t, lon_t):
    """테스트 → 최근접 훈련 사운딩 거리(km) — 3D 단위벡터 KD-tree."""
    def xyz(la, lo):
        la, lo = np.deg2rad(la), np.deg2rad(lo); return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    tree = cKDTree(xyz(lat_t, lon_t)); d, i = tree.query(xyz(lat_q, lon_q)); return 2 * R_EARTH * np.arcsin(np.clip(d / 2, 0, 1)), i


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="time:2,space:0"); ap.add_argument("--vfold", type=int, default=1)
    ap.add_argument("--ks", default="16,32,64,128"); ap.add_argument("--out", default="experiments/rung1")
    ap.add_argument("--model-preds", default="experiments/pilot2_seeds")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    df = load(); y = df.xco2.to_numpy(); lab = (df.label == "L_train").to_numpy(); P = df[["x_km", "y_km"]].to_numpy(); td = df.tdays.to_numpy()
    print(f"사운딩 {len(df):,} · L_train {lab.sum():,} · 준비 {time.time()-t0:.0f}s")
    params_all = {}
    for cfg in a.configs.split(","):
        scheme, fold = cfg.split(":"); fold = int(fold); tr, val, te = M17.nested_masks(df, scheme, fold, lab, a.vfold)
        trend = trend_fit(df, tr); res = y - trend
        rm = lambda m, pred: float(np.sqrt(((pred - y)[m] ** 2).mean()))
        print(f"\n== {scheme} fold {fold}: train {tr.sum():,} / val {val.sum():,} / test {te.sum():,} · trend RMSE val {rm(val, trend):.3f} test {rm(te, trend):.3f}")
        ev = empirical_variogram(P[tr], res[tr], td[tr]); ev.to_csv(f"{a.out}/variogram_{scheme}{fold}.csv", index=False)
        fits = {}
        for nested in (False, True):
            p = fit_variogram(ev, nested); fits[nested] = p
            lbl = ("c0 c1 r1 c2 r2 a" if nested else "c0 c1 r a").split(); print(f"  변동함수 {'중첩' if nested else '단일'}: " + " ".join(f"{l}={v:.3g}" for l, v in zip(lbl, p)))
        best = (np.inf, None)
        for nested in (False, True):
            for k in [int(x) for x in a.ks.split(",")]:
                pv = trend[val] + local_kriging(P[tr], td[tr], res[tr], P[val], td[val], fits[nested], nested, k)
                r = float(np.sqrt(((pv - y[val]) ** 2).mean())); print(f"  val {'중첩' if nested else '단일'} k={k:3d}: RMSE {r:.3f}", flush=True)
                if r < best[0]: best = (r, (nested, k))
        nested, k = best[1]; print(f"  선택: {'중첩' if nested else '단일'} k={k} (val {best[0]:.3f})")
        pk = trend.copy(); pk[te] = trend[te] + local_kriging(P[tr], td[tr], res[tr], P[te], td[te], fits[nested], nested, k)
        dmin, inear = haversine_min(df.latitude.to_numpy()[te], df.longitude.to_numpy()[te], df.latitude.to_numpy()[tr], df.longitude.to_numpy()[tr])
        dtmin = np.abs(td[te] - td[tr][inear])
        params_all[cfg] = dict(nested=bool(nested), k=k, params=[float(v) for v in fits[nested]], val_rmse=best[0], trend_test=rm(te, trend), krig_test=rm(te, pk))
        print(f"  test RMSE: trend {rm(te, trend):.3f} · kriging {rm(te, pk):.3f}")
        # 모델 예측 (물리 3시드 평균 · 절제 3시드 평균) 결합
        out = pd.DataFrame(dict(row_idx=df.row_idx[te].to_numpy(), xco2=y[te], pred_trend=trend[te], pred_krig=pk[te], d_min=dmin, dt_min=dtmin))
        for kind in ("phys", "nophys"):
            fs = sorted(glob.glob(f"{a.model_preds}/pred_{scheme}{fold}_s*_{kind}.parquet"))
            if fs:
                m = pd.concat([pd.read_parquet(f)[["row_idx", "pred"]] for f in fs]).groupby("row_idx").pred.mean()
                out[f"pred_{kind}"] = out.row_idx.map(m).to_numpy()
        out.to_parquet(f"{a.out}/rung1_{scheme}{fold}.parquet", index=False)
        out["stratum"] = pd.cut(out.d_min, STRATA, right=False, labels=[f"{STRATA[i]}–{STRATA[i+1]}" for i in range(len(STRATA) - 1)])
        cols = [c for c in ("pred_trend", "pred_krig", "pred_phys", "pred_nophys") if c in out]
        rows = []
        for s_, g in out.groupby("stratum", observed=False):
            r = dict(stratum=str(s_), n=len(g), dt_min_med=float(g.dt_min.median()) if len(g) else np.nan)
            for c in cols: r[c.replace("pred_", "rmse_")] = float(np.sqrt(((g[c] - g.xco2) ** 2).mean())) if len(g) else np.nan
            rows.append(r)
        r = dict(stratum="전체", n=len(out), dt_min_med=float(out.dt_min.median()))
        for c in cols: r[c.replace("pred_", "rmse_")] = float(np.sqrt(((out[c] - out.xco2) ** 2).mean()))
        rows.append(r); st = pd.DataFrame(rows); st.to_csv(f"{a.out}/strata_{scheme}{fold}.csv", index=False)
        print(st.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    json.dump(params_all, open(f"{a.out}/params.json", "w"), indent=1, ensure_ascii=False); print(f"\n총 {time.time()-t0:.0f}s → {a.out}/")
