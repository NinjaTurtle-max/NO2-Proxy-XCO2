"""기준선: 크리깅 · LightGBM. GNN 과 같은 분할(train.nested_masks space 0, vfold 1) · 같은 시험 사운딩.

크리깅: kriging.py (OLS 추세 + 시공간 변동함수 + 국소 단순 크리깅). 모형(단일 · 중첩) · 이웃 수 k 는 val 부표본 RMSE 로 고른다.
LightGBM: 입력 = 배경 11변수 + no2_24h(인과 NO₂, build_no2_features.py) + ERA5 경계층 u · v (사운딩 시각, GNN 과 같은 4노드 가중),
  목표 = xco2 − μ_train, 학습률 0.05, 상한 3,000회, 조기 종료 100회(val RMSE), num_leaves {31, 63, 255} 중 val 로 선택, seed 0.
출력 (<out>/): kriging/{pred_krig.parquet, variogram.csv, krige_params.json} · lgbm_no2_wind/{pred_lgbm_no2wind.parquet, model_lgbm_no2wind.txt} · summary.csv · info.json
사용: python scripts/baselines/run_baselines.py --idx-dir data/processed/index --no2-feat runs/no2_features/no2_features.parquet --out runs [--steps krige,lgbm]
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kriging as K  # noqa: E402

COLS = ["row_idx", "time", "label", "latitude", "longitude", "xco2", "doy", "hour", "step_h", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3",
        "bg_t2m", "bg_blh", "bg_sp", "bg_z850", "bg_thk", "fold_time", "fold_space", "space_block", "year"]
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)  # MB


def wind_at_soundings(big, months, era5_dir):
    """사운딩 시각의 ERA5 경계층 u · v (4노드 이중선형 가중)."""
    from no2xco2.data.era5 import open_wind
    u = np.full(len(big), np.nan); v = np.full(len(big), np.nan); ymr = big.ym.to_numpy()
    for ym in months:
        ii = np.flatnonzero(ymr == ym)
        with open_wind(os.path.join(era5_dir, f"era5_wind_{ym}_z100.nc"), ym) as W:
            U_, V_ = W.u_pbl.values, W.v_pbl.values
        t = big.step_h.to_numpy()[ii]; au = np.zeros(len(ii)); av = np.zeros(len(ii))
        for k in range(4):
            nk = big[f"n{k}"].to_numpy()[ii]; wk = big[f"w{k}"].to_numpy()[ii]; au += wk * U_[t, nk // 201, nk % 201]; av += wk * V_[t, nk // 201, nk % 201]
        u[ii] = au; v[ii] = av
    assert np.isfinite(u).all() and np.isfinite(v).all()
    return u, v


def main():
    ap = argparse.ArgumentParser(description="크리깅 · LightGBM 기준선")
    ap.add_argument("--idx-dir", default="data/processed/index"); ap.add_argument("--no2-feat", default="runs/no2_features/no2_features.parquet")
    ap.add_argument("--era5-dir", default="data/raw/era5_wind"); ap.add_argument("--out", default="runs")
    ap.add_argument("--pred-ref", default=None, help="선택: GNN pred parquet — 시험 행 집합이 같은지 확인")
    ap.add_argument("--steps", default="krige,lgbm"); ap.add_argument("--leaves", default="31,63,255"); ap.add_argument("--ks", default="16,32,64,128")
    ap.add_argument("--months", default="202001-202412"); ap.add_argument("--n-var", type=int, default=4000); ap.add_argument("--n-val", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); t0 = time.time()
    from no2xco2.data import loader as L
    from no2xco2.train import nested_masks
    a0, a1 = a.months.split("-"); months = [f"{y}{m:02d}" for y in range(2020, 2025) for m in range(1, 13) if a0 <= f"{y}{m:02d}" <= a1]
    big = pd.concat([L.load_oco(ym, a.idx_dir, columns=COLS) for ym in months], ignore_index=True); assert big.row_idx.is_unique
    y = big.xco2.to_numpy(np.float64); lab = (big.label == "L_train").to_numpy(); ri = big.row_idx.to_numpy()
    tr, va, te = nested_masks(big, "space", 0, lab, 1); mu = float(y[tr].mean())
    if a.pred_ref:
        ref = pd.read_parquet(a.pred_ref, columns=["row_idx", "test"])
        assert np.array_equal(np.sort(ri[te]), np.sort(ref.row_idx.to_numpy()[ref.test.to_numpy(bool)])), "시험 행이 GNN 과 다름"
    print(f"적재 {time.time() - t0:.0f}s · 행 {len(big):,} · train {tr.sum():,} / val {va.sum():,} / test {te.sum():,} · μ_train {mu:.4f} · RSS {rss():.0f} MB", flush=True)
    Xbg = L.bg_features(big).astype(np.float64); rows = []
    met = lambda p: dict(rmse=float(np.sqrt(np.mean((p - y[te]) ** 2))), bias=float(np.mean(p - y[te])), r2=float(1 - np.sum((p - y[te]) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)))
    if "krige" in a.steps:
        from scipy.spatial import cKDTree
        out = os.path.join(a.out, "kriging"); os.makedirs(out, exist_ok=True)
        t1 = time.time(); xyz = K.unit_xyz(big.latitude.to_numpy(), big.longitude.to_numpy())
        tdays = ((pd.to_datetime(big.time) - pd.Timestamp("2020-01-01")) / pd.Timedelta(days=1)).to_numpy(np.float64)
        m_, s_ = Xbg[tr].mean(0), Xbg[tr].std(0) + 1e-9; Z = np.column_stack([np.ones(len(big)), (Xbg - m_) / s_]); beta, *_ = np.linalg.lstsq(Z[tr], y[tr], rcond=None)
        trend = Z @ beta; res = y - trend; del Z
        ev = K.empirical_variogram(xyz[tr], res[tr], tdays[tr], a.n_var); ev.to_csv(os.path.join(out, "variogram.csv"), index=False)
        fits = {n: K.fit_variogram(ev, n) for n in (False, True)}; Xtr, ttr, rtr = xyz[tr], tdays[tr], res[tr]
        trees = {n: cKDTree(np.column_stack([K.R_EARTH * Xtr, fits[n][-1] * ttr])) for n in (False, True)}
        rng = np.random.default_rng(0); vi = rng.choice(np.flatnonzero(va), min(a.n_val, int(va.sum())), replace=False); best = (np.inf, None)
        for n in (False, True):
            for k in [int(v) for v in a.ks.split(",")]:
                pv, _ = K.predict_local(trees[n], Xtr, ttr, rtr, xyz[vi], tdays[vi], fits[n], n, k, fits[n][-1], 64, a.threads)
                r = float(np.sqrt(np.mean((trend[vi] + pv - y[vi]) ** 2))); print(f"  크리깅 val {'중첩' if n else '단일'} k={k}: {r:.4f}", flush=True)
                if r < best[0]: best = (r, (n, k))
        n, k = best[1]; ti = np.flatnonzero(te)
        with open(os.path.join(out, "krige_params.json"), "w") as fh:
            json.dump(dict(nested=bool(n), k=int(k), variogram={str(nn): [float(x) for x in fits[nn]] for nn in fits}, trend_beta=[float(x) for x in beta],
                           trend_mean=[float(x) for x in m_], trend_std=[float(x) for x in s_], features=list(L.BG_FEATS)), fh, indent=1)
        pk, _ = K.predict_local(trees[n], Xtr, ttr, rtr, xyz[ti], tdays[ti], fits[n], n, k, fits[n][-1], 64, a.threads); p = trend[ti] + pk
        pd.DataFrame(dict(row_idx=ri[ti], pred=p, pred_trend=trend[ti])).to_parquet(os.path.join(out, "pred_krig.parquet"), index=False)
        r = dict(model="kriging", choice=f"{'nested' if n else 'single'} k={k}", val_rmse=best[0], secs=time.time() - t1, rss_mb=rss(), **met(p), trend_rmse=met(trend[ti])["rmse"])
        rows.append(r); print(f"크리깅: test RMSE {r['rmse']:.4f} (추세만 {r['trend_rmse']:.4f}) · R² {r['r2']:.4f} · {r['secs']:.0f}s", flush=True)
    if "lgbm" in a.steps:
        import lightgbm as lgb
        out = os.path.join(a.out, "lgbm_no2_wind"); os.makedirs(out, exist_ok=True)
        big["ym"] = pd.to_datetime(big.time).dt.strftime("%Y%m")
        f = pd.read_parquet(a.no2_feat, columns=["row_idx", "ym", "no2_24h"]); f["ym"] = f.ym.astype(str)
        g = big[["row_idx", "ym"]].merge(f, on=["ym", "row_idx"], how="left", validate="one_to_one"); assert len(g) == len(big)
        g["u_pbl"], g["v_pbl"] = wind_at_soundings(big, months, a.era5_dir)
        P0 = dict(objective="regression", metric="rmse", learning_rate=0.05, deterministic=True, force_row_wise=True, num_threads=a.threads, verbosity=-1, max_bin=255,
                  bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, seed=0)
        extra = ["no2_24h", "u_pbl", "v_pbl"]; t1 = time.time()
        X = np.column_stack([Xbg] + [g[c].to_numpy(np.float64)[:, None] for c in extra]).astype(np.float32); names = list(L.BG_FEATS) + extra
        dtr = lgb.Dataset(X[tr], y[tr] - mu, feature_name=names, free_raw_data=False, params={"max_bin": 255, "feature_pre_filter": False}); dva = lgb.Dataset(X[va], y[va] - mu, reference=dtr, free_raw_data=False)
        best = None
        for lv in [int(x) for x in a.leaves.split(",")]:
            b = lgb.train(dict(P0, num_leaves=lv), dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            bi = b.best_iteration or b.current_iteration(); v = float(np.sqrt(np.mean((b.predict(X[va], num_iteration=bi) + mu - y[va]) ** 2)))
            print(f"  LightGBM leaves {lv}: best_iter {bi} val {v:.4f} · {time.time() - t1:.0f}s", flush=True)
            if best is None or v < best[0]: best = (v, lv, bi, b)
        p = best[3].predict(X[te], num_iteration=best[2]) + mu
        pd.DataFrame(dict(row_idx=ri[te], pred=p)).to_parquet(os.path.join(out, "pred_lgbm_no2wind.parquet"), index=False)
        best[3].save_model(os.path.join(out, "model_lgbm_no2wind.txt"), num_iteration=best[2])
        r = dict(model="LightGBM", choice=f"leaves {best[1]} iter {best[2]}", val_rmse=best[0], secs=time.time() - t1, rss_mb=rss(), **met(p)); rows.append(r)
        print(f"LightGBM: test RMSE {r['rmse']:.4f} · R² {r['r2']:.4f} · {r['secs']:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "baselines_summary.csv"), index=False)
    with open(os.path.join(a.out, "baselines_info.json"), "w") as fh:
        json.dump(dict(mu_train=mu, n=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())), total_sec=time.time() - t0), fh, indent=1)
    print(f"끝 · 총 {time.time() - t0:.0f}s → {a.out}", flush=True)


if __name__ == "__main__":
    main()
