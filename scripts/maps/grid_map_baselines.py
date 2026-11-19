"""Fig. 3 지도 (기준선): 크리깅 · LightGBM 을 GNN 과 같은 사례일 · 같은 격자 질의(grid_202001_feats.parquet)에서 예측한다.
크리깅 = run_baselines.py 와 같은 계산 (OLS 추세 + 단일 지수 변동함수, k = 128; 변동함수는 저장된 variogram.csv 를 다시 맞춤).
LightGBM = 저장된 model_lgbm_no2wind.txt, 입력 = 배경 11변수 + no2_24h(같은 화소 · 4노드 가중 · 시각까지 24 h) + ERA5 경계층 u · v.
수용 검사: 같은 계산으로 2020-01 시험 사운딩을 예측해 저장된 예측을 재현해야 격자를 저장한다.
단계별로 따로 실행 (LightGBM 과 scipy 를 한 프로세스에 올리지 않음): python scripts/maps/grid_map_baselines.py krige | feats | lgbm"""
import importlib.util, os, sys
import numpy as np, pandas as pd

D = "runs/maps"; B = "runs"; KRIGE_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines", "kriging.py")
IDX60 = "data/processed/index"; MONTHS = [f"{y}{m:02d}" for y in range(2020, 2025) for m in range(1, 13)]
if sys.argv[1] != "lgbm":
    from no2xco2.data import loader as L
    Z = np.load(f"{D}/grid_map_202001_m60.npz"); tr = Z["tr"]; g = pd.read_parquet(f"{D}/grid_202001_feats.parquet")
    n = np.cumsum([0] + [len(pd.read_parquet(f"{IDX60}/oco_{m}.parquet", columns=["row_idx"])) for m in MONTHS]); assert n[-1] == len(tr)
    jan = L.load_oco("202001", IDX60, columns=L.OCO_COLS)
    te_j = (jan.fold_space.to_numpy() == 0) & (jan.label == "L_train").to_numpy()
tdays = lambda df: ((pd.to_datetime(df["time"]) - pd.Timestamp("2020-01-01")) / pd.Timedelta(days=1)).to_numpy(np.float64)


def krige():
    spec = importlib.util.spec_from_file_location("kriging", KRIGE_PY); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
    from scipy.spatial import cKDTree
    X, Y, XYZ, TD = [], [], [], []
    for i, m in enumerate(MONTHS):
        sl = tr[n[i]:n[i + 1]]; d = L.load_oco(m, IDX60, columns=L.OCO_COLS)
        X.append(L.bg_features(d).astype(np.float64)[sl]); Y.append(d.xco2.to_numpy(np.float64)[sl])
        XYZ.append(K.unit_xyz(d.latitude.to_numpy()[sl], d.longitude.to_numpy()[sl])); TD.append(tdays(d)[sl])
    X, Y, XYZ, TD = (np.concatenate(v) for v in (X, Y, XYZ, TD))
    m_, s_ = X.mean(0), X.std(0) + 1e-9; beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), (X - m_) / s_]), Y, rcond=None)
    trend = lambda x: np.column_stack([np.ones(len(x)), (x.astype(np.float64) - m_) / s_]) @ beta
    res = Y - trend(X); ev = pd.read_csv(f"{B}/kriging/variogram.csv"); p = K.fit_variogram(ev, False); a = p[-1]
    tree = cKDTree(np.column_stack([K.R_EARTH * XYZ, a * TD]))
    # 수용 검사: 2020-01 시험 사운딩
    qj = jan[te_j]; pk, _ = K.predict_local(tree, XYZ, TD, res, K.unit_xyz(qj.latitude.to_numpy(), qj.longitude.to_numpy()), tdays(qj), p, False, 128, a, 64, 6)
    pj = trend(L.bg_features(qj)) + pk
    ref = pd.read_parquet(f"{B}/kriging/pred_krig.parquet").set_index("row_idx").pred.reindex(qj.row_idx).to_numpy(); err = float(np.max(np.abs(pj - ref)))
    print(f"크리깅 2020-01 시험 {len(qj):,}행 재현 max|차이| {err:.2e} ppm · 변동함수 {np.round(p, 3).tolist()}", flush=True); assert err < 1e-3, "크리깅 재현 실패"
    pg, _ = K.predict_local(tree, XYZ, TD, res, K.unit_xyz(g.latitude.to_numpy(), g.longitude.to_numpy()), tdays(g), p, False, 128, a, 64, 6)
    np.save(f"{D}/grid_202001_krig.npy", (trend(L.bg_features(g)) + pg).reshape(121, 201).astype(np.float32))
    jd = pd.DataFrame(dict(row_idx=qj.row_idx.to_numpy(), krig=pj)); jd.to_parquet(f"{D}/grid_202001_krig_snd.parquet", index=False); print("크리깅 격자 저장", flush=True)


def no2_24h(df):  # build_no2_features.py 정의: 노드 값 = 창 [t−23, t] 화소의 Σw·no2/Σw, 사운딩 = 값 있는 노드끼리 가중 재정규화
    t = pd.read_parquet("data/processed/index/trop_hi_202001.parquet", columns=["step_g", "no2"] + [f"n{k}" for k in range(4)] + [f"w{k}" for k in range(4)])
    out = np.full(len(df), np.nan); steps = df.step_h.to_numpy()
    for s in np.unique(steps):
        w = t[(t.step_g >= s - 23) & (t.step_g <= s)]; num = np.zeros(121 * 201); den = np.zeros(121 * 201)
        for k in range(4):
            np.add.at(num, w[f"n{k}"].to_numpy(), w[f"w{k}"].to_numpy() * w.no2.to_numpy()); np.add.at(den, w[f"n{k}"].to_numpy(), w[f"w{k}"].to_numpy())
        mnode = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan); ii = np.flatnonzero(steps == s); a = np.zeros(len(ii)); b = np.zeros(len(ii))
        for k in range(4):
            v = mnode[df[f"n{k}"].to_numpy()[ii]]; wk = df[f"w{k}"].to_numpy()[ii]; ok = np.isfinite(v); a += np.where(ok, wk * np.nan_to_num(v), 0); b += np.where(ok, wk, 0)
        out[ii] = np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)
    return out


def uv(df):
    from no2xco2.data.era5 import open_wind
    W = open_wind("data/raw/era5_wind/era5_wind_202001_z100.nc", "202001"); U, V = W.u_pbl.values, W.v_pbl.values; t = df.step_h.to_numpy(); au = np.zeros(len(df)); av = np.zeros(len(df))
    for k in range(4):
        nk = df[f"n{k}"].to_numpy(); wk = df[f"w{k}"].to_numpy(); au += wk * U[t, nk // 201, nk % 201]; av += wk * V[t, nk // 201, nk % 201]
    return au, av


def feats():  # netCDF 를 읽는 피처 계산 (LightGBM 은 이 프로세스에 올리지 않음)
    X = lambda df: np.column_stack([L.bg_features(df), no2_24h(df)[:, None], np.column_stack(uv(df))]).astype(np.float32)
    qj = jan[te_j].reset_index(drop=True)
    np.savez(f"{D}/grid_202001_lgbm_X.npz", Xj=X(qj), rowj=qj.row_idx.to_numpy(), Xg=X(g), mu=float(Z["mu"])); print("LightGBM 입력 저장", flush=True)


def lgbm():  # numpy · pandas · lightgbm 만 사용
    import lightgbm as lgb
    A = np.load(f"{D}/grid_202001_lgbm_X.npz"); bst = lgb.Booster(model_file=f"{B}/lgbm_no2_wind/model_lgbm_no2wind.txt"); mu = float(A["mu"])
    pj = bst.predict(A["Xj"]) + mu
    ref = pd.read_parquet(f"{B}/lgbm_no2_wind/pred_lgbm_no2wind.parquet").set_index("row_idx").pred.reindex(A["rowj"]).to_numpy(); err = float(np.max(np.abs(pj - ref)))
    print(f"LightGBM 2020-01 시험 {len(pj):,}행 재현 max|차이| {err:.2e} ppm", flush=True); assert err < 1e-4, "LightGBM 재현 실패"
    np.save(f"{D}/grid_202001_lgbm.npy", (bst.predict(A["Xg"]) + mu).reshape(121, 201).astype(np.float32))
    pd.DataFrame(dict(row_idx=A["rowj"], lgbm=pj)).to_parquet(f"{D}/grid_202001_lgbm_snd.parquet", index=False); print("LightGBM 격자 저장", flush=True)


if __name__ == "__main__":
    {"krige": krige, "feats": feats, "lgbm": lgbm}[sys.argv[1]]()
