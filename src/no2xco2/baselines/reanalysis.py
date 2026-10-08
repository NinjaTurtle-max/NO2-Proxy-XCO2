"""rung 0 평가: CAMS EGG4 XCO2 · CarbonTracker 컬럼 평균 CO2 를 OCO 사운딩 위치·시각에 보간 → OCO 대비 오차, 거리 층, 파일럿 폴드 test 비교.

EGG4  tcco2 (ppm, 0.75°, 3 h) → 3선형 보간 (lat, lon, time).
CT    co2 (34층, 2°×3°, 3 h 평균) → 층 air_mass 가중 컬럼 평균 (OCO 평균핵·사전정보 미적용 — 미구현) → 3선형 보간.
편향: 재분석은 OCO v11 X2019 스케일과 다를 수 있어 (a) 원값 RMSE (b) L_train 전체 평균 편향 제거 후 RMSE 를 함께 보고.
TCCON: --tccon-dir 가 있으면 5 사이트 GGG2020 .public.qc.nc 를 읽어 2020-01 창에서 OCO(≤100 km, ±1 h)·EGG4·CT 를 TCCON xco2 와 비교 (B3 참조값).
출력: <out>/rung0_202001.parquet (row_idx, xco2, egg4, ct), strata_*.csv, tccon_202001.csv, summary.json
"""
import argparse, glob, os, json, time
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
import no2xco2.baselines.kriging as M18
from no2xco2.config import NAS_TCCON
R = 6371.0


def interp_egg4(path, lat, lon, t):
    with xr.open_dataset(path) as ds:  # 파일 핸들 닫음 (QA D5)
        v = ds.tcco2.sortby("latitude")
        return v.interp(latitude=xr.DataArray(lat, dims="p"), longitude=xr.DataArray(lon, dims="p"), valid_time=xr.DataArray(t, dims="p")).values


def interp_ct(path, lat, lon, t):
    with xr.open_dataset(path) as ds:  # 파일 핸들 닫음 (QA D5)
        x = (ds.co2 * ds.air_mass).sum("level") / ds.air_mass.sum("level")
        return x.interp(latitude=xr.DataArray(lat, dims="p"), longitude=xr.DataArray(lon, dims="p"), time=xr.DataArray(t, dims="p")).values


def rmse(a, b, m): r = (a - b)[m]; r = r[np.isfinite(r)]; return float(np.sqrt((r ** 2).mean())), float(r.mean()), int(len(r))


def read_tccon(f):
    with xr.open_dataset(f) as ds:  # 파일 핸들 닫음 (QA D5)
        t = pd.to_datetime(ds["time"].values)
        return pd.DataFrame(dict(time=t, xco2=ds["xco2"].values, lat=float(ds["lat"].values[0]) if ds["lat"].ndim else float(ds["lat"]), lon=float(ds["long"].values[0]) if ds["long"].ndim else float(ds["long"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--egg4", default="data/raw/pilot_202001/egg4_xco2_202001.nc"); ap.add_argument("--ct", default="data/raw/pilot_202001/ct_ea_202001.nc")
    ap.add_argument("--tccon-dir", default=NAS_TCCON); ap.add_argument("--out", default="experiments/rung0")
    ap.add_argument("--rung1", default="experiments/rung1"); ap.add_argument("--rung3", default="experiments/rung3")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    df = M18.load(); y = df.xco2.to_numpy(); lab = (df.label == "L_train").to_numpy(); tt = pd.to_datetime(df.time).values
    lat = df.latitude.to_numpy(); lon = df.longitude.to_numpy()
    egg4 = interp_egg4(a.egg4, lat, lon, tt); ct = interp_ct(a.ct, lat, lon, tt)
    out = pd.DataFrame(dict(row_idx=df.row_idx, xco2=y, egg4=egg4, ct=ct, label=df.label)); out.to_parquet(f"{a.out}/rung0_202001.parquet", index=False)
    summ = {}
    print(f"사운딩 {len(df):,} · EGG4 보간 결측 {np.isnan(egg4).mean():.1%} · CT 결측 {np.isnan(ct).mean():.1%}")
    for name, p in (("EGG4", egg4), ("CT", ct)):
        r, b, n = rmse(p, y, lab); rb, _, _ = rmse(p - b, y, lab)
        summ[name] = dict(rmse=r, bias=b, rmse_debiased=rb, n=n); print(f"{name}: L_train n={n:,} RMSE {r:.3f} bias {b:+.3f} → 편향 제거 후 RMSE {rb:.3f}")
    # 파일럿 폴드 test 비교 (동일 test 사운딩): 추세·크리깅·GAT·물리·절제 + rung0 (원값·편향제거)
    for cfg in ("time2", "space0"):
        st = pd.read_parquet(f"{a.rung1}/rung1_{cfg}.parquet")
        g3 = glob.glob(f"{a.rung3}/pred_{cfg}_s*.parquet")
        if g3:
            m3 = pd.concat([pd.read_parquet(f)[["row_idx", "pred"]] for f in g3]).groupby("row_idx").pred.mean(); st["pred_gat"] = st.row_idx.map(m3).to_numpy()
        st = st.merge(out[["row_idx", "egg4", "ct"]], on="row_idx")
        for name in ("egg4", "ct"): st[f"pred_{name}_deb"] = st[name] - summ[name.upper()]["bias"]
        st["stratum"] = pd.cut(st.d_min, M18.STRATA, right=False, labels=[f"{M18.STRATA[i]}–{M18.STRATA[i+1]}" for i in range(len(M18.STRATA) - 1)])
        cols = [c for c in ("egg4", "ct", "pred_egg4_deb", "pred_ct_deb", "pred_trend", "pred_krig", "pred_gat", "pred_phys", "pred_nophys") if c in st]
        rows = []
        for s_, g in list(st.groupby("stratum", observed=False)) + [("전체", st)]:
            if len(g) == 0: continue
            rows.append(dict(stratum=str(s_), n=len(g), **{c.replace("pred_", "rmse_") if c.startswith("pred_") else f"rmse_{c}": float(np.sqrt(np.nanmean((g[c] - g.xco2) ** 2))) for c in cols}))
        sdf = pd.DataFrame(rows); sdf.to_csv(f"{a.out}/strata_{cfg}.csv", index=False); print(f"\n[{cfg}]"); print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # TCCON
    files = sorted(glob.glob(os.path.join(a.tccon_dir, "*.public.qc.nc"))) if os.path.isdir(a.tccon_dir) else []
    if files:
        rows = []; sxyz = M18.__dict__.get("xyz")
        def xyz(la, lo):
            la, lo = np.deg2rad(la), np.deg2rad(lo); return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        tree = cKDTree(xyz(lat[lab], lon[lab])); idx_lab = np.where(lab)[0]
        for f in files:
            tc = read_tccon(f); tc = tc[(tc.time >= "2020-01-01") & (tc.time < "2020-02-01")]
            site = os.path.basename(f)[:2]; n_tc = len(tc)
            if n_tc == 0: rows.append(dict(site=site, n_tccon=0)); continue
            # OCO 공동배치: ≤100 km, ±1 h → 사운딩 평균 vs TCCON ±1 h 평균
            near = tree.query_ball_point(xyz(np.array([tc.lat.iloc[0]]), np.array([tc.lon.iloc[0]]))[0], 2 * np.sin(100 / (2 * R)))
            near = idx_lab[np.array(near, dtype=int)] if len(near) else np.array([], int)
            pairs = []
            for d, g in df.iloc[near].groupby(pd.to_datetime(df.time.iloc[near]).dt.floor("h")):
                tw = tc[(tc.time >= d - pd.Timedelta("1h")) & (tc.time <= d + pd.Timedelta("2h"))]
                if len(tw) >= 5: pairs.append(dict(t=d, tccon=tw.xco2.mean(), oco=g.xco2.mean(), n_oco=len(g), egg4=np.nanmean(out.egg4.values[g.index]), ct=np.nanmean(out.ct.values[g.index])))
            pr = pd.DataFrame(pairs)
            # EGG4·CT 를 TCCON 지점·시각에 직접 보간 (공동배치 무관, 1월 전체)
            eg_site = interp_egg4(a.egg4, np.full(n_tc, tc.lat.iloc[0]), np.full(n_tc, tc.lon.iloc[0]), tc.time.values); ct_site = interp_ct(a.ct, np.full(n_tc, tc.lat.iloc[0]), np.full(n_tc, tc.lon.iloc[0]), tc.time.values)
            r = dict(site=site, lat=tc.lat.iloc[0], lon=tc.lon.iloc[0], n_tccon=n_tc, egg4_minus_tccon=float(np.nanmean(eg_site - tc.xco2)), ct_minus_tccon=float(np.nanmean(ct_site - tc.xco2)),
                     egg4_rmse_tccon=float(np.sqrt(np.nanmean((eg_site - tc.xco2) ** 2))), ct_rmse_tccon=float(np.sqrt(np.nanmean((ct_site - tc.xco2) ** 2))), n_pairs_oco=len(pr))
            if len(pr): r.update(oco_minus_tccon=float((pr.oco - pr.tccon).mean()), oco_rmse_tccon=float(np.sqrt(((pr.oco - pr.tccon) ** 2).mean())))
            rows.append(r)
        tdf = pd.DataFrame(rows); tdf.to_csv(f"{a.out}/tccon_202001.csv", index=False); print("\n[TCCON 2020-01]"); print(tdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("\nTCCON 디렉토리 없음 (NAS 미마운트) → TCCON 비교 미측정")
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1); print(f"\n총 {time.time()-t0:.0f}s → {a.out}/")
