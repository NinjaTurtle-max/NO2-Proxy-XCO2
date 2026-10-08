"""TCCON 검증 파이프라인 (B3): 5년 공동배치 쌍 생성 → OCO · CT · EGG4 vs TCCON 오차 → 블록 부트스트랩 CI → 모델 예측 평가 함수.

매칭 규칙 (결정 12 B3 구현, Laughner 2024 GGG2020 XCO2 사용):
  사운딩 ≤ 100 km (haversine) · 같은 UTC 시각(시간 단위 floor) · TCCON 관측 ≥ 5개/시간 · OCO L_train ≥ 1개
  쌍 값 = TCCON 시간 평균 xco2, OCO 시간 평균 xco2 (사운딩 수·표준편차 보존)
rung 0: CT 컬럼 평균(air_mass 가중, 평균핵 미적용)을 사이트 위치·시각에 3선형 보간; EGG4 는 2020년만.
부트스트랩: 블록 = (사이트, UTC 일). 1,000회. RMSE 의 95% CI 반폭 → B3 임계 0.42 ppm 대비 검정력 판단 입력.
모델 평가: evaluate(pred: DataFrame[row_idx, pred]) → 쌍별 모델 시간 평균 → 같은 표.
출력: <out>/tccon_pairs_5yr.parquet, tccon_metrics_5yr.csv, per_site.csv
"""
import argparse, glob, os, time, json
import numpy as np, pandas as pd, xarray as xr, pyarrow.dataset as pds
from no2xco2.config import NAS_CAMS, NAS_CT, NAS_MOUNT, NAS_OCO_NODES, NAS_TCCON, wait_nas
from no2xco2.train import block_bootstrap
R = 6371.0; RADIUS_KM = 100.0; MIN_TCCON = 5


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.deg2rad, (la1, lo1, la2, lo2)); a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_tccon(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.public.qc.nc"))):
        s = os.path.basename(f)[:2]
        with xr.open_dataset(f) as ds:  # 파일 핸들 닫음 (QA D5)
            out[s] = dict(lat=float(ds["lat"].values.ravel()[0]), lon=float(ds["long"].values.ravel()[0]),
                          df=pd.DataFrame(dict(time=pd.to_datetime(ds["time"].values), xco2=ds["xco2"].values.astype(float))))
    return out


def build_pairs(tccon, oco_dir):
    d = pds.dataset(oco_dir, format="parquet", partitioning="hive")
    tab = d.to_table(columns=["row_idx", "latitude", "longitude", "time", "xco2", "label"], filter=(pds.field("label") == "L_train")).to_pandas()
    tab["hour"] = pd.to_datetime(tab.time).dt.floor("h"); rows = []; members = []
    for s, v in tccon.items():
        near = tab[hav(tab.latitude.values, tab.longitude.values, v["lat"], v["lon"]) <= RADIUS_KM]
        tc = v["df"].copy(); tc["hour"] = tc.time.dt.floor("h"); tg = tc.groupby("hour").xco2.agg(["mean", "std", "count"]); tg = tg[tg["count"] >= MIN_TCCON]
        og = near.groupby("hour").agg(oco=("xco2", "mean"), oco_sd=("xco2", "std"), n_oco=("xco2", "size"), lat=("latitude", "mean"), lon=("longitude", "mean"))
        j = og.join(tg, how="inner").rename(columns={"mean": "tccon", "std": "tccon_sd", "count": "n_tccon"}); j["site"] = s; j["site_lat"] = v["lat"]; j["site_lon"] = v["lon"]
        rows.append(j.reset_index()); members.append(near[near.hour.isin(j.index)][["row_idx", "hour"]].assign(site=s))
    return pd.concat(rows, ignore_index=True), pd.concat(members, ignore_index=True)


def ct_at(pairs, ct_dir):
    out = np.full(len(pairs), np.nan)
    for ym, g in pairs.groupby(pairs.hour.dt.strftime("%Y%m")):
        f = os.path.join(ct_dir, f"ct_ea_{ym}.nc")
        if not os.path.exists(f): continue
        with xr.open_dataset(f) as ds:  # 파일 핸들 닫음 (QA D5)
            x = (ds.co2 * ds.air_mass).sum("level") / ds.air_mass.sum("level")
            out[g.index] = x.interp(latitude=xr.DataArray(g.site_lat.values, dims="p"), longitude=xr.DataArray(g.site_lon.values, dims="p"), time=xr.DataArray(g.hour.values, dims="p")).values
    return out


def egg4_at(pairs, cams_dir):
    out = np.full(len(pairs), np.nan)
    for ym, g in pairs.groupby(pairs.hour.dt.strftime("%Y%m")):
        f = os.path.join(cams_dir, f"egg4_xco2_{ym}.nc")
        if not os.path.exists(f): continue
        with xr.open_dataset(f) as ds:  # 파일 핸들 닫음 (QA D5)
            out[g.index] = ds.tcco2.sortby("latitude").interp(latitude=xr.DataArray(g.site_lat.values, dims="p"), longitude=xr.DataArray(g.site_lon.values % 360, dims="p"), valid_time=xr.DataArray(g.hour.values, dims="p")).values
    return out


def block_boot(res, blocks, n=1000, seed=0, stat=lambda r: np.sqrt((r ** 2).mean())):
    rng = np.random.default_rng(seed); ub = np.unique(blocks); idx = {b: np.where(blocks == b)[0] for b in ub}; vals = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub), replace=True); ii = np.concatenate([idx[b] for b in pick]); vals.append(stat(res[ii]))
    return np.percentile(vals, [2.5, 97.5])


def metrics(pairs, col, name):
    ok = np.isfinite(pairs[col]) & np.isfinite(pairs.tccon); r = (pairs[col] - pairs.tccon)[ok].to_numpy(); blocks = (pairs.site + "_" + pairs.hour.dt.strftime("%Y%m%d"))[ok].to_numpy()
    if len(r) == 0: return dict(model=name, n=0)
    lo, hi = block_boot(r, blocks); blo, bhi = block_boot(r, blocks, stat=np.mean)
    return dict(model=name, n=int(len(r)), n_blocks=int(len(np.unique(blocks))), bias=float(r.mean()), bias_ci=f"[{blo:+.3f}, {bhi:+.3f}]", rmse=float(np.sqrt((r ** 2).mean())), rmse_ci=f"[{lo:.3f}, {hi:.3f}]", rmse_ci_halfwidth=float((hi - lo) / 2), mad=float(np.median(np.abs(r - np.median(r)))))


def evaluate(pairs, members, pred, name):
    """pred: DataFrame[row_idx, pred, xco2(, test)] (사운딩 단위 모델 예측·관측) → 쌍별 시간 평균 → metrics 행 목록.
    평가 표본 (E-2, 사용자 09-24): pred 에 test 열이 있으면 두 표본을 병기한다.
      test 한정 (주 판정): 쌍의 모델 값 = 그 시간의 test 사운딩만의 평균, test 사운딩이 없는 쌍은 제외
      공동배치 전체: 훈련 사운딩을 포함한 공동배치 L_train 사운딩 전체 평균 (훈련 행 채점 포함 — 참고용)
    각 표본에서 모델·OCO·CT 를 같은 쌍(모델·CT 모두 유한)으로 계산해 n 을 맞춘다 (D3-2). OCO 행은 그 표본의 **같은 사운딩** 관측 평균
    (test 한정이면 test 사운딩만; 쌍의 L_train 전체 평균 pairs.oco 가 아님 — QA X1). 반환: (행 목록, {표본: 쌍 DataFrame})."""
    samples = ([("test 한정", pred[pred["test"].astype(bool)])] if "test" in pred else []) + [("공동배치 전체", pred)]
    rows, out = [], {}
    for lab, pr in samples:
        mm = members.merge(pr[["row_idx", "pred", "xco2"]], on="row_idx", how="inner")
        m = mm.groupby(["site", "hour"]).agg(**{name: ("pred", "mean"), "oco_s": ("xco2", "mean"), "n_snd": ("pred", "size")}).reset_index()
        p = pairs.merge(m, on=["site", "hour"], how="inner"); p = p[np.isfinite(p[name]) & np.isfinite(p.ct)]
        for col, nm in ((name, name), ("oco_s", "OCO"), ("ct", "CT2022/NRT")):
            rows.append(dict(metrics(p, col, f"{nm} ({lab} 쌍)"), sample=lab, n_snd=int(p.n_snd.sum())))
        out[lab] = p
    return rows, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tccon-dir", default=NAS_TCCON); ap.add_argument("--oco-dir", default=NAS_OCO_NODES)  # config 상수 (QA P10)
    ap.add_argument("--ct-dir", default=NAS_CT); ap.add_argument("--cams-dir", default=NAS_CAMS)
    ap.add_argument("--out", default="experiments/tccon"); ap.add_argument("--pred", default=None, help="선택: 모델 예측 parquet (row_idx, pred[, test]) — test 열이 있으면 test 한정 행(주 판정) 병기 (E-2)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    if any(os.path.abspath(p).startswith(NAS_MOUNT.rstrip("/") + "/") for p in (a.tccon_dir, a.oco_dir, a.ct_dir, a.cams_dir)):
        wait_nas()  # NAS 입력이 있을 때만 마운트 대기 (QA X10)
    tccon = load_tccon(a.tccon_dir); print("TCCON 사이트", {k: (v["lat"], v["lon"], len(v["df"])) for k, v in tccon.items()}, flush=True)
    pairs, members = build_pairs(tccon, a.oco_dir); print(f"공동배치 쌍 {len(pairs):,} (사운딩 {len(members):,}) {time.time()-t0:.0f}s", flush=True)
    pairs["ct"] = ct_at(pairs, a.ct_dir); print(f"CT 보간 완료 (결측 {np.isnan(pairs.ct).mean():.1%}) {time.time()-t0:.0f}s", flush=True)
    pairs["egg4"] = egg4_at(pairs, a.cams_dir); print(f"EGG4 보간 완료 (2020 쌍 {int(np.isfinite(pairs.egg4).sum())}) {time.time()-t0:.0f}s", flush=True)
    pairs.to_parquet(f"{a.out}/tccon_pairs_5yr.parquet", index=False); members.to_parquet(f"{a.out}/tccon_members_5yr.parquet", index=False)
    rows = [dict(metrics(pairs, c, n), sample="전체 쌍") for c, n in (("oco", "OCO"), ("ct", "CT2022/NRT"), ("egg4", "EGG4 (2020)"))]
    p20 = pairs[pairs.hour.dt.year == 2020]; rows += [dict(metrics(p20, c, n), sample="2020 쌍") for c, n in (("oco", "OCO (2020)"), ("ct", "CT (2020)"))]
    if a.pred:
        pr = pd.read_parquet(a.pred)
        if "xco2" not in pr:
            raise ValueError(f"{a.pred}: xco2 열 필요 — OCO 행을 모델과 같은 사운딩으로 계산 (QA X1)")
        pr = pr[["row_idx", "pred", "xco2"] + (["test"] if "test" in pr else [])]
        if "test" not in pr:
            print("  pred 에 test 열 없음 → 공동배치 전체(훈련 사운딩 포함) 행만 산출 — 주 판정(test 한정) 불가", flush=True)
        mr, _ = evaluate(pairs, members, pr, "MODEL"); rows += mr
    mdf = pd.DataFrame(rows); mdf.to_csv(f"{a.out}/tccon_metrics_5yr.csv", index=False); print("\n" + mdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    ps = []
    for s, g in pairs.groupby("site"):
        r = dict(site=s, n_pairs=len(g), years=f"{g.hour.dt.year.min()}–{g.hour.dt.year.max()}")
        for c, n in (("oco", "OCO"), ("ct", "CT"), ("egg4", "EGG4")):
            ok = np.isfinite(g[c]); d = (g[c] - g.tccon)[ok]; r[f"{n}_bias"] = float(d.mean()) if ok.any() else np.nan; r[f"{n}_rmse"] = float(np.sqrt((d ** 2).mean())) if ok.any() else np.nan; r[f"{n}_n"] = int(ok.sum())
        ps.append(r)
    sdf = pd.DataFrame(ps); sdf.to_csv(f"{a.out}/per_site.csv", index=False); print("\n" + sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # B3 검정력 입력: |RMSE_ours − RMSE_ref| ≤ 0.42 를 판정하려면 ΔRMSE CI 반폭이 0.42 보다 작아야 함 → CT vs OCO 의 ΔRMSE CI
    ok = np.isfinite(pairs.ct) & np.isfinite(pairs.oco) & np.isfinite(pairs.tccon); ra = (pairs.oco - pairs.tccon)[ok].to_numpy(); rb = (pairs.ct - pairs.tccon)[ok].to_numpy(); blocks = (pairs.site + "_" + pairs.hour.dt.strftime("%Y%m%d"))[ok].to_numpy()
    if not ok.any():  # 유효 쌍 0 → 빈 부트스트랩 대신 명시적 실패 (QA P4)
        raise ValueError("B3: CT·OCO·TCCON 모두 유한한 쌍 0")
    ub = np.unique(blocks); lo, hi = block_bootstrap(ra, rb, blocks); pt = np.sqrt((rb ** 2).mean()) - np.sqrt((ra ** 2).mean())  # train.block_bootstrap 과 같은 알고리즘·시드 (QA P4 복제 제거)
    b3 = dict(delta_rmse_ct_minus_oco=float(pt), ci=[float(lo), float(hi)], ci_halfwidth=float((hi - lo) / 2), b3_threshold=0.42, n_pairs=int(ok.sum()), n_blocks=int(len(ub)))
    json.dump(b3, open(f"{a.out}/b3_power.json", "w"), indent=1); print(f"\nΔRMSE(CT − OCO) vs TCCON: {pt:+.3f} ppm, 95% CI [{lo:+.3f}, {hi:+.3f}], 반폭 {(hi-lo)/2:.3f} (B3 임계 0.42) · 쌍 {ok.sum()} · 블록 {len(ub)}")
    print(f"총 {time.time()-t0:.0f}s → {a.out}/")
