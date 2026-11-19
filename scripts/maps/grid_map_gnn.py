"""Fig. 3 지도 (GNN): 60개월 수송 켬 · 끔 모델을 사례일 2020-01-31 의 0.25° 전 격자에서 디코드한다.
분할 · μ · 배경 스케일러는 학습과 같게 60개월 전체로 계산한다 (버퍼 · 시간 val 제외가 달을 넘나들어 달별 계산 불가).
사례일 = 2020-01 중 시험 지역 사운딩이 가장 많은 UTC 날 (결과를 보기 전에 정함).
격자 질의 시각 = 경도별 현지 13:30 (UTC 정수 시 = round(13.5 − 경도/15)), 질의 위치 = 격자점, 배경 5변수 = 그 시각 ERA5 격자값.
수용 검사: 같은 롤아웃의 실제 사운딩 디코드가 저장된 예측(1월 행)을 켬 · 끔 모두 재현해야 지도를 저장한다 (max|차| ≤ 1e-4 ppm).
출력: runs/maps/grid_map_202001_m60.npz · grid_202001_feats.parquet (기준선 지도 입력). 실행: 저장소 루트에서 python scripts/maps/grid_map_gnn.py"""
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
import no2xco2.model as M
from no2xco2.data import loader as L
from no2xco2.data.encoder_index import bilinear
from no2xco2.data.era5 import open_wind
from no2xco2.train import nested_masks

D = "runs/maps"
RUNS = {"on": "runs/gnn_m60_on/state_space0_s0_phys.pt", "off": "runs/gnn_m60_off/state_space0_s0_nophys.pt"}
PRED = {"on": "runs/gnn_m60_on/pred_space0_s0_phys.parquet", "off": "runs/gnn_m60_off/pred_space0_s0_nophys.parquet"}
MONTHS = [f"{y}{m:02d}" for y in range(2020, 2025) for m in range(1, 13)]; YM = "202001"; dev = torch.device("cpu")
IDX, ERA5 = L.DEFAULT_IDX, L.DEFAULT_ERA5; NLAT, NLON = 121, 201; os.makedirs(D, exist_ok=True)

# 분할 · μ · 배경 스케일러: 학습과 같은 60개월 · space 0 · vfold 1 (배경 특징은 달별로 훈련 행만 모아 fit_bg_scaler 와 같은 순서 · 연산)
MC = ["row_idx", "time", "label", "latitude", "longitude", "xco2", "fold_time", "fold_space", "space_block", "year"]
big = pd.concat([L.load_oco(m, IDX, columns=MC) for m in MONTHS], ignore_index=True)
import pyarrow.parquet as pq
n = np.cumsum([0] + [pq.ParquetFile(f"{IDX}/oco_{m}.parquet").metadata.num_rows for m in MONTHS])
slices = [slice(int(n[i]), int(n[i + 1])) for i in range(len(MONTHS))]; assert n[-1] == len(big)
y = big.xco2.to_numpy(np.float64); lab = (big.label == "L_train").to_numpy()
tr, val, te = nested_masks(big, "space", 0, lab, 1); mu = y[tr].mean(); print(f"μ_train {mu:.4f} · train {tr.sum():,} · test {te.sum():,}", flush=True); del big
xs = []
for m, s_ in zip(MONTHS, slices):
    dfm = L.load_oco(m, IDX, columns=L.OCO_COLS); assert len(dfm) == s_.stop - s_.start
    xs.append(L.bg_features(dfm)[tr[s_]]); del dfm
x = np.concatenate(xs); bgm, bgs = __import__("torch").tensor(x.mean(0)), __import__("torch").tensor(x.std(0) + 1e-6); del xs, x
dfs = [L.load_oco(YM, IDX, columns=L.OCO_COLS)]; sl = slices[0]
st = L.load_no2_stats(IDX); mu_no2, sd_no2 = float(st["mean"]), float(st["std"])
jan = dfs[0]; te_j = te[sl]
day_te = pd.to_datetime(jan.time[te_j]).dt.day.value_counts(); DAY = int(day_te.idxmax())
print(f"μ_train {mu:.4f} · 사례일 2020-01-{DAY:02d} (시험 사운딩 {int(day_te.max())}, 2위 {int(day_te.iloc[1])})")

# 격자 질의 DataFrame
lat = 20 + 0.25 * np.arange(NLAT); lon = 100 + 0.25 * np.arange(NLON); LA, LO = np.meshgrid(lat, lon, indexing="ij")
LA, LO = LA.ravel(), LO.ravel(); hr = np.round(13.5 - LO / 15).astype(int); step = (DAY - 1) * 24 + hr
nodes, w, _ = bilinear(LA, LO); ii, jj = np.divmod(np.arange(NLAT * NLON), NLON)
W = open_wind(f"{ERA5}/era5_wind_{YM}_z100.nc", YM); S = xr.open_dataset(f"data/raw/era5_syn/era5_syn_{YM}.nc")
assert abs(float(S.lat[0]) - 20) < 1e-6
g = pd.DataFrame(dict(latitude=LA, longitude=LO, step_h=step.astype(np.int32), hour=hr.astype(np.float32),
                      time=pd.Timestamp(f"2020-01-{DAY:02d}") + pd.to_timedelta(hr, "h"), doy=np.int16(DAY)))
for k in range(4):
    g[f"n{k}"] = nodes[:, k].astype(np.int32); g[f"w{k}"] = w[:, k].astype(np.float32)
for v, src in (("t2m", W), ("blh", W), ("sp", W), ("z850", S), ("thk", S)):
    g[f"bg_{v}"] = src[v].values[step, ii, jj].astype(np.float32)
g.to_parquet(f"{D}/grid_202001_feats.parquet", index=False)  # 크리깅 · LightGBM 격자 예측 입력
uv = dict(u=W.u_pbl.values[(DAY - 1) * 24 + 5], v=W.v_pbl.values[(DAY - 1) * 24 + 5])  # 그림용 바람: 05 UTC (현지 정오 전후)


def merge(qa, qb, off):  # 같은 스텝의 사운딩 질의 + 격자 질의 (rows · seg 오프셋)
    if qa is None:
        return dict(qb, rows=qb["rows"] + off)
    return dict(node=torch.cat([qa["node"], qb["node"]]), edge=torch.cat([qa["edge"], qb["edge"]]), seg=torch.cat([qa["seg"], qb["seg"] + qa["n"]]),
                n=qa["n"] + qb["n"], bg=torch.cat([qa["bg"], qb["bg"]]),
                rows=np.concatenate([qa["rows"], qb["rows"] + off]))


out = {}
for arm, path in RUNS.items():
    s = torch.load(path, weights_only=False); assert s["n_bg"] == L.N_BG
    model = M.Model(8, dev, n_bg=L.N_BG).to(dev); model.load_state_dict(s["params"], strict=False); model.eval()
    if not s["physics"]:
        model.phys.advect = lambda h, u, v: h; model.phys.diffuse = lambda h: h
    assert s["physics"] == (arm == "on")
    qs = L.load_qry(jan, dev); L.apply_bg_scaler(qs, bgm, bgs); qg = L.load_qry(g, dev); L.apply_bg_scaler(qg, bgm, bgs)
    q = {t: merge(qs.get(t), qg[t], len(jan)) for t in qg}; q.update({t: v for t, v in qs.items() if t not in qg})
    px = L.load_px(YM, IDX, mu_no2, sd_no2, dev); U, V, T = L.load_wind(YM, ERA5, dev)
    yy = np.concatenate([(y - mu)[sl], np.zeros(len(g))]); tm = np.zeros(len(yy), bool)
    with torch.no_grad():
        p = M.run(model, px, q, U, V, T, yy, tm, None, chunk=48, px_frac=1.0)
    ref = pd.read_parquet(PRED[arm]).pred.to_numpy()[sl]; err = np.nanmax(np.abs(p[:len(jan)] + mu - ref))
    print(f"{arm}: 1월 사운딩 {len(jan):,} pred 재현 max|차이| {err:.2e} ppm · β {float(model.beta.detach()):+.3f}")
    assert err <= 1e-4, f"{arm}: 저장 pred 재현 실패 ({err:.2e})"
    out[arm] = (p[len(jan):] + mu).reshape(NLAT, NLON); out[f"{arm}_snd"] = p[:len(jan)] + mu

# 그림용 관측 NO₂ (그날 qa ≥ 0.75 화소의 노드 가중 평균, mol m⁻²) · 그날 시험 지역 관측 점
e = pd.read_parquet(f"data/processed/encoder_index/enc_tropomi_{YM}.parquet", columns=["step_h", "qa", "no2_tvcd"] + [f"n{k}" for k in range(4)] + [f"w{k}" for k in range(4)])
e = e[(e.step_h // 24 == DAY - 1) & (e.qa >= 0.75)]; num = np.zeros(NLAT * NLON); den = np.zeros(NLAT * NLON)
for k in range(4):
    np.add.at(num, e[f"n{k}"].to_numpy(), e[f"w{k}"].to_numpy() * e.no2_tvcd.to_numpy()); np.add.at(den, e[f"n{k}"].to_numpy(), e[f"w{k}"].to_numpy())
no2 = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan).reshape(NLAT, NLON)
# 지도용 0.05° 장: 화소 위치 = 4노드 좌표의 이중선형 가중합(이중선형 가중의 정의상 원위치 복원) → 0.05° 칸 평균
plat = sum(e[f"w{k}"].to_numpy() * (20 + 0.25 * (e[f"n{k}"].to_numpy() // NLON)) for k in range(4))
plon = sum(e[f"w{k}"].to_numpy() * (100 + 0.25 * (e[f"n{k}"].to_numpy() % NLON)) for k in range(4))
fi = np.clip(((plat - 20) / 0.05).astype(int), 0, 599); fj = np.clip(((plon - 100) / 0.05).astype(int), 0, 999)
s_ = np.zeros(600 * 1000); c_ = np.zeros(600 * 1000); np.add.at(s_, fi * 1000 + fj, e.no2_tvcd.to_numpy()); np.add.at(c_, fi * 1000 + fj, 1)
no2f = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan).reshape(600, 1000)
dd = pd.to_datetime(jan.time).dt.day.to_numpy() == DAY; m = te_j & dd
np.savez_compressed(f"{D}/grid_map_202001_m60.npz", lat=lat, lon=lon, on=out["on"], off=out["off"], no2=no2, no2f=no2f, tr=tr, hr=hr, step=step, u=uv["u"], v=uv["v"], day=DAY, mu=mu,
                    snd_lat=jan.latitude.to_numpy()[m], snd_lon=jan.longitude.to_numpy()[m], snd_obs=jan.xco2.to_numpy()[m],
                    snd_on=out["on_snd"][m], snd_off=out["off_snd"][m], snd_row=jan.row_idx.to_numpy()[m], snd_time=jan.time.astype("int64").to_numpy()[m] if hasattr(jan.time, "dt") else pd.to_datetime(jan.time).astype("int64").to_numpy()[m])
print(f"격자 켬 {np.nanpercentile(out['on'], [2, 50, 98]).round(2)} · 관측 점 {m.sum()} · NO₂ 화소 {len(e):,} → {D}/grid_map_202001_m60.npz")
