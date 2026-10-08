"""rung 3 대조군: 관측=노드 그래프 어텐션 (물리 없음, 격자 잠재 없음), 2020-01, 파일럿 2차와 동일 분할.

문헌: 'Bao 2025 IJDE 관측=노드 GAT' 는 검색으로 확인되지 않음(2026-09-14). 가장 가까운 Yin et al. 2026 TGIS (Dual-Transformer,
      dynamic heterogeneous graph, CAMS 보정) 는 본문 미확인. 여기서는 일반형 — 사양은 우리 설계.
노드: OCO 사운딩 (L_train + L_masked; 라벨은 L_train 만). 특징 = 결정 8 허용 변수 7종 (lat, lon, DOY, hour, t2m, blh, sp).
엣지 (시공간 거리 인접, 물리 없음):
  (a) 화소→사운딩: qa≥0.75 TROPOMI 화소, 같은 UTC 일 & ≤ 50 km 에서 최근접 64개 + 전날/다음날 & ≤ 200 km 에서 64개. 속성 (d km, Δt h, no2 표준화, qa)
  (b) 사운딩→사운딩: ≤ 250 km & |Δt| ≤ 3 일 최근접 32개, 속성 (d km, Δt h). 이웃 특징만 전달 (이웃 XCO2 라벨 미사용 — 라벨 전파 없음)
모델: 엣지 어텐션 1-hop 집계 (타입별 분리, 16.EdgeAttention 재사용) → z_px, z_snd → ŷ = μ + g(bg) + β·NO2_local + MLP([z_px, z_snd])
      NO2_local = 화소 엣지의 거리 가중 평균 no2 (식별 가능한 β, 우리 모델 결정 7 과 대응)
학습: 사운딩 미니배치 4,096, Adam, MSE(L_train), 중첩 검증 조기 종료 (17.nested_masks), 시드 3, 화소 전량 평가.
비교: 같은 test 사운딩에서 우리 물리 (3시드 평균) 와 짝지어 일블록 부트스트랩 ΔRMSE, 거리 층별 RMSE.
"""
import argparse, glob, os, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch, torch.nn as nn
from scipy.spatial import cKDTree
import no2xco2.model as M16; import no2xco2.train as M17; import no2xco2.baselines.kriging as M18; 

NLON = 201; R = 6371.0


def xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon); return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def chord2km(d): return 2 * R * np.arcsin(np.clip(d / 2, 0, 1))


def build_pixel_edges(df, month="202001", qa_min=0.75, k0=64, r0=50.0, k1=64, r1=200.0):
    enc = pq.read_table(f"data/processed/encoder_index/enc_tropomi_{month}.parquet", columns=["step_h", "qa", "no2_tvcd", "n0", "w1", "w2", "w3"]).to_pandas()
    enc = enc[enc.qa >= qa_min].reset_index(drop=True)
    i0 = enc.n0.to_numpy() // NLON; j0 = enc.n0.to_numpy() % NLON
    plat = 20 + 0.25 * (i0 + (enc.w2 + enc.w3).to_numpy()); plon = 100 + 0.25 * (j0 + (enc.w1 + enc.w3).to_numpy())
    pday = enc.step_h.to_numpy() // 24; ph = enc.step_h.to_numpy(); pno2 = ((enc.no2_tvcd - enc.no2_tvcd.mean()) / enc.no2_tvcd.std()).to_numpy(np.float32); pqa = enc.qa.to_numpy(np.float32)
    sday = df.step_h.to_numpy() // 24; sxyz = xyz(df.latitude.to_numpy(), df.longitude.to_numpy())
    src, dst, attr = [], [], []
    for d in np.unique(sday):
        si = np.where(sday == d)[0]
        for dd, k, r in ((0, k0, r0), (-1, k1, r1), (1, k1, r1)):
            pi = np.where(pday == d + dd)[0]
            if len(pi) == 0: continue
            tree = cKDTree(xyz(plat[pi], plon[pi])); dist, idx = tree.query(sxyz[si], k=k, distance_upper_bound=2 * np.sin(r / (2 * R)))
            ok = np.isfinite(dist); s_rep = np.repeat(si[:, None], k, 1)[ok]; p_sel = pi[idx[ok]]; dk = chord2km(dist[ok])
            dt = (ph[p_sel] - df.step_h.to_numpy()[s_rep]).astype(np.float32)
            src.append(p_sel); dst.append(s_rep); attr.append(np.column_stack([dk / 100, dt / 24, pno2[p_sel], pqa[p_sel]]).astype(np.float32))
    src = np.concatenate(src); dst = np.concatenate(dst); attr = np.concatenate(attr)
    return dict(src=src, dst=dst, attr=attr, n_px=len(enc))


def build_sounding_edges(df, k=32, r=250.0, dt_days=3):
    lat = df.latitude.to_numpy(); lon = df.longitude.to_numpy(); t = df.step_h.to_numpy() / 24.0
    tree = cKDTree(np.column_stack([xyz(lat, lon), np.zeros(len(df))]))  # 공간만 → 후처리로 시간 제한
    dist, idx = tree.query(np.column_stack([xyz(lat, lon), np.zeros(len(df))]), k=k * 4 + 1, distance_upper_bound=2 * np.sin(r / (2 * R)))
    src, dst, attr = [], [], []
    for i in range(len(df)):
        j = idx[i][np.isfinite(dist[i])]; j = j[j != i]; j = j[np.abs(t[j] - t[i]) <= dt_days][:k]
        if len(j) == 0: continue
        src.append(j); dst.append(np.full(len(j), i)); attr.append(np.column_stack([chord2km(dist[i][np.isin(idx[i], j)][:len(j)]) / 100, (t[j] - t[i]) * 24 / 24]).astype(np.float32))
    return dict(src=np.concatenate(src), dst=np.concatenate(dst), attr=np.concatenate(attr))


class ObsGAT(nn.Module):
    def __init__(self, d=16):
        super().__init__(); self.d = d
        self.px = M16.EdgeAttention(d_src=2, d_edge=2, d=d)     # 화소 (no2, qa) · 엣지 (d, Δt)
        self.sd = M16.EdgeAttention(d_src=7, d_edge=2, d=d)     # 이웃 사운딩 bg 7 · 엣지 (d, Δt)
        self.g = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, 1)); self.beta = nn.Parameter(torch.zeros(1))
        self.head = nn.Sequential(nn.Linear(2 * d + 7, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, bg, pe, se, n):
        h0 = torch.zeros(n, self.d, device=bg.device)
        zp, _ = self.px(pe["src"], pe["edge"], h0[pe["dst"]], pe["dst"], n)
        zs, _ = self.sd(se["src"], se["edge"], h0[se["dst"]], se["dst"], n)
        w = 1.0 / (pe["edge"][:, 0] + 0.1); num = torch.zeros(n, device=bg.device).index_add(0, pe["dst"], w * pe["src"][:, 0]); den = torch.zeros(n, device=bg.device).index_add(0, pe["dst"], w)
        no2_local = torch.where(den > 0, num / den.clamp(min=1e-9), torch.zeros_like(num))
        return self.g(bg).squeeze(-1) + self.beta * no2_local + self.head(torch.cat([zp, zs, bg], -1)).squeeze(-1), no2_local


def batch_edges(E, sel_idx, remap, dev, src_feat=None):
    m = np.isin(E["dst"], sel_idx); s = E["src"][m]; dlocal = remap[E["dst"][m]]
    return dict(src=torch.tensor(src_feat[s] if src_feat is not None else s, device=dev), dst=torch.tensor(dlocal, device=dev), edge=torch.tensor(E["attr"][m], device=dev), n=len(sel_idx))


def predict(model, bg_all, PE, SE, sfeat, idx, dev, bs=8192):
    out = np.full(len(bg_all), np.nan); model.eval()
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            sel = idx[b:b + bs]; remap = np.full(len(bg_all), -1); remap[sel] = np.arange(len(sel))
            pe = batch_edges(PE, sel, remap, dev, PE["feat"]); se = batch_edges(SE, sel, remap, dev, sfeat)
            pe["src"] = pe["src"][:, :2]; pe["edge"] = pe["edge"][:, :2] if pe["edge"].shape[1] > 2 else pe["edge"]
            yhat, _ = model(torch.tensor(bg_all[sel], device=dev), pe, se, len(sel)); out[sel] = yhat.cpu().numpy()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="time:2,space:0"); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--vfold", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--bs", type=int, default=4096); ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--out", default="experiments/rung3"); ap.add_argument("--model-preds", default="experiments/pilot2_seeds"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = torch.device(a.device); t0 = time.time()
    df = M18.load(); y = df.xco2.to_numpy(np.float64); lab = (df.label == "L_train").to_numpy(); mu = y[lab].mean()
    bg = np.column_stack([df.latitude, df.longitude, df.doy, df.hour, df.t2m, df.blh, df.sp]).astype(np.float32); bg = (bg - bg.mean(0)) / (bg.std(0) + 1e-6)
    PE = build_pixel_edges(df); PE["feat"] = PE["attr"][:, 2:4]; PE["attr"] = PE["attr"][:, :2]
    SE = build_sounding_edges(df)
    print(f"사운딩 {len(df):,} · 화소 엣지 {len(PE['src']):,} (사운딩당 {len(PE['src'])/len(df):.0f}) · 사운딩 엣지 {len(SE['src']):,} (사운딩당 {len(SE['src'])/len(df):.1f}) · 준비 {time.time()-t0:.0f}s", flush=True)
    days = (df.step_h.to_numpy() // 24)
    rows = []
    for cfg in a.configs.split(","):
        scheme, fold = cfg.split(":"); fold = int(fold); tr, val, te = M17.nested_masks(df, scheme, fold, lab, a.vfold)
        print(f"\n== {scheme} fold {fold}: train {tr.sum():,} / val {val.sum():,} / test {te.sum():,}", flush=True)
        tr_idx = np.where(tr)[0]; preds = []
        for seed in [int(s) for s in a.seeds.split(",")]:
            torch.manual_seed(seed); np.random.seed(seed); model = ObsGAT(a.d).to(dev); opt = torch.optim.Adam(model.parameters(), a.lr)
            yt = torch.tensor(y - mu, dtype=torch.float32, device=dev); best = (np.inf, -1, None)
            for ep in range(a.epochs):
                t1 = time.time(); model.train(); perm = np.random.permutation(tr_idx)
                for b in range(0, len(perm), a.bs):
                    sel = perm[b:b + a.bs]; remap = np.full(len(df), -1); remap[sel] = np.arange(len(sel))
                    pe = batch_edges(PE, sel, remap, dev, PE["feat"]); se = batch_edges(SE, sel, remap, dev, bg)
                    yhat, _ = model(torch.tensor(bg[sel], device=dev), pe, se, len(sel)); loss = ((yhat - yt[sel]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                pv = predict(model, bg, PE, SE, bg, np.where(val)[0], dev); rv = float(np.sqrt(np.nanmean((pv[val] - (y - mu)[val]) ** 2)))
                if rv < best[0]: best = (rv, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
                if ep % 5 == 0 or ep == a.epochs - 1: print(f"  [s{seed}] ep {ep:2d} val {rv:.3f} β {model.beta.item():+.3f} {time.time()-t1:.0f}s", flush=True)
            if best[2] is None:  # 유효 val 에폭 0 → 명시적 실패 (train5·train.fit_one 과 같은 가드, QA D3)
                raise RuntimeError(f"[{scheme}{fold} s{seed}] 유효한 val 에폭 없음 (epochs={a.epochs})")
            model.load_state_dict(best[2]); pred = predict(model, bg, PE, SE, bg, np.arange(len(df)), dev) + mu; preds.append(pred)
            r = pred - y; rt = float(np.sqrt(np.nanmean(r[tr] ** 2))); rte = float(np.sqrt(np.nanmean(r[te] ** 2)))
            rows.append(dict(scheme=scheme, fold=fold, seed=seed, best_ep=best[1], val=best[0], train=rt, test=rte, beta=model.beta.item()))
            print(f"  → s{seed}: best ep {best[1]} val {best[0]:.3f} train {rt:.3f} test {rte:.3f} β {model.beta.item():+.3f}", flush=True)
            pd.DataFrame(dict(row_idx=df.row_idx, pred=pred, xco2=y, test=te)).to_parquet(f"{a.out}/pred_{scheme}{fold}_s{seed}.parquet", index=False)
        # 우리 물리 3시드 평균과 짝 비교 (test)
        pm = np.mean(preds, 0)
        fs = sorted(glob.glob(f"{a.model_preds}/pred_{scheme}{fold}_s*_phys.parquet"))
        if not fs:  # 물리 예측 없음 → 짝 비교·거리 층 건너뜀, GAT 행은 summary 에 남김 (QA D4)
            print(f"  물리 예측 없음 ({a.model_preds}/pred_{scheme}{fold}_s*_phys.parquet) → B1·거리 층 건너뜀", flush=True)
            pd.DataFrame(rows).to_csv(f"{a.out}/summary.csv", index=False); continue
        ours = pd.concat([pd.read_parquet(f)[["row_idx", "pred"]] for f in fs]).groupby("row_idx").pred.mean(); po = df.row_idx.map(ours).to_numpy()
        ra, rb = (po - y)[te], (pm - y)[te]; ok = np.isfinite(ra) & np.isfinite(rb)
        d_pt = np.sqrt((rb[ok] ** 2).mean()) - np.sqrt((ra[ok] ** 2).mean()); lo, hi = M17.block_bootstrap(ra[ok], rb[ok], days[te][ok])
        rows.append(dict(scheme=scheme, fold=fold, seed="B1", delta_gat_minus_phys=d_pt, ci_lo=lo, ci_hi=hi, gat_test=float(np.sqrt((rb[ok] ** 2).mean())), phys_test=float(np.sqrt((ra[ok] ** 2).mean()))))
        print(f"  B1 {scheme}{fold}: RMSE GAT(3시드 평균) {rows[-1]['gat_test']:.3f} vs 물리 {rows[-1]['phys_test']:.3f} · Δ(GAT−물리) {d_pt:+.3f} CI [{lo:+.3f}, {hi:+.3f}]", flush=True)
        # 거리 층
        st = pd.read_parquet(f"experiments/rung1/rung1_{scheme}{fold}.parquet"); st["pred_gat"] = st.row_idx.map(pd.Series(pm, index=df.row_idx)).to_numpy()
        st["stratum"] = pd.cut(st.d_min, M18.STRATA, right=False, labels=[f"{M18.STRATA[i]}–{M18.STRATA[i+1]}" for i in range(len(M18.STRATA) - 1)])
        cols = [c for c in ("pred_trend", "pred_krig", "pred_gat", "pred_phys", "pred_nophys") if c in st]
        srows = []
        for s_, g in list(st.groupby("stratum", observed=False)) + [("전체", st)]:
            if len(g) == 0: continue
            srows.append(dict(stratum=str(s_), n=len(g), **{c.replace("pred_", "rmse_"): float(np.sqrt(((g[c] - g.xco2) ** 2).mean())) for c in cols}))
        sdf = pd.DataFrame(srows); sdf.to_csv(f"{a.out}/strata_{scheme}{fold}.csv", index=False); print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    pd.DataFrame(rows).to_csv(f"{a.out}/summary.csv", index=False); print(f"\n총 {time.time()-t0:.0f}s → {a.out}/")
