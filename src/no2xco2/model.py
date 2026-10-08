"""B판 모델 (결정 1–8 전 부품). prep()/__main__ 은 구 파일럿 전용 경로(월별 NO₂ μσ·배경 전 행 fit = D2/D4 미적용, 결정 1 미적용) — 5년 학습은 train5 + loader 를 쓴다 (QA S8).

상태  h ∈ R^{N×d}, N = 121×201 격자 노드, d 채널.
스텝  h ← 물리(h): 반라그랑주 이류(ERA5 z100 바람, cos lat 계량) → 확산 Δt·D·∇²h (D = D_cap·σ(θ_D), 채널별 등방, D_cap = min_lat 0.9·D_max = 28,245 m²/s; 승인 2026-09-22 S1(b)) → 감쇠 h·exp(−Δt·k) (k = exp(logk), 승인 S2)
      화소 있는 스텝: h ← 주입(h, 화소): 엣지(화소→4 노드) 어텐션 집계 u, 게이트 g=σ(W[h,u]) → h=(1−g)h+g·tanh(Wu)  (덮어쓰기 아님)
디코드 사운딩 ← 4 노드 엣지 어텐션 → z ∈ R^d → NO2_latent = w·z (스칼라) ; ŷ = μ + g(배경) + β·NO2_latent
      g = MLP(배경 n_bg 변수: 구 prep 7 = 위도·경도·일·시각·t2m·blh·sp / train5 = loader.BG_FEATS, 개수 loader.N_BG) n_bg→32→1, β 스칼라 (계절·지역 분해는 5년에서)
학습  월 연속 롤아웃, 48스텝 청크 단위 절단 역전파(청크 경계에서 상태 detach), 손실 = 청크 내 train 사운딩 MSE.
평가  분할(13)의 train_mask/test — 시간·공간 폴드. 출력: train/test RMSE, β, D, k, 벽시계.
어텐션 커널은 인코더/디코더 타입별 분리(SPIN형). 물리 연산자에는 어텐션 없음.
"""
import argparse, glob, os, time, math
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as Fn, torch.utils.checkpoint
import no2xco2.data.splits as splits_mod
from no2xco2.config import GRID_LAT0, GRID_D

NLAT, NLON = 121, 201; N = NLAT * NLON; DT = 3600.0; KM = 111.2e3 * GRID_D


# ---------------- 물리 연산자 ----------------
class Physics(nn.Module):
    def __init__(self, d, dev):
        super().__init__()
        # 확산 계수 D = D_cap·sigmoid(thetaD): 매끄러운 상한(기울기 소실 없음), 채널별 스칼라(등방), D_cap = 전 위도 최소 FTCS 한계의 0.9배 (승인 2026-09-22 S1(b)).
        # 초기 D 5,000 m²/s → thetaD0 = logit(5000 / D_cap). Dmax·D_cap 은 __init__ 아래에서 계량 버퍼 뒤에 정의.
        self.logk = nn.Parameter(torch.full((1,), math.log(1 / (24 * 3600.0))))  # k 초기 1/일
        lat = GRID_LAT0 + GRID_D * torch.arange(NLAT, dtype=torch.float32)
        self.register_buffer("coslat", torch.cos(torch.deg2rad(lat))[:, None])
        ii, jj = torch.meshgrid(torch.arange(NLAT, dtype=torch.float32), torch.arange(NLON, dtype=torch.float32), indexing="ij")
        self.register_buffer("ii", ii); self.register_buffer("jj", jj)
        self.register_buffer("dy2", torch.tensor(KM ** 2)); self.register_buffer("dx2", (KM * self.coslat) ** 2)
        # 명시 확산(FTCS) 안정 한계 DT·D·(2/dx²+2/dy²) ≤ 1 → 위도별 D_max (20°N 50,335 · 50°N 31,383 m²/s); 0.9배를 상한으로 (승인 2026-09-22, QA R1)
        self.register_buffer("Dmax", 0.9 / (DT * (2 / self.dx2 + 2 / self.dy2)))  # [NLAT,1] 위도별 0.9·D_max (참고·검증용)
        self.register_buffer("D_cap", self.Dmax.min())  # 28,245 m²/s (50°N) — 전 위도 안정
        self.thetaD = nn.Parameter(torch.full((d,), math.log(5000.0 / (float(self.D_cap) - 5000.0))))  # logit(5000/D_cap)

    def advect(self, h, u, v):  # h [d,NLAT,NLON]
        si = self.ii - v * DT / KM; sj = self.jj - u * DT / (KM * self.coslat)
        i0 = torch.floor(si); j0 = torch.floor(sj); ti = (si - i0)[None]; tj = (sj - j0)[None]
        inside = ((i0 >= 0) & (i0 < NLAT - 1) & (j0 >= 0) & (j0 < NLON - 1))[None].float()
        i0 = i0.clamp(0, NLAT - 2).long(); j0 = j0.clamp(0, NLON - 2).long()
        g = lambda di, dj: h[:, i0 + di, j0 + dj]
        out = (1 - ti) * (1 - tj) * g(0, 0) + (1 - ti) * tj * g(0, 1) + ti * (1 - tj) * g(1, 0) + ti * tj * g(1, 1)
        return out * inside

    def diffuse(self, h):
        hp = Fn.pad(h[None], (1, 1, 1, 1), mode="replicate")[0]
        c = hp[:, 1:-1, 1:-1]; n = hp[:, :-2, 1:-1]; s = hp[:, 2:, 1:-1]; w = hp[:, 1:-1, :-2]; e = hp[:, 1:-1, 2:]
        lap = (n - c) / self.dy2 + (s - c) / self.dy2 + (w - c) / self.dx2 + (e - c) / self.dx2
        D = self.D()[:, None, None]  # 채널별 등방 D ∈ (0, D_cap)
        return h + DT * D * lap

    def D(self) -> torch.Tensor:
        """유효 확산 계수 [d] (m²/s) = D_cap·σ(θ_D). 보고값도 이 값."""
        return self.D_cap * torch.sigmoid(self.thetaD)

    def forward(self, h, u, v):
        h = self.diffuse(self.advect(h, u, v))
        return h * torch.exp(-DT * torch.exp(self.logk))  # 감쇠 exp(−Δt·k): k 무제약이어도 무조건 안정 (승인 2026-09-22 S2; 구식 1−Δt·k 는 k > 1/Δt 에서 부호 반전)


# ---------------- 엣지 어텐션 (타입별 분리) ----------------
def segment_softmax(logit, seg, nseg):
    m = torch.full((nseg,), -1e9, device=logit.device).scatter_reduce(0, seg, logit, "amax")
    e = torch.exp(logit - m[seg]); z = torch.zeros(nseg, device=logit.device).index_add(0, seg, e)
    return e / (z[seg] + 1e-12)


class EdgeAttention(nn.Module):
    """엣지 (소스 특징, 엣지 속성, 타깃 잠재) → 어텐션 가중 집계. 인코더(화소→노드)·디코더(노드→사운딩) 각각 별도 인스턴스."""
    def __init__(self, d_src, d_edge, d, hidden=32):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(d_src + d_edge + d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.att = nn.Sequential(nn.Linear(d_src + d_edge + d, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, src, edge, tgt_h, seg, nseg):
        x = torch.cat([src, edge, tgt_h], -1); a = segment_softmax(self.att(x).squeeze(-1), seg, nseg)
        return torch.zeros(nseg, tgt_h.shape[-1], device=src.device).index_add(0, seg, a[:, None] * self.msg(x)), a


class Model(nn.Module):
    def __init__(self, d, dev, n_bg=7, n_beta=1):
        super().__init__(); self.d = d
        self.phys = Physics(d, dev)
        self.enc = EdgeAttention(d_src=2, d_edge=3, d=d)            # 화소: (no2 표준화, qa) ; 엣지: (w, dy, dx)
        self.gate = nn.Linear(2 * d, d); self.upd = nn.Linear(d, d)
        self.dec = EdgeAttention(d_src=d, d_edge=3, d=d)            # 노드 잠재 → 사운딩 ; 엣지: (w, dy, dx)
        self.readout = nn.Linear(d, 1); self.beta = nn.Parameter(torch.zeros(n_beta))  # n_beta = 1 (스칼라, 현행) 또는 12 (월별 β_m, 결정 7-a 승인 09-24)
        self.g = nn.Sequential(nn.Linear(n_bg, 32), nn.GELU(), nn.Linear(32, 1))  # n_bg: 구 prep 7, train5 는 loader.n_bg(harmonics) (기본 11 = loader.N_BG, --harmonics 3 이면 15; TS-1)

    def inject(self, h, px):  # h [N,d]; px: dict(node [E], edge [E,3], src [E,2])
        # 채널 0 = 물리 NO2 채널: 표준화 no2_tvcd 의 4점 가중평균을 관측 노드에 직접 퇴적 (학습 가중 없음 → β 부호·단위 식별)
        w = px["edge"][:, 0]; num = torch.zeros(N, device=h.device).index_add(0, px["node"], w * px["src"][:, 0])
        den = torch.zeros(N, device=h.device).index_add(0, px["node"], w); touched = den > 0
        phys = torch.where(touched, num / den.clamp(min=1e-9), h[:, 0])
        # 채널 1..d-1 = 학습 채널: 엣지 어텐션 집계 → 게이트 결합 (덮어쓰기 아님)
        u, _ = self.enc(px["src"], px["edge"], h[px["node"]], px["node"], N)
        g = torch.sigmoid(self.gate(torch.cat([h, u], -1))) * touched[:, None]
        hl = (1 - g) * h + g * torch.tanh(self.upd(u))
        return torch.cat([phys[:, None], hl[:, 1:]], -1)

    def decode(self, h, q):  # q: dict(node [4S], edge [4S,3], seg [4S], bg [S,n_bg])
        # NO2_latent = 물리 채널 0 의 4점 이중선형 디코드 (학습 가중 없음). 학습 채널은 어텐션 디코드 → g 의 잔차 보정 항.
        no2_lat = torch.zeros(q["n"], device=h.device).index_add(0, q["seg"], q["edge"][:, 0] * h[q["node"], 0])
        z, a = self.dec(h[q["node"]], q["edge"], torch.zeros(len(q["seg"]), self.d, device=h.device), q["seg"], q["n"])
        beta = self.beta if self.beta.numel() == 1 else self.beta[q["month"]]  # 월별 모드: 사운딩 UTC 월(0–11)의 β_m
        return self.g(q["bg"]).squeeze(-1) + beta * no2_lat + self.readout(z).squeeze(-1), no2_lat, a


# ---------------- 데이터 준비 ----------------
def prep(dev, month="202001", qa_min=0.75):
    """구 파일럿 전용 (QA S8): 구 encoder_index 경로, 월별 NO₂ μσ, 배경 7변수를 전 행으로 표준화(누수) — train.py·obs_gat 호환용. 신규 실험은 loader 사용."""
    enc = pq.read_table(f"data/processed/encoder_index/enc_tropomi_{month}.parquet",
                        columns=["step_h", "qa", "no2_tvcd", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3"]).to_pandas()
    enc = enc[enc.qa >= qa_min]; mu, sd = enc.no2_tvcd.mean(), enc.no2_tvcd.std()
    ty = (enc.w2 + enc.w3).to_numpy(np.float32); tx = (enc.w1 + enc.w3).to_numpy(np.float32)
    off = {0: (ty, tx), 1: (ty, tx - 1), 2: (ty - 1, tx), 3: (ty - 1, tx - 1)}  # 노드 k 기준 화소 오프셋(셀 단위)
    px = {}
    for s, g in enc.groupby("step_h"):
        idx = g.index.to_numpy(); nodes, edge, src = [], [], []
        for k in range(4):
            sel = enc.index.get_indexer(idx)
            nodes.append(g[f"n{k}"].to_numpy()); edge.append(np.stack([g[f"w{k}"].to_numpy(np.float32), off[k][0][sel], off[k][1][sel]], 1))
            src.append(np.stack([((g.no2_tvcd - mu) / sd).to_numpy(np.float32), g.qa.to_numpy(np.float32)], 1))
        px[int(s)] = dict(node=torch.tensor(np.concatenate(nodes), device=dev), edge=torch.tensor(np.concatenate(edge), device=dev),
                          src=torch.tensor(np.concatenate(src), device=dev))
    eo = pq.read_table(f"data/processed/encoder_index/enc_oco_{month}.parquet").to_pandas()
    oco = pq.read_table(sorted(glob.glob(f"data/raw/pilot_{month}/oco_nodes/*.parquet")), columns=["row_idx", "latitude", "longitude", "time", "label", "xco2"]).to_pandas()
    df = oco.merge(eo[["row_idx", "step_h", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3"]], on="row_idx"); df = splits_mod.assign(df)
    from no2xco2.data.era5 import open_wind
    ds = open_wind(f"data/raw/era5_wind/era5_wind_{month}_z100.nc", month)  # 위도 오름차순 (2026-09-22 정정)
    U = torch.tensor(ds["u_pbl"].values, device=dev); V = torch.tensor(ds["v_pbl"].values, device=dev); T = U.shape[0]
    t = np.clip(df.step_h.to_numpy(), 0, T - 1); met = {}
    for v in ("t2m", "blh", "sp"):
        A = ds[v].values; acc = np.zeros(len(df))
        for k in range(4):
            n = df[f"n{k}"].to_numpy(); acc += df[f"w{k}"].to_numpy() * A[t, n // NLON, n % NLON]
        met[v] = acc
    tt = pd.to_datetime(df["time"])
    bg = np.column_stack([df.latitude, df.longitude, tt.dt.day, tt.dt.hour + tt.dt.minute / 60, met["t2m"], met["blh"], met["sp"]]).astype(np.float32)
    bg = (bg - bg.mean(0)) / (bg.std(0) + 1e-6)
    ty = (df.w2 + df.w3).to_numpy(np.float32); tx = (df.w1 + df.w3).to_numpy(np.float32)
    qry = {}
    for s, g in df.groupby("step_h"):
        pos = df.index.get_indexer(g.index); nodes, edge = [], []
        for k, (oy, ox) in enumerate([(ty, tx), (ty, tx - 1), (ty - 1, tx), (ty - 1, tx - 1)]):
            nodes.append(g[f"n{k}"].to_numpy()); edge.append(np.stack([g[f"w{k}"].to_numpy(np.float32), oy[pos], ox[pos]], 1))
        S = len(g); seg = np.tile(np.arange(S), 4)
        qry[int(s)] = dict(node=torch.tensor(np.concatenate(nodes), device=dev), edge=torch.tensor(np.concatenate(edge), device=dev),
                           seg=torch.tensor(seg, device=dev), n=S, bg=torch.tensor(bg[pos], device=dev), rows=pos)
    return px, qry, df, U, V, T


def run(model, px, qry, U, V, T, y, tr_mask, opt=None, chunk=48, px_frac=1.0, h0=None, return_h=False, w=None, gen=None):
    """월 연속 롤아웃. opt 있으면 청크마다 train 사운딩 손실로 역전파. 반환: 전체 예측 벡터 (return_h 면 (pred, 월 말 상태 h)).
    h0: 전월 말 상태 (D3 warm start). None 이면 0 콜드스타트.
    w: 사운딩 손실 가중 (y 와 같은 길이, 결정 13-a 승인 09-24). None 이면 현행(청크 내 평균), 있으면 청크 내 가중 평균 Σw·r²/Σw.
    gen: 평가(opt=None) 에서도 화소 부표본을 쓰려면 torch.Generator 를 준다 (DR-1 깨끗한 val 선택의 px_frac < 1, 09-29). 학습 경로는 현행대로 전역 난수."""
    pred = np.full(len(y), np.nan); dev = U.device
    h = torch.zeros(N, model.d, device=dev) if h0 is None else h0.detach().to(dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev); trm = torch.tensor(tr_mask, device=dev)
    wt = None if w is None else torch.tensor(w, dtype=torch.float32, device=dev)
    for c0 in range(0, T, chunk):
        h = h.detach(); losses = []; n_tr = 0
        for t in range(c0, min(c0 + chunk, T)):
            h = model.phys(h.T.reshape(model.d, NLAT, NLON), U[t], V[t]).reshape(model.d, N).T
            if t in px:
                p = px[t]
                if px_frac < 1.0 and (opt is not None or gen is not None):  # 화소 부표본: 학습(전역 난수, 현행) 또는 평가(gen 고정 시드); 화소 단위 4엣지 유지
                    P = len(p["node"]) // 4; keep = (torch.rand(P, device=h.device) if gen is None else torch.rand(P, generator=gen).to(h.device)) < px_frac; keep4 = keep.repeat(4)
                    p = dict(node=p["node"][keep4], edge=p["edge"][keep4], src=p["src"][keep4])
                h = torch.utils.checkpoint.checkpoint(model.inject, h, p, use_reentrant=False) if opt is not None else model.inject(h, p)
            if t in qry:
                q = qry[t]; yhat, _, _ = model.decode(h, q); rows = torch.tensor(q["rows"], device=dev)
                pred[q["rows"]] = yhat.detach().cpu().numpy()
                m = trm[rows]
                if opt is not None and m.any():
                    if wt is None:
                        losses.append(((yhat[m] - yt[rows][m]) ** 2).sum()); n_tr += int(m.sum())
                    else:
                        ww = wt[rows][m]; losses.append((ww * (yhat[m] - yt[rows][m]) ** 2).sum()); n_tr += float(ww.sum())
        if opt is not None and losses:
            loss = torch.stack(losses).sum() / n_tr; opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return (pred, h.detach()) if return_h else pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="time"); ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--d", type=int, default=8); ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--chunk", type=int, default=24); ap.add_argument("--px-frac", type=float, default=0.25); ap.add_argument("--no-physics", action="store_true", help="절제: 이류·확산 없이 (감쇠만)")
    a = ap.parse_args(); dev = torch.device(a.device); torch.manual_seed(0)
    t0 = time.time(); px, qry, df, U, V, T = prep(dev); print(f"준비 {time.time()-t0:.0f}s · 주입 스텝 {len(px)} · 질의 스텝 {len(qry)} · 사운딩 {len(df):,}")
    y = df.xco2.to_numpy(np.float64); lab = (df.label == "L_train").to_numpy(); mu = y[lab].mean()
    _col = "year" if a.scheme == "year" else f"fold_{a.scheme}"  # year 스킴은 fold_year 열이 없음 (QA T3′)
    tr = splits_mod.train_mask(df, a.scheme, a.fold) & lab; te = (df[_col].to_numpy() == a.fold) & lab
    model = Model(a.d, dev).to(dev)
    if a.no_physics:
        model.phys.advect = lambda h, u, v: h; model.phys.diffuse = lambda h: h
    opt = torch.optim.Adam(model.parameters(), a.lr)
    print(f"{a.scheme} fold {a.fold}: train {tr.sum():,} / test {te.sum():,} · 파라미터 {sum(p.numel() for p in model.parameters()):,} · device {dev}")
    hist = []
    for ep in range(a.epochs):
        t1 = time.time(); model.train(); pred = run(model, px, qry, U, V, T, y - mu, tr, opt, chunk=a.chunk, px_frac=a.px_frac)
        r = pred - (y - mu); rt = np.sqrt(np.nanmean(r[tr] ** 2)); rv = np.sqrt(np.nanmean(r[te] ** 2))
        D = model.phys.D().detach().cpu().numpy(); k = torch.exp(model.phys.logk).item()
        hist.append((ep, rt, rv)); print(f"ep {ep:2d}  train RMSE {rt:.3f}  test RMSE {rv:.3f}  β {model.beta.item():+.3f}  D 중앙 {np.median(D):.0f} m²/s  수명 1/k {1/k/3600:.1f} h  {time.time()-t1:.0f}s", flush=True)
    best = min(hist, key=lambda x: x[1]); print(f"\n최종 ep {hist[-1][0]}: train {hist[-1][1]:.3f} / test {hist[-1][2]:.3f} ppm · 예측 결측 {np.isnan(pred[lab]).mean():.1%} · 총 {time.time()-t0:.0f}s")
    os.makedirs("data/processed/model_pilot", exist_ok=True)
    pd.DataFrame(dict(row_idx=df.row_idx, pred=pred + mu, xco2=y, label=df.label, train=tr, test=te)).to_parquet(f"data/processed/model_pilot/pred_{a.scheme}{a.fold}_d{a.d}{'_nophys' if a.no_physics else ''}.parquet", index=False)
