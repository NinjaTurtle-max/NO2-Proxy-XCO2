"""물리 주입 잠재 격자 그래프 모델.

상태  h ∈ R^{N×d}, N = 121×201 격자 노드(0.25°), d 채널.
스텝  (1 h) 물리: 반라그랑주 이류(ERA5 경계층 바람, cos 위도 계량) → 확산 Δt·D·∇²h (D = D_cap·σ(θ_D), 채널별 등방,
      D_cap = 전 위도 FTCS 안정 한계의 0.9배 = 28,245 m²/s) → 감쇠 h·exp(−Δt·k), k = exp(logk).
      화소가 있는 스텝: 주입. 채널 0 = 표준화 NO₂ 의 4점 가중 평균(학습 가중 없음),
      채널 1..d−1 = 엣지 어텐션 집계 u, 게이트 g = σ(W[h, u]) → h = (1 − g)h + g·tanh(Wu).
디코드 ŷ = g(배경 11변수) + β·NO₂_latent + r(z),  NO₂_latent = 채널 0 의 4점 이중선형 값, z = 4노드 엣지 어텐션.
학습  월 연속 롤아웃, 48스텝 청크 절단 역전파(청크 경계에서 상태 detach), 손실 = 청크 안 훈련 사운딩 MSE.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.utils.checkpoint

from no2xco2.config import GRID_D, GRID_LAT0

NLAT, NLON = 121, 201; N = NLAT * NLON; DT = 3600.0; KM = 111.2e3 * GRID_D


# ---------------- 물리 연산자 ----------------
class Physics(nn.Module):
    def __init__(self, d, dev):
        super().__init__()
        self.logk = nn.Parameter(torch.full((1,), math.log(1 / (24 * 3600.0))))  # k 초기 1/일
        lat = GRID_LAT0 + GRID_D * torch.arange(NLAT, dtype=torch.float32)
        self.register_buffer("coslat", torch.cos(torch.deg2rad(lat))[:, None])
        ii, jj = torch.meshgrid(torch.arange(NLAT, dtype=torch.float32), torch.arange(NLON, dtype=torch.float32), indexing="ij")
        self.register_buffer("ii", ii); self.register_buffer("jj", jj)
        self.register_buffer("dy2", torch.tensor(KM ** 2)); self.register_buffer("dx2", (KM * self.coslat) ** 2)
        # 명시 확산(FTCS) 안정 한계 Δt·D·(2/dx² + 2/dy²) ≤ 1 → 위도별 D_max 의 0.9배, 그 최솟값(50°N)을 상한으로
        self.register_buffer("Dmax", 0.9 / (DT * (2 / self.dx2 + 2 / self.dy2)))
        self.register_buffer("D_cap", self.Dmax.min())
        self.thetaD = nn.Parameter(torch.full((d,), math.log(5000.0 / (float(self.D_cap) - 5000.0))))  # 초기 D = 5,000 m²/s

    def advect(self, h, u, v):  # h [d, NLAT, NLON]
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
        return h + DT * self.D()[:, None, None] * lap

    def D(self) -> torch.Tensor:
        """유효 확산 계수 [d] (m²/s) = D_cap·σ(θ_D)."""
        return self.D_cap * torch.sigmoid(self.thetaD)

    def forward(self, h, u, v):
        h = self.diffuse(self.advect(h, u, v))
        return h * torch.exp(-DT * torch.exp(self.logk))


# ---------------- 엣지 어텐션 ----------------
def segment_softmax(logit, seg, nseg):
    m = torch.full((nseg,), -1e9, device=logit.device).scatter_reduce(0, seg, logit, "amax")
    e = torch.exp(logit - m[seg]); z = torch.zeros(nseg, device=logit.device).index_add(0, seg, e)
    return e / (z[seg] + 1e-12)


class EdgeAttention(nn.Module):
    """엣지 (소스 특징, 엣지 속성, 타깃 잠재) → 어텐션 가중 집계. 인코더(화소 → 노드)와 디코더(노드 → 사운딩)가 각자 인스턴스를 가진다."""
    def __init__(self, d_src, d_edge, d, hidden=32):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(d_src + d_edge + d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.att = nn.Sequential(nn.Linear(d_src + d_edge + d, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, src, edge, tgt_h, seg, nseg):
        x = torch.cat([src, edge, tgt_h], -1); a = segment_softmax(self.att(x).squeeze(-1), seg, nseg)
        return torch.zeros(nseg, tgt_h.shape[-1], device=src.device).index_add(0, seg, a[:, None] * self.msg(x)), a


class Model(nn.Module):
    def __init__(self, d, dev, n_bg=11):
        super().__init__(); self.d = d
        self.phys = Physics(d, dev)
        self.enc = EdgeAttention(d_src=2, d_edge=3, d=d)            # 화소: (NO₂ 표준화, qa) · 엣지: (w, dy, dx)
        self.gate = nn.Linear(2 * d, d); self.upd = nn.Linear(d, d)
        self.dec = EdgeAttention(d_src=d, d_edge=3, d=d)            # 노드 잠재 → 사운딩 · 엣지: (w, dy, dx)
        self.readout = nn.Linear(d, 1); self.beta = nn.Parameter(torch.zeros(1))
        self.g = nn.Sequential(nn.Linear(n_bg, 32), nn.GELU(), nn.Linear(32, 1))

    def inject(self, h, px):  # h [N, d]; px: dict(node [E], edge [E, 3], src [E, 2])
        # 채널 0: 표준화 NO₂ 의 4점 가중 평균을 관측 노드에 직접 둔다 (학습 가중 없음 → β 의 부호 · 단위가 정해짐)
        w = px["edge"][:, 0]; num = torch.zeros(N, device=h.device).index_add(0, px["node"], w * px["src"][:, 0])
        den = torch.zeros(N, device=h.device).index_add(0, px["node"], w); touched = den > 0
        phys = torch.where(touched, num / den.clamp(min=1e-9), h[:, 0])
        # 채널 1..d−1: 엣지 어텐션 집계 → 게이트 결합
        u, _ = self.enc(px["src"], px["edge"], h[px["node"]], px["node"], N)
        g = torch.sigmoid(self.gate(torch.cat([h, u], -1))) * touched[:, None]
        hl = (1 - g) * h + g * torch.tanh(self.upd(u))
        return torch.cat([phys[:, None], hl[:, 1:]], -1)

    def decode(self, h, q):  # q: dict(node [4S], edge [4S, 3], seg [4S], n, bg [S, n_bg])
        no2_lat = torch.zeros(q["n"], device=h.device).index_add(0, q["seg"], q["edge"][:, 0] * h[q["node"], 0])
        z, a = self.dec(h[q["node"]], q["edge"], torch.zeros(len(q["seg"]), self.d, device=h.device), q["seg"], q["n"])
        yhat = self.g(q["bg"]).squeeze(-1) + self.beta * no2_lat + self.readout(z).squeeze(-1)
        return yhat, no2_lat, a


def run(model, px, qry, U, V, T, y, tr_mask, opt=None, chunk=48, px_frac=1.0, h0=None, return_h=False, gen=None):
    """한 달 연속 롤아웃. opt 가 있으면 청크마다 훈련 사운딩 손실로 역전파한다. 반환: 예측 벡터 (return_h 면 (예측, 월 말 상태)).
    h0: 앞 달 말 상태 (없으면 0 에서 시작). gen: 평가(opt=None)에서 화소 부표본을 쓸 때의 고정 난수 생성기."""
    pred = np.full(len(y), np.nan); dev = U.device
    h = torch.zeros(N, model.d, device=dev) if h0 is None else h0.detach().to(dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev); trm = torch.tensor(tr_mask, device=dev)
    for c0 in range(0, T, chunk):
        h = h.detach(); losses = []; n_tr = 0
        for t in range(c0, min(c0 + chunk, T)):
            h = model.phys(h.T.reshape(model.d, NLAT, NLON), U[t], V[t]).reshape(model.d, N).T
            if t in px:
                p = px[t]
                if px_frac < 1.0 and (opt is not None or gen is not None):  # 화소 부표본 (화소마다 4엣지를 함께 남김)
                    P = len(p["node"]) // 4; keep = (torch.rand(P, device=h.device) if gen is None else torch.rand(P, generator=gen).to(h.device)) < px_frac; keep4 = keep.repeat(4)
                    p = dict(node=p["node"][keep4], edge=p["edge"][keep4], src=p["src"][keep4])
                h = torch.utils.checkpoint.checkpoint(model.inject, h, p, use_reentrant=False) if opt is not None else model.inject(h, p)
            if t in qry:
                q = qry[t]; yhat, _, _ = model.decode(h, q); rows = torch.tensor(q["rows"], device=dev)
                pred[q["rows"]] = yhat.detach().cpu().numpy()
                m = trm[rows]
                if opt is not None and m.any():
                    losses.append(((yhat[m] - yt[rows][m]) ** 2).sum()); n_tr += int(m.sum())
        if opt is not None and losses:
            loss = torch.stack(losses).sum() / n_tr; opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return (pred, h.detach()) if return_h else pred
