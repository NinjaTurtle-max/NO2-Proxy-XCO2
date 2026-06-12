"""
pde_residual.py

[Academic Justification for PDE Source Scaling Constraint]
본 스크립트는 PINN의 편미분 방정식(PDE) 소스항 스케일러(alpha_pde)에 시그모이드(Sigmoid) 기반의
역 매개변수화(Reparameterization)를 적용하여 [0.5, 2.0]의 하드 바운드(Box-constraint)를 강제합니다.
이 제약 범위는 단순한 최적화 휴리스틱이 아니라, 다음의 최신 대기과학/역산 모델링 문헌에 근거한
물리적 불확실성 한계(Physical Uncertainty Bounds)를 엄밀히 차용한 것입니다.

1. 고해상도 인벤토리(ODIAC) 공간 할당 오차 (Spatial Disaggregation Error)
- 학술적 근거: Oda, T., Maksyutov, S., and Andres, R. J. (2018). "The Open-source Data Inventory for Anthropogenic CO2, version 2016 (ODIAC2016): a global monthly fossil fuel CO2 gridded emissions data product for tracer transport simulations and surface flux inversions". Earth System Science Data (ESSD).
- 인용문: "Andres et al. (2016), for example, estimated the uncertainty associated with CDIAC gridded emissions data on a per grid cell basis with an average of 120% and a range of 4.0 to 190% (2σ)." / "Hogue et al. (2016) looked closely at CDIAC gridded emissions data over the US domain and estimated the uncertainty associated with the 1x1 emissions grids as ±150%."
- 해석: 고해상도 격자 단위의 인벤토리 불확실성이 최대 ±150%~190%에 달하므로, 본 모델의 [-50%, +100%] (0.5~2.0) 스케일 제약은 과적합을 방지하는 매우 보수적이고 물리적으로 타당한 신뢰 구간입니다.

2. 생물권 흡수량(Biospheric Flux)의 모델 간 추정 편차 (Factor of 2)
- 학술적 근거: Crowell, S. et al. (2019). "The 2015–2016 carbon cycle as seen from OCO-2 and the global in situ network". Atmospheric Chemistry and Physics (ACP).
- 인용문: "For example, in tropical northern Africa, the LN and LG mean seasonal amplitude (i.e., max minus min flux) was about 1 PgC per month, while in the prior fluxes, the amplitude was about 0.4 PgC per month." / "The strongest of these deviations is evident in northern Africa, where annual net fluxes of carbon were 1.5±0.6 PgC yr-1 for LN and 0.8±0.6 PgC yr-1 for LG."
- 해석: OCO-2 위성 역산 결과(LN, LG)는 기존 사전 플럭스(Prior) 대비 생태계 흡수/배출 진폭을 약 2.5배(0.4 -> 1.0) 또는 2배 가까이(0.8 -> 1.5) 크게 추정합니다. 따라서 최대 2.0배 증폭을 허용하는 상한선은 위성 역산에서 흔히 관찰되는 생태학적 'Factor of 2' 과소평가를 보정하는 정당한 물리적 역산 궤적입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EARTH_R = 6_371_000.0
DEG_TO_M = 111_320.0
RESOLUTION_DEG = 0.25
LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 150.0
lat_edges = np.arange(LAT_MIN, LAT_MAX + RESOLUTION_DEG, RESOLUTION_DEG)
lon_edges = np.arange(LON_MIN, LON_MAX + RESOLUTION_DEG, RESOLUTION_DEG)
LAT_CENTERS = (lat_edges[:-1] + lat_edges[1:]) / 2
LON_CENTERS = (lon_edges[:-1] + lon_edges[1:]) / 2
N_LAT, N_LON = len(LAT_CENTERS), len(LON_CENTERS)


def compute_grid_spacing():
    dy = RESOLUTION_DEG * DEG_TO_M
    lat_rad = np.deg2rad(LAT_CENTERS)
    dx = RESOLUTION_DEG * DEG_TO_M * np.cos(lat_rad)
    return dy, torch.tensor(dx, dtype=torch.float32)


class CentralDiffKernels(nn.Module):
    def __init__(self):
        super().__init__()
        k_dy = torch.tensor([[[-0.5], [0.0], [0.5]]], dtype=torch.float32)
        self.register_buffer("k_dy", k_dy.unsqueeze(0))
        k_dx = torch.tensor([[[-0.5, 0.0, 0.5]]], dtype=torch.float32)
        self.register_buffer("k_dx", k_dx.unsqueeze(0))
        k_dyy = torch.tensor([[[1.0], [-2.0], [1.0]]], dtype=torch.float32)
        self.register_buffer("k_dyy", k_dyy.unsqueeze(0))
        k_dxx = torch.tensor([[[1.0, -2.0, 1.0]]], dtype=torch.float32)
        self.register_buffer("k_dxx", k_dxx.unsqueeze(0))
        dy, dx = compute_grid_spacing()
        self.register_buffer("dy", torch.tensor(dy, dtype=torch.float32))
        self.register_buffer("dx", dx)
        self.register_buffer("inv_dx", (1.0 / dx).view(1, 1, -1, 1))
        self.register_buffer("inv_dx2", (1.0 / dx ** 2).view(1, 1, -1, 1))
        self.register_buffer("inv_dy", 1.0 / self.dy)
        self.register_buffer("inv_dy2", 1.0 / self.dy ** 2)

    def grad_y(self, C):
        return F.conv2d(C, self.k_dy, padding=(1, 0)) * self.inv_dy

    def grad_x(self, C):
        return F.conv2d(C, self.k_dx, padding=(0, 1)) * self.inv_dx

    def laplacian(self, C):
        d2y = F.conv2d(C, self.k_dyy, padding=(1, 0)) * self.inv_dy2
        d2x = F.conv2d(C, self.k_dxx, padding=(0, 1)) * self.inv_dx2
        return d2x + d2y


class UpwindDiffKernels(nn.Module):
    def __init__(self):
        super().__init__()
        k_dy_back = torch.tensor([[[-1.0], [1.0], [0.0]]], dtype=torch.float32)
        self.register_buffer("k_dy_back", k_dy_back.unsqueeze(0))
        k_dx_back = torch.tensor([[[-1.0, 1.0, 0.0]]], dtype=torch.float32)
        self.register_buffer("k_dx_back", k_dx_back.unsqueeze(0))
        k_dy_fwd = torch.tensor([[[0.0], [-1.0], [1.0]]], dtype=torch.float32)
        self.register_buffer("k_dy_fwd", k_dy_fwd.unsqueeze(0))
        k_dx_fwd = torch.tensor([[[0.0, -1.0, 1.0]]], dtype=torch.float32)
        self.register_buffer("k_dx_fwd", k_dx_fwd.unsqueeze(0))
        dy, dx = compute_grid_spacing()
        self.register_buffer("dy", torch.tensor(dy, dtype=torch.float32))
        self.register_buffer("inv_dx", (1.0 / dx).view(1, 1, -1, 1))
        self.register_buffer("inv_dy", 1.0 / torch.tensor(dy, dtype=torch.float32))

    def advection_upwind(self, C, u, v):
        dC_dx_back = F.conv2d(C, self.k_dx_back, padding=(0, 1)) * self.inv_dx
        dC_dx_fwd = F.conv2d(C, self.k_dx_fwd, padding=(0, 1)) * self.inv_dx
        u_pos = torch.clamp(u, min=0)
        u_neg = torch.clamp(u, max=0)
        adv_x = u_pos * dC_dx_back + u_neg * dC_dx_fwd
        dC_dy_back = F.conv2d(C, self.k_dy_back, padding=(1, 0)) * self.inv_dy
        dC_dy_fwd = F.conv2d(C, self.k_dy_fwd, padding=(1, 0)) * self.inv_dy
        v_pos = torch.clamp(v, min=0)
        v_neg = torch.clamp(v, max=0)
        adv_y = v_pos * dC_dy_back + v_neg * dC_dy_fwd
        return adv_x + adv_y


class FourierFeatures(nn.Module):
    def __init__(self, in_ch, mapping_size=64, sigma=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_ch, mapping_size) * sigma,
                              requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.size()
        xp = x.permute(0, 2, 3, 1)
        proj = torch.matmul(xp, self.B).permute(0, 3, 1, 2)
        return torch.cat([torch.sin(2 * np.pi * proj),
                          torch.cos(2 * np.pi * proj)], dim=1)


class PDEResidual(nn.Module):
    """Advection 제거 + 비선형 Source + discrepancy + Box-constrained scaling."""

    def __init__(self, use_upwind=True, init_log_K=9.2, init_log_lambda=-13.0,
                 c_scale: float = 1.4434, u_scale: float = 5.0,
                 learn_pde_alpha: bool = True):
        super().__init__()
        self.use_upwind = use_upwind
        self.central = CentralDiffKernels()
        if use_upwind:
            self.upwind = UpwindDiffKernels()

        self.log_K = nn.Parameter(torch.tensor(init_log_K, dtype=torch.float32))
        self.log_lambda = nn.Parameter(torch.tensor(init_log_lambda, dtype=torch.float32))
        self.log_K_bounds = (6.9, 11.5)
        self.log_lambda_bounds = (-16.1, -11.5)

        # 기후/식생 기반 Box-constrained 영역 내 학습 파라미터 설정
        # Hard clipping의 구배 사멸 방지를 위해 Sigmoid 기반 역 매개변수화 적용
        # alpha_pde = 0.5 + 1.5 * sigmoid(w_raw). 초기값 1.0을 위해 w_raw ≈ -0.6931
        if learn_pde_alpha:
            self.pde_alpha_raw = nn.Parameter(torch.tensor(-0.6931))
        else:
            self.register_buffer('pde_alpha_raw', torch.tensor(-0.6931))

        self.register_buffer('c_scale', torch.tensor(float(c_scale), dtype=torch.float32))
        self.register_buffer('u_scale', torch.tensor(float(u_scale), dtype=torch.float32))

        self.S_NN = nn.Sequential(
            nn.Conv2d(4, 32, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 1), nn.SiLU(),
            nn.Conv2d(32, 1, 1)
        )
        # [P1] delta_eta 제거: 공간 편향 네트워크가 PDE 잔차를 통째로 흡수하는
        # trivial escape 경로를 차단. 모듈 등록은 체크포인트 호환성 유지를 위해 보존하되,
        # forward에서 호출하지 않는다.
        # [P2] sigma 10.0 → 1.0: 고주파 Fourier 기저의 공간 암기 용량 억제.
        self.delta_eta = nn.Sequential(
            FourierFeatures(in_ch=2, mapping_size=64, sigma=1.0),
            nn.Conv2d(128, 32, 1), nn.SiLU(),
            nn.Conv2d(32, 1, 1)
        )
        lat_grid, lon_grid = torch.meshgrid(
            torch.linspace(-1, 1, N_LAT),
            torch.linspace(-1, 1, N_LON),
            indexing='ij'
        )
        xy = torch.stack([lon_grid, lat_grid], dim=0).unsqueeze(0)
        self.register_buffer('xy_grid', xy)

    @property
    def K(self):
        return torch.exp(torch.clamp(self.log_K, *self.log_K_bounds))

    @property
    def lam(self):
        return torch.exp(torch.clamp(self.log_lambda, *self.log_lambda_bounds))

    def forward(self, C_hat, u, v, S_flux=None, gamma_clim=None, mask=None,
                C_hat_prev=None, dt=86400.0, no2=None, blh=None, ndvi=None,
                blh_prev=None):
        if C_hat_prev is not None:
            dC_dt = (C_hat - C_hat_prev) / dt
        else:
            dC_dt = torch.zeros_like(C_hat)

        advection = torch.zeros_like(C_hat)        # E1: advection 제거
        diffusion = -self.K * self.central.laplacian(C_hat)
        
        if blh_prev is not None and blh is not None:
            # E4: 3D 대기 연직 혼합(Dilution)을 2D PDE 반응항으로 Parametrize
            blh_safe = blh.clamp(min=1e-6)
            dilution_rate = (blh - blh_prev) / (dt * blh_safe)
            lambda_dynamic = torch.clamp(dilution_rate, min=0.0) + self.lam
            relaxation = lambda_dynamic * C_hat
        else:
            relaxation = self.lam * C_hat

        if no2 is not None and blh is not None and ndvi is not None:
            # E6: 식생, 국지 프록시, 그리고 인위적 배출량을 결합하여 물리적 상호작용 유도 (4채널 또는 3채널)
            if self.S_NN[0].in_channels == 3:
                bio_source = self.S_NN(torch.cat([no2, blh, ndvi], dim=1))
            else:
                s_flux_for_nn = S_flux if S_flux is not None else torch.zeros_like(no2)
                bio_source = self.S_NN(torch.cat([no2, blh, ndvi, s_flux_for_nn], dim=1))
            if gamma_clim is not None:
                # 기후 변동성 변조 계수(gamma_clim)가 생물권 소스항 강도를 스케일링하도록 결합
                bio_source = bio_source * torch.exp(gamma_clim)
            
            if S_flux is not None:
                # 물리적 인위적 사전 배출량(S_flux)과 융합하여 정합성 보장
                source = S_flux + bio_source
            else:
                source = bio_source
        elif S_flux is not None:
            source = S_flux * torch.exp(gamma_clim) if gamma_clim is not None else S_flux
        else:
            source = torch.zeros_like(C_hat)

        # Box-constrained clamping (단위 보정용 스케일러 단독 유지, 덧셈 바이어스는 완전 제거)
        # 구배 사멸 방지를 위한 시그모이드 기반 Reparameterization: [0.5, 2.0]
        if hasattr(self, 'pde_alpha_raw'):
            alpha_clamped = 0.5 + 1.5 * torch.sigmoid(self.pde_alpha_raw)
        else:
            alpha_clamped = torch.tensor(1.0, device=C_hat.device) # Fallback

        adjusted_source = alpha_clamped * source

        # [P1] delta_eta 호출 제거: 공간 좌표만으로 PDE 잔차를 흡수하는 trivial escape 차단.
        # delta_eta 모듈은 체크포인트 호환성을 위해 등록 유지하나, 잔차 계산에서 배제.
        residual_si = dC_dt + advection + diffusion + relaxation - adjusted_source

        L0 = self.central.dx[self.central.dx.shape[0] // 2].clamp(min=1.0)
        adv_scale = (self.u_scale * self.c_scale / L0).clamp(min=1e-12)
        residual = residual_si / adv_scale
        # [P4] hard clamp → soft tanh cap:
        # clamp(-10,10)는 |residual| > 10 구간의 gradient를 완전히 소거하여
        # 레짐 역전(2023년)처럼 물리 잔차가 크게 폭발하는 구간의 학습 신호를 차단했음.
        # tanh(r/30)*30은 연속적으로 스케일을 낮추므로 gradient가 살아있어
        # 물리 위반이 클수록 더 강하게 억제하도록 역전파 신호가 유지됨.
        residual = torch.nan_to_num(residual, nan=0.0, posinf=30.0, neginf=-30.0)
        residual = 30.0 * torch.tanh(residual / 30.0)
        if mask is not None:
            residual = residual * mask

        diagnostics = {
            "K": self.K.item(), "lambda": self.lam.item(),
            "adv_scale": adv_scale.item(),
            "source_norm": source.abs().mean().item(),
            "delta_norm": 0.0,  # [P1] delta_eta 제거 후 상수 0으로 고정 (진단 필드 호환성 유지)
            "residual_norm": residual.abs().mean().item(),
        }
        return residual, diagnostics

    def pde_loss(self, C_hat, u, v, S_flux=None, gamma_clim=None, mask=None,
                 C_hat_prev=None, dt=86400.0):
        residual, diag = self.forward(C_hat, u, v, S_flux, gamma_clim, mask,
                                      C_hat_prev=C_hat_prev, dt=dt)
        if mask is not None:
            loss = (residual ** 2).sum() / mask.sum().clamp(min=1)
        else:
            loss = (residual ** 2).mean()
        return loss, diag