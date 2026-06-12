"""
base_pinn.py
=============
길 B(XCO₂ 이상장 재구성) 6종 PINN의 공유 베이스.

설계 원칙:
  · 6종이 다른 부분 = **인코더**(x_seq → 공간특징)뿐. 나머지(헤드·SR source·PDE·게이트)는 공유.
  · 살아있는 물리코어 재사용: core/pde_residual.PDEResidual (등방성 Laplacian + source + anti-escape).
  · SR source = Path A가 실증한 선형항 S_anthro = softplus(β)·NO₂ (β₀=0.0081 ppm/(μmol m⁻²)).
    └ 소실된 복잡 SRPrior 대체. LOYO에서 선형 β만이 강건했으므로 물리적으로도 정직한 선택.
  · trivial escape 차단: (1) PDEResidual 내부 delta_eta 제거·tanh soft cap,
    (2) 본 베이스의 anti-mean 정규화항(공간평균 수렴 페널티)을 손실에 제공.

forward 계약 (6종 공통):
  입력  : x_seq (B,T,C,H,W), lat_norm/lon_norm (B,1,H,W), no2/u/v/blh/ndvi (B,1,H,W),
          C_prev (B,1,H,W) optional
  출력  : c_hat (B,1,H,W), log_var (B,1,H,W), pde_res (B,1,H,W),
          s_anthro (B,1,H,W), gate (B,1,H,W)

각 아키텍처는 `encode(self, x_seq, lat_norm, lon_norm) -> (B, hidden, H, W)` 만 구현.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pde_residual import PDEResidual

# Path A 실증 NO₂ 선형계수 (시드 CV<1%, r=1.0)
BETA_NO2_INIT = 0.0081


class BaseSpatioTemporalPINN(nn.Module):
    """6종 PINN 공유 베이스. 서브클래스는 encode()만 구현."""

    def __init__(self, in_channels: int, hidden: int = 64,
                 beta_init: float = BETA_NO2_INIT, use_pde: bool = True,
                 logvar_init: float = -2.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.use_pde = use_pde

        # ── 공유 디코더 헤드 (encode 출력 → 평균/로그분산) ──
        self.head_mean = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=1), nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
        )
        self.head_logvar = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.constant_(self.head_logvar.bias, logvar_init)

        # ── SR source: S_anthro = softplus(β)·NO₂ (Path A 선형항) ──
        # softplus⁻¹(0.0081) ≈ log(exp(0.0081)-1) ≈ -4.81 → 양수 보장 + 학습 가능
        inv = math.log(math.expm1(beta_init)) if beta_init > 0 else -5.0
        self.beta_raw = nn.Parameter(torch.tensor(float(inv)))

        # ── SR 수용 게이트 (픽셀별 0~1; 레짐 의존 차단/수용) ──
        self.gate = nn.Sequential(
            nn.Conv2d(hidden, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid(),
        )

        # ── 물리코어 ──
        if use_pde:
            self.pde = PDEResidual()

    # 서브클래스 구현 지점 ────────────────────────────────────────────────
    def encode(self, x_seq: torch.Tensor,
               lat_norm: torch.Tensor, lon_norm: torch.Tensor,
               obs_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("아키텍처별 인코더에서 구현: (B,T,C,H,W)→(B,hidden,H,W)")

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw)

    # 공유 forward ───────────────────────────────────────────────────────
    def forward(self, x_seq, lat_norm, lon_norm, no2, u, v,
                blh=None, ndvi=None, C_prev=None, obs_mask=None, dt: float = 86400.0):
        feat = self.encode(x_seq, lat_norm, lon_norm, obs_mask)  # (B,hidden,H,W)
        c_hat = self.head_mean(feat)                            # (B,1,H,W)
        log_var = self.head_logvar(feat).clamp(-2.0, 1.0)
        gate = self.gate(feat)                                  # (B,1,H,W) ∈(0,1)

        # SR 선형 source (게이트로 레짐 의존 수용)
        s_anthro = gate * self.beta * no2                       # (B,1,H,W)

        if self.use_pde:
            s_flux = s_anthro
            pde_res, _ = self.pde(c_hat, u, v, S_flux=s_flux,
                                  no2=no2, blh=blh, ndvi=ndvi, C_hat_prev=C_prev, dt=dt)
        else:
            pde_res = torch.zeros_like(c_hat)
        return c_hat, log_var, pde_res, s_anthro, gate

    # trivial escape 방지용 보조 정규화 (손실에서 사용) ──────────────────
    @staticmethod
    def anti_mean_penalty(c_hat: torch.Tensor) -> torch.Tensor:
        """
        예측이 공간평균(자명해)으로 붕괴하는 것을 페널티.
        공간 분산이 0에 가까우면 큰 손실 → 평탄해 탈출 차단.
        L = 1 / (var_spatial + ε) 대신, -log(var) 형태로 수치 안정.
        """
        var = c_hat.var(dim=(2, 3), keepdim=False)              # (B,1)
        return (-torch.log(var.clamp(min=1e-6))).mean()
