"""
losses_adaptive.py — v4 공통 동적 물리 손실 가중치 + Trivial Escape 통합 정규화
================================================================================
docs/v4_dual_architecture_spec_20260612.html §2 스펙 구현.

구성 요소:
  1. DynamicPhysicsWeighter — 그래디언트-노름 균형 λ_pde 스케줄러.
       λ̂_e = ḡ_data / (ḡ_pde + ε)          (인코더 파라미터 기준 L2 노름 비)
       λ_pde^(e) = clip(ρ·λ_pde^(e−1) + (1−ρ)·λ̂_e, [λ_min, λ_max]) · r(e)
     · ρ=0.9 (시간상수 10 epoch ≈ Phase 2 길이의 1/4 — 배치 노이즈 평활,
       phase 전환 추적)
     · r(e) = 기존 커리큘럼 램프(P1: r_min 고정, P2: 0→1 선형, P3: 1)
     · F3 전례(이중 곱셈으로 PDE 가중 소멸) 방지: CurriculumScheduler의
       lambdas['pde'] 절대값을 본 스케줄러 출력으로 **대체**한다(곱 아님).
     · AdaptivePhysicsLoss의 SNR 픽셀 가중은 분포(어디에), 본 스케줄러는
       전역 스케일(얼마나) — 직교라 병존.

  2. TrivialEscapeRegularizer — 공간평균 편법 수렴 차단 통합 페널티.
       L_esc = w_am·(−log σ²_pred) + w_vf·E[max(0, 1 − σ_pred/σ_target)²]
     · 첫 항 = base_pinn.anti_mean_penalty와 동일식(여기로 흡수 — 중복 부과 금지)
     · 둘째 항 = 분산 하한: 예측장 표준편차가 관측 타겟 표준편차에 못 미치면
       벌점 — 평균 수렴의 직접 차단(관측픽셀 기준, 샘플별).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DynamicPhysicsWeighter:
    """그래디언트-노름 균형 동적 λ_pde 스케줄러 (상태 보유, nn.Module 아님)."""

    def __init__(self, total_epochs: int,
                 rho: float = 0.9,
                 lambda_min: float = 0.05, lambda_max: float = 5.0,
                 phase1_end: float = 0.3, phase2_end: float = 0.6,
                 ramp_min: float = 0.05):
        self.total_epochs = total_epochs
        self.rho = rho
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.p1 = int(total_epochs * phase1_end)
        self.p2 = int(total_epochs * phase2_end)
        self.ramp_min = ramp_min
        self.lam = 1.0                 # EMA 상태 (램프 곱 전 λ)
        self._ratios: list[float] = []  # 에폭 내 관측된 ḡ_data/ḡ_pde

    # ── 커리큘럼 램프 r(e): 기존 3-phase 의미 유지 ──
    def ramp(self, epoch: int) -> float:
        if epoch < self.p1:
            return self.ramp_min
        if epoch < self.p2:
            return (epoch - self.p1) / max(self.p2 - self.p1, 1)
        return 1.0

    @staticmethod
    def _grad_norm(loss: torch.Tensor, params: list[torch.Tensor]) -> float:
        grads = torch.autograd.grad(loss, params, retain_graph=True,
                                    allow_unused=True)
        sq = 0.0
        for g in grads:
            if g is not None:
                sq += float(g.detach().pow(2).sum())
        return sq ** 0.5

    def observe(self, model: nn.Module,
                loss_data: torch.Tensor, loss_pde: torch.Tensor) -> None:
        """배치의 L_data·L_pde 그래디언트 노름 비를 측정(backward 전 호출).

        측정 대상 = 인코더 공유 파라미터(pde 모듈 제외 — P1 동결로 grad 부재).
        retain_graph=True라 이후 total.backward()와 충돌 없음.
        """
        params = [p for n, p in model.named_parameters()
                  if p.requires_grad and not n.startswith("pde.")]
        if not params:
            return
        g_data = self._grad_norm(loss_data, params)
        g_pde = self._grad_norm(loss_pde, params)
        if g_pde > 1e-12 and g_data > 1e-12:
            self._ratios.append(g_data / g_pde)

    def update(self) -> None:
        """에폭 말 호출 — 관측 비율 평균으로 EMA 갱신."""
        if self._ratios:
            lam_hat = sum(self._ratios) / len(self._ratios)
            self.lam = self.rho * self.lam + (1.0 - self.rho) * lam_hat
            self.lam = min(max(self.lam, self.lambda_min), self.lambda_max)
        self._ratios = []

    def get_lambdas(self, epoch: int) -> dict:
        """CurriculumScheduler.get_lambdas 호환 — 'pde'를 동적 값으로 대체."""
        return {"data": 1.0, "pde": self.lam * self.ramp(epoch), "reg": 1.0}

    def state(self) -> dict:
        return {"lam_ema": self.lam}


class TrivialEscapeRegularizer(nn.Module):
    """anti-mean(−log 분산) + 분산 하한 통합 페널티. 중복 부과 방지를 위해
    dynamic 모드에서는 base_pinn.anti_mean_penalty 대신 이것만 사용한다."""

    def __init__(self, w_anti_mean: float = 0.01, w_var_floor: float = 0.1):
        super().__init__()
        self.w_am = w_anti_mean
        self.w_vf = w_var_floor

    def forward(self, c_hat: torch.Tensor, target: torch.Tensor,
                obs_mask: torch.Tensor) -> torch.Tensor:
        # ── anti-mean: 공간 분산 0 수렴 차단 (base_pinn과 동일식) ──
        var = c_hat.var(dim=(2, 3), keepdim=False)               # (B,1)
        l_am = (-torch.log(var.clamp(min=1e-6))).mean()

        # ── 분산 하한: σ_pred < σ_target(관측픽셀)이면 벌점 ──
        m = (obs_mask > 0.5).float()
        n = m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        t_mean = (target * m).sum(dim=(2, 3), keepdim=True) / n
        t_var = ((target - t_mean) ** 2 * m).sum(dim=(2, 3), keepdim=True) / n
        sigma_t = t_var.clamp(min=1e-12).sqrt().squeeze(-1).squeeze(-1)  # (B,1)
        sigma_p = var.clamp(min=1e-12).sqrt()                            # (B,1)
        l_vf = torch.relu(1.0 - sigma_p / (sigma_t + 1e-6)).pow(2).mean()

        return self.w_am * l_am + self.w_vf * l_vf
