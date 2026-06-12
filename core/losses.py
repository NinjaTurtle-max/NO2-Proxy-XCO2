"""
losses.py
==========
PI-ConvLSTM Hybrid 모델용 손실 함수 모음.

L_total = λ₁·L_data + λ₂·L_pde + λ₃·L_reg

구성:
  - L_data: Heteroscedastic Huber NLL (관측 마스크 적용, log σ² 클리핑)
  - L_pde:  PDE 잔차 MSE (EMA running scale 정규화)
  - L_reg:  Prediction-SR Anchoring + 비음수 제약 + SR 파라미터 정규화
  - CurriculumScheduler: 3-phase λ 스케줄링

Reference:
  - Kendall & Gal (2017) — heteroscedastic uncertainty
  - Raissi et al. (2019) — PINN loss balancing
  - 설계서 §3.1-3.4
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────
# 경계 마스크 유틸리티 (Warning #2: Conv2D 패딩 왜곡 차단)
# ─────────────────────────────────────────────────────────────────
def make_boundary_mask(H: int, W: int,
                       device=None,
                       pad: int = 2) -> torch.Tensor:
    """
    외곽 `pad` 픽셀을 0으로, 내부를 1로 채운 경계 마스크 생성.

    Conv2D zero-padding으로 인해 경계 영역의 PDE 잔차는 노이만 경계
    조건에 오염된 값임. 이 마스크를 residual² 에 element-wise 곱하면
    오염된 경계 기여가 역전파에서 배제됨.

    Args:
        H, W: 격자 높이·너비 (픽셀)
        device: torch device
        pad: 마스킹할 외곽 픽셀 수 (기본 2 — 3×3 커널의 stencil 반경)

    Returns:
        mask: (1, 1, H, W) float32, interior=1 / boundary=0
    """
    mask = torch.ones(1, 1, H, W, dtype=torch.float32, device=device)
    # 상·하 경계
    mask[:, :, :pad, :]  = 0.0
    mask[:, :, -pad:, :] = 0.0
    # 좌·우 경계
    mask[:, :, :, :pad]  = 0.0
    mask[:, :, :, -pad:] = 0.0
    return mask


# ─────────────────────────────────────────────────────────────────
# L_data: Heteroscedastic Huber NLL
# ─────────────────────────────────────────────────────────────────
class HeteroscedasticHuberNLL(nn.Module):
    """
    관측 불확실성을 학습하는 Huber NLL Loss.

    모델이 Ĉ(x,t)와 log σ²(x,t)를 동시 출력:
      |e| < δ: L = 0.5·e²/σ² + 0.5·log σ²
      |e| ≥ δ: L = δ·(|e| - 0.5δ)/σ² + 0.5·log σ²

    obs_mask=1인 격자에서만 계산 (OCO-2 sparse coverage).
    n_soundings로 가중: 관측 수가 많은 격자에 더 높은 가중치.

    설계서 §3.1: log σ² ∈ [-3, 3] 클리핑 (외부에서 전달).

    Args:
        delta: Huber 전환점 (|e| < δ → L2, else L1)
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, log_var, obs_mask, n_soundings=None):
        """
        Args:
            pred:       (B, 1, H, W) 예측 XCO₂ anomaly
            target:     (B, 1, H, W) 관측 XCO₂ anomaly
            log_var:    (B, 1, H, W) 예측 log σ² (이미 클리핑된 상태)
            obs_mask:   (B, 1, H, W) 관측 마스크 (1=관측, 0=미관측)
            n_soundings:(B, 1, H, W) optional, 관측 수 (가중치)
        Returns:
            loss: scalar
        """
        # [중요 bugfix] target에 포함된 NaN 결측값을 0.0으로 클린 대체 (어차피 obs_mask=0에 의해 차단됨)
        target_clean = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        # 잔차
        e = pred - target_clean
        abs_e = e.abs()

        # Huber loss (per-pixel)
        huber = torch.where(
            abs_e < self.delta,
            0.5 * e.pow(2),
            self.delta * (abs_e - 0.5 * self.delta)
        )

        # Heteroscedastic NLL: huber/σ² + 0.5·log(σ²)
        precision = torch.exp(-log_var)  # 1/σ²
        nll = huber * precision + 0.5 * log_var

        # 관측 마스크 적용
        nll = nll * obs_mask

        # n_soundings 가중치 (sqrt 스케일링으로 극단값 완화)
        if n_soundings is not None:
            # n_soundings에 포함된 NaN도 클린 대체
            n_soundings_clean = torch.where(torch.isnan(n_soundings), torch.ones_like(n_soundings), n_soundings)
            weights = torch.sqrt(n_soundings_clean.clamp(min=1.0))
            
            obs_weights = weights[obs_mask > 0]
            if len(obs_weights) > 0:
                weights = weights / obs_weights.mean()  # 정규화
            else:
                weights = torch.ones_like(weights)
                
            nll = nll * weights

        # 관측 격자 수로 나눠 평균
        n_obs = obs_mask.sum().clamp(min=1.0)
        return nll.sum() / n_obs


# ─────────────────────────────────────────────────────────────────
# Adaptive Physics Loss (SNR-based Weighting)
# ─────────────────────────────────────────────────────────────────
class AdaptivePhysicsLoss(nn.Module):
    """
    제공된 로직 기반: SNR(신호 대 잡음비)에 따른 적응형 가중치 제어.

    1. SNR = |Target| / σ (확실성이 높고 아노말리가 클수록 정보량 증대)
    2. Data Weight: log1p 스케일링으로 Outlier 발산 억제
    3. PDE Weight: Hotspot에서는 제약 완화, Background에서는 물리 법칙 지배

    [F3 수정] 내부 lambda_pde 제거: CurriculumScheduler의 lambdas['pde']가 PDE 가중치를
    단일 경로로 제어하도록 통일. 기존 lambda_pde=0.1 내부 스케일링은 커리큘럼 값과 이중
    곱셈되어 실효 PDE 가중치가 의도의 1/10 수준으로 소멸되는 버그를 유발했음.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha      # Data 가중치 민감도
        self.beta = beta        # PDE 가중치 이완(Relaxation) 강도
        # [F3] lambda_pde 제거 — 커리큘럼 lambdas['pde']가 단독 제어

    def forward(self, pred, target, log_var, pde_residual, huber_per_pixel, obs_mask):
        """
        [Final Self-Adaptive Engine Version] 
        1. Hybrid SNR: 정보 희귀성(Z-score)과 학습 난이도(Error)를 결합하여 가중치 산출
        2. Cauchy PDE Decay: 핫스팟 주변부에서 더 끈질기게 물리 가이드를 유지
        3. Norm-then-Clamp: 전체 에너지 보존 후 개별 픽셀의 폭주를 방지
        4. Balanced log1p: PDE 학습이 초기 단계에서 소외되지 않도록 그래디언트 통로 확보
        """
        # [방어 1] Uncertainty Clamping (-2.0 ~ 1.0)
        log_var = torch.clamp(log_var, min=-2.0, max=1.0)
        B = target.size(0)
        
        # ── 가중치 계산 루프 (Meta-Guide Loop: No Gradients) ──
        with torch.no_grad():
            sigma_dt = torch.exp(0.5 * log_var).detach()
            target_dt = target.detach()
            pred_dt = pred.detach()
            valid_mask = (obs_mask > 0.5)

            # [통계 1] Per-sample Statistics: 샘플별 관측 기반 통계 산출
            n_valid = valid_mask.sum(dim=(1, 2, 3), keepdim=True)
            has_obs = (n_valid > 0).float()
            
            t_mean = (target_dt * valid_mask).sum(dim=(1, 2, 3), keepdim=True) / (n_valid + 1e-6)
            t_std = torch.sqrt(((target_dt - t_mean)**2 * valid_mask).sum(dim=(1, 2, 3), keepdim=True) / (n_valid + 1e-6)) + 1e-6
            
            # Fallback: 관측치가 없는 샘플은 전역 통계 사용
            f_mean = target_dt.mean(dim=(1, 2, 3), keepdim=True)
            f_std = target_dt.std(dim=(1, 2, 3), keepdim=True) + 1e-6
            final_mean = has_obs * t_mean + (1 - has_obs) * f_mean
            final_std = has_obs * t_std + (1 - has_obs) * f_std
            
            # [결정 1] Hybrid SNR: 정보 희귀성 / (불확실성 + 학습 난이도)
            error_dt = (target_dt - pred_dt).abs()
            z_anom = (target_dt - final_mean) / final_std
            snr = z_anom.abs() / (sigma_dt + 0.1 * error_dt + 1e-6)

            # [수정 1] Dynamic SNR Fallback: nanquantile 실패 시 전체 Max SNR로 회귀
            snr_masked = snr.masked_fill(~valid_mask, float('nan'))
            max_snr = torch.nanquantile(snr_masked.view(B, -1), 0.95, dim=1).view(B, 1, 1, 1)
            
            # 관측치가 없어 NaN이 발생한 경우, 지도의 전체 SNR 중 최댓값을 대안으로 사용
            snr_max_fallback = snr.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
            max_snr = torch.where(torch.isnan(max_snr), snr_max_fallback, max_snr)
            max_snr = max_snr + 1e-6

            # [결정 2] Data Weighting: Norm-then-Clamp 순서 엄수
            raw_data_weight = 1.0 + self.alpha * torch.log1p(snr)
            dw_mean = ((raw_data_weight * valid_mask).sum(dim=(1, 2, 3), keepdim=True) / (n_valid + 1e-6)).clamp(min=0.5, max=2.0)
            
            blend = torch.clamp(n_valid / 10.0, max=1.0)
            norm_dw = (raw_data_weight / (dw_mean + 1e-6))
            data_weight = (blend * norm_dw + (1.0 - blend) * 1.0).clamp(min=0.1, max=5.0)

            # [결정 3] Cauchy-style PDE Decay: 핫스팟 주변부 가이드 강화
            # BUG-8 fix: pde_weight도 obs_mask 기반으로 계산. 미관측 격자에서 SNR 계산이
            # 의미 없으므로 pde_decay를 관측 마스크 내에서만 정규화.
            pde_decay = 1.0 / (1.0 + (snr / max_snr).pow(2))
            pde_weight_raw = 0.1 + 0.9 * pde_decay
            # 관측 마스크 내 평균으로 정규화 (obs_mask=0 영역은 정규화 기준에서 제외)
            pde_weight_denom = (pde_weight_raw * valid_mask).sum(dim=(1, 2, 3), keepdim=True) / (n_valid + 1e-6)
            pde_weight = pde_weight_raw / (pde_weight_denom + 1e-6)

        # ── 최종 손실 합산 (Learning Loop: Gradients Active) ──
        # [방어 2] Precision Safeguard (max=10.0, 최소 sigma ~0.31 ppm)
        precision = torch.exp(-log_var).clamp(max=10.0)
        nll_data_pixel = huber_per_pixel * precision + 0.5 * log_var
        
        # 관측 지점에 대해서만 Data Loss 계산
        weighted_data_loss = (nll_data_pixel * data_weight * obs_mask).sum() / (obs_mask.sum() + 1e-6)
        
        # BUG-8 fix: PDE Loss를 obs_mask에서만 계산하여 Data Loss와 동일한 공정한 비교기준 확보.
        # 미관측 격자(obs_mask=0)에서의 PDE 잔차는 C_hat의 가상 값에 기반한 근거 없는 신호이므로
        # 이를 Loss에 포함하면 Data Loss(~20% 격자)를 PDE Loss(~100% 격자)가 압도하여
        # 옵티마이저가 관측 피팅 대신 PDE 완성을 목적 함수로 학습하게 됨.
        #
        # [P4] tanh 클리핑 + log1p 이중 압축 제거 → 직접 r².clamp(max=100.0).
        # 이전 방식: tanh 포화로 |r|≫10 영역의 그래디언트 소멸 + log1p로 PDE loss 체감.
        # 결과적으로 물리 위반이 클수록 페널티가 약해지는 역방향 특성이 있었음.
        # clamp(max=100.0)은 수치 발산만 방지하고 그래디언트 소멸은 일으키지 않음.
        res_sq_clamped = pde_residual.pow(2).clamp(max=100.0)
        n_obs = obs_mask.sum().clamp(min=1.0)
        pde_loss_val = (res_sq_clamped * pde_weight * obs_mask).sum() / n_obs
        # [F3] lambda_pde 내부 스케일링 제거 — 커리큘럼이 외부에서 단일 제어
        return weighted_data_loss, pde_loss_val, data_weight, pde_weight

# ─────────────────────────────────────────────────────────────────
# L_pde: PDE 잔차 MSE (EMA Running Scale 정규화)
# ─────────────────────────────────────────────────────────────────
class PDEResidualLoss(nn.Module):
    """
    PDE 잔차의 MSE Loss + EMA running scale 정규화.
    L_data와 L_pde 간 ~10⁸배 스케일 차이를 자동 보정.
    """
    def __init__(self, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('running_scale', torch.tensor(1.0))
        self.has_initialized = False

    def forward(self, pde_residual, spatial_mask=None, is_raw_val=False):
        """
        Args:
            pde_residual: (B, 1, H, W) PDE 잔차 R(Ĉ) 또는 이미 계산된 scalar loss (if is_raw_val=True)
            spatial_mask:  (1, 1, H, W) 유효 영역 마스크
            is_raw_val: True이면 pde_residual을 이미 계산된 scalar loss로 간주
        """
        if is_raw_val:
            raw_loss = pde_residual
        else:
            B, _, H, W = pde_residual.shape
            bnd_mask = make_boundary_mask(H, W, device=pde_residual.device)
            combined_mask = bnd_mask if spatial_mask is None else bnd_mask * spatial_mask
            r_sq = pde_residual.pow(2) * combined_mask
            n_valid = combined_mask.sum().clamp(min=1.0) * B
            raw_loss = r_sq.sum() / n_valid

        # [P3] EMA 갱신 제거 → 훈련 첫 배치 스케일로 1회 고정.
        # EMA 방식에서는 과적합이 진행되어 PDE 잔차가 작아지면 running_scale도 감소하여
        # "잔차 감소 → 정규화 약화 → 과적합 심화"의 양성 피드백이 발생했음.
        # 고정 스케일은 훈련 전반에 걸쳐 PDE 신호의 절대적 크기를 일정하게 유지.
        with torch.no_grad():
            if not self.has_initialized:
                self.running_scale.copy_(raw_loss.detach().clamp(min=1e-12))
                self.has_initialized = True

        return raw_loss / self.running_scale.clamp(min=1e-12)


# ─────────────────────────────────────────────────────────────────
# L_reg: Prediction-SR Anchoring + 비음수 + 파라미터 정규화
# ─────────────────────────────────────────────────────────────────
class RegularizationLoss(nn.Module):
    """
    정규화 손실 (설계서 §3.4):

    L_reg = β(t)·mean(Ĉ - S_anthro)²        [Prediction-SR Anchoring]
          + γ·mean(ReLU(-Ĉ))²                [비음수 제약]
          + 0.01·[(α_SR-1)² + max(0,-β_SR)²] [SR 파라미터 정규화]
          + var_reg·mean(log_var²)            [분산 정규화]

    Phase 1에서 β(t)가 크면 NN 출력이 SR 표면에 가깝게 초기화됨.
    Phase 3에서 β(t)→0으로 SR 앵커 해제.

    Args:
        anchor_weight: 전체 L_reg 스케일 (미사용, 호환성)
        var_reg_weight: 분산 정규화 가중치
        gamma_nonneg: 비음수 제약 가중치
    """

class RegularizationLoss(nn.Module):
    def __init__(self, anchor_weight: float = 1.0, var_reg_weight: float = 0.01,
                 gamma_nonneg: float = 0.0):
        super().__init__()
        self.anchor_weight = anchor_weight
        self.var_reg_weight = var_reg_weight
        self.gamma = gamma_nonneg  # 기본 0.0 — 비음수 제약 비활성

    def forward(self, c_hat, s_anthro, beta_t, sr_prior=None, log_var=None, target=None):
        loss = torch.tensor(0.0, device=c_hat.device if c_hat is not None else 'cpu')

        # Prediction-SR anchoring (s_anthro = 농도 prior ppm; SR 수정 후 의미 회복)
        if c_hat is not None and s_anthro is not None:
            if target is not None:
                # [Academic Justification] Anthropogenic prior only applies to positive enhancements.
                # Mask out negative anomaly regions to allow the model to freely learn negative anomalies.
                pos_mask = (target >= 0.0).float()
                diff_sq = (c_hat - s_anthro.detach()).pow(2) * pos_mask
                n_pos = pos_mask.sum().clamp(min=1.0)
                loss = loss + beta_t * (diff_sq.sum() / n_pos)
            else:
                loss = loss + beta_t * (c_hat - s_anthro.detach()).pow(2).mean()

        # (제거됨) 비음수 제약 — anomaly에 부적합

        # SR 파라미터 안정화
        if sr_prior is not None:
            loss = loss + 0.01 * sr_prior.get_sr_anchor_loss()

        # 분산 정규화
        if log_var is not None:
            log_var_clipped = log_var.clamp(-2.0, 1.0)
            loss = loss + self.var_reg_weight * log_var_clipped.pow(2).mean()

        return loss



# ─────────────────────────────────────────────────────────────────
# Curriculum Scheduler
# ─────────────────────────────────────────────────────────────────
class CurriculumScheduler:
    """
    3-Phase Curriculum λ 스케줄러.

    Phase 1 (ep 1~p1): SR Anchoring
      - λ₁=1.0, λ₂=0.0, λ₃=1.0
      - β(t): cosine annealing β₀ → β₀/2

    Phase 2 (ep p1~p2): Climate Injection + PDE 도입
      - λ₁=1.0, λ₂=0→pde_max, λ₃=1.0
      - β(t): β₀/2 → β₀/10

    Phase 3 (ep p2~end): Physics Fine-tune
      - λ₁=1.0, λ₂=pde_max, λ₃=1.0
      - β(t) → 0 (SR anchor 해제)

    설계서 §Curriculum Table 기반.

    Args:
        total_epochs: 전체 에폭 수
        phase1_end: Phase 1 종료 에폭 (비율, 0-1)
        phase2_end: Phase 2 종료 에폭 (비율, 0-1)
        lambda_pde_max: PDE loss 최대 가중치
        beta_0: 초기 SR anchoring 강도
    """

    def __init__(self, total_epochs: int = 100,
                 phase1_end: float = 0.3,
                 phase2_end: float = 0.6,
                 lambda_pde_max: float = 1.0,
                 beta_0: float = 1.0):
        self.total_epochs = total_epochs
        self.p1 = int(total_epochs * phase1_end)
        self.p2 = int(total_epochs * phase2_end)
        self.lambda_pde_max = lambda_pde_max
        self.beta_0 = beta_0

    def get_lambdas(self, epoch: int) -> dict:
        """
        현재 에폭에 따른 λ₁, λ₂, λ₃ 반환.

        Args:
            epoch: 현재 에폭 (0-indexed)
        Returns:
            dict with 'data', 'pde', 'reg' keys
        """
        if epoch < self.p1:
            # Phase 1: SR Anchoring
            # [P5] lambda_pde 0.0 → 0.05: Phase 1에서도 최소 물리 제약을 유지하여
            # 초기 에폭 데이터 암기 고착화를 방지. 0.05는 data loss를 압도하지 않는 보수적 값.
            lambda_data = 1.0
            lambda_pde = 0.05
            lambda_reg = 1.0

        elif epoch < self.p2:
            # Phase 2: Climate Injection + PDE 점진 도입
            t = (epoch - self.p1) / max(self.p2 - self.p1, 1)
            lambda_data = 1.0
            lambda_pde = self.lambda_pde_max * t
            lambda_reg = 1.0

        else:
            # Phase 3: Physics Fine-tune — PDE 최대
            lambda_data = 1.0
            lambda_pde = self.lambda_pde_max
            lambda_reg = 1.0

        return {
            'data': lambda_data,
            'pde': lambda_pde,
            'reg': lambda_reg,
        }

    def get_beta(self, epoch: int) -> float:
        """
        SR anchoring β(t) — cosine annealing.

        설계서: β(t) = β₀ · (1 + cos(πt/T)) / 2
        Phase 1: β₀ → β₀/2
        Phase 2: β₀/2 → β₀/10
        Phase 3: β₀/10 → 0
        """
        import math
        if epoch < self.p1:
            # Phase 1: cosine β₀ → β₀/2
            # β(t) = end + (start - end) * (1 + cos(πt)) / 2
            t = epoch / max(self.p1, 1)
            return self.beta_0 * (0.5 + 0.5 * (1.0 + math.cos(math.pi * t)) / 2.0)

        elif epoch < self.p2:
            # Phase 2: linear β₀/2 → β₀/10
            t = (epoch - self.p1) / max(self.p2 - self.p1, 1)
            return self.beta_0 * (0.5 - 0.4 * t)

        else:
            # Phase 3: linear β₀/10 → 0
            t = (epoch - self.p2) / max(self.total_epochs - self.p2, 1)
            return self.beta_0 * 0.1 * (1.0 - t)

    def get_phase(self, epoch: int) -> int:
        """현재 phase 번호 (1, 2, 3)"""
        if epoch < self.p1:
            return 1
        elif epoch < self.p2:
            return 2
        return 3

    def apply_curriculum(self, model, epoch: int) -> dict:
        """
        Warning #5: Phase별 파라미터 동적 동결(Freeze) 제어.

        Phase 1 (epoch < p1):
          - PDE 관련 파라미터 (pde.log_K, pde.log_lambda) 동결
          - Climate 모듈 (cie.*, scm.*) 동결
          → 이유: 초기 학습에서 모델이 데이터를 먼저 피팅하도록 유도.
                  PDE 제약을 초기부터 가하면 SR 앵커링 전에 물리 파라미터가
                  무의미한 방향으로 학습될 위험이 있음.

        Phase 2+ (epoch >= p1):
          - 모든 파라미터 학습 활성화 (requires_grad = True)
          → 이유: SR 앵커에 의해 C_hat의 초기 스케일이 수렴된 이후
                  물리 파라미터와 기후 모듈을 함께 Fine-tune.

        Args:
            model: ArchAPinn 또는 ArchBConvLSTM 인스턴스
            epoch: 현재 에폭 (0-indexed)

        Returns:
            status: dict — 각 그룹의 현재 freeze 상태 로그
        """
        phase = self.get_phase(epoch)
        freeze_pde = (phase == 1)
        freeze_climate = (phase == 1)

        status = {}

        # PDE 파라미터 (log_K, log_lambda in PDEResidual)
        pde_module = getattr(model, 'pde', None)
        if pde_module is not None:
            for name, p in pde_module.named_parameters():
                p.requires_grad = not freeze_pde
            status['pde'] = 'FROZEN' if freeze_pde else 'ACTIVE'
        else:
            status['pde'] = 'N/A (no pde module)'

        # Climate 모듈: ClimateIndexEncoder (cie)
        cie_module = getattr(model, 'cie', None)
        if cie_module is not None:
            for p in cie_module.parameters():
                p.requires_grad = not freeze_climate
            status['cie'] = 'FROZEN' if freeze_climate else 'ACTIVE'
        else:
            status['cie'] = 'N/A (use_climate=False)'

        # Climate 모듈: SpatialClimateModulator (scm)
        scm_module = getattr(model, 'scm', None)
        if scm_module is not None:
            for p in scm_module.parameters():
                p.requires_grad = not freeze_climate
            status['scm'] = 'FROZEN' if freeze_climate else 'ACTIVE'
        else:
            status['scm'] = 'N/A (use_climate=False)'

        status['phase'] = phase
        status['epoch'] = epoch
        return status


# ─────────────────────────────────────────────────────────────────
# Combined Loss
# ─────────────────────────────────────────────────────────────────
class PIConvLSTMLoss(nn.Module):
    """
    전체 손실 함수:
    L_total = λ₁·L_data + λ₂·L_pde + λ₃·L_reg

    CurriculumScheduler와 함께 사용하여 각 phase에서
    적절한 가중치로 학습.

    Args:
        huber_delta: Huber 전환점
        anchor_weight: RegularizationLoss 내부 스케일
        var_reg_weight: 분산 정규화 가중치
    """

class PIConvLSTMLoss(nn.Module):
    def __init__(self, huber_delta: float = 1.0, anchor_weight: float = 1.0, var_reg_weight: float = 0.01,
                 alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.huber_delta = huber_delta
        # [F3] AdaptivePhysicsLoss에서 lambda_pde 파라미터 제거
        self.adaptive_engine = AdaptivePhysicsLoss(alpha=alpha, beta=beta)
        self.l_pde_norm = PDEResidualLoss()  # 스케일 보정용 유지
        self.l_reg = RegularizationLoss(anchor_weight, var_reg_weight)

    def forward(self, pred, target, log_var, obs_mask,
                pde_residual, lambdas, s_anthro, beta_t,
                sr_prior=None, spatial_mask=None, n_soundings=None,
                sr_gate=None):
        
        log_var_clipped = log_var.clamp(-2.0, 1.0)
        
        # 1. Huber 기초값 계산 (Per-pixel)
        target_clean = torch.where(torch.isnan(target), torch.zeros_like(target), target)
        e = pred - target_clean
        abs_e = e.abs()
        huber_pixel = torch.where(abs_e < self.huber_delta, 0.5 * e.pow(2), 
                                  self.huber_delta * (abs_e - 0.5 * self.huber_delta))

        # 2. Adaptive Physics Loss 엔진 가동
        loss_data, loss_pde_raw, d_w, p_w = self.adaptive_engine(
            pred, target_clean, log_var_clipped, pde_residual, huber_pixel, obs_mask
        )

        # 3. PDE 스케일 보정 적용
        loss_pde = self.l_pde_norm(loss_pde_raw, spatial_mask, is_raw_val=True)
        
        # 4. Regularization
        loss_reg = self.l_reg(pred, s_anthro, beta_t, sr_prior, log_var_clipped, target_clean)

        # [F1] Gate Regularization: 엔트로피 정규화 (0.5로 당김).
        # 기존 L_gate = (1-G).mean()은 게이트를 항상 1(완전 개방) 방향으로 강제하여
        # 2023년 음의 레짐에서도 SR prior 신호가 차단되지 못하는 구조적 문제를 유발했음.
        # (G-0.5)^2는 게이트가 0.5에서 자유롭게 이탈할 수 있도록 허용하면서,
        # 완전 개방/완전 차단 양 극단으로의 조기 고착만 방지함.
        if sr_gate is not None:
            loss_gate = (sr_gate - 0.5).pow(2).mean()
        else:
            loss_gate = torch.tensor(0.0, device=pred.device)

        # 최종 가중 합산 (lambda_reg를 Gate 정규화에도 공유하거나 별도 가중치 부여)
        # 여기서는 lambda_reg의 10% 수준을 게이트 정규화에 할당 (비선형성 우선순위)
        total = (lambdas['data'] * loss_data
                 + lambdas['pde'] * loss_pde
                 + lambdas['reg'] * loss_reg
                 + lambdas['reg'] * 0.1 * loss_gate)

        # [수치 방어 4] 발산 발생 시 해당 배치 무시
        if not torch.isfinite(total):
            print(f"  [Warning] Non-finite loss detected. Skipping batch...")
            total = torch.tensor(0.0, device=total.device, requires_grad=True)

        loss_dict = {
            'total': total.item(),
            'data': loss_data.item(),
            'pde': loss_pde.item(),
            'reg': loss_reg.item(),
            'gate': loss_gate.item(),
            'lambda_data': lambdas['data'],
            'lambda_pde': lambdas['pde'],
            'lambda_reg': lambdas['reg'],
            # E2: PCGrad용 텐서 원본 유지
            'data_tensor': loss_data,
            'pde_tensor': loss_pde,
            'reg_tensor': loss_reg,
            'gate_tensor': loss_gate,
        }

        return total, loss_dict

