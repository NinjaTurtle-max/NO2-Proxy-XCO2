"""
arch_b_convlstm.py
====================
Architecture B: ConvLSTM 중심 PI-ConvLSTM.

ConvLSTM이 시공간 예측을 주도하고, PDE는 정규화 제약으로 작용.
시계열 윈도우(T_window=7~14일)의 공간 피처를 ConvLSTM으로 처리.

설계 특징:
  - Conv2D encoder → ConvLSTM core → Conv2D decoder
  - PDE 잔차는 finite difference로 계산 (Conv2D 커널 사용)
  - 비정상 PDE: t-1 시점 hidden을 디코딩하여 ∂C/∂t = (C_t - C_{t-1})/dt 계산
  - PDE loss는 λ₂를 보수적으로 (0→0.3) 올려 과적합 방지
  - Climate conditioning: z_clim을 ConvLSTM hidden에 결합

Ablation 실험:
  - B1: ConvLSTM only (PDE/SR 없음)
  - B2: PI-ConvLSTM (SR + PDE 정규화)
  - B3: PI-ConvLSTM + α (SR + PDE + climate)

Reference:
  - Shi et al. (2015) — Convolutional LSTM for precipitation nowcasting
  - Rao et al. (2023) — Physics-informed ConvLSTM (PhyDNet)
  - LeVeque (2007) — Backward Euler 시간 차분
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sr_prior import SRPrior
from .pde_residual import PDEResidual, FourierFeatures
from .climate_encoder import ClimateIndexEncoder
from .spatial_modulator import SpatialClimateModulator
from .extrapolation_anchor import ContinuousTimeAnchor


# ─────────────────────────────────────────────────────────────────
# ConvLSTM Cell
# ─────────────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    """
    단일 ConvLSTM 셀.

    i = σ(W_xi * x + W_hi * h + b_i)
    f = σ(W_xf * x + W_hf * h + b_f)
    g = tanh(W_xg * x + W_hg * h + b_g)
    o = σ(W_xo * x + W_ho * h + b_o)
    c' = f ⊙ c + i ⊙ g
    h' = o ⊙ tanh(c')

    Args:
        input_dim: 입력 채널 수
        hidden_dim: hidden state 채널 수
        kernel_size: convolution 커널 크기
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # 입력 + hidden → 4 gates (i, f, g, o) 를 한 번에 계산
        self.gates = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True
        )

        # forget gate bias 초기화 (forget gate를 열어서 gradient flow 보장)
        nn.init.constant_(self.gates.bias[hidden_dim:2*hidden_dim], 1.0)

    def forward(self, x, state):
        """
        Args:
            x:     (B, input_dim, H, W)
            state: tuple of (h, c), each (B, hidden_dim, H, W)
        Returns:
            h_next, c_next: each (B, hidden_dim, H, W)
        """
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        """Zero-initialized hidden state."""
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


# ─────────────────────────────────────────────────────────────────
# Architecture B: PI-ConvLSTM
# ─────────────────────────────────────────────────────────────────
class ArchBConvLSTM(nn.Module):
    """
    ConvLSTM 중심 PI-ConvLSTM 아키텍처.

    구조:
      1) Spatial Encoder: Conv2D layers (각 시점의 공간 피처 추출)
      2) ConvLSTM Core: 시공간 통합 (여러 시점을 순차 처리)
      3) Climate Conditioning: CIE → z_clim을 decoder에 concat
      4) Decoder: Conv2D → Ĉ(x,t), log σ²(x,t)
      5) PDE Regularizer: finite difference로 R(Ĉ) 계산

    Args:
        in_channels: 시점별 입력 채널 수
        encoder_channels: spatial encoder 출력 채널
        lstm_hidden: ConvLSTM hidden 채널
        num_lstm_layers: ConvLSTM 레이어 수
        sr_expr_id: SR 수식 ID
        use_sr: SR prior 사용 여부
        use_pde: PDE 정규화 사용 여부
        use_climate: +α 모듈 사용 여부
        climate_dim: CIE hidden 차원
    """

    def __init__(self,
                 in_channels: int = 6,
                 encoder_channels: int = 32,
                 lstm_hidden: int = 64,
                 num_lstm_layers: int = 2,
                 sr_expr_id: str = 'eq10',
                 use_sr: bool = True,
                 use_pde: bool = True,
                 use_climate: bool = False,
                 climate_dim: int = 32,
                 dropout: float = 0.0,
                 # [P2] 디코더 Fourier skip-connection 기본값 False:
                 # lstm_out + t_emb + fourier_coords 결합이 시공간 암기를 촉진하므로 비활성화.
                 use_decoder_fourier: bool = False,
                 # [P3] Fourier 입력 드롭아웃: 훈련 시 Fourier 채널을 무작위 제로잉하여
                 # 모델이 고정된 좌표 패턴(공간 lookup table)을 암기하지 못하도록 방지.
                 # Dropout2d는 채널 단위로 제로잉하므로 공간 구조 암기를 더 효과적으로 억제.
                 fourier_input_dropout: float = 0.2):
        super().__init__()

        self.use_sr = use_sr
        self.use_pde = use_pde
        self.use_climate = use_climate
        self.num_lstm_layers = num_lstm_layers
        self.use_decoder_fourier = use_decoder_fourier
        self.fourier_input_dropout = fourier_input_dropout

        # 1) Spatial Encoder: (B, in_channels, H, W) → (B, encoder_channels, H, W)
        # SR output, 좌표, seasonal을 포함하면 채널 증가
        # E3: Fourier Feature 좌표 임베딩 (128 채널 추가)
        # [P2] sigma 10.0 → 1.0: 고주파 사영이 공간 좌표를 XCO₂에 직접 암기하는
        # shortcut 경로를 억제. σ=1.0은 ~100km 이상의 광역 공간 구조만 표현.
        self.coords_fourier = FourierFeatures(in_ch=2, mapping_size=64, sigma=1.0)
        
        enc_in = in_channels + 2 + 2 + 128  # + cos/sin_doy + lat/lon_norm + fourier_coords
        if use_sr:
            enc_in += 1  # + S_anthro

        # [F4] BatchNorm2d → InstanceNorm2d(affine=True):
        # BatchNorm은 훈련 기간(2020-2022) 통계를 running_mean/var로 고정하여
        # 2023 레짐 전환 시 피처 분포 불일치를 증폭시킴.
        # InstanceNorm은 샘플별 독립 정규화로 running stats를 일체 사용하지 않으므로
        # 연도간 분포 변화에 구조적으로 강건함.
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(enc_in, encoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.InstanceNorm2d(encoder_channels, affine=True),
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.InstanceNorm2d(encoder_channels, affine=True),
        )

        # 2) ConvLSTM Core: 여러 레이어 스택
        self.lstm_cells = nn.ModuleList()
        for i in range(num_lstm_layers):
            cell_input = encoder_channels if i == 0 else lstm_hidden
            self.lstm_cells.append(
                ConvLSTMCell(cell_input, lstm_hidden, kernel_size=3)
            )

        # Decoder
        # BUG-4 fix: ContinuousTimeAnchor(t_emb_map)는 use_climate 여부와 무관하게
        # 항상 decoder에 concat된다. (use_climate=True일 때는 clim_emb_map = t_emb_map + z_clim_map
        # 으로 element-wise 합산하므로 채널 수는 climate_dim으로 동일.)
        # 따라서 dec_in_channels = lstm_hidden + climate_dim (+ 128 if fourier) 으로 고정.
        self.time_anchor = ContinuousTimeAnchor(hidden_dim=climate_dim)
        dec_in_channels = lstm_hidden + climate_dim  # t_emb_map 항상 포함
        if use_decoder_fourier:
            dec_in_channels += 128

        # Decoder: dropout > 0 이면 Spatial Dropout 삽입 (과적합 방지)
        dec_layers = [
            nn.Conv2d(dec_in_channels, lstm_hidden, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if dropout > 0:
            dec_layers.append(nn.Dropout2d(p=dropout))
        dec_layers.extend([
            nn.Conv2d(lstm_hidden, lstm_hidden // 2, kernel_size=1),
            nn.GELU(),
        ])
        if dropout > 0:
            dec_layers.append(nn.Dropout2d(p=dropout))
        self.decoder = nn.Sequential(*dec_layers)

        self.head_mean = nn.Conv2d(lstm_hidden // 2, 1, kernel_size=1)
        self.head_logvar = nn.Conv2d(lstm_hidden // 2, 1, kernel_size=1)
        nn.init.constant_(self.head_logvar.bias, -2.0)

        # 4) SR Prior 및 Gating Layer (Issue #4: Uncertainty Fusion)
        if use_sr:
            self.sr_prior = SRPrior(expr_id=sr_expr_id, use_odiac=True)
            # SR 정보를 얼마나 수용할지 결정하는 픽셀 단위 게이트
            # Conv2D(입력 피처들 -> 1채널 게이트)
            self.sr_gate = nn.Sequential(
                nn.Conv2d(enc_in - 1, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.Sigmoid()
            )
            # 초기에는 물리를 신뢰하도록 Bias 조정 (G ≈ 0.88)
            nn.init.constant_(self.sr_gate[-2].bias, 2.0)

        # 5) PDE Residual (선택적)
        if use_pde:
            self.pde = PDEResidual()

        # 6) Climate modules (선택적)
        if use_climate:
            self.cie = ClimateIndexEncoder(
                input_dim=3, hidden_dim=climate_dim
            )
            self.scm = SpatialClimateModulator(
                climate_dim=climate_dim, spatial_features=3
            )

    def forward(self, x_seq, lat_norm, lon_norm,
                cos_doy_seq, sin_doy_seq,
                no2_seq, odiac_seq, u10_seq, v10_seq,
                blh_seq=None, wind_speed_seq=None,
                climate_seq=None, ndvi=None, lat_raw=None,
                t_norm=None, obs_mask=None):
        """
        ... (생략된 docstring) ...
        Returns:
            c_hat:    (B, 1, H, W) 마지막 시점 예측
            log_var:  (B, 1, H, W) 마지막 시점 log σ²
            pde_res:  (B, 1, H, W) PDE 잔차
            s_anthro: (B, 1, H, W) SR source term
            gamma_clim: (B, 1, H, W) 기후 변조 계수
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        lat_for_sr = lat_raw if lat_raw is not None else lat_norm
        states = []
        for cell in self.lstm_cells:
            states.append(cell.init_hidden(B, H, W, device))

        lstm_out_prev = None
        sr_gate_val = None  # BUG-5 fix: 올바른 변수명 사용
        s_anthro_t = None

        # 시간 루프: 각 시점을 순차 처리
        for t in range(T):
            x_t = x_seq[:, t]
            cos_doy_t = cos_doy_seq[:, t]
            sin_doy_t = sin_doy_seq[:, t]

            cos_spatial = cos_doy_t.expand(B, 1, H, W)
            sin_spatial = sin_doy_t.expand(B, 1, H, W)
            lat_exp = lat_norm.expand(B, 1, H, W)
            lon_exp = lon_norm.expand(B, 1, H, W)

            # E3: Fourier Feature 좌표 임베딩
            lat_grid, lon_grid = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing='ij'
            )
            xy = torch.stack([lon_grid, lat_grid], dim=0).unsqueeze(0).expand(B, 2, H, W)
            fourier_coords = self.coords_fourier(xy)

            # [P3] Fourier 채널 드롭아웃: 훈련 시 위경도 좌표 패턴 암기 방지.
            # F.dropout2d는 (B, C, H, W) 채널 단위로 전체 공간을 제로잉하여,
            # 단순히 픽셀 단위로 노이즈를 추가하는 것보다 공간 lookup table 암기를 더 강하게 차단.
            if self.fourier_input_dropout > 0:
                fourier_coords = F.dropout2d(fourier_coords, p=self.fourier_input_dropout,
                                             training=self.training)

            # 기본 공간 피처 결합 (+ Fourier Features)
            base_feats = torch.cat([x_t, cos_spatial, sin_spatial, lat_exp, lon_exp, fourier_coords], dim=1)

            if self.use_sr:
                no2_t_raw = no2_seq[:, t]              # ← RAW NO2 (핵심 수정, 병목①)
                odiac_t   = odiac_seq[:, t]           # ppm/s 변환 완료된 raw
                blh_t     = blh_seq[:, t] if blh_seq is not None else None
                ws_t      = wind_speed_seq[:, t] if wind_speed_seq is not None else None
                # 농도 prior (ppm) — 앵커/리턴/인코더 피처
                s_anthro_t = self.sr_prior(
                    no2_t_raw, odiac_t, lat_for_sr, cos_doy_t, sin_doy_t, blh_t, ws_t
                )
                # PDE source용 배출 플럭스 (ppm/s) — 마지막 t 값 보관
                s_flux_t = self.sr_prior.forward_flux(odiac_t)   # ← 추가 (병목②)
                g = self.sr_gate(base_feats)
                sr_gate_val = g  # BUG-5 fix: 루프 내 마지막 시점의 게이트 값을 보관
                s_anthro_gated = g * s_anthro_t
                enc_in = torch.cat([base_feats, s_anthro_gated], dim=1)
            else:
                enc_in = base_feats

            # Spatial encoding
            spatial_feat = self.spatial_encoder(enc_in)

            # ConvLSTM forward
            for i, cell in enumerate(self.lstm_cells):
                inp = spatial_feat if i == 0 else states[i-1][0]
                states[i] = cell(inp, states[i])

            if t == T - 2:
                lstm_out_prev = states[-1][0]

        # 마지막 시점의 hidden state로 디코딩
        lstm_out = states[-1][0]

        # ContinuousTimeAnchor 적용 (계절/추세 분리 구조)
        cos_last = cos_doy_seq[:, -1].reshape(B, 1)
        sin_last = sin_doy_seq[:, -1].reshape(B, 1)
        if t_norm is None:
            t_norm = torch.zeros(B, 1, 1, 1, device=device)
        t_emb = self.time_anchor(cos_last, sin_last, t_norm)
        t_emb_map = t_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)

        # [F6] t-1 시점용 별도 t_emb_prev 계산.
        # 기존: c_hat(t)와 c_hat_prev(t-1) 모두 동일한 t_emb_map을 사용하여
        # dC/dt = (C_hat - C_hat_prev)/dt에서 시간 앵커 기여가 정확히 상쇄되었음.
        # 수정: t-1 시점의 cos/sin_doy와 t_norm_prev를 별도 계산하여 주입함으로써
        # ∂C/∂t가 LSTM hidden 차이 + 시간 앵커 차이를 모두 반영하도록 복원.
        if T >= 2:
            cos_prev = cos_doy_seq[:, -2].reshape(B, 1)
            sin_prev = sin_doy_seq[:, -2].reshape(B, 1)
            # t_norm_prev: 타겟 기준 1일 이전 (1/(5×365.25) ≈ 5.48e-4)
            t_norm_prev = t_norm - (1.0 / (5.0 * 365.25))
            t_emb_prev = self.time_anchor(cos_prev, sin_prev, t_norm_prev)
            t_emb_map_prev = t_emb_prev.view(B, -1, 1, 1).expand(-1, -1, H, W)
        else:
            t_emb_map_prev = t_emb_map  # T=1 fallback (quasi-steady)
        
        # FourierFeatures 디코더 스킵 커넥션 결합용 생성 (고주파 공간 매핑)
        lat_grid_dec, lon_grid_dec = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        xy_dec = torch.stack([lon_grid_dec, lat_grid_dec], dim=0).unsqueeze(0).expand(B, 2, H, W)
        fourier_coords_dec = self.coords_fourier(xy_dec)

        # 기후 변조 변수 계산 및 결합 분기
        # BUG-4 fix: use_climate=False 여부에 관계없이 t_emb_map을 항상 decoder에 연결.
        # ContinuousTimeAnchor는 연도 추세를 학습하므로, Baseline 실험에서도 반드시 필요.
        # 이를 누락하면 2023/2024 Test set 외삽 시 연도 추세 정보가 전혀 없어 R² < 0 을 유발.
        gamma_clim = torch.zeros(B, 1, H, W, device=device)
        if self.use_climate and climate_seq is not None:
            z_clim = self.cie(climate_seq)
            gamma_clim = self.scm(z_clim, lat_norm, lon_norm, ndvi)
            z_clim_map = z_clim.view(B, -1, 1, 1).expand(-1, -1, H, W)
            
            # 시간 추세와 기후 변동성을 정합하게 단일 기후 차원으로 병합 (t_emb_map + z_clim_map)
            clim_emb_map = t_emb_map + z_clim_map

            if self.use_decoder_fourier:
                dec_in = torch.cat([lstm_out, clim_emb_map, fourier_coords_dec], dim=1)
            else:
                dec_in = torch.cat([lstm_out, clim_emb_map], dim=1)
        else:
            # BUG-4 fix: use_climate=False여도 t_emb_map(연도 추세)은 항상 decoder에 포함
            if self.use_decoder_fourier:
                dec_in = torch.cat([lstm_out, t_emb_map, fourier_coords_dec], dim=1)
            else:
                dec_in = torch.cat([lstm_out, t_emb_map], dim=1)

        # Decoder (시점 t)
        h = self.decoder(dec_in)
        c_hat = self.head_mean(h)
        log_var = self.head_logvar(h)

        # SR source term (마지막 시점)
        if self.use_sr:
            s_anthro = s_anthro_t  # 마지막 시점
        else:
            s_anthro = torch.zeros(B, 1, H, W, device=device)

        # ── t-1 시점 디코딩 (PDE 시간 미분항용) ──
        # lstm_out_prev (t==T-2에서 저장)를 동일 decoder로 디코딩하여 c_hat_prev 도출.
        # dropout 충돌 방지: c_hat_prev 디코딩 시에는 decoder를 eval 모드로 강제하여
        #   c_hat과 c_hat_prev가 서로 다른 dropout mask를 통과하는 것을 방지한다.
        #   (서로 다른 mask면 dC_dt에 dropout 노이즈가 섞여 물리적 시간 변화가 오염됨)
        # gradient는 정상적으로 흐른다 — eval 모드는 dropout/BN 통계만 끄고
        #   autograd 그래프 추적은 그대로 유지되므로 PDE loss가 ConvLSTM까지 역전파됨.
        # T < 2 이면 lstm_out_prev가 None → c_hat_prev=None → PDE가 준정상으로 fallback.
        c_hat_prev = None
        if self.use_pde and lstm_out_prev is not None:
            if self.use_climate:
                # [F6] t-1 시점 기후 임베딩: t_emb_map_prev + z_clim_map
                clim_emb_map_prev = t_emb_map_prev + z_clim_map
                if self.use_decoder_fourier:
                    dec_in_prev = torch.cat([lstm_out_prev, clim_emb_map_prev, fourier_coords_dec], dim=1)
                else:
                    dec_in_prev = torch.cat([lstm_out_prev, clim_emb_map_prev], dim=1)
            else:
                # [F6] t-1 시점 별도 t_emb_map_prev 사용 (t_emb_map 재사용 금지)
                if self.use_decoder_fourier:
                    dec_in_prev = torch.cat([lstm_out_prev, t_emb_map_prev, fourier_coords_dec], dim=1)
                else:
                    dec_in_prev = torch.cat([lstm_out_prev, t_emb_map_prev], dim=1)

            # decoder의 dropout을 일시적으로 비활성화 (c_hat과 동일 경로 보장).
            # no_grad는 쓰지 않는다 — c_hat_prev에도 gradient가 흘러야 PDE 시간항이
            # ConvLSTM의 t-1 hidden까지 학습 신호를 전달할 수 있다.
            _was_training = self.decoder.training
            self.decoder.eval()
            h_prev = self.decoder(dec_in_prev)
            c_hat_prev = self.head_mean(h_prev)
            if _was_training:
                self.decoder.train()

        # PDE Residual (마지막 시점, 비정상 — ∂C/∂t 포함)
        if self.use_pde:
            u10_last = u10_seq[:, -1]
            v10_last = v10_seq[:, -1]
            s_flux_last = s_flux_t if self.use_sr else None   # 농도→플럭스 교체
            
            no2_last = no2_seq[:, -1] if no2_seq is not None else None
            blh_last = blh_seq[:, -1] if blh_seq is not None else None
            
            blh_prev = None
            if blh_seq is not None and blh_seq.shape[1] >= 2:
                blh_prev = blh_seq[:, -2]
            
            # BUG-6 fix: obs_mask를 PDE에 전달하여 관측된 격자에서만 물리 잔차를 계산.
            # 미관측 격자(obs_mask=0)에서의 PDE 잔차는 근거 없는 가상 값이므로 배제.
            pde_mask = obs_mask  # (B, 1, H, W) 또는 None
            pde_res, _ = self.pde(c_hat, u10_last, v10_last, s_flux_last, gamma_clim,
                                  C_hat_prev=c_hat_prev, dt=86400.0,
                                  no2=no2_last, blh=blh_last, ndvi=ndvi,
                                  blh_prev=blh_prev, mask=pde_mask)
        else:
            pde_res = torch.zeros(B, 1, H, W, device=device)

        # 최종 게이트 값 결정 (Ablation 모니터링용)
        # BUG-5 fix: sr_gate_val에 루프 내 마지막 게이트 값이 보관되어 있음.
        # 기존 코드는 sr_gate가 None인 채로 유지되어 항상 torch.ones를 반환했음.
        if not self.use_sr or sr_gate_val is None:
            sr_gate = torch.ones(B, 1, H, W, device=device)
        else:
            sr_gate = sr_gate_val

        return c_hat, log_var, pde_res, s_anthro, gamma_clim, sr_gate