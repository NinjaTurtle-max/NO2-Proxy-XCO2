"""
arch_cnn_lstm_v4.py — [Version 3-v4] Dilated CNN + GroupNorm-ConvGRU PINN
==========================================================================
v3(arch_cnn_lstm) 대비 3개 상향 (docs/v4_dual_architecture_spec_20260612.html §1-A):

  1. Dilated CNN 인코더 — 블록 2·3에 dilation 2·4 적용.
     수용영역 13격자 → 31격자 (베리오그램 e-folding range 19.35격자의 1.6배,
     파라미터 증가 0). 53km(2격자) range 주장은 실측(537km)과 달라 기각됨.
  2. GroupNorm-ConvGRU + recurrent zoneout(p=0.1) — 경향(lag) 채널의 고주파
     잡음에 은닉상태가 과적합되는 것을 억제. 배치 4라 BatchNorm 불가 →
     GroupNorm(8)로 프레임별 통계 독립 보장.
  3. 경향 채널 1×1 게이트 — lag 채널(말미 n_lag개)에 학습 가능한 채널별
     시그모이드 게이트(초기 0.5). 모델이 잡음질 경향 채널을 스스로 감쇠.

forward 계약·헤드·SR source·PDE는 BaseSpatioTemporalPINN 공유 (encode만 구현).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base_pinn import BaseSpatioTemporalPINN


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm(8) — 채널이 8로 안 나뉘면 LayerNorm 등가(1그룹)."""
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class GNConvGRUCell(nn.Module):
    """GroupNorm + zoneout ConvGRU 셀.

    게이트 전활성과 후보 상태 전활성에 GroupNorm을 적용해 경향 채널 유입 시
    은닉상태 분포 폭주를 막고, 학습 중 zoneout(z 게이트 일부를 이전 상태
    유지로 강제)으로 시간축 과적합을 억제한다.
    """

    def __init__(self, in_dim: int, hid: int, k: int = 3, zoneout: float = 0.1):
        super().__init__()
        self.conv_zr = nn.Conv2d(in_dim + hid, 2 * hid, k, padding=k // 2)
        self.conv_h = nn.Conv2d(in_dim + hid, hid, k, padding=k // 2)
        self.norm_zr = _gn(2 * hid)
        self.norm_h = _gn(hid)
        self.hid = hid
        self.zoneout = zoneout

    def forward(self, x, h):
        zr = torch.sigmoid(self.norm_zr(self.conv_zr(torch.cat([x, h], dim=1))))
        z, r = zr.chunk(2, dim=1)
        hh = torch.tanh(self.norm_h(self.conv_h(torch.cat([x, r * h], dim=1))))
        h_new = (1 - z) * h + z * hh
        if self.zoneout > 0.0:
            if self.training:
                keep = torch.bernoulli(
                    torch.full_like(h, self.zoneout))      # 1=이전 상태 유지
                h_new = keep * h + (1 - keep) * h_new
            else:
                h_new = self.zoneout * h + (1 - self.zoneout) * h_new
        return h_new


class CNNLSTMv4PINN(BaseSpatioTemporalPINN):
    def __init__(self, in_channels: int, time_steps: int, hidden: int = 64,
                 lag_channels: int = 0, zoneout: float = 0.1, **kw):
        super().__init__(in_channels=in_channels, hidden=hidden, **kw)
        self.time_steps = time_steps
        self.lag_channels = lag_channels
        # 경향 채널 게이트 (초기 0 → sigmoid 0.5 = 반개방 시작)
        if lag_channels > 0:
            self.lag_gate = nn.Parameter(torch.zeros(lag_channels))
        # Dilated 2D CNN 공간 인코더 (RF 7→31격자, 프레임별 가중치 공유)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden, 3, padding=1), nn.GELU(),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4), nn.GELU(),
        )
        self.gru = GNConvGRUCell(hidden, hidden, zoneout=zoneout)

    def encode(self, x_seq, lat_norm, lon_norm, obs_mask=None):
        B, T, C, H, W = x_seq.shape
        if self.lag_channels > 0:
            g = torch.sigmoid(self.lag_gate).view(1, 1, -1, 1, 1)
            x_seq = torch.cat(
                [x_seq[:, :, :C - self.lag_channels],
                 x_seq[:, :, C - self.lag_channels:] * g], dim=2)
        h = x_seq.new_zeros(B, self.hidden, H, W)
        for t in range(T):
            xt = torch.cat([x_seq[:, t], lat_norm, lon_norm], dim=1)
            feat = self.cnn(xt)          # 공간 먼저 (dilated CNN)
            h = self.gru(feat, h)        # 시간 순차 (GN-GRU + zoneout)
        return h
