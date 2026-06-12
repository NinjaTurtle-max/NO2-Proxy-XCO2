"""
arch_cnn_lstm.py — [Version 3] CNN + LSTM PINN
===============================================
순수 2D CNN 공간 인코더로 격자 간 공간 연속성을 먼저 포착(프레임별 가중치 공유)한 뒤,
평탄화 없이 ConvGRU 서순으로 시간축을 전달하여 물리 잔차를 추적하는 순차 아키텍처.
V2(ConvLSTM 코어+어텐션)와 달리, 공간(CNN)과 시간(GRU)을 명시적으로 분리한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base_pinn import BaseSpatioTemporalPINN


class ConvGRUCell(nn.Module):
    """ConvGRU 셀 (평탄화 없이 시공간 유지)."""

    def __init__(self, in_dim: int, hid: int, k: int = 3):
        super().__init__()
        self.conv_zr = nn.Conv2d(in_dim + hid, 2 * hid, k, padding=k // 2)
        self.conv_h = nn.Conv2d(in_dim + hid, hid, k, padding=k // 2)
        self.hid = hid

    def forward(self, x, h):
        zr = torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1)))
        z, r = zr.chunk(2, dim=1)
        hh = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * hh


class CNNLSTMPINN(BaseSpatioTemporalPINN):
    def __init__(self, in_channels: int, time_steps: int, hidden: int = 64, **kw):
        super().__init__(in_channels=in_channels, hidden=hidden, **kw)
        self.time_steps = time_steps
        # 2D CNN 공간 인코더 (프레임별 공유, 3-블록 residual-lite)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden, 3, padding=1), nn.GELU(),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.gru = ConvGRUCell(hidden, hidden)

    def encode(self, x_seq, lat_norm, lon_norm, obs_mask=None):
        B, T, C, H, W = x_seq.shape
        h = x_seq.new_zeros(B, self.hidden, H, W)
        for t in range(T):
            xt = torch.cat([x_seq[:, t], lat_norm, lon_norm], dim=1)
            feat = self.cnn(xt)          # 공간 먼저 (CNN)
            h = self.gru(feat, h)        # 시간 순차 (GRU)
        return h
