"""
arch_mlp.py — [Version 6] MLP (Point-wise Baseline PINN)
========================================================
공간 자기상관을 인코더에서 배제한 대조군. 격자별 1D 벡터를 1×1 Conv로 독립 연산하여
공간 혼합을 일절 하지 않는다. 공간 연속성(53km/등방성 Laplacian)은 오직 손실 단의
PDE 잔차에서만 부과된다 → "물리 제약의 순수 기여도"를 측정하는 baseline.

시간축은 채널로 평탄화(B,T,C,H,W → B,T·C,H,W)하여 point-wise MLP에 입력.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base_pinn import BaseSpatioTemporalPINN


class MLPPINN(BaseSpatioTemporalPINN):
    """Point-wise MLP 인코더 (1×1 Conv only). 공간 결합 없음."""

    def __init__(self, in_channels: int, time_steps: int, hidden: int = 64,
                 depth: int = 4, **kw):
        super().__init__(in_channels=in_channels, hidden=hidden, **kw)
        self.time_steps = time_steps
        # 입력: 시간축 평탄화(T·C) + 좌표 2채널
        din = in_channels * time_steps + 2
        layers = [nn.Conv2d(din, hidden, kernel_size=1), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(hidden, hidden, kernel_size=1), nn.GELU()]
        self.mlp = nn.Sequential(*layers)

    def encode(self, x_seq, lat_norm, lon_norm, obs_mask=None):
        B, T, C, H, W = x_seq.shape
        x = x_seq.reshape(B, T * C, H, W)                 # 시간축 → 채널 평탄화
        x = torch.cat([x, lat_norm, lon_norm], dim=1)     # 좌표 결합
        return self.mlp(x)                                # (B,hidden,H,W) — 공간혼합 없음
