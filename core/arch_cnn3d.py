"""
arch_cnn3d.py — [Version 4] 3D CNN PINN (시공간 볼륨 컨볼루션)
==========================================================
과거 시간축을 3D 볼륨 차원(D=T)으로 인입하여 시공간 확산을 순수 Conv3D 커널
가중치로 근사하는 고속 물리 대리 모델. 시간 차원을 conv로 점진 축약 후 마지막에
T=1로 풀링하여 (B,hidden,H,W) 공간 특징으로 환원.
입력 재배열: (B,T,C,H,W) → (B,C,T,H,W).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base_pinn import BaseSpatioTemporalPINN


class CNN3DPINN(BaseSpatioTemporalPINN):
    def __init__(self, in_channels: int, time_steps: int, hidden: int = 64, **kw):
        super().__init__(in_channels=in_channels, hidden=hidden, **kw)
        self.time_steps = time_steps
        # 좌표 2채널 결합 → in_channels+2
        cin = in_channels + 2
        # Conv3D: 시간(D) 커널 3, 공간 3×3. 공간 padding 유지, 시간은 점진 축약.
        self.conv3d = nn.Sequential(
            nn.Conv3d(cin, hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.GELU(),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.Conv3d(hidden, hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.GELU(),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.Conv3d(hidden, hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.GELU(),
        )

    def encode(self, x_seq, lat_norm, lon_norm, obs_mask=None):
        B, T, C, H, W = x_seq.shape
        # 좌표를 시간축으로 broadcast 결합
        coord = torch.stack([lat_norm, lon_norm], dim=2)          # (B,1,2,H,W)? -> 정리
        lat_t = lat_norm.unsqueeze(1).expand(B, T, 1, H, W)
        lon_t = lon_norm.unsqueeze(1).expand(B, T, 1, H, W)
        x = torch.cat([x_seq, lat_t, lon_t], dim=2)               # (B,T,C+2,H,W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()                 # (B,C+2,T,H,W)
        feat = self.conv3d(x)                                     # (B,hidden,T,H,W)
        return feat.mean(dim=2)                                   # 시간 평균 풀링 → (B,hidden,H,W)
