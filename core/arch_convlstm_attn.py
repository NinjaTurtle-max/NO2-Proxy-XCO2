"""
arch_convlstm_attn.py — [Version 2] ConvLSTM + Spatial Attention PINN
=====================================================================
시간 연속성을 보존하는 ConvLSTM 코어 위에, NO₂ 핫스팟 신호와 연동되어 국지 물리
확산을 증폭시키는 Spatial Attention Map을 결합한다. 마지막 시점 hidden에 어텐션
가중을 곱해 핫스팟 영역의 특징을 강조한 뒤 base 헤드로 전달.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base_pinn import BaseSpatioTemporalPINN


class ConvLSTMCell(nn.Module):
    """표준 ConvLSTM 셀 (입력/망각/출력/셀 게이트 동시 conv)."""

    def __init__(self, in_dim: int, hid: int, k: int = 3):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(in_dim + hid, 4 * hid, k, padding=k // 2)

    def forward(self, x, h, c):
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = z.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class SpatialAttention(nn.Module):
    """NO₂ 핫스팟 + hidden 기반 픽셀별 어텐션 맵 (0~1) 생성·곱셈 증폭."""

    def __init__(self, hid: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(hid, hid // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(hid // 2, 1, 1), nn.Sigmoid(),
        )

    def forward(self, h):
        return h * (1.0 + self.att(h))   # 잔차형 증폭 (어텐션=0이어도 정보보존)


class ConvLSTMAttnPINN(BaseSpatioTemporalPINN):
    def __init__(self, in_channels: int, time_steps: int, hidden: int = 64,
                 num_layers: int = 2, **kw):
        super().__init__(in_channels=in_channels, hidden=hidden, **kw)
        self.time_steps = time_steps
        self.num_layers = num_layers
        # 프레임별 공간 인코더 (좌표 2채널 결합)
        self.frame_enc = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden, 3, padding=1), nn.GELU(),
            nn.InstanceNorm2d(hidden, affine=True),
        )
        self.cells = nn.ModuleList(
            [ConvLSTMCell(hidden, hidden) for _ in range(num_layers)]
        )
        self.attn = SpatialAttention(hidden)

    def encode(self, x_seq, lat_norm, lon_norm, obs_mask=None):
        B, T, C, H, W = x_seq.shape
        hs = [x_seq.new_zeros(B, self.hidden, H, W) for _ in range(self.num_layers)]
        cs = [x_seq.new_zeros(B, self.hidden, H, W) for _ in range(self.num_layers)]
        for t in range(T):
            xt = torch.cat([x_seq[:, t], lat_norm, lon_norm], dim=1)
            inp = self.frame_enc(xt)
            for li, cell in enumerate(self.cells):
                hs[li], cs[li] = cell(inp, hs[li], cs[li])
                inp = hs[li]
        return self.attn(hs[-1])         # 마지막 hidden에 spatial attention
