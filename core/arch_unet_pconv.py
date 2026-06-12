"""
arch_unet_pconv.py — [Version 5] U-Net + Partial Convolution PINN
================================================================
OCO-2 월별 관측 희소성으로 인한 가짜 경계면 왜곡을 막기 위해, 유효 관측 마스크
영역만 가중치를 갱신하는 Partial Convolution(Liu et al. 2018) 기반 인코더-디코더.
스킵 연결로 다중 스케일 공간 구조를 복원한다. 마스크는 obs_mask(없으면 전체 유효)로
초기화되어 PartialConv 전파 과정에서 점진 갱신된다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_pinn import BaseSpatioTemporalPINN


class PartialConv2d(nn.Module):
    """Liu et al.(2018) Partial Convolution. 유효 영역만 정규화·갱신."""

    def __init__(self, cin, cout, k=3, stride=1, pad=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, pad, bias=True)
        self.register_buffer("wmask", torch.ones(1, 1, k, k))
        self.k, self.stride, self.pad = k, stride, pad
        self.slide = k * k

    def forward(self, x, m):
        # m: (B,1,H,W) 유효=1
        with torch.no_grad():
            msum = F.conv2d(m, self.wmask, stride=self.stride, padding=self.pad)
            new_mask = (msum > 0).float()
            ratio = self.slide / (msum + 1e-8)
            ratio = ratio * new_mask
        out = self.conv(x * m)
        bias = self.conv.bias.view(1, -1, 1, 1)
        out = (out - bias) * ratio + bias
        out = out * new_mask
        return out, new_mask


class PConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.pc = PartialConv2d(cin, cout, k=3, stride=stride, pad=1)
        self.norm = nn.InstanceNorm2d(cout, affine=True)
        self.act = nn.GELU()

    def forward(self, x, m):
        x, m = self.pc(x, m)
        return self.act(self.norm(x)), m


class UNetPConvPINN(BaseSpatioTemporalPINN):
    def __init__(self, in_channels: int, time_steps: int, hidden: int = 64, **kw):
        super().__init__(in_channels=in_channels, hidden=hidden, **kw)
        self.time_steps = time_steps
        cin = in_channels * time_steps + 2                  # 시간평탄화 + 좌표
        h = hidden
        # 인코더 (2단 다운샘플)
        self.e1 = PConvBlock(cin, h, stride=1)
        self.e2 = PConvBlock(h, h, stride=2)                # H/2
        self.e3 = PConvBlock(h, h, stride=2)                # H/4
        # 디코더 (업샘플 + 스킵)
        self.d2 = PConvBlock(h + h, h, stride=1)
        self.d1 = PConvBlock(h + h, h, stride=1)
        self.out = PConvBlock(h + h, h, stride=1)

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def encode(self, x_seq, lat_norm, lon_norm, obs_mask=None):
        B, T, C, H, W = x_seq.shape
        x = x_seq.reshape(B, T * C, H, W)
        x = torch.cat([x, lat_norm, lon_norm], dim=1)
        m = obs_mask if obs_mask is not None else x.new_ones(B, 1, H, W)

        s1, m1 = self.e1(x, m)                              # (B,h,H,W)
        s2, m2 = self.e2(s1, m1)                            # H/2
        s3, m3 = self.e3(s2, m2)                            # H/4
        # 업: s3 → s2 스케일, 특징은 concat(채널 결합), 마스크는 union(max, 1채널 유지)
        u2 = self._up(s3, s2); um2 = self._up(m3, m2)
        d2, dm2 = self.d2(torch.cat([u2, s2], 1), torch.max(um2, m2))
        u1 = self._up(d2, s1); um1 = self._up(dm2, m1)
        d1, dm1 = self.d1(torch.cat([u1, s1], 1), torch.max(um1, m1))
        out, _ = self.out(torch.cat([d1, s1], 1), torch.max(dm1, m1))
        return out
