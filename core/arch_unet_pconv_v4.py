"""
arch_unet_pconv_v4.py — [Version 5-v4] 이중 스트림 U-Net + Partial Convolution PINN
===================================================================================
v5(arch_unet_pconv) 대비 3개 상향 (docs/v4_dual_architecture_spec_20260612.html §1-B):

  1. **이중 스트림 (v4 최대 기대 이득)** — v5는 `x*m`으로 dense 물리장
     (NO₂·u·v·t2m·z500·div850·계절·경향)까지 관측마스크로 0이 되는 구조적
     정보 손실이 있었음. v4는 희소 스트림(y_in·in_mask → PConv)과 밀집
     스트림(물리장 → 일반 Conv)을 분리해 인코더 각 단계에서 융합한다.
     dense 채널은 마스크 갱신에 영향받지 않으므로 국지 NO₂ 배출 코어가
     구조적으로 보존된다.
  2. 첫 희소 블록 k=5 + 분수 마스크 갱신 — 이진(msum>0) 대신 연속 신뢰도
     m_new = msum/slide. 경계 신뢰도를 보존해 핫스팟 스미어링 억제.
     (±2격자 ≈ 55km 국지 문맥.)
  3. 인코더 3단 다운샘플(H/2·H/4·H/8) — 베리오그램 유효 range 58격자 대응.
     보틀넥(H/8=15)의 3×3이 원해상도 24격자를 덮어 RF가 range를 상회.

forward 계약·헤드·SR source·PDE는 BaseSpatioTemporalPINN 공유 (encode만 구현).
희소 채널 = 각 프레임의 [0]=y_in, [1]=in_mask (시간 평탄화 2T채널),
PConv 마스크 = 프레임별 in_mask의 union(max) — in_mask 채널이 프레임별
유효성을 보존하므로 union으로도 구분 가능.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_pinn import BaseSpatioTemporalPINN


class PartialConv2dFrac(nn.Module):
    """Partial Convolution(Liu et al. 2018) + 분수 마스크 갱신.

    이진 갱신(msum>0 → 1)은 단일 유효 이웃만으로 픽셀을 완전 신뢰 처리해
    희소 경계에서 신뢰도 정보를 소실한다. 분수 갱신 m_new = msum/slide는
    연속 신뢰도를 다음 층으로 전파한다 (m ∈ [0,1] 연속값 허용).
    """

    def __init__(self, cin, cout, k=3, stride=1, pad=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, pad, bias=True)
        self.register_buffer("wmask", torch.ones(1, 1, k, k))
        self.stride, self.pad = stride, pad
        self.slide = k * k

    def forward(self, x, m):
        # m: (B,1,H,W) 신뢰도 ∈ [0,1]
        with torch.no_grad():
            msum = F.conv2d(m, self.wmask, stride=self.stride, padding=self.pad)
            valid = (msum > 1e-8).float()
            ratio = self.slide / (msum + 1e-8) * valid
            new_mask = (msum / self.slide).clamp(0.0, 1.0)
        out = self.conv(x * m)
        bias = self.conv.bias.view(1, -1, 1, 1)
        out = (out - bias) * ratio + bias
        out = out * valid
        return out, new_mask


class PConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        self.pc = PartialConv2dFrac(cin, cout, k=k, stride=stride, pad=k // 2)
        self.norm = nn.InstanceNorm2d(cout, affine=True)
        self.act = nn.GELU()

    def forward(self, x, m):
        x, m = self.pc(x, m)
        return self.act(self.norm(x)), m


class ConvBlock(nn.Module):
    """밀집 스트림용 일반 Conv 블록 (마스크 없음)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1)
        self.norm = nn.InstanceNorm2d(cout, affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UNetPConvV4PINN(BaseSpatioTemporalPINN):
    N_SPARSE_PER_FRAME = 2          # [0]=y_in, [1]=in_mask

    def __init__(self, in_channels: int, time_steps: int, hidden: int = 64,
                 lag_channels: int = 0, **kw):
        super().__init__(in_channels=in_channels, hidden=hidden, **kw)
        self.time_steps = time_steps
        self.lag_channels = lag_channels   # dense 스트림에 포함(별도 처리 불필요)
        T = time_steps
        h = hidden
        hs, hd = h // 2, h // 2            # 희소/밀집 스트림 채널 (융합 후 h)
        c_sparse = self.N_SPARSE_PER_FRAME * T
        c_dense = (in_channels - self.N_SPARSE_PER_FRAME) * T + 2  # +좌표

        # ── 희소 스트림 인코더 (PConv, 첫 블록 k=5 분수 마스크) ──
        self.s1 = PConvBlock(c_sparse, hs, k=5, stride=1)
        self.s2 = PConvBlock(hs, hs, k=3, stride=2)         # H/2
        self.s3 = PConvBlock(hs, hs, k=3, stride=2)         # H/4
        self.s4 = PConvBlock(hs, hs, k=3, stride=2)         # H/8

        # ── 밀집 스트림 인코더 (일반 Conv) ──
        self.d1 = ConvBlock(c_dense, hd, stride=1)
        self.d2 = ConvBlock(hd, hd, stride=2)               # H/2
        self.d3 = ConvBlock(hd, hd, stride=2)               # H/4
        self.d4 = ConvBlock(hd, hd, stride=2)               # H/8

        # ── 디코더 (융합 특징 + 신뢰도 마스크 채널, 일반 Conv) ──
        # 입력 = up(하위) h + 스킵(hs+hd) + 마스크 1
        self.u3 = ConvBlock(h + hs + hd + 1, h)             # H/8 → H/4
        self.u2 = ConvBlock(h + hs + hd + 1, h)             # H/4 → H/2
        self.u1 = ConvBlock(h + hs + hd + 1, h)             # H/2 → H
        self.fuse_bottom = ConvBlock(hs + hd + 1, h)        # H/8 보틀넥 융합

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def encode(self, x_seq, lat_norm, lon_norm, obs_mask=None):
        B, T, C, H, W = x_seq.shape
        ns = self.N_SPARSE_PER_FRAME
        xs = x_seq[:, :, :ns].reshape(B, ns * T, H, W)            # 희소(y_in·in_mask)
        xd = x_seq[:, :, ns:].reshape(B, (C - ns) * T, H, W)      # 밀집(물리장·계절·경향)
        xd = torch.cat([xd, lat_norm, lon_norm], dim=1)

        # PConv 마스크 = 프레임별 in_mask union (in_mask 채널이 프레임 구분 보존)
        in_masks = x_seq[:, :, 1]                                  # (B,T,H,W)
        m = in_masks.max(dim=1, keepdim=True)[0]                   # (B,1,H,W)
        if obs_mask is not None:
            m = torch.max(m, obs_mask)

        # ── 인코더: 단계별 (희소 PConv ∥ 밀집 Conv) ──
        f1, m1 = self.s1(xs, m);   g1 = self.d1(xd)                # H
        f2, m2 = self.s2(f1, m1);  g2 = self.d2(g1)                # H/2
        f3, m3 = self.s3(f2, m2);  g3 = self.d3(g2)                # H/4
        f4, m4 = self.s4(f3, m3);  g4 = self.d4(g3)                # H/8

        # ── 보틀넥 융합 → 디코더 (스킵 = 희소∥밀집∥신뢰도) ──
        b = self.fuse_bottom(torch.cat([f4, g4, m4], dim=1))       # H/8
        x = self.u3(torch.cat([self._up(b, f3), f3, g3, m3], dim=1))   # H/4
        x = self.u2(torch.cat([self._up(x, f2), f2, g2, m2], dim=1))   # H/2
        x = self.u1(torch.cat([self._up(x, f1), f1, g1, m1], dim=1))   # H
        return x
