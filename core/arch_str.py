"""
arch_str.py — [Version 1] STR (Spatiotemporal Transformer PINN)
==============================================================
공간 패치 임베딩 + 시간축 Self-Attention + 국지 Spatial Local-Attention 결합.
변동도가 정당화한 단거리 공간 상관(2격자 반경=directive 명세)을 무력화하지 않도록,
공간 어텐션은 5×5(반경 2) 윈도우로 마스킹된 Local Attention만 수행한다.

흐름:
  프레임별 임베딩 → 시간 Self-Attn(per-pixel, T축) → 마지막 시점 → Local Spatial-Attn → 특징
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_pinn import BaseSpatioTemporalPINN


class LocalSpatialAttention(nn.Module):
    """5×5(반경 2) 윈도우 내 Local Self-Attention. 단거리 공간상관 보존."""

    def __init__(self, dim: int, window: int = 5):
        super().__init__()
        self.dim = dim
        self.win = window
        self.pad = window // 2
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5

    def forward(self, x):
        B, Cc, H, W = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)            # 각 (B,dim,H,W)
        # 이웃 윈도우 추출: (B, dim*win², H*W)
        k_w = F.unfold(k, self.win, padding=self.pad).view(B, Cc, self.win**2, H * W)
        v_w = F.unfold(v, self.win, padding=self.pad).view(B, Cc, self.win**2, H * W)
        q_f = q.view(B, Cc, 1, H * W)
        attn = (q_f * k_w).sum(1) * self.scale              # (B,win²,H*W)
        attn = attn.softmax(dim=1)
        out = (attn.unsqueeze(1) * v_w).sum(2)              # (B,dim,H*W)
        return self.proj(out.view(B, Cc, H, W))


class STRPINN(BaseSpatioTemporalPINN):
    def __init__(self, in_channels: int, time_steps: int, hidden: int = 64,
                 n_heads: int = 4, **kw):
        super().__init__(in_channels=in_channels, hidden=hidden, **kw)
        self.time_steps = time_steps
        # 패치(픽셀) 임베딩 — 1×1 + 3×3로 국소 컨텍스트
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden, 3, padding=1), nn.GELU(),
        )
        # 시간축 Self-Attention (per-pixel 토큰, 길이 T)
        self.temporal_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.tnorm = nn.LayerNorm(hidden)
        # 국지 공간 어텐션
        self.local_attn = LocalSpatialAttention(hidden, window=5)
        self.snorm = nn.GroupNorm(1, hidden)

    def encode(self, x_seq, lat_norm, lon_norm, obs_mask=None):
        B, T, C, H, W = x_seq.shape
        # 프레임별 임베딩
        feats = []
        for t in range(T):
            xt = torch.cat([x_seq[:, t], lat_norm, lon_norm], dim=1)
            feats.append(self.embed(xt))                   # (B,hid,H,W)
        f = torch.stack(feats, dim=1)                      # (B,T,hid,H,W)
        # 시간 Self-Attention: 토큰=시간, 픽셀 병렬
        f = f.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, self.hidden)
        att, _ = self.temporal_attn(f, f, f)
        f = self.tnorm(att + f)                            # residual
        f = f.reshape(B, H, W, T, self.hidden)[:, :, :, -1]    # 마지막 시점 (B,H,W,hid)
        f = f.permute(0, 3, 1, 2).contiguous()             # (B,hid,H,W)
        # 국지 공간 어텐션 + residual
        f = self.snorm(f + self.local_attn(f))
        return f
