"""
15_striping_audit.py — 재구성장 스와스 줄무늬(striping) 정량 감사
==================================================================
배경: 14번 시각화에서 재구성장이 OCO-2 스와스 방향 줄무늬 텍스처를 보임.
관측 변동도는 등방(anisotropy ratio 1.14 < 1.25,
data/variogram/variogram_results.json) → 재구성장이 특정 방향으로 비등방이면
물리 신호가 아니라 샘플링 기하 인공물로 판정.

방법: 재구성장(dense)의 방향별 반변동도 γ_d(h), d ∈ {E-W, N-S, NE-SW, NW-SE}.
  · 등거리 비교: 대각 lag는 ×√2 셀 거리 → km 축으로 보간 후 비교
  · 지표 1: 고정 거리(≈111 km)에서 γ 비율 max/min  (단파장 줄무늬 민감)
  · 지표 2: e-folding range 비율 max/min
  · 관측 기준선: 1.14 (등방 판정 임계 1.25)

실행: python pipeline/15_striping_audit.py --arch unet_pconv [--ckpt ...]
산출: outputs/pinn_b/striping_audit_{tag}.json + 동명 PNG
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import ARCHITECTURES, V4_ARCHS                 # noqa: E402
from core.dataset import make_datasets, SEQ_LEN          # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "recon_viz", ROOT / "pipeline" / "14_recon_visualize.py")
recon_viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recon_viz)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

CELL_KM = 0.25 * 111.0                                   # ≈27.75 km
DIRS = {"E-W": (0, 1), "N-S": (1, 0), "NE-SW": (1, 1), "NW-SE": (1, -1)}
MAX_LAG_KM = 600.0
REF_KM = 111.0                                            # 고정 거리 지표(≈4셀)


def directional_gamma(z: np.ndarray, di: int, dj: int, max_lag: int):
    """단일 (H,W) 장의 방향별 반변동도. 반환: (거리km, γ) 배열."""
    H, W = z.shape
    ds, gs = [], []
    step_km = CELL_KM * np.hypot(di, dj)
    for h in range(1, max_lag + 1):
        si, sj = di * h, dj * h
        a = z[max(si, 0):H + min(si, 0), max(sj, 0):W + min(sj, 0)]
        b = z[max(-si, 0):H + min(-si, 0), max(-sj, 0):W + min(-sj, 0)]
        d = a - b
        ds.append(h * step_km)
        gs.append(0.5 * float(np.mean(d ** 2)))
    return np.array(ds), np.array(gs)


def efold_range(dist: np.ndarray, gamma: np.ndarray) -> float:
    """sill의 (1−e⁻¹) 도달 거리 (선형 보간)."""
    sill = gamma[-5:].mean()
    target = sill * (1.0 - np.exp(-1.0))
    idx = np.argmax(gamma >= target)
    if idx == 0:
        return float(dist[0])
    x0, x1 = dist[idx - 1], dist[idx]
    y0, y1 = gamma[idx - 1], gamma[idx]
    return float(x0 + (x1 - x0) * (target - y0) / max(y1 - y0, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else ROOT / "outputs" / "pinn_b" / args.arch / "best.pt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    tag = ckpt_path.parent.name
    lag = bool(ck_args.get("lag_features", False))

    _, va, _, meta = make_datasets(seed=int(ck_args.get("seed", 42)),
                                   lag_features=lag, verbose=False)
    kw = {"lag_channels": meta["n_lag_channels"]} if args.arch in V4_ARCHS else {}
    model = ARCHITECTURES[args.arch](in_channels=meta["n_channels"], time_steps=SEQ_LEN,
                                     hidden=int(ck_args.get("hidden", 64)), **kw)
    model.load_state_dict(ck["model"])
    frames = recon_viz.infer_val(model, DataLoader(va, batch_size=4),
                                 torch.device(args.device))

    # ── 방향별 γ: 12개월 평균 (월별 공간평균 제거 후) ──
    max_lag = int(MAX_LAG_KM / CELL_KM)
    curves = {}
    for name, (di, dj) in DIRS.items():
        gs = []
        for f in frames:
            z = f["pred"] - f["pred"].mean()
            d, g = directional_gamma(z, di, dj, int(max_lag / np.hypot(di, dj)))
            gs.append((d, g))
        dist = gs[0][0]
        gamma = np.mean([g for _, g in gs], axis=0)
        curves[name] = (dist, gamma)

    # ── 지표: 공통 km 그리드 보간 후 고정거리 γ·e-fold range ──
    grid = np.arange(CELL_KM, MAX_LAG_KM, CELL_KM)
    interp = {n: np.interp(grid, d, g) for n, (d, g) in curves.items()}
    g_ref = {n: float(np.interp(REF_KM, grid, gi)) for n, gi in interp.items()}
    ranges = {n: efold_range(grid, gi) for n, gi in interp.items()}
    aniso_gamma = max(g_ref.values()) / max(min(g_ref.values()), 1e-12)
    aniso_range = max(ranges.values()) / max(min(ranges.values()), 1e-12)

    result = {
        "tag": tag, "ckpt": str(ckpt_path),
        "gamma_at_111km": g_ref, "efold_range_km": ranges,
        "anisotropy_gamma_111km": round(aniso_gamma, 3),
        "anisotropy_efold_range": round(aniso_range, 3),
        "obs_baseline_anisotropy": 1.14,
        "isotropy_threshold": 1.25,
        "verdict": ("ANISOTROPIC — 스와스 인공물 의심"
                    if max(aniso_gamma, aniso_range) > 1.25 else
                    "isotropic — 관측 기준선과 부합"),
    }
    out_json = ROOT / "outputs" / "pinn_b" / f"striping_audit_{tag}.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for n, gi in interp.items():
        ax.plot(grid, gi, label=n)
    ax.axvline(REF_KM, color="gray", ls=":", lw=1)
    ax.set_xlabel("거리 (km)"); ax.set_ylabel("반변동도 γ (ppm²)")
    ax.set_title(f"{tag} 재구성장 방향별 변동도 — "
                 f"비등방비 γ@111km={aniso_gamma:.2f} (관측 1.14, 임계 1.25)")
    ax.legend()
    fig.savefig(out_json.with_suffix(".png"), dpi=150, bbox_inches="tight")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
