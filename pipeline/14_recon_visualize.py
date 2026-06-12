"""
14_recon_visualize.py — 학습된 PINN 체크포인트의 val 2023 재구성 시각화
========================================================================
체크포인트(best.pt)를 로드해 val 2023을 추론하고 3종 그림 생성:
  (a) {arch}_recon_maps.png    월 4종(1·4·7·10월): 입력(50%) / 재구성 / 전체 관측
  (b) {arch}_scatter.png       hidden 픽셀 pred vs true 산점도(밀도 색) + R²
  (c) {arch}_monthly_r2.png    2023 월별 hidden R² 막대

실행:
  python pipeline/14_recon_visualize.py --arch unet_pconv
  python pipeline/14_recon_visualize.py --arch cnn_lstm \
         --ckpt outputs/pinn_b/cnn_lstm_s123/best.pt
추론은 기본 CPU (학습 중 MPS 점유 회피).
"""
from __future__ import annotations

import argparse
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

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

DT_MONTH = 86400.0 * 30.44


@torch.no_grad()
def infer_val(model, loader, device):
    """val 12개월 추론 → 월별 (pred, target, in_mask, hidden_mask) 누적."""
    model.eval()
    out = []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        c_hat, *_ = model(b["x_seq"], b["lat_norm"], b["lon_norm"],
                          no2=b["no2"], u=b["u"], v=b["v"], blh=None,
                          ndvi=b["ndvi"], C_prev=None,
                          obs_mask=b["in_mask"], dt=DT_MONTH)
        for i in range(c_hat.shape[0]):
            out.append({k: b[k][i, 0].cpu().numpy()
                        for k in ["target", "in_mask", "hidden_mask", "obs_mask"]}
                       | {"pred": c_hat[i, 0].cpu().numpy(),
                          "t_index": int(b["t_index"][i])})
    return sorted(out, key=lambda d: d["t_index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--ckpt", default=None, help="기본: outputs/pinn_b/{arch}/best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default=None, help="기본: outputs/pinn_b/figs")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else ROOT / "outputs" / "pinn_b" / args.arch / "best.pt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    lag = bool(ck_args.get("lag_features", False))
    hidden = int(ck_args.get("hidden", 64))
    tag = ckpt_path.parent.name                         # 예: unet_pconv, base_150ep_s42

    _, va, _, meta = make_datasets(seed=int(ck_args.get("seed", 42)),
                                   lag_features=lag, verbose=False)
    device = torch.device(args.device)
    kw = {"lag_channels": meta["n_lag_channels"]} if args.arch in V4_ARCHS else {}
    model = ARCHITECTURES[args.arch](in_channels=meta["n_channels"],
                                     time_steps=SEQ_LEN, hidden=hidden, **kw).to(device)
    model.load_state_dict(ck["model"])
    print(f"[viz] {tag}: ep{ck.get('epoch')} ckpt metrics={ck.get('metrics')}")

    frames = infer_val(model, DataLoader(va, batch_size=4), device)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "pinn_b" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── (a) 월 4종 맵: 입력(50%) / 재구성 / 전체 관측 ──
    sel = [0, 3, 6, 9]                                   # 2023-01, 04, 07, 10
    vmax = 1.5
    fig, axes = plt.subplots(len(sel), 3, figsize=(13, 3.1 * len(sel)))
    for r, mi in enumerate(sel):
        f = frames[mi]
        shown = [np.where(f["in_mask"] > 0, f["target"], np.nan),
                 f["pred"],
                 np.where(f["obs_mask"] > 0, f["target"], np.nan)]
        titles = [f"2023-{mi+1:02d} 입력 관측(50%)", "재구성", "전체 관측(정답)"]
        for c in range(3):
            ax = axes[r, c]
            im = ax.imshow(shown[c], origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(titles[c], fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.55, label="ΔXCO₂_local (ppm)")
    fig.suptitle(f"{tag} — val 2023 재구성 (hidden R²={ck['metrics']['hidden_r2']:+.3f})",
                 fontsize=13)
    fig.savefig(out_dir / f"{tag}_recon_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── (b) hidden 픽셀 산점도 ──
    ys = np.concatenate([f["target"][f["hidden_mask"] > 0] for f in frames])
    ps = np.concatenate([f["pred"][f["hidden_mask"] > 0] for f in frames])
    ss_res = float(((ys - ps) ** 2).sum())
    r2 = 1.0 - ss_res / float(((ys - ys.mean()) ** 2).sum())
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    hb = ax.hexbin(ys, ps, gridsize=60, cmap="viridis", mincnt=1)
    lim = [min(ys.min(), ps.min()), max(ys.max(), ps.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("관측 ΔXCO₂_local (ppm)"); ax.set_ylabel("재구성 (ppm)")
    ax.set_title(f"{tag} — hidden 픽셀 (n={len(ys):,}, R²={r2:+.3f})")
    fig.colorbar(hb, ax=ax, label="픽셀 수")
    fig.savefig(out_dir / f"{tag}_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── (c) 월별 hidden R² ──
    r2m = []
    for f in frames:
        m = f["hidden_mask"] > 0
        if m.sum() < 10:
            r2m.append(np.nan); continue
        y, p = f["target"][m], f["pred"][m]
        r2m.append(1.0 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.bar(np.arange(1, 13), r2m, color="#0f4c81")
    ax.axhline(0.70, color="crimson", ls="--", lw=1, label="변동도 천장 0.70")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(1, 13)); ax.set_xlabel("2023년 월"); ax.set_ylabel("hidden R²")
    ax.set_title(f"{tag} — 월별 gap-filling 성능"); ax.legend(fontsize=8)
    fig.savefig(out_dir / f"{tag}_monthly_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz] 풀링 hidden R²={r2:+.4f} → {out_dir}/{tag}_*.png 3종 저장")


if __name__ == "__main__":
    main()
