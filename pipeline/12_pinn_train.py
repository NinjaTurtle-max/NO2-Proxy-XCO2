"""
12_pinn_train.py — 길 B PINN 학습 + 6종 벤치마크
=================================================
core/ARCHITECTURES 6종을 동일 조건으로 학습하고 재구성 R²를 변동도 천장(0.70)과 비교.

학습 설정 (docs/research_synthesis_decision_20260611.html §7):
  · 타겟: ΔXCO₂_local (anomaly_A − 월별 공간중앙값 BG, Hakkarainen 2016)
  · 분할: train 2020–2022 / val 2023 (core/dataset.py)
  · 손실: PIConvLSTMLoss (data NLL + adaptive PDE + reg + gate) × 3-phase Curriculum
          + anti-mean 페널티(자명해 차단, base_pinn 제공)
  · 평가 (둘 다 보고):
      hidden R² — val 관측의 50%만 입력, 숨긴 50%에서 R² (gap-filling, 1차 벤치마크)
      full R²   — val 관측 전부 입력, 관측픽셀 R² (temporal 일반화)

실행:
  python pipeline/12_pinn_train.py --arch mlp --epochs 60          # 단일
  python pipeline/12_pinn_train.py --arch all --epochs 60          # 6종 벤치마크
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import ARCHITECTURES                      # noqa: E402
from core.dataset import make_datasets, N_CHANNELS, SEQ_LEN  # noqa: E402
from core.losses import PIConvLSTMLoss, CurriculumScheduler  # noqa: E402

OUT_DIR = ROOT / "outputs" / "pinn_b"
DT_MONTH = 86400.0 * 30.44                          # 월 평균 초 (PDE dt)
CEILING = 0.70                                      # 변동도 gap-filling 천장


def get_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forward_batch(model, batch, device):
    b = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    c_hat, log_var, pde_res, s_anthro, gate = model(
        b["x_seq"], b["lat_norm"], b["lon_norm"],
        no2=b["no2"], u=b["u"], v=b["v"],
        blh=None, ndvi=b["ndvi"], C_prev=None,
        obs_mask=b["in_mask"], dt=DT_MONTH,
    )
    return b, c_hat, log_var, pde_res, s_anthro, gate


@torch.no_grad()
def evaluate(model, loader, device, mask_key: str):
    """마스크(hidden_mask 또는 obs_mask) 픽셀에서 풀링 R²·RMSE."""
    model.eval()
    se, n, ys, preds = 0.0, 0.0, [], []
    for batch in loader:
        b, c_hat, *_ = forward_batch(model, batch, device)
        m = b[mask_key]
        ys.append((b["target"] * m).flatten()[m.flatten() > 0].cpu())
        preds.append((c_hat * m).flatten()[m.flatten() > 0].cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(preds).numpy()
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rmse = float(np.sqrt(ss_res / len(y)))
    return r2, rmse, len(y)


def train_one(arch: str, args, datasets) -> dict:
    tr_ds, va_ds, va_full_ds, meta = datasets
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    vl = DataLoader(va_ds, batch_size=args.batch, num_workers=0)
    vl_full = DataLoader(va_full_ds, batch_size=args.batch, num_workers=0)

    model = ARCHITECTURES[arch](in_channels=N_CHANNELS, time_steps=SEQ_LEN,
                                hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n━━ [{arch}] params={n_params/1e3:.0f}k device={device} ━━")

    criterion = PIConvLSTMLoss().to(device)
    sched = CurriculumScheduler(total_epochs=args.epochs,
                                lambda_pde_max=args.lambda_pde, beta_0=args.beta0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out = OUT_DIR / arch
    out.mkdir(parents=True, exist_ok=True)
    best = {"hidden_r2": -np.inf, "epoch": -1}
    history, patience = [], 0
    t0 = time.time()

    for ep in range(args.epochs):
        sched.apply_curriculum(model, ep)               # Phase1: PDE 파라미터 동결
        lambdas = sched.get_lambdas(ep)
        beta_t = sched.get_beta(ep)

        model.train()
        ep_loss, n_batch = 0.0, 0
        for batch in tl:
            b, c_hat, log_var, pde_res, s_anthro, gate = forward_batch(model, batch, device)
            total, ld = criterion(
                pred=c_hat, target=b["target"], log_var=log_var,
                obs_mask=b["obs_mask"], pde_residual=pde_res,
                lambdas=lambdas, s_anthro=s_anthro, beta_t=beta_t, sr_gate=gate,
            )
            total = total + args.anti_mean * model.anti_mean_penalty(c_hat)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(total.detach())
            n_batch += 1

        hid_r2, hid_rmse, _ = evaluate(model, vl, device, "hidden_mask")
        full_r2, full_rmse, _ = evaluate(model, vl_full, device, "obs_mask")
        history.append({"epoch": ep, "phase": sched.get_phase(ep),
                        "loss": ep_loss / max(n_batch, 1),
                        "hidden_r2": hid_r2, "full_r2": full_r2,
                        "beta": float(model.beta.detach())})

        if hid_r2 > best["hidden_r2"]:
            best = {"hidden_r2": hid_r2, "hidden_rmse": hid_rmse,
                    "full_r2": full_r2, "full_rmse": full_rmse, "epoch": ep}
            torch.save({"model": model.state_dict(), "arch": arch,
                        "epoch": ep, "metrics": best, "args": vars(args)},
                       out / "best.pt")
            patience = 0
        else:
            patience += 1

        if ep % 5 == 0 or patience == 0:
            print(f"  ep{ep:3d} P{sched.get_phase(ep)} loss={ep_loss/max(n_batch,1):8.3f} "
                  f"hidden R²={hid_r2:+.3f} full R²={full_r2:+.3f} β={float(model.beta.detach()):.4f}")
        if patience >= args.patience:
            print(f"  조기종료 ep{ep} (best ep{best['epoch']} hidden R²={best['hidden_r2']:+.3f})")
            break

    best["minutes"] = (time.time() - t0) / 60.0
    best["n_params"] = n_params
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "metrics.json").write_text(json.dumps(best, indent=2))
    print(f"  [{arch}] best ep{best['epoch']}: hidden R²={best['hidden_r2']:+.3f} "
          f"full R²={best['full_r2']:+.3f} ({best['minutes']:.1f}분)")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all",
                    help="all 또는 콤마목록: " + ",".join(ARCHITECTURES))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda_pde", type=float, default=1.0)
    ap.add_argument("--beta0", type=float, default=1.0, help="SR anchoring 초기 강도")
    ap.add_argument("--anti_mean", type=float, default=0.01, help="자명해 페널티 가중치")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    archs = list(ARCHITECTURES) if args.arch == "all" else args.arch.split(",")
    datasets = make_datasets(seed=args.seed)

    results = {}
    for a in archs:
        results[a] = train_one(a, args, datasets)

    # ── 벤치마크 표 ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "benchmark.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'='*64}\n6종 벤치마크 — val 2023, 변동도 천장 R²={CEILING}\n{'='*64}")
    print(f"{'arch':16s} {'hidden R²':>10s} {'full R²':>9s} {'천장대비':>8s} {'params':>8s}")
    for a, m in sorted(results.items(), key=lambda kv: -kv[1]["hidden_r2"]):
        print(f"{a:16s} {m['hidden_r2']:+10.3f} {m['full_r2']:+9.3f} "
              f"{m['hidden_r2']/CEILING*100:7.1f}% {m['n_params']/1e3:7.0f}k")


if __name__ == "__main__":
    main()
