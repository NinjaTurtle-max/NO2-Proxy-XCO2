"""
13_hpo_v4.py — v4 이중 아키텍처 하이퍼파라미터 최적화
=====================================================
docs/v4_dual_architecture_spec_20260612.html §4.

Stage 1 (스크리닝, 60ep): 아키 2종 × 설정 8종.
  base / lr↑ / lr↓ / hidden↑ / rho↑ / var_floor↑ + 절제 2종(legacy 손실, lag 채널 off)
  — 절제 2종이 v4 신규 요소(동적 손실·경향 채널)의 기여를 분리한다.
Stage 2 (확증, 150ep × seeds 42·123·7): 아키별 stage 1 최고 설정.
  시드 규약은 기존 시드 스터디와 동일(데이터 마스킹·모델 초기화 모두 seed).

산출: outputs/pinn_b_v4/hpo_results.json + 설정별 디렉터리(metrics/history/best.pt)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.dataset import make_datasets  # noqa: E402

# 12_pinn_train 모듈 로드 (숫자 시작 파일명 → importlib 경유)
_spec = importlib.util.spec_from_file_location(
    "pinn_train", ROOT / "pipeline" / "12_pinn_train.py")
pinn_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pinn_train)

OUT = ROOT / "outputs" / "pinn_b_v4"
ARCHS = ["cnn_lstm_v4", "unet_pconv_v4"]
SEEDS_STAGE2 = [42, 123, 7]

BASE = dict(epochs=60, hidden=64, batch=4, lr=3e-4, lambda_pde=1.0, beta0=1.0,
            anti_mean=0.01, patience=15, seed=42, device="auto",
            loss="dynamic", rho=0.9, var_floor=0.1, lag_features=True)

# 설정명 → BASE 오버라이드 (스크리닝 격자)
CONFIGS = {
    "base":        {},
    "lr_hi":       {"lr": 1e-3},
    "lr_lo":       {"lr": 1e-4},
    "hidden96":    {"hidden": 96},
    "rho095":      {"rho": 0.95},
    "vf03":        {"var_floor": 0.3},
    "abl_legacy":  {"loss": "legacy"},      # 절제: 동적 손실 기여
    "abl_nolag":   {"lag_features": False},  # 절제: 경향 채널 기여
}


def run_one(arch: str, cfg_name: str, overrides: dict, datasets_cache: dict) -> dict:
    args = SimpleNamespace(**{**BASE, **overrides})
    key = args.lag_features
    if key not in datasets_cache or datasets_cache[key][1] != args.seed:
        datasets_cache[key] = (
            make_datasets(seed=args.seed, lag_features=key, verbose=False), args.seed)
    datasets = datasets_cache[key][0]

    pinn_train.OUT_DIR = OUT / cfg_name                  # 설정별 분리 저장
    t0 = time.time()
    best = pinn_train.train_one(arch, args, datasets)
    best["config"] = cfg_name
    best["overrides"] = overrides
    best["wall_min"] = (time.time() - t0) / 60.0
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"stage1": {}, "stage2": {}}
    cache: dict = {}

    # ── Stage 1: 스크리닝 ──
    for arch in ARCHS:
        results["stage1"][arch] = {}
        for cfg_name, ov in CONFIGS.items():
            print(f"\n##### STAGE1 {arch} / {cfg_name} #####")
            results["stage1"][arch][cfg_name] = run_one(arch, cfg_name, ov, cache)
            (OUT / "hpo_results.json").write_text(json.dumps(results, indent=2))

    # ── Stage 1 요약 + 최고 설정 선택 ──
    print(f"\n{'='*70}\nSTAGE 1 요약 (60ep, hidden R²)\n{'='*70}")
    best_cfg = {}
    for arch in ARCHS:
        rows = sorted(results["stage1"][arch].items(),
                      key=lambda kv: -kv[1]["hidden_r2"])
        for name, m in rows:
            print(f"  {arch:14s} {name:11s} hidden={m['hidden_r2']:+.3f} "
                  f"full={m['full_r2']:+.3f} ep{m['epoch']}")
        best_cfg[arch] = rows[0][0]
        print(f"  → {arch} 최고: {best_cfg[arch]}")

    # ── Stage 2: 최고 설정 150ep × 3 seeds ──
    for arch in ARCHS:
        cfg_name = best_cfg[arch]
        ov = deepcopy(CONFIGS[cfg_name])
        ov["epochs"] = 150
        results["stage2"][arch] = {"config": cfg_name, "seeds": {}}
        for s in SEEDS_STAGE2:
            print(f"\n##### STAGE2 {arch} / {cfg_name} / seed {s} #####")
            ov["seed"] = s
            r = run_one(arch, f"{cfg_name}_150ep_s{s}", ov, cache)
            results["stage2"][arch]["seeds"][str(s)] = r
            (OUT / "hpo_results.json").write_text(json.dumps(results, indent=2))
        hid = np.array([results["stage2"][arch]["seeds"][str(s)]["hidden_r2"]
                        for s in SEEDS_STAGE2])
        results["stage2"][arch]["hidden_r2_mean"] = float(hid.mean())
        results["stage2"][arch]["hidden_r2_std"] = float(hid.std(ddof=1))
        (OUT / "hpo_results.json").write_text(json.dumps(results, indent=2))

    # ── 최종 표 ──
    print(f"\n{'='*70}\nSTAGE 2 최종 (150ep × seeds {SEEDS_STAGE2}, 천장 0.70)\n{'='*70}")
    for arch in ARCHS:
        m = results["stage2"][arch]
        print(f"  {arch:14s} cfg={m['config']:11s} "
              f"hidden R²={m['hidden_r2_mean']:+.4f}±{m['hidden_r2_std']:.4f} "
              f"({m['hidden_r2_mean']/0.70*100:.1f}%)")
    print("HPO DONE")


if __name__ == "__main__":
    main()
