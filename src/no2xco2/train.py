"""분할 마스크 · UTC 일 블록 · 짝 부트스트랩 (학습 · 평가 · 기준선이 함께 쓴다).

중첩 분할: test = 폴드 f. val = 훈련 가능 사운딩 중 직교 블록 — 공간 시험이면 시간 폴드 v 의 블록(버퍼 포함),
시간 시험이면 공간 폴드 v 의 칸(버퍼 포함). train = 훈련 가능 사운딩 − val − val 버퍼.
"""
import numpy as np
import pandas as pd

import no2xco2.data.splits as splits_mod


def nested_masks(df, scheme, fold, lab, vfold):
    tr_all = splits_mod.train_mask(df, scheme, fold) & lab
    other = "space" if scheme == "time" else "time"
    val = tr_all & (df[f"fold_{other}"].to_numpy() == vfold)
    tr = tr_all & splits_mod.train_mask(df, other, vfold)
    te_col = "year" if scheme == "year" else f"fold_{scheme}"
    te = (df[te_col].to_numpy() == fold) & lab
    return tr, val, te


def day_blocks(t) -> np.ndarray:
    """UTC 일 블록 번호 = 1970-01-01 기준 일수 (시각 해상도와 무관)."""
    return np.asarray(pd.to_datetime(np.asarray(t))).astype("datetime64[D]").astype(np.int64)


def block_bootstrap(res_a, res_b, days, n=1000, seed=0):
    """ΔRMSE = RMSE(b) − RMSE(a) 의 UTC 일 블록 부트스트랩 95 % 구간. res_*: 같은 사운딩의 잔차, days: 블록 번호."""
    rng = np.random.default_rng(seed); ud = np.unique(days); idx = {d: np.where(days == d)[0] for d in ud}; out = []
    for _ in range(n):
        pick = rng.choice(ud, len(ud), replace=True); ii = np.concatenate([idx[d] for d in pick])
        out.append(np.sqrt((res_b[ii] ** 2).mean()) - np.sqrt((res_a[ii] ** 2).mean()))
    return np.percentile(out, [2.5, 97.5])
