"""블록 교차 검증 분할 (무작위 CV 대신). 연속 7일 시간 블록 + 5° 공간 블록, 버퍼 포함.

블록 (사운딩 단위):
  time_block  = (date − 2020-01-01).days // 7
  space_block = floor((lat − 20)/5) * 10 + floor((lon − 100)/5)   (6 × 10 = 60 칸)
  year        = 연도
폴드:
  fold_time   = time_block % K   (K = 5, 교차 배치)
  fold_space  = 60칸을 고정 시드 난수로 K개 폴드에 12칸씩 배정
훈련 가능 행 (train_mask): 시험 블록과 버퍼를 뺀 행
  시간 버퍼 = 시험 날 ±2일 (48 h 롤아웃보다 길게)
  공간 버퍼 = 시험 칸 경계에서 1° 이내
"""
import numpy as np
import pandas as pd

K = 5; T0 = pd.Timestamp("2020-01-01"); SEED = 20260908
T_BUF_DAYS = 2; S_BUF_DEG = 1.0


def assign(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["time"])
    df["time_block"] = ((t - T0).dt.days // 7).astype(np.int32)
    ci = np.floor((df["latitude"] - 20) / 5).clip(0, 5).astype(np.int32)
    cj = np.floor((df["longitude"] - 100) / 5).clip(0, 9).astype(np.int32)
    df["space_block"] = (ci * 10 + cj).astype(np.int32)
    df["year"] = t.dt.year.astype(np.int16)
    df["fold_time"] = (df["time_block"] % K).astype(np.int8)
    rng = np.random.default_rng(SEED); cell_fold = rng.permutation(np.arange(60) % K)
    df["fold_space"] = cell_fold[df["space_block"]].astype(np.int8)
    return df


def train_mask(df: pd.DataFrame, scheme: str, fold: int) -> np.ndarray:
    """scheme ∈ {time, space, year}. 반환: 훈련에 쓸 수 있는 행 (시험 · 버퍼 제외)."""
    if scheme == "time":
        test = df["fold_time"].to_numpy() == fold
        t = pd.to_datetime(df["time"]).to_numpy().astype("datetime64[D]")
        test_days = np.unique(t[test]); buf = np.zeros(len(df), bool)
        for d in test_days:
            buf |= (np.abs((t - d).astype(int)) <= T_BUF_DAYS)
        return ~test & ~buf
    if scheme == "space":
        test = df["fold_space"].to_numpy() == fold
        cells = np.unique(df.loc[test, "space_block"]); buf = np.zeros(len(df), bool)
        lat = df["latitude"].to_numpy(); lon = df["longitude"].to_numpy()
        for c in cells:
            ci, cj = divmod(int(c), 10); la0, lo0 = 20 + 5 * ci, 100 + 5 * cj
            buf |= (lat >= la0 - S_BUF_DEG) & (lat <= la0 + 5 + S_BUF_DEG) & (lon >= lo0 - S_BUF_DEG) & (lon <= lo0 + 5 + S_BUF_DEG)
        return ~test & ~buf
    if scheme == "year":
        return df["year"].to_numpy() != fold
    raise ValueError(scheme)
