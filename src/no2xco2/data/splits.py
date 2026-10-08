"""결정 10 분할: 랜덤 CV 배제. 연속 7일 시간 블록 + 5° 공간 블록 (버퍼 1°) + leave-one-year-out.

블록 정의 (OCO 사운딩 단위):
  time_block  = (date − 2020-01-01).days // 7            (5년 → 262 블록)
  space_block = floor((lat−20)/5) * 10 + floor((lon−100)/5) (6×10 = 60 셀)
  year        = 연도
폴드 배정:
  fold_time   = time_block % K       (K=5, 교차 배치)
  fold_space  = 셀을 시드 고정 난수로 K개 폴드에 균등 배정
훈련 마스크 (train_mask): 테스트 블록 제외 + 버퍼 제외
  시간 버퍼 = 테스트 블록 양측 2일 (lag 48 h 초과 회피)
  공간 버퍼 = 테스트 셀 경계에서 1° 이내 (Mahoney 2023·Valavi 2019: 버퍼 = 변동함수 range; 우리 range 14.4 km 상회)
출력: <out>/splits_YYYYMM.parquet (row_idx, time_block, space_block, year, fold_time, fold_space) + 요약.
"""
import argparse, glob, os
import numpy as np, pandas as pd, pyarrow.parquet as pq

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
    rng = np.random.default_rng(SEED); cell_fold = rng.permutation(np.arange(60) % K)  # 60셀 → 12셀/폴드
    df["fold_space"] = cell_fold[df["space_block"]].astype(np.int8)
    return df


def train_mask(df: pd.DataFrame, scheme: str, fold: int) -> np.ndarray:
    """scheme ∈ {time, space, year}. 반환: 훈련에 쓸 수 있는 행 (테스트·버퍼 제외)."""
    if scheme == "time":
        test = df["fold_time"].to_numpy() == fold
        t = pd.to_datetime(df["time"]).to_numpy().astype("datetime64[D]")
        test_days = np.unique(t[test]); buf = np.zeros(len(df), bool)
        for d in test_days:  # 테스트 일 ±2일
            buf |= (np.abs((t - d).astype(int)) <= T_BUF_DAYS)
        return ~test & ~buf
    if scheme == "space":
        test = df["fold_space"].to_numpy() == fold
        cells = np.unique(df.loc[test, "space_block"]); buf = np.zeros(len(df), bool)
        lat = df["latitude"].to_numpy(); lon = df["longitude"].to_numpy()
        for c in cells:  # 테스트 셀 경계 1° 확장 영역
            ci, cj = divmod(int(c), 10); la0, lo0 = 20 + 5 * ci, 100 + 5 * cj
            buf |= (lat >= la0 - S_BUF_DEG) & (lat <= la0 + 5 + S_BUF_DEG) & (lon >= lo0 - S_BUF_DEG) & (lon <= lo0 + 5 + S_BUF_DEG)
        return ~test & ~buf
    if scheme == "year":
        return df["year"].to_numpy() != fold
    raise ValueError(scheme)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="202001")
    ap.add_argument("--oco-dir", default="data/raw/pilot_202001/oco_nodes")
    ap.add_argument("--out", default="data/processed/splits")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.oco_dir, "*.parquet")))
    df = pq.read_table(files, columns=["row_idx", "latitude", "longitude", "time", "label"]).to_pandas()
    df = assign(df)
    df[["row_idx", "time_block", "space_block", "year", "fold_time", "fold_space"]].to_parquet(
        os.path.join(a.out, f"splits_{a.month}.parquet"), compression="zstd", index=False)
    n = len(df); lab = (df["label"] == "L_train").to_numpy()  # label ∈ {L_train, L_masked} (문자열)
    print(f"사운딩 {n:,} (L_train {lab.sum():,} / L_masked {(~lab).sum():,}) · time_block {df.time_block.nunique()}개 · space_block {df.space_block.nunique()}개 (사운딩 있는 셀)")
    print(f"{'scheme':6s} {'fold':>4s} {'test':>9s} {'buffer':>9s} {'train':>9s} {'buf%':>6s}")
    for scheme, folds in (("time", range(K)), ("space", range(K))):
        for f in folds:
            tm = train_mask(df, scheme, f)
            test = (df[f"fold_{scheme}"] == f).to_numpy(); buf = ~tm & ~test
            print(f"{scheme:6s} {f:4d} {test.sum():9,} {buf.sum():9,} {tm.sum():9,} {100*buf.mean():5.1f}%")
