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


SCENE_GAP_S = 300.0  # 장면 = 같은 위성, 직전 사운딩과 300 s 초과 간격이면 새 장면 (결정 11 ① W1, P1 정의)


def _scene_sorted(df: pd.DataFrame, lab: np.ndarray):
    """L_train 행을 (위성 문자열, 시각) 순으로 정렬한 표와 장면 번호. P1 `experiments/p1_revisit_20261027/split_check_20261027_pi.py` 와 같은 절차.
    반환: (o = L_train 부분집합 안 정렬 순서, d = 정렬된 표[t, sat, time], scene = 정렬 순서의 장면 번호)."""
    d = df.loc[lab, ["time", "satellite"]].reset_index(drop=True)
    d["t"] = (pd.to_datetime(d.time) - T0).dt.total_seconds() / 86400.0
    d["sat"] = d.satellite.astype(str)
    o = np.lexsort((d.t.to_numpy(), d.sat.to_numpy())); d = d.iloc[o].reset_index(drop=True)
    scene = (d.groupby("sat").t.diff().fillna(1e9) * 86400 > SCENE_GAP_S).cumsum().to_numpy()
    return o, d, scene


def scene_ids(df: pd.DataFrame, lab: np.ndarray) -> np.ndarray:
    """행마다 장면 번호 (L_train 행만, 나머지 −1). 번호는 P1 과 같다(정렬 순 누적합)."""
    o, _, scene = _scene_sorted(df, lab)
    sub = np.empty(int(lab.sum()), np.int64); sub[o] = scene
    out = np.full(len(df), -1, np.int64); out[lab] = sub
    return out


def scene_holdout(df: pd.DataFrame, lab: np.ndarray, seed: int = 0, f_test: float = 0.190, f_val_cum: float = 0.345):
    """결정 11 ① 주 평가 분할 (사용자 확정 10-27, 설계표 §10.3 U3): 장면 단위 월별 층화 무작위, 버퍼 없음.
    장면 월 = 첫 사운딩의 UTC 월. default_rng(seed) 를 1 → 12월 순으로 소비, 달마다 장면 순열 뒤 누적 사운딩 비율 ≤ f_test → test,
    ≤ f_val_cum → val, 나머지 train. P1 split_check 와 같은 절차 (seed 0: 훈련 1,521,459 · val 357,732 · test 420,391 · test 장면 371).
    반환: (role [len(df)] ∈ {train, val, test, none(비 L_train)}, scene_id [len(df)] (비 L_train −1))."""
    o, d, scene = _scene_sorted(df, lab)
    d["scene"] = scene; d["month"] = pd.to_datetime(d.time).dt.month
    rng = np.random.default_rng(seed)
    role = pd.Series("train", index=d.scene.unique())
    scn = d.groupby("scene").agg(n=("t", "size"), month=("month", "first"))
    for _, g in scn.groupby("month"):
        ids = g.index.to_numpy()[rng.permutation(len(g))]; cum = np.cumsum(g.n.loc[ids].to_numpy()) / g.n.sum()
        role.loc[ids[cum <= f_test]] = "test"
        role.loc[ids[(cum > f_test) & (cum <= f_val_cum)]] = "val"
    sub_role = np.empty(int(lab.sum()), object); sub_role[o] = role.loc[d.scene].to_numpy()
    sub_sid = np.empty(int(lab.sum()), np.int64); sub_sid[o] = scene
    r = np.full(len(df), "none", object); r[lab] = sub_role
    sid = np.full(len(df), -1, np.int64); sid[lab] = sub_sid
    return r, sid


def upstream_lag(df: pd.DataFrame, sid: np.ndarray, up: np.ndarray, q: np.ndarray, r_km: float = 600.0) -> np.ndarray:
    """결정 11 ① W7 시차 (P1 `split_check_20261027_pi.py` lags 와 같은 정의): 1° 칸 (floor(lat−20), floor(lon−100)) 중심 거리,
    q 행의 (장면, 칸)마다 칸 중심이 r_km 안인 up 행 (장면, 칸) 중 다른 장면이면서 시각(장면-칸 평균)이 앞선 가장 최근 것까지 시차(일).
    r_km = 0 이면 같은 칸만. 반환 [len(df)]: q 행 = 그 (장면, 칸) 의 시차 (직전 없음 inf), 나머지 NaN. 사운딩 비율 = q 행 단위 집계 (P1 의 n 가중과 같음)."""
    from scipy.spatial import cKDTree
    t = (pd.to_datetime(df["time"]) - T0).dt.total_seconds().to_numpy() / 86400.0
    ci = np.floor(df["latitude"].to_numpy() - 20).astype(np.int64); cj = np.floor(df["longitude"].to_numpy() - 100).astype(np.int64)
    m = up | q; cells = np.unique(np.c_[ci[m], cj[m]], axis=0); key = {(a, b): k for k, (a, b) in enumerate(map(tuple, cells))}
    cell = np.full(len(df), -1, np.int64); cell[m] = [key[(a, b)] for a, b in zip(ci[m], cj[m])]
    clat = np.deg2rad(20 + cells[:, 0] + 0.5); clon = np.deg2rad(100 + cells[:, 1] + 0.5)
    xyz = np.c_[np.cos(clat) * np.cos(clon), np.cos(clat) * np.sin(clon), np.sin(clat)]; tree = cKDTree(xyz)
    U = pd.DataFrame(dict(scene=sid[up], cell=cell[up], t=t[up])).groupby(["scene", "cell"]).t.mean().reset_index()
    Q = pd.DataFrame(dict(scene=sid[q], cell=cell[q], t=t[q])).groupby(["scene", "cell"]).t.mean().reset_index()
    Tb = {k: (g.sort_values("t").t.to_numpy(), g.sort_values("t").scene.to_numpy()) for k, g in U.groupby("cell")}
    chord = 2 * np.sin(r_km / (2 * 6371.0)) + 1e-9; best = np.full(len(Q), np.inf)
    for a, ga in Q.groupby("cell"):
        nb = tree.query_ball_point(xyz[a], chord) if r_km > 0 else [a]
        ta, sa = ga.t.to_numpy(), ga.scene.to_numpy(); b_ = np.full(len(ta), np.inf)
        for b in nb:
            if b not in Tb:
                continue
            tb, sb = Tb[b]; j = np.searchsorted(tb, ta, side="left") - 1
            j = np.where((j >= 0) & (sb[np.clip(j, 0, None)] == sa), j - 1, j)  # 같은 장면이면 하나 앞 (장면-칸 1개라 한 번이면 충분)
            b_ = np.minimum(b_, np.where(j >= 0, ta - tb[np.clip(j, 0, None)], np.inf))
        best[ga.index.to_numpy()] = b_
    Q["lag"] = best
    out = np.full(len(df), np.nan)
    out[q] = pd.DataFrame(dict(scene=sid[q], cell=cell[q])).merge(Q[["scene", "cell", "lag"]], on=["scene", "cell"], how="left").lag.to_numpy()
    return out


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
