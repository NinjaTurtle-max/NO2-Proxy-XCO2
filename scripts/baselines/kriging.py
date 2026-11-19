"""시공간 크리깅 기준선 (학습 없음 — 훈련 사운딩의 XCO₂ 만 쓴다).

  추세: 훈련 행 OLS, 입력 = 절편 + 배경 11변수 (loader.bg_features, 훈련 평균 · 표준편차로 표준화).
  잔차 r = y − 추세. 변동함수: 훈련 부표본의 경험 시공간 변동함수 (거리 ≤ 1,000 km × 시차 ≤ 10일) →
    지수 모형 γ(h, u) = c0 + c1(1 − e^{−d/r}) [단일] · + c2(1 − e^{−d/r2}) [중첩], d = sqrt(h² + (a·u)²).
  거리는 대원 거리, 이웃 탐색은 단위벡터 [R·xyz, a·t] 4차원 KD-tree. 이웃 공분산 대각 = c0 + c1(+c2).
  예측: 국소 단순 크리깅(잔차 평균 0) — 이웃 k 개의 훈련 잔차를 가중해 추세에 더한다.
run_baselines.py 가 이 모듈을 쓴다. 자기 시험: python scripts/baselines/kriging.py
"""
import numpy as np
import pandas as pd

R_EARTH = 6371.0
def unit_xyz(lat, lon):
    la, lo = np.deg2rad(np.asarray(lat, np.float64)), np.deg2rad(np.asarray(lon, np.float64))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def gc_from_chord(chord):
    return 2 * R_EARTH * np.arcsin(np.clip(chord / 2, 0.0, 1.0))


def gamma_model(p, h, u, nested):
    d = np.sqrt(h ** 2 + (p[-1] * u) ** 2)
    if nested:
        c0, c1, r1, c2, r2, _ = p
        return c0 + c1 * (1 - np.exp(-d / r1)) + c2 * (1 - np.exp(-d / r2))
    c0, c1, r, _ = p
    return c0 + c1 * (1 - np.exp(-d / r))


def sill_of(p, nested):
    return p[0] + p[1] + (p[3] if nested else 0.0)


def cov_nonug(p, h, u, nested):
    """공분산에서 너겟을 뺀 부분 = sill − γ(h,u) (γ(0) = c0 이므로 h = 0 에서 c1(+c2))."""
    return sill_of(p, nested) - gamma_model(p, h, u, nested)


def empirical_variogram(xyz, res, tdays, n_sub, seed=0, hmax=1000.0, umax=10.0):
    rng = np.random.default_rng(seed); i = rng.choice(len(xyz), min(n_sub, len(xyz)), replace=False)
    X, r, t = xyz[i], res[i], tdays[i]
    dot = np.clip(X @ X.T, -1.0, 1.0); h = gc_from_chord(np.sqrt(np.maximum(2.0 - 2.0 * dot, 0.0))); u = np.abs(t[:, None] - t[None, :]); g = 0.5 * (r[:, None] - r[None, :]) ** 2
    iu = np.triu_indices(len(X), 1); h, u, g = h[iu], u[iu], g[iu]
    hb = np.array([0, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 700, 1000]); ub = np.array([0, 0.5, 1, 2, 3, 5, 7, 10]); rows = []
    for a0, b0 in zip(hb[:-1], hb[1:]):
        for c0_, d0 in zip(ub[:-1], ub[1:]):
            m = (h >= a0) & (h < b0) & (u >= c0_) & (u < d0)
            if m.sum() >= 30:
                rows.append(((a0 + b0) / 2, (c0_ + d0) / 2, g[m].mean(), int(m.sum())))
    return pd.DataFrame(rows, columns=["h", "u", "gamma", "n"])


def fit_variogram(ev, nested):
    from scipy.optimize import least_squares
    w = np.sqrt(ev.n.to_numpy()); s = ev.gamma.max()
    if nested:
        p0 = [0.1 * s, 0.4 * s, 30, 0.5 * s, 400, 50]; lb = [0, 0, 1, 0, 50, 1]; ub = [s, 2 * s, 200, 2 * s, 5000, 1000]
    else:
        p0 = [0.1 * s, 0.9 * s, 100, 50]; lb = [0, 0, 1, 1]; ub = [s, 2 * s, 5000, 1000]
    f = lambda p: w * (gamma_model(p, ev.h.to_numpy(), ev.u.to_numpy(), nested) - ev.gamma.to_numpy())
    return least_squares(f, p0, bounds=(lb, ub)).x


def krige_batch(Xq, tq, Xn, tn, rn, p, nested):
    """배치 국소 단순 크리깅. Xq [B,3] · tq [B] · Xn [B,k,3] · tn [B,k] · rn [B,k] (잔차). 이웃 공분산 대각 = c0 + c1(+c2). 반환 [B] 잔차 예측."""
    B, k = tn.shape; sill = sill_of(p, nested)
    chord = np.sqrt(((Xn[:, :, None, :] - Xn[:, None, :, :]) ** 2).sum(-1)); h = gc_from_chord(chord); u = np.abs(tn[:, :, None] - tn[:, None, :])
    K = cov_nonug(p, h, u, nested) + (p[0] + 1e-6 * sill) * np.eye(k)[None]  # 대각 너겟 c0 + 수치 안정 지터
    h0 = gc_from_chord(np.sqrt(((Xn - Xq[:, None, :]) ** 2).sum(-1))); u0 = np.abs(tn - tq[:, None]); k0 = cov_nonug(p, h0, u0, nested)
    try:
        w = np.linalg.solve(K, k0[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        w = np.stack([np.linalg.lstsq(K[i], k0[i], rcond=None)[0] for i in range(B)])
    return (w * rn).sum(1)


def predict_local(tree, Xtr, ttr, rtr, Xq, tq, p, nested, k, a, batch=64, workers=4):
    """KD-tree [R·xyz, a·t] 4차원 k-최근접 → 배치 크리깅. 반환 (잔차 예측 [Q], dt_min_k [Q])."""
    out = np.empty(len(Xq)); dtm = np.empty(len(Xq))
    for s in range(0, len(Xq), batch):
        sl = slice(s, min(s + batch, len(Xq)))
        _, idx = tree.query(np.column_stack([R_EARTH * Xq[sl], a * tq[sl]]), k=k, workers=workers)
        out[sl] = krige_batch(Xq[sl], tq[sl], Xtr[idx], ttr[idx], rtr[idx], p, nested)
        dtm[sl] = np.abs(ttr[idx] - tq[sl][:, None]).min(1)
    return out, dtm


def dt_space_nearest(Xq, tq, Xtr, ttr, workers=4):
    """공간 최근접 훈련 사운딩까지 대원 거리(km)와 그 사운딩과의 시차(일)."""
    from scipy.spatial import cKDTree
    d, i = cKDTree(Xtr).query(Xq, k=1, workers=workers)
    return gc_from_chord(d), np.abs(ttr[i] - tq)


def selftest():
    """합성 자기 시험 (데이터 없음)."""
    from scipy.spatial import cKDTree
    rs = np.random.RandomState(0)
    # 1. 대원 거리: chord 공식 vs 표준 haversine
    la1, lo1, la2, lo2 = rs.uniform(-80, 80, 2000), rs.uniform(-180, 180, 2000), rs.uniform(-80, 80, 2000), rs.uniform(-180, 180, 2000)
    h1 = gc_from_chord(np.sqrt(((unit_xyz(la1, lo1) - unit_xyz(la2, lo2)) ** 2).sum(1)))
    a_ = np.sin(np.deg2rad(la2 - la1) / 2) ** 2 + np.cos(np.deg2rad(la1)) * np.cos(np.deg2rad(la2)) * np.sin(np.deg2rad(lo2 - lo1) / 2) ** 2; h2 = 2 * R_EARTH * np.arcsin(np.sqrt(a_))
    assert np.abs(h1 - h2).max() < 1e-6, np.abs(h1 - h2).max()
    # 2. 배치 풀이 = 이웃별 순진 구현
    p = np.array([0.3, 0.7, 150.0, 40.0]); nested = False; k = 6; B = 5
    Xn = unit_xyz(rs.uniform(30, 40, (B, k)).ravel(), rs.uniform(120, 130, (B, k)).ravel()).reshape(B, k, 3); tn = rs.uniform(0, 20, (B, k)); rn = rs.randn(B, k)
    Xq = unit_xyz(rs.uniform(30, 40, B), rs.uniform(120, 130, B)); tq = rs.uniform(0, 20, B); got = krige_batch(Xq, tq, Xn, tn, rn, p, nested); naive = np.zeros(B)
    for b in range(B):
        K = np.zeros((k, k)); k0 = np.zeros(k)
        for i in range(k):
            for j in range(k):
                hh = 2 * R_EARTH * np.arcsin(min(np.linalg.norm(Xn[b, i] - Xn[b, j]) / 2, 1.0)); K[i, j] = p[0] + p[1] - (p[0] + p[1] * (1 - np.exp(-np.sqrt(hh ** 2 + (p[3] * abs(tn[b, i] - tn[b, j])) ** 2) / p[2])))
            K[i, i] += p[0] + 1e-6 * (p[0] + p[1])
            h0 = 2 * R_EARTH * np.arcsin(min(np.linalg.norm(Xn[b, i] - Xq[b]) / 2, 1.0)); k0[i] = p[0] + p[1] - (p[0] + p[1] * (1 - np.exp(-np.sqrt(h0 ** 2 + (p[3] * abs(tn[b, i] - tq[b])) ** 2) / p[2])))
        naive[b] = np.linalg.solve(K, k0) @ rn[b]
    assert np.abs(got - naive).max() < 1e-9, np.abs(got - naive).max()
    # 3. 대각 너겟 효과: 질의가 훈련점과 같은 위치·시각이면 c0 = 0 에서 잔차를 거의 그대로, c0 가 크면 줄여서 예측
    Xs = unit_xyz([35.0, 35.5, 36.0, 36.5], [125.0, 125.2, 125.4, 125.6])[None]; ts = np.array([[1.0, 1.1, 1.2, 1.3]]); rs_ = np.array([[1.0, 0.5, -0.5, 0.2]])
    ratios = []
    for c0 in (0.0, 5.0):
        pp = np.array([c0, 1.0, 100.0, 10.0]); pr = krige_batch(Xs[:, 0], ts[:, 0], Xs, ts, rs_, pp, False)[0]; ratios.append(pr / rs_[0, 0])
    assert ratios[0] > 0.99 and ratios[1] < 0.6, ratios
    # 4. dt 두 열: 공간 최근접(시차 큰) vs 이웃 k 안 최소 시차
    Xtr = unit_xyz([35.0, 35.3, 40.0], [125.0, 125.0, 130.0]); ttr = np.array([10.0, 14.0, 14.1]); Xq1 = unit_xyz([35.01], [125.0]); tq1 = np.array([14.05])
    d_min, dts = dt_space_nearest(Xq1, tq1, Xtr, ttr); tree = cKDTree(np.column_stack([R_EARTH * Xtr, 5.0 * ttr])); _, idx = tree.query(np.column_stack([R_EARTH * Xq1, 5.0 * tq1]), k=3)
    dtm = np.abs(ttr[idx] - tq1[:, None]).min(1)
    assert abs(dts[0] - 4.05) < 1e-9 and abs(dtm[0] - 0.05) < 1e-9 and d_min[0] < 5.0, (dts, dtm, d_min)
    print(f"자기 시험 통과: 대원 거리 최대 |차| {np.abs(h1 - h2).max():.1e} km · 배치 풀이 대 순진 최대 |차| {np.abs(got - naive).max():.1e} · 너겟 효과 비 c0=0: {ratios[0]:.3f} c0=5: {ratios[1]:.3f} · dt_space_nearest {dts[0]:.2f} d · dt_min_k {dtm[0]:.2f} d")


if __name__ == "__main__":
    selftest()
