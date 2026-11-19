"""인과 NO₂ 사운딩 피처 (LightGBM 기준선 입력). 학습 없음.

  - 화소: build_index 산출 trop_hi_YYYYMM.parquet (qa ≥ 0.75; 시각 step_g · no2 · 4노드 n0..n3 · 이중선형 가중 w0..w3) — 모델 주입과 같은 4노드 가중.
  - 노드 값: 창 안 전 화소의 가중 평균 Σw·no2 / Σw (그 노드로 가는 가중이 없으면 결측).
  - 사운딩 값: 4노드 (n_k, w_k) 중 값이 있는 노드로 가중 재정규화 Σ w_k·m_k / Σ w_k (4노드 모두 결측이면 결측, nn_* = 값 있는 노드 수).
  - 인과 창: 사운딩 시각 t(시) 까지. no2_24h = 화소 시각 s ∈ [t−23, t], no2_30d = s ∈ [t−719, t]. 같은 시의 화소는 포함, t 이후 화소는 쓰지 않는다.
    창이 2020-01-01 00 UTC 앞으로 넘어가면 있는 구간만 평균한다.
구현: 시 단위 희소 기여를 720 시간 보관하는 슬라이딩 창 누적 (달은 연속이어야 함). 달 끝마다 보관분으로 누적을 다시 합산한다.
산출: <out>/no2_features.parquet (row_idx, ym, step_g, no2_24h, no2_30d, nn_24h, nn_30d) · no2_features_summary.csv (달별 비결측 비율 등).
사용: python scripts/data/build_no2_features.py --months 202001-202412 --out runs/no2_features   (자기 시험: --selftest)
"""
import argparse
import calendar
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

N = 121 * 201  # 잠재 격자 노드 수
W24, W30 = 24, 720  # 창 길이 (시간)
EPS = 1e-9  # 창 안 노드 가중 합이 이 값 이하면 결측
T0 = pd.Timestamp("2020-01-01")


def month_h0(ym):
    return int((pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01") - T0) / pd.Timedelta(hours=1))


def month_hours(ym):
    return calendar.monthrange(int(ym[:4]), int(ym[4:]))[1] * 24


def parse_months(s):
    a, b = s.split("-"); out = []; y, m = int(a[:4]), int(a[4:])
    while f"{y}{m:02d}" <= b:
        out.append(f"{y}{m:02d}"); m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


class CausalNO2:
    """시 단위 슬라이딩 창 누적. process_month 를 달 순서대로 부른다."""

    def __init__(self):
        self.num = {W24: np.zeros(N), W30: np.zeros(N)}
        self.den = {W24: np.zeros(N), W30: np.zeros(N)}
        self.store = {}  # 시(step) → (노드 idx, Σw·no2, Σw)
        self.next_h = None

    def _apply(self, c, sign, wins):
        idx, nm, dn = c
        for w in wins:
            self.num[w][idx] += sign * nm; self.den[w][idx] += sign * dn

    def resync(self, h):
        """보관분에서 [h−719, h] · [h−23, h] 누적을 정확히 다시 합산, 이전 누적과의 최대 차이를 반환하고 교체."""
        num = {W24: np.zeros(N), W30: np.zeros(N)}; den = {W24: np.zeros(N), W30: np.zeros(N)}
        for s, (idx, nm, dn) in self.store.items():
            for w in (W24, W30):
                if h - w < s <= h:
                    num[w][idx] += nm; den[w][idx] += dn
        drift = max(max(np.abs(num[w] - self.num[w]).max(), np.abs(den[w] - self.den[w]).max()) for w in (W24, W30))
        self.num, self.den = num, den
        return float(drift)

    def sample(self, win, nodes, wts):
        """nodes [S,4] · wts [S,4] → (값 [S], 값 있는 노드 수 [S])."""
        d = self.den[win][nodes]; ok = d > EPS
        m = np.where(ok, self.num[win][nodes] / np.where(ok, d, 1.0), 0.0)
        sw = (wts * ok).sum(1)
        val = np.where(sw > 0, (wts * ok * m).sum(1) / np.where(sw > 0, sw, 1.0), np.nan)
        return val, ok.sum(1)

    def process_month(self, ym, px, snd):
        """px: DataFrame(step_g, qa, no2, n0..n3, w0..w3) · snd: DataFrame(step_g, n0..n3, w0..w3). 반환 (no2_24h, no2_30d, nn_24h, nn_30d) — snd 행 순서."""
        h0, T = month_h0(ym), month_hours(ym)
        assert self.next_h is None or self.next_h == h0, f"달이 연속이 아님: 기대 시 {self.next_h}, {ym} 시작 {h0}"
        po = np.argsort(px.step_g.to_numpy(), kind="stable"); sg = px.step_g.to_numpy()[po]
        assert len(sg) == 0 or (sg.min() >= h0 and sg.max() < h0 + T), f"{ym}: 화소 step_g 가 달 범위 밖"
        pn = np.stack([px[f"n{k}"].to_numpy()[po] for k in range(4)], 1); pw = np.stack([px[f"w{k}"].to_numpy(np.float64)[po] for k in range(4)], 1)
        pv = px.no2.to_numpy(np.float64)[po]
        pb = np.searchsorted(sg, np.arange(h0, h0 + T + 1))
        so = np.argsort(snd.step_g.to_numpy(), kind="stable"); ss = snd.step_g.to_numpy()[so]
        assert len(ss) == 0 or (ss.min() >= h0 and ss.max() < h0 + T), f"{ym}: 사운딩 step_g 가 달 범위 밖"
        sn = np.stack([snd[f"n{k}"].to_numpy()[so] for k in range(4)], 1); sw_ = np.stack([snd[f"w{k}"].to_numpy(np.float64)[so] for k in range(4)], 1)
        sb = np.searchsorted(ss, np.arange(h0, h0 + T + 1))
        out = {k: np.full(len(snd), np.nan) for k in ("v24", "v30")}; nn = {k: np.zeros(len(snd), np.int8) for k in ("n24", "n30")}
        for i in range(T):
            h = h0 + i; a, b = pb[i], pb[i + 1]
            if b > a:
                nodes = pn[a:b].ravel(); ww = pw[a:b].ravel(); val = np.repeat(pv[a:b], 4)
                dn = np.bincount(nodes, weights=ww, minlength=N); nm = np.bincount(nodes, weights=ww * val, minlength=N)
                idx = np.flatnonzero(dn); c = (idx.astype(np.int32), nm[idx], dn[idx]); self.store[h] = c; self._apply(c, +1.0, (W24, W30))
            if (h - W24) in self.store:
                self._apply(self.store[h - W24], -1.0, (W24,))
            if (h - W30) in self.store:
                self._apply(self.store.pop(h - W30), -1.0, (W30,))
            c0, c1 = sb[i], sb[i + 1]
            if c1 > c0:
                v24, k24 = self.sample(W24, sn[c0:c1], sw_[c0:c1]); v30, k30 = self.sample(W30, sn[c0:c1], sw_[c0:c1])
                rows = so[c0:c1]; out["v24"][rows] = v24; out["v30"][rows] = v30; nn["n24"][rows] = k24; nn["n30"][rows] = k30
        self.next_h = h0 + T
        drift = self.resync(h0 + T - 1)
        return out["v24"], out["v30"], nn["n24"], nn["n30"], drift


def selftest():
    """합성 화소 · 사운딩으로 슬라이딩 창 결과를 정의대로 전수 계산한 값과 비교 (창 경계 · 달 경계 · 시작 앞 잘림 · 부분 결측 노드)."""
    rs = np.random.RandomState(1); months = ["202001", "202002"]; TT = month_hours("202001") + month_hours("202002")
    P = 3000; nodes_pool = np.arange(500, 520)  # 작은 노드 풀 → 한 노드에 여러 화소·시간이 겹침
    pxs = pd.DataFrame(dict(step_g=rs.randint(0, TT, P).astype(np.int32), qa=np.float32(0.9), no2=rs.rand(P).astype(np.float32) * 1e-4))
    for k in range(4):
        pxs[f"n{k}"] = rs.choice(nodes_pool, P).astype(np.int32)
    wr = rs.rand(P, 4) + 0.01; wr /= wr.sum(1, keepdims=True)
    for k in range(4):
        pxs[f"w{k}"] = wr[:, k].astype(np.float32)
    # 경계 시험용 고정 화소: 노드 700 에 사운딩 t=800 기준 h−24, h−23, h−720, h−719, h, h+1 시 화소
    t = 800; fixed = [(t - 24, 1.0), (t - 23, 2.0), (t - 720, 4.0), (t - 719, 8.0), (t, 16.0), (t + 1, 32.0)]
    extra = pd.DataFrame(dict(step_g=[np.int32(s) for s, _ in fixed], qa=np.float32(0.9), no2=[np.float32(v) for _, v in fixed], n0=np.int32(700), n1=np.int32(700), n2=np.int32(700), n3=np.int32(700),
                              w0=np.float32(1.0), w1=np.float32(0.0), w2=np.float32(0.0), w3=np.float32(0.0)))
    pxs = pd.concat([pxs, extra], ignore_index=True)
    S = 400; sn = rs.choice(np.r_[nodes_pool, [700, 900]], (S, 4)).astype(np.int32); sw = rs.rand(S, 4) + 0.01; sw /= sw.sum(1, keepdims=True)
    snd = pd.DataFrame(dict(step_g=rs.randint(0, TT, S).astype(np.int32)))
    snd.loc[0, "step_g"] = t; sn[0] = [700, 900, 900, 900]; sw[0] = [1.0, 0.0, 0.0, 0.0]  # 경계 시험 사운딩 (노드 700 만 가중 1)
    snd.loc[1, "step_g"] = 5  # 시작 앞 잘림 (t < 23)
    for k in range(4):
        snd[f"n{k}"] = sn[:, k]; snd[f"w{k}"] = sw[:, k].astype(np.float32)
    # 전수 기준값
    spx_n = np.stack([pxs[f"n{k}"] for k in range(4)], 1); spx_w = np.stack([pxs[f"w{k}"].to_numpy(np.float64) for k in range(4)], 1); spx_v = pxs.no2.to_numpy(np.float64); spx_s = pxs.step_g.to_numpy()
    def brute(ti, node4, w4, win):
        sel = (spx_s >= ti - win + 1) & (spx_s <= ti); num = {}; den = {}
        for p in np.flatnonzero(sel):
            for k in range(4):
                num[spx_n[p, k]] = num.get(spx_n[p, k], 0.0) + spx_w[p, k] * spx_v[p]; den[spx_n[p, k]] = den.get(spx_n[p, k], 0.0) + spx_w[p, k]
        ws = 0.0; acc = 0.0; cnt = 0
        for k in range(4):
            d = den.get(node4[k], 0.0)
            if d > EPS:
                ws += w4[k]; acc += w4[k] * num[node4[k]] / d; cnt += 1
        return (acc / ws if ws > 0 else np.nan), cnt
    st = CausalNO2(); res = []
    for ym in months:
        h0, T = month_h0(ym), month_hours(ym)
        pm = pxs[(pxs.step_g >= h0) & (pxs.step_g < h0 + T)]; sm = snd[(snd.step_g >= h0) & (snd.step_g < h0 + T)]
        v24, v30, n24, n30, drift = st.process_month(ym, pm, sm); res.append((sm, v24, v30, n24, n30, drift))
    maxdiff = 0.0; checked = 0; bad = []
    for sm, v24, v30, n24, n30, drift in res:
        for j in range(len(sm)):
            ti = int(sm.step_g.iloc[j]); node4 = [int(sm[f"n{k}"].iloc[j]) for k in range(4)]; w4 = [float(sm[f"w{k}"].iloc[j]) for k in range(4)]
            for win, v, n in ((W24, v24, n24), (W30, v30, n30)):
                bv, bc = brute(ti, node4, w4, win); checked += 1
                if np.isnan(bv) != np.isnan(v[j]) or bc != n[j] or (not np.isnan(bv) and abs(bv - v[j]) > 1e-9 * max(1.0, abs(bv))):
                    bad.append((ti, win, bv, v[j], bc, n[j]))
                elif not np.isnan(bv):
                    maxdiff = max(maxdiff, abs(bv - v[j]))
    # 경계 사운딩 값: 노드 700 에서 24 h 창 = {t−23: 2, t: 16} → 평균 9 ; 30 d 창 = {t−719: 8, t−23: 2, t: 16} (t−24: 1 도 포함, t−720 제외, t+1 제외) → (1+2+8+16)/4 = 6.75
    sm0, v24, v30, n24, n30, _ = res[1]  # t = 800 은 2월 (1월 말 화소가 창에 걸치는 달 경계 시험)
    # (고정 화소 외에 무작위 화소가 노드 700 에 없으므로 노드 700 의 값은 고정 화소만 반영)
    j00 = [j for j in range(len(sm0)) if int(sm0.step_g.iloc[j]) == t and int(sm0.n0.iloc[j]) == 700][0]
    e24, e30 = (2.0 + 16.0) / 2, (1.0 + 2.0 + 8.0 + 16.0) / 4
    assert abs(v24[j00] - e24) < 1e-9 and abs(v30[j00] - e30) < 1e-9, f"경계 시험 실패: 24h {v24[j00]} (기대 {e24}) 30d {v30[j00]} (기대 {e30})"
    trunc = res[0][0].step_g.to_numpy() == 5; assert trunc.any()
    print(f"자기 시험 통과: 전수 대조 {checked}건 불일치 {len(bad)} · 최대 |차| {maxdiff:.2e} · 월 끝 누적 드리프트 {[f'{r[5]:.2e}' for r in res]} · 경계(24 h: h−23 포함 · h−24 제외 / 30 d: h−719 포함 · h−720 제외 · 같은 시 포함 · 다음 시 제외) 기대값 {e24} · {e30} 일치 · 시작 앞 잘림 사운딩 {int(trunc.sum())}개 처리")
    assert not bad, bad[:5]


def main():
    ap = argparse.ArgumentParser(description="인과 NO₂ 사운딩 피처")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--out", default="runs/no2_features")
    ap.add_argument("--idx-dir", default="data/processed/index"); ap.add_argument("--months", default="202001-202412")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    months = parse_months(a.months); outp = os.path.join(a.out, "no2_features.parquet")
    assert months[0] == "202001", "첫 달은 202001 (창이 비어 시작한다)"
    t0 = time.time(); st = CausalNO2(); parts = []; rows = []
    for ym in months:
        t1 = time.time()
        px = pq.read_table(os.path.join(a.idx_dir, f"trop_hi_{ym}.parquet")).to_pandas()
        snd = pq.read_table(os.path.join(a.idx_dir, f"oco_{ym}.parquet"), columns=["row_idx", "step_g", "label", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3"]).to_pandas()
        assert float(px.qa.min()) >= 0.75, f"{ym}: trop_hi 에 qa < 0.75 화소"
        v24, v30, n24, n30, drift = st.process_month(ym, px, snd)
        parts.append(pd.DataFrame(dict(row_idx=snd.row_idx.to_numpy(), ym=ym, step_g=snd.step_g.to_numpy(), no2_24h=v24.astype(np.float32), no2_30d=v30.astype(np.float32), nn_24h=n24, nn_30d=n30)))
        lt = (snd.label == "L_train").to_numpy()
        rows.append(dict(ym=ym, n_px=len(px), n_snd=len(snd), n_Ltrain=int(lt.sum()), nonmiss_24h=float(np.isfinite(v24).mean()), nonmiss_30d=float(np.isfinite(v30).mean()),
                         nonmiss_24h_Ltrain=float(np.isfinite(v24[lt]).mean()), nonmiss_30d_Ltrain=float(np.isfinite(v30[lt]).mean()), drift=drift, secs=time.time() - t1))
        r = rows[-1]; print(f"{ym}: 화소 {r['n_px']:,} · 사운딩 {r['n_snd']:,} · 비결측 24h {r['nonmiss_24h']:.3f} 30d {r['nonmiss_30d']:.3f} · 드리프트 {drift:.1e} · {r['secs']:.0f}s", flush=True)
        del px, snd
    full = pd.concat(parts, ignore_index=True)
    assert not full.duplicated(["ym", "row_idx"]).any(), "(ym, row_idx) 중복"
    os.makedirs(a.out, exist_ok=True); full.to_parquet(outp, compression="zstd", index=False)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "no2_features_summary.csv"), index=False)
    print(f"전체 {len(full):,}행 · 비결측 24h {np.isfinite(full.no2_24h).mean():.4f} 30d {np.isfinite(full.no2_30d).mean():.4f} · {time.time() - t0:.0f}s → {outp}", flush=True)


if __name__ == "__main__":
    main()
