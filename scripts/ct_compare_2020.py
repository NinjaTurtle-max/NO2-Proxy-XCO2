"""12개월(2020) 동일 테스트 사운딩에서 CarbonTracker(rung 0) vs 우리 모델 비교 (남은 일 1번, 2026-09-22).

CT: co2 34층 air_mass 가중 컬럼 평균(OCO 평균핵 미적용) → (lat, lon, time) 3선형 보간. 스케일 편향은 **훈련 행에서만** 추정해 제거한 값도 병기.
비교: time fold 2 (우리 = lc2020_12m 물리, 결정 1 배경) · space fold 0 (우리 = train5_2020_space0 물리/절제).
출력: <out>/{summary.csv, monthly_time2.csv, strata_space0.csv} + 표준출력. 기본 <out> = experiments/ct_compare_2020_<생성일>, 이미 있으면 중단 (N-2: 덮어쓰기 없음).
평가 표본 규약 (D-3, 사용자 09-24; docs/decisiond3_gate_2026-09-24.md): 모든 표에 n 열. 같은 n 끼리만 비교.
  공통 = test ∧ CT 보간 유한 · 전체 = 폴드 test 전체 (train5 summary 의 test_clean_all 과 연결; --eval-months 없는 런은 test_clean 과 같음, QA X5)
열·행 형식 (기존 소비자 호환, QA Y1 제안 09-24 — 기존 이름의 의미 유지, 공통 값은 `_공통` 접미로 추가):
  summary.csv : CT_raw · CT_debiased(train) = 공통(CT 유한 행) · ours_* = 전체 · ours_*_공통 = 공통 · Δ = 공통. sample 열로 표본 명시
  monthly_time2.csv / strata_space0.csv : n · SD · {모델}_rmse/_bias(층별은 {모델}) = 기존 정의(CT 열은 CT 유한 행 = 공통, ours 열은 전체)
                                          + n_공통 · SD_공통(월별) · ours_*_rmse_공통 / ours_*_bias_공통 (층별 ours_*_공통)
"""
import argparse
import datetime
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from no2xco2.baselines.reanalysis import interp_ct  # rung 0 과 같은 CT 컬럼 평균·3선형 보간 (중복 제거, QA 2026-09-22)
from no2xco2.data import loader as L
from no2xco2.train import SLAB, STRATA, block_bootstrap, day_blocks, nested_masks  # 거리 층 정의 공유 (QA B5)

MONTHS = [f"2020{m:02d}" for m in range(1, 13)]


def stats(p, y, m):
    r = (p - y)[m]; ok = np.isfinite(r); r = r[ok]; v = np.var(y[m][ok])
    return dict(n=int(ok.sum()), rmse=float(np.sqrt((r ** 2).mean())), bias=float(r.mean()), r2=float(1 - (r ** 2).mean() / v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ct-dir", default="data/raw/ct_ea"); ap.add_argument("--out", default=f"experiments/ct_compare_2020_{datetime.date.today():%Y%m%d}")
    ap.add_argument("--idx-dir", default=L.DEFAULT_IDX)
    ap.add_argument("--time-pred", default="experiments/lc2020_12m/pred_time2_s0_phys.parquet")
    ap.add_argument("--space-pred", default="experiments/train5_2020_space0/pred_space0_s0_phys.parquet"); ap.add_argument("--space-pred-nophys", default="experiments/train5_2020_space0/pred_space0_s0_nophys.parquet")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "summary.csv")):  # N-2: 결과는 생성일 새 디렉토리, 덮어쓰기 없음
        raise FileExistsError(f"{a.out}/summary.csv 존재 — 다른 --out 을 지정")
    os.makedirs(a.out, exist_ok=True)
    dfs = [L.load_oco(m, a.idx_dir, columns=["row_idx", "latitude", "longitude", "label", "xco2", "time", "fold_time", "fold_space", "space_block", "year"]) for m in MONTHS]
    ct = np.concatenate([interp_ct(os.path.join(a.ct_dir, f"ct_ea_{m}.nc"), d.latitude.to_numpy(), d.longitude.to_numpy(), pd.to_datetime(d.time).values) for m, d in zip(MONTHS, dfs)])
    big = pd.concat(dfs, ignore_index=True); y = big.xco2.to_numpy(); lab = (big.label == "L_train").to_numpy(); mon = pd.to_datetime(big.time).dt.month.to_numpy()
    days = day_blocks(big.time)
    print(f"사운딩 {len(big):,} · CT 보간 결측 {np.isnan(ct).mean():.2%}")
    rows = []
    for scheme, fold, preds in (("time", 2, {"ours_phys": a.time_pred}), ("space", 0, {"ours_phys": a.space_pred, "ours_nophys": a.space_pred_nophys})):
        tr, val, te = nested_masks(big, scheme, fold, lab, 1)
        com = te & np.isfinite(ct)  # D-3 공통 표본
        b_tr = float(np.nanmean((ct - y)[tr]))  # CT 스케일 편향: 훈련 행에서만
        cand = {"CT_raw": ct, "CT_debiased(train)": ct - b_tr}
        for k, f in preds.items():
            if not os.path.exists(f):  # 없는 예측을 조용히 건너뛰면 우리·Δ 행 없는 표가 정상 종료됨 (QA X6)
                raise FileNotFoundError(f"{k} 예측 없음: {f}")
            p = pd.read_parquet(f)
            assert len(p) == len(big) and (p.row_idx.to_numpy() == big.row_idx.to_numpy()).all(), f"{f}: row_idx 불일치"
            assert np.allclose(p.xco2.to_numpy(), y), f"{f}: xco2 불일치 — 다른 인덱스로 만든 예측"
            cand[k] = p.pred.to_numpy()
        print(f"\n== {scheme} fold {fold}: test {te.sum():,} (공통 {com.sum():,}) · test SD {y[te].std():.3f} · CT 훈련행 편향 {b_tr:+.3f}")
        for k, p in cand.items():  # 기존 행 이름 = 기존 의미 (CT = 유한 행 = 공통, ours = 전체); ours 공통은 `_공통` 행 추가 (QA Y1)
            specs = ((k, "전체", te), (f"{k}_공통", "공통", com)) if k.startswith("ours") else ((k, "공통", com),)
            for name, smp, m in specs:
                s = stats(p, y, m); s.update(scheme=scheme, fold=fold, model=name, sample=smp); rows.append(s)
                print(f"  {name:22s} [{smp}] n {s['n']:,} RMSE {s['rmse']:.3f} bias {s['bias']:+.3f} R² {s['r2']:.3f}")
        if "ours_phys" in cand:
            ra = (cand["ours_phys"] - y)[te]; rb = (cand["CT_debiased(train)"] - y)[te]; ok = np.isfinite(ra) & np.isfinite(rb)
            d = np.sqrt((rb[ok] ** 2).mean()) - np.sqrt((ra[ok] ** 2).mean()); lo, hi = block_bootstrap(ra[ok], rb[ok], days[te][ok])
            print(f"  ΔRMSE(CT_deb − 우리 물리) = {d:+.3f} ppm, 일블록 95% CI [{lo:+.3f}, {hi:+.3f}]  (|Δ| ≤ 0.42 = B3 동등 기준)")
            rows.append(dict(scheme=scheme, fold=fold, model="Δ(CT_deb−ours)", sample="공통", n=int(ok.sum()), delta=d, ci_lo=lo, ci_hi=hi))  # train5 summary 와 같은 열 이름 (QA 2026-09-22: 이전엔 lo·hi 가 bias·r2 열에 들어갔음)
        if scheme == "time":
            mrows = []
            for m in range(1, 13):
                sel = te & (mon == m); sc = com & (mon == m); r = {"월": m, "n": int(sel.sum()), "SD": float(y[sel].std()), "n_공통": int(sc.sum()), "SD_공통": float(y[sc].std())}
                for k, p in cand.items():  # 기존 열 = 기존 정의 (CT 는 유한 행, ours 는 전체); ours 공통 열 추가 (D-3, QA Y1)
                    s = stats(p, y, sel); r[f"{k}_rmse"] = s["rmse"]; r[f"{k}_bias"] = s["bias"]
                    if k.startswith("ours"):
                        s = stats(p, y, sc); r[f"{k}_rmse_공통"] = s["rmse"]; r[f"{k}_bias_공통"] = s["bias"]
                mrows.append(r)
            mt = pd.DataFrame(mrows); mt.to_csv(f"{a.out}/monthly_time2.csv", index=False)
            pd.set_option("display.width", 220); print(mt.round(2).to_string(index=False))
        else:
            tree = BallTree(np.deg2rad(big.loc[tr, ["latitude", "longitude"]].to_numpy()), metric="haversine")
            dkm = tree.query(np.deg2rad(big.loc[te, ["latitude", "longitude"]].to_numpy()), k=1)[0][:, 0] * 6371
            srows = []
            for a0, a1, lb in zip(STRATA[:-1], STRATA[1:], SLAB):
                m = (dkm >= a0) & (dkm < a1); mc = m & np.isfinite(ct[te])
                if m.sum() == 0: continue
                r = {"층 km": lb, "n": int(m.sum()), "n_공통": int(mc.sum())}
                for k, p in cand.items():  # 기존 열 = 기존 정의 (CT 는 유한 행, ours 는 전체); ours 공통 열 추가 (D-3, QA Y1)
                    rr = (p - y)[te]; r[k] = float(np.sqrt(np.nanmean(rr[m] ** 2)))
                    if k.startswith("ours"):
                        r[f"{k}_공통"] = float(np.sqrt(np.nanmean(rr[mc] ** 2)))
                srows.append(r)
            st = pd.DataFrame(srows); st.to_csv(f"{a.out}/strata_space0.csv", index=False); print(st.round(3).to_string(index=False))
    pd.DataFrame(rows).to_csv(f"{a.out}/summary.csv", index=False); print(f"\n→ {a.out}/summary.csv")


if __name__ == "__main__":
    main()
