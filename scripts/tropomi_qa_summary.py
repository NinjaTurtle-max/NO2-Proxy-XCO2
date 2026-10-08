"""_tropomi_ea_v2 전수 qa 스캔(granule CSV) → 연도·처리형·버전별 집계표.

입력: data/processed/tropomi_qa_scan.csv (scripts 외부 qa_scan 이 만든 granule 단위 파일)
출력: docs/results/tropomi_qa_by_{year,procver,yearproc}_<생성일>.csv + 표준출력 표. 같은 이름이 있으면 중단 (N-2 생성일 규약, 덮어쓰기 없음)
집계는 화소 가중(합계 후 비율). 중앙값 열은 granule 중앙값의 중앙값(화소 가중 아님).
"""
import argparse
import datetime
import os

import numpy as np
import pandas as pd

BINS = ["qa_0.00_0.50", "qa_0.50_0.66", "qa_0.66_0.70", "qa_0.70_0.75", "qa_0.75_0.80", "qa_0.80_0.90", "qa_0.90_0.95", "qa_0.95_1.00"]


def agg(g: pd.DataFrame) -> pd.Series:
    npx = g.n_pix.sum(); hi = g.qa_ge075.sum()
    w = g.qa_ge075.fillna(0).to_numpy(); neg = g.no2_neg_frac_hi.fillna(0).to_numpy()
    out = {
        "granules": len(g), "empty": int((g.n_pix == 0).sum()), "pix_M": npx / 1e6, "qa_nan": int(g.qa_nan.fillna(0).sum()),
        "pix_ge075_M": hi / 1e6, "frac_ge075": hi / npx if npx else np.nan,
        "frac_0.66_0.75": (g["qa_0.66_0.70"].fillna(0).sum() + g["qa_0.70_0.75"].fillna(0).sum()) / npx if npx else np.nan,
        "frac_ge095": g["qa_0.95_1.00"].fillna(0).sum() / npx if npx else np.nan,
        "no2_med_hi_medG": g.no2_med_hi.median(), "neg_frac_hi_w": (w * neg).sum() / w.sum() if w.sum() else np.nan,
        "cf_med_hi_medG": g.cf_med_hi.median(), "date_min": g.date.min(), "date_max": g.date.max(),
    }
    for b in BINS:
        out[b] = g[b].fillna(0).sum() / npx if npx else np.nan
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed/tropomi_qa_scan.csv")
    ap.add_argument("--out", default="docs/results")
    a = ap.parse_args()
    d = pd.read_csv(a.inp, dtype={"date": str})
    d["year"] = d.date.str[:4]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    tabs = {"year": d.groupby("year").apply(agg), "procver": d.groupby(["proc", "ver"]).apply(agg), "yearproc": d.groupby(["year", "proc"]).apply(agg)}
    day = f"{datetime.date.today():%Y-%m-%d}"
    outs = {k: os.path.join(a.out, f"tropomi_qa_by_{k}_{day}.csv") for k in tabs}
    if any(os.path.exists(p) for p in outs.values()):  # N-2: 같은 날 재생성은 --out 을 바꿔서
        raise FileExistsError(f"이미 있음: {[p for p in outs.values() if os.path.exists(p)]}")
    for k, t in tabs.items():
        t.to_csv(outs[k])
        print(f"\n## by {k}\n", t.drop(columns=BINS).to_string())
    print("\n## 전체 qa 구간 분포 (화소 비율)\n", agg(d)[BINS].to_string())
    dup = d[d.n_pix > 0].groupby("orbit").proc.nunique()
    print(f"\n동일 orbit 이 RPRO·OFFL 양쪽에 있는 수: {(dup > 1).sum()} / orbit {len(dup)}")
    print("qa_min 분포:", d.qa_min.dropna().round(2).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
