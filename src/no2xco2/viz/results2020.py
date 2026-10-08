"""2020년 결과 그림 (설계 승인 2026-09-22): 거리 층별 RMSE 막대 · 월별 편향/RMSE 선 · 학습 곡선 · 잔차 지도(0.25° 격자) · 결정 1 전후 월별 R².

입력은 experiments/* 산출물만 읽는다. 층 계산(2020-01 3시드)은 scripts/ct_compare_2020.py 와 같은 정의:
최근접 **훈련** 사운딩(nested_masks(space, 0, vfold 1) 의 tr) 까지 haversine 거리, 층 경계 STRATA.
"""
import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from no2xco2.config import GRID_D, GRID_LAT0, GRID_LON0

# 격자점 수 = 20–50°N · 100–150°E / 0.25° + 1 (era5.NLAT·encoder_index.NLON 과 같은 값 — test_viz 가 대조). 상수만 쓰려고 torch·xarray 모듈을 적재하지 않는다 (QA W9)
NLAT, NLON = int(round(30 / GRID_D)) + 1, int(round(50 / GRID_D)) + 1


def _L():
    from no2xco2.data import loader  # torch 적재 — 인덱스를 읽는 함수에서만
    return loader


def _train():
    import no2xco2.train as T  # 층 경계 STRATA·SLAB 은 train 한 곳 (QA B5) · nested_masks
    return T
from no2xco2.viz import C_CT, C_DIV_MID, C_DIV_NEG, C_DIV_POS, C_GRID, C_MUTED, C_OURS, C_TEXT2, COLOR, LABEL, save, styled

R_EARTH_KM = 6371.0


# ---------------------------------------------------------------- ct_compare 출력 형식 (QA Y1)
def ct_format(df: pd.DataFrame) -> str:
    """ct_compare 월별·층별 CSV 형식 (AI 세션 합의 09-24 02:5x 판).
    "d3": 기존 열 의미 유지(n·SD·ours_* = 테스트 전체) + 공통 표본(테스트 ∧ CT 유한) 열 n_공통·SD_공통·ours_*_공통 추가. CT 열은 두 판 모두 CT 유한 행.
    "old": 구 판 (n·SD·모델 열, 공통 열 없음)."""
    return "d3" if "n_공통" in df.columns else "old"


def ct_sample_note(csv_path: str, scheme: str, fold: int) -> str:
    """구 형식 CSV 는 CT 열(CT 보간 있는 행)과 우리 열(테스트 전체)의 표본이 다르다 (QA Z1). 같은 디렉토리 summary 에서 n 을 읽어 문구로.
    summary 가 없으면 n 없이 표기."""
    f = os.path.join(os.path.dirname(csv_path), "summary.csv")
    try:
        sm = pd.read_csv(f); n_ct = int(ct_summary_row(sm, "CT_debiased(train)", scheme, fold, sample="공통").n); n_us = int(ct_summary_row(sm, "ours_phys", scheme, fold).n)
        return f"표본 다름: CT = CT 보간 있는 테스트 행 (n {n_ct:,}) · 우리 = 테스트 전체 (n {n_us:,})"
    except (OSError, ValueError, KeyError):
        return "표본 다름: CT = CT 보간 있는 테스트 행 · 우리 = 테스트 전체"


def ct_summary_row(ct: pd.DataFrame, model: str, scheme: str, fold: int, sample: str = "전체") -> pd.Series:
    """ct_compare summary 에서 한 행. D-3 판: ours_* 행 = 전체, ours_*_공통 행 = 공통, CT 행 = 공통 (sample 열로 확인)."""
    m = (ct.model == model) & (ct.scheme == scheme) & (ct.fold == fold)
    if "sample" in ct.columns and m.any():
        m &= ct["sample"] == sample
    r = ct[m]
    if len(r) != 1:
        raise ValueError(f"ct summary {model} {scheme}:{fold} sample={sample}: 행 {len(r)}개")
    return r.iloc[0]


# ---------------------------------------------------------------- 1. 거리 층별 RMSE (space fold 0)
def strata_202001_3seed(idx_dir: str, exp_dirs: list[str], seeds=(0, 1, 2), fold: int = 0, vfold: int = 1) -> pd.DataFrame:
    """2020-01 space fold 0 테스트 사운딩을 최근접 훈련 사운딩 거리 층으로 나눠 시드·모델별 RMSE. 행: 층·n·model·seed·rmse (+ '전체' 층)."""
    T = _train(); STRATA, SLAB, nested_masks = T.STRATA, T.SLAB, T.nested_masks
    df = _L().load_oco("202001", idx_dir, columns=["row_idx", "latitude", "longitude", "label", "xco2", "time", "fold_time", "fold_space", "space_block", "year"])
    lab = (df.label == "L_train").to_numpy(); y = df.xco2.to_numpy()
    tr, _, te = nested_masks(df, "space", fold, lab, vfold)
    tree = BallTree(np.deg2rad(df.loc[tr, ["latitude", "longitude"]].to_numpy()), metric="haversine")
    dkm = tree.query(np.deg2rad(df.loc[te, ["latitude", "longitude"]].to_numpy()), k=1)[0][:, 0] * R_EARTH_KM
    rows = []
    for s in seeds:
        for model in ("phys", "nophys"):
            name = f"pred_space{fold}_s{s}_{model}.parquet"; cands = [os.path.join(d, name) for d in exp_dirs if os.path.exists(os.path.join(d, name))]
            if len(cands) != 1:
                raise FileNotFoundError(f"{name}: {exp_dirs} 에서 {len(cands)}개 발견 (정확히 1개여야 함 — 누락은 SD 를 미측정으로, 중복은 출처 불명으로 만든다)")
            p = pd.read_parquet(cands[0])
            if len(p) != len(df) or not (p.row_idx.to_numpy() == df.row_idx.to_numpy()).all():  # 길이 먼저 (QA W6)
                raise ValueError(f"row_idx 순서 불일치: {cands[0]}")
            if not (p.test.to_numpy() == te).all():
                raise ValueError(f"test 마스크 불일치: {cands[0]}")
            r = (p.pred.to_numpy() - y)[te]; key = f"ours_{model}"
            fin = np.isfinite(r)  # n 과 RMSE 를 같은 행(유한 잔차)으로 (QA Y8)
            for a0, a1, lb in zip(STRATA[:-1], STRATA[1:], SLAB):
                m = (dkm >= a0) & (dkm < a1) & fin
                if m.sum() == 0:
                    continue
                rows.append({"층 km": lb, "n": int(m.sum()), "model": key, "seed": s, "rmse": float(np.sqrt(np.nanmean(r[m] ** 2))), "src": cands[0]})
            rows.append({"층 km": "전체", "n": int(fin.sum()), "model": key, "seed": s, "rmse": float(np.sqrt(np.nanmean(r ** 2))), "src": cands[0]})
    return pd.DataFrame(rows)


def _bars(ax, labels, series: dict, err: dict | None = None, width=0.26):
    """series: {key: values[len(labels)]}. 막대 사이 2 px 급 간격은 width 합 < 1 로 확보."""
    x = np.arange(len(labels)); k = len(series); off = (np.arange(k) - (k - 1) / 2) * width
    for i, (key, v) in enumerate(series.items()):
        e = err.get(key) if err else None
        ax.bar(x + off[i], v, width * 0.92, color=COLOR[key], label=LABEL[key], yerr=e, error_kw=dict(ecolor=C_TEXT2, capsize=3, lw=1))
        top = v if e is None else v + np.nan_to_num(e)
        for xi, vi, ti in zip(x + off[i], v, top):
            if np.isfinite(vi):
                ax.text(xi, ti + 0.05, f"{vi:.2f}", ha="center", va="bottom", fontsize=7, color=C_TEXT2)  # 오차막대 캡 위
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, None)


@styled
def plot_strata(ct_csv: str, seed_df: pd.DataFrame, date: str, out_dir: str | None = None) -> tuple[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw=dict(width_ratios=[3, 3]))
    # (a) 2020 12개월 — ct_compare 산출 그대로
    ct = pd.read_csv(ct_csv); keys = [k for k in ("CT_debiased(train)", "ours_phys", "ours_nophys") if k in ct.columns]
    if ct_format(ct) == "d3":  # 막대 = 공통 표본(CT 보간 있는 행)에서 비교: CT 열 + ours_*_공통. 라벨 n 공통 (전체)
        xl = [f"{l}\nn {nc:,} ({na:,})" for l, nc, na in zip(ct["층 km"], ct["n_공통"], ct["n"])]; smp = " · 공통 표본 (괄호 = 전체)"
        col = {k: (f"{k}_공통" if k.startswith("ours") else k) for k in keys}
    else:
        xl = [f"{l}\nn {n:,}" for l, n in zip(ct["층 km"], ct["n"])]; col = {k: k for k in keys}
        smp = ""; axes[0].text(0.5, -0.30, ct_sample_note(ct_csv, "space", 0) + " · 층 라벨 n = 테스트 전체", transform=axes[0].transAxes, ha="center", va="top", fontsize=6.5, color=C_TEXT2)  # QA Z1
    _bars(axes[0], xl, {k: ct[col[k]].to_numpy() for k in keys})
    axes[0].set_title("(a) 2020 12개월 · space fold 0 · seed 0 (배경 9변수) · 층 = 훈련 12개월 기준" + ("\n" + smp.strip(" ·") if smp else "") + ("" if "ours_nophys" in keys else "  [절제: 미측정]"), fontsize=9)
    axes[0].set_ylabel("테스트 RMSE (ppm)"); axes[0].set_xlabel("최근접 훈련 사운딩 거리 (km)")
    # (b) 2020-01 3시드 평균 ± SD
    g = seed_df[seed_df["층 km"] != "전체"].groupby(["층 km", "model"]).agg(mean=("rmse", "mean"), sd=("rmse", "std"), k=("seed", "nunique")).reset_index()
    labs = [l for l in _train().SLAB if l in set(g["층 km"])]
    k = int(g.k.max()); series, err = {}, {}
    for key in ("ours_phys", "ours_nophys"):
        sub = g[g.model == key].set_index("층 km").reindex(labs); series[key] = sub["mean"].to_numpy(); err[key] = sub["sd"].to_numpy()
    nr = seed_df[seed_df["층 km"] != "전체"].groupby("층 km").n.agg(["min", "max"]).reindex(labs)  # 모델·시드 전체에서 (QA Z2)
    nlab = [f"{int(a):,}" if a == b else f"{int(a):,}–{int(b):,}" for a, b in zip(nr["min"], nr["max"])]
    _bars(axes[1], [f"{l}\nn {n}" for l, n in zip(labs, nlab)], series, err if k >= 2 else None)  # 1시드면 SD 미측정 → 오차막대 없음
    axes[1].set_title(f"(b) 2020-01 · space fold 0 · {k}시드 " + ("평균 ± SD" if k >= 2 else "(SD 미측정)") + "\n층 = 훈련 1월 기준", fontsize=9); axes[1].set_xlabel("최근접 훈련 사운딩 거리 (km)")
    ymax = max(np.nanmax(ct[[col[k] for k in keys]].to_numpy()), *[np.nanmax(series[q] + (np.nan_to_num(err[q]) if k >= 2 else 0)) for q in series]) * 1.25
    for ax in axes:
        ax.set_ylim(0, ymax); ax.legend(loc="upper left", ncol=1)
    fig.suptitle("거리 층별 테스트 RMSE — 물리 주입 vs 절제 vs CarbonTracker", y=1.13)
    fig.text(0.5, 1.05, "층 = 최근접 훈련 사운딩까지 거리 → 훈련 집합이 다르면 같은 사운딩도 다른 층에 들어간다: 두 패널 사이 층 비교 불가", ha="center", va="top", fontsize=7.5, color=C_TEXT2)
    table = pd.concat([ct.assign(panel="a_2020_12m"), seed_df.assign(panel="b_202001_3seed")], ignore_index=True)
    return save(fig, table, "strata_space0", date, out_dir)


# ---------------------------------------------------------------- 2. 월별 편향·RMSE (time fold 2)
@styled
def plot_monthly(csv: str, date: str, out_dir: str | None = None, sd_note_month: int = 7, name: str = "monthly_time2", title: str | None = None, note: str | None = None) -> tuple[str, str]:
    mt = pd.read_csv(csv); m = mt["월"].to_numpy(); d3 = ct_format(mt) == "d3"
    n_lab = mt["n_공통"] if d3 else mt["n"]; sd = mt["SD_공통"] if d3 else mt["SD"]  # D-3: CT 와의 비교는 공통 표본 (n·SD 도 공통)
    ours = "ours_phys_rmse_공통" if d3 else "ours_phys_rmse"; ours_b = "ours_phys_bias_공통" if d3 else "ours_phys_bias"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 2], hspace=0.12))
    a1.bar(m, sd, width=0.6, color=C_GRID, label="테스트 XCO₂ SD" + (" (공통 표본)" if d3 else ""), zorder=1)
    a1.plot(m, mt["CT_raw_rmse"], color=C_CT, ls=(0, (3, 2)), lw=1.5, marker=None, label=LABEL["CT_raw"], zorder=2)
    a1.plot(m, mt["CT_debiased(train)_rmse"], color=C_CT, marker="o", label=LABEL["CT_debiased(train)"], zorder=3)
    a1.plot(m, mt[ours], color=C_OURS, marker="o", label=LABEL["ours_phys"] + (" · 공통 표본" if d3 else ""), zorder=4)
    if d3:  # 우리 전체 표본 = 참고선
        a1.plot(m, mt["ours_phys_rmse"], color=C_OURS, ls=(0, (3, 2)), lw=1.2, marker=None, label=LABEL["ours_phys"] + " · 전체 표본 (참고)", zorder=3)
    a1.set_ylabel("테스트 RMSE (ppm)"); a1.set_ylim(0, None); a1.legend(loc="upper left", ncol=2)
    a1.set_title(title or "월별 테스트 RMSE·편향 — 2020 · time fold 2 · seed 0 (우리 = lc2020_12m 물리, 결정 1 배경)")
    if not d3:  # 구 형식: CT 와 우리 표본이 다름을 명기 (QA Z1). 월 라벨 n 은 우리(테스트 전체) 기준
        note = (note + "\n" if note else "") + ct_sample_note(csv, "time", 2) + " · 월 라벨 n = 테스트 전체"
    if note:
        a1.text(0.5, 1.13, note, transform=a1.transAxes, ha="center", va="bottom", fontsize=7.5, color=C_TEXT2)
    a2.axhline(0, color=C_TEXT2, lw=1, zorder=1)
    a2.plot(m, mt["CT_debiased(train)_bias"], color=C_CT, marker="o", label=LABEL["CT_debiased(train)"], zorder=3)
    a2.plot(m, mt[ours_b], color=C_OURS, marker="o", label=LABEL["ours_phys"] + (" · 공통 표본" if d3 else ""), zorder=4)  # 상단과 같은 표본 표기 (QA §17)
    a2.set_ylabel("편향 = 예측 − 관측 (ppm)"); a2.legend(loc="best", ncol=2)  # 데이터와 겹치지 않는 자리
    a2.set_xticks(m); a2.set_xticklabels([f"{int(mm)}월\nn {int(n):,}" for mm, n in zip(m, n_lab)], fontsize=7); a2.set_xlabel("" if not d3 else "n = 공통 표본 (CT 보간 있는 테스트 행) · 파란 점선 = 우리 전체 표본")
    if sd_note_month in set(m):
        r = mt[mt["월"] == sd_note_month].iloc[0]; o = sd[mt["월"] != sd_note_month]; r_sd = float(sd[mt["월"] == sd_note_month].iloc[0])
        a1.annotate(f"{sd_note_month}월: SD {r_sd:.1f}, 우리 RMSE {r[ours]:.2f}\n(다른 달 SD {o.min():.1f}–{o.max():.1f}, 별도 취급)", xy=(sd_note_month, r[ours]),
                    xytext=(sd_note_month + 1.3, r[ours] + 0.2), fontsize=7, color=C_TEXT2, arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=1))
    return save(fig, mt, name, date, out_dir)


# ---------------------------------------------------------------- 3. 학습 곡선 (훈련 개월 수)
def learning_curve_table(dirs: dict[int, str], scheme: str = "time", fold: int = 2, seed: int = 0) -> pd.DataFrame:
    """런별 summary 행 + n_test. 세 런의 고정 테스트 행(pred parquet 의 test==True 인 row_idx 집합)이 같은지 검사한다 (--eval-months 가 다른 런 혼입 방지)."""
    rows = []; ref = None
    for n_m, d in sorted(dirs.items()):
        p = pd.read_parquet(os.path.join(d, f"pred_{scheme}{fold}_s{seed}_phys.parquet"), columns=["row_idx", "test"]); ids = np.sort(p.row_idx.to_numpy()[p.test.to_numpy()])
        if ref is None:
            ref = ids
        elif len(ids) != len(ref) or not (ids == ref).all():
            raise ValueError(f"{d}: 고정 테스트 행 집합이 다른 런과 다름 (n {len(ids):,} vs {len(ref):,})")
        s = pd.read_csv(os.path.join(d, "summary.csv")); s = s[(s.physics.astype(str) == "True") & (s.scheme == scheme) & (s.fold == fold)]
        if len(s) != 1:
            raise ValueError(f"{d}/summary.csv: {scheme}:{fold} 물리 행 {len(s)}개 (1개여야 함)")
        s = s.iloc[0]
        rows.append(dict(train_months=n_m, months=s.months, test_clean=s.test_clean, test_clean_all=s.test_clean_all, beta=s.beta, best_ep=int(s.best_ep), epochs_run=int(s.epochs_run),
                         converged=int(s.best_ep) < int(s.epochs_run) - 1, seed=int(s.seed), scheme=scheme, fold=fold, n_test=int(len(ids))))
    return pd.DataFrame(rows)


@styled
def plot_learning_curve(dirs: dict[int, str], date: str, out_dir: str | None = None, scheme: str = "time", fold: int = 2, table: pd.DataFrame | None = None) -> tuple[str, str]:
    """table: learning_curve_table 결과를 이미 계산했으면 넘긴다 (두 번 읽지 않도록, QA N9)."""
    t = learning_curve_table(dirs, scheme, fold) if table is None else table; x = t.train_months.to_numpy(); n_test = int(t.n_test.iloc[0])
    if sorted(t.train_months) != sorted(dirs) or not ((t.scheme == scheme) & (t.fold == fold)).all():  # 넘겨받은 표가 dirs·폴드와 같은 런인지 (QA W10)
        raise ValueError(f"table 이 dirs {sorted(dirs)} · {scheme}:{fold} 와 다름")
    seeds = sorted(t.seed.unique()); eps = sorted(t.epochs_run.unique())
    # 고정 테스트 구간 라벨: 실제 행은 --eval-months 로 정해진다. 최단 런의 훈련 구간과 같다는 가정을 검사한다 (QA W3)
    short = t.loc[t.train_months.idxmin()]
    if short.test_clean != short.test_clean_all:
        raise ValueError("최단 런의 test_clean ≠ test_clean_all — 고정 테스트 구간이 그 런의 훈련 구간과 다름, 라벨을 정할 수 없음")
    fixed = short.months  # 최단 런은 고정 테스트 = 자기 폴드 테스트 전체 (위 검사) → 구간 = 그 런의 월
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.2, 5), sharex=True, gridspec_kw=dict(height_ratios=[3, 2], hspace=0.12))
    # test_clean_all = 각 런의 훈련 구간에 속한 폴드 테스트 행 → 런마다 테스트 집합이 다르다 (QA B1). 선으로 잇지 않고 점마다 구간을 적는다
    a1.plot(x, t.test_clean_all, color=C_MUTED, marker="o", ls="none", label="참고: 런별 폴드 테스트 (test_clean_all, 구간이 런마다 다름 → 서로 비교 불가)", zorder=2)
    for xi, yi, mo in zip(x, t.test_clean_all, t.months):
        if mo != fixed:  # 테스트 구간 = 고정 테스트인 런은 두 값이 같은 행 집합 → 파란 라벨로 충분
            a1.text(xi + 0.25, yi, f"{yi:.3f}\n테스트 {mo}", ha="left", va="center", fontsize=6.5, color=C_MUTED)
    a1.plot(x, t.test_clean, color=C_OURS, lw=2, zorder=3, label=f"고정 테스트 {fixed} (test_clean)")
    for xi, yi, c in zip(x, t.test_clean, t.converged):
        a1.plot(xi, yi, marker="o", color=C_OURS, mfc=C_OURS if c else "white", mec=C_OURS if not c else "white", zorder=4)
        a1.text(xi, yi - 0.025, f"{yi:.3f}", ha="center", va="top", fontsize=7, color=C_TEXT2)  # 마커 아래 (위쪽은 test_clean_all 선과 겹침)
    a1.plot([], [], marker="o", color=C_OURS, mfc="white", mec=C_OURS, ls="none", label="빈 마커 = 미수렴 (best_ep = 마지막 에폭)")
    a1.set_ylabel("테스트 RMSE (ppm)"); a1.legend(loc="upper left"); a1.set_ylim(min(t.test_clean.min(), t.test_clean_all.min()) - 0.1, max(t.test_clean_all.max(), t.test_clean.max()) + 0.15)
    a1.set_title(f"학습 곡선 — 훈련 개월 수 vs 테스트 RMSE ({scheme} fold {fold} · seed {','.join(map(str, seeds))} · {','.join(map(str, eps))} ep · 물리)"
                 + f"\n고정 테스트 {fixed} n {n_test:,} (세 런 동일 행 집합 검사됨)", fontsize=9)
    a2.plot(x, t.beta, color=C_OURS, marker="o", zorder=3)
    for xi, yi in zip(x, t.beta):
        a2.text(xi, yi + 0.008, f"{yi:+.3f}", ha="center", va="bottom", fontsize=7, color=C_TEXT2)
    a2.set_ylabel("β (NO₂ 계수)"); a2.set_xlabel("훈련 개월 수"); a2.set_xticks(x); a2.set_xticklabels([f"{n}개월\n({mo})" for n, mo in zip(x, t.months)], fontsize=7)
    lo, hi = min(0.0, t.beta.min()), max(0.0, t.beta.max()); pad = 0.3 * (hi - lo or 0.1); a2.set_ylim(lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0))
    return save(fig, t, "learning_curve_2020", date, out_dir)


# ---------------------------------------------------------------- 4. 잔차 지도 (0.25° 잠재 격자 셀 평균)
MONTHS_2020 = [f"2020{m:02d}" for m in range(1, 13)]
NCELL_LAT, NCELL_LON = NLAT - 1, NLON - 1  # 셀 수 (격자점 121×201 사이)


@functools.lru_cache(maxsize=4)
def _oco_test_base(idx_dir: str, scheme: str, fold: int, months: tuple[str, ...], vfold: int) -> tuple[pd.DataFrame, np.ndarray]:
    """oco 인덱스 이어붙이기 + nested_masks 테스트 마스크. pred 와 무관 → 같은 (폴드, 월) 은 한 번만 계산 (QA B6). 호출자는 복사본을 쓴다."""
    nested_masks = _train().nested_masks
    big = pd.concat([_L().load_oco(m, idx_dir, columns=["row_idx", "latitude", "longitude", "time", "inside", "xco2", "label", "fold_time", "fold_space", "space_block", "year"]) for m in months], ignore_index=True)
    lab = (big.label == "L_train").to_numpy(); _, _, te = nested_masks(big, scheme, fold, lab, vfold)
    return big, te


def load_test_rows(pred_path: str, idx_dir: str, scheme: str, fold: int, months: list[str] = MONTHS_2020, vfold: int = 1) -> pd.DataFrame:
    """pred parquet 을 oco 인덱스(row_idx 순서 일치 검사)에 붙여 폴드 테스트 행만 반환. 열: row_idx, latitude, longitude, time, inside, xco2, pred, res.
    테스트 마스크는 nested_masks(scheme, fold) 로 재계산한다 — parquet 의 test 열은 --eval-months 런(lc2020_*)에서 부분 구간만 True 라서 그대로 쓰지 않는다;
    대신 parquet test ⊆ 재계산 마스크를 검사한다."""
    base, te = _oco_test_base(idx_dir, scheme, fold, tuple(months), vfold); big = base[["row_idx", "latitude", "longitude", "time", "inside", "xco2"]].copy()
    p = pd.read_parquet(pred_path)
    if len(p) != len(big) or not (p.row_idx.to_numpy() == big.row_idx.to_numpy()).all():
        raise ValueError(f"row_idx 순서 불일치: {pred_path} vs oco {months[0]}–{months[-1]}")
    if not np.allclose(p.xco2.to_numpy(), big.xco2.to_numpy(), equal_nan=True):
        raise ValueError(f"xco2 불일치: {pred_path}")
    pt = p.test.to_numpy()
    if not te[pt].all():
        raise ValueError(f"parquet test 열이 {scheme}:{fold} 테스트 마스크의 부분집합이 아님: {pred_path}")
    big["pred"] = p.pred.to_numpy(); big["res"] = big.pred - big.xco2
    return big.loc[te, ["row_idx", "latitude", "longitude", "time", "inside", "xco2", "pred", "res"]].reset_index(drop=True)


def residual_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """테스트 행 → 셀(i, j) 별 n·mean_res·rmse. 셀 배정 i = floor((lat−20)/0.25), j = floor((lon−100)/0.25). 격자 밖(inside=False) 은 제외하고 수만 기록.
    월 부분집합은 호출자가 df 를 걸러서 넘긴다."""
    fin = np.isfinite(df.res.to_numpy()); ins = df.inside.to_numpy(); ok = ins & fin; d = df[ok]
    # 셀 = 격자점 사이 (NLAT−1)×(NLON−1) = 120×200. 북·동 경계(lat 50.0 · lon 150.0) 사운딩은 마지막 셀로 (가상 셀 없음, QA N4)
    i = np.floor((d.latitude.to_numpy() - GRID_LAT0) / GRID_D).astype(int).clip(0, NCELL_LAT - 1); j = np.floor((d.longitude.to_numpy() - GRID_LON0) / GRID_D).astype(int).clip(0, NCELL_LON - 1)
    g = pd.DataFrame({"i": i, "j": j, "res": d.res.to_numpy()}).groupby(["i", "j"]).res.agg(n="size", mean_res="mean", rmse=lambda r: float(np.sqrt((r ** 2).mean()))).reset_index()
    g["lat_s"] = GRID_LAT0 + GRID_D * g.i; g["lon_w"] = GRID_LON0 + GRID_D * g.j  # 셀 남·서 경계
    g["lat_c"] = g.lat_s + GRID_D / 2; g["lon_c"] = g.lon_w + GRID_D / 2          # 셀 중심
    r_all = df.res.to_numpy()[fin]  # 전체 RMSE·편향 = 유한 잔차 전체(격자 밖 포함) — train5 summary test_clean_all 과 같은 정의
    meta = dict(n_test=int(len(df)), n_outside=int((~ins & fin).sum()), n_nan=int((~fin).sum()), n_cells=int(len(g)),  # outside·nan 은 상호배타 (nan 우선)
                rmse=float(np.sqrt((r_all ** 2).mean())) if len(r_all) else float("nan"), bias=float(r_all.mean()) if len(r_all) else float("nan"), bias_grid=float(np.average(g.mean_res, weights=g.n)) if len(g) else float("nan"))  # 격자 안 유한 행 0 → nan (QA N5)
    return g[["i", "j", "lat_s", "lon_w", "lat_c", "lon_c", "n", "mean_res", "rmse"]], meta


def _field(g: pd.DataFrame, col: str) -> np.ndarray:
    A = np.full((NCELL_LAT, NCELL_LON), np.nan); A[g.i.to_numpy(), g.j.to_numpy()] = g[col].to_numpy(); return A


def _map_axes(fig, pos):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree()); ax.set_extent([GRID_LON0, GRID_LON0 + GRID_D * NCELL_LON, GRID_LAT0, GRID_LAT0 + GRID_D * NCELL_LAT], crs=ccrs.PlateCarree())  # 100–150E · 20–50N
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, edgecolor=C_TEXT2, zorder=3)
    gl = ax.gridlines(draw_labels=True, xlocs=range(100, 151, 10), ylocs=range(20, 51, 10), lw=0.5, color=C_GRID, zorder=2); gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": C_TEXT2}
    return ax, ccrs.PlateCarree()


@styled
def plot_residual_map(g: pd.DataFrame, meta: dict, title: str, name: str, date: str, out_dir: str | None = None, diff: pd.DataFrame | None = None,
                      diff_label: str = "RMSE 차 (절제 − 물리)", q: float = 98.0, stat_run: str | None = None) -> tuple[str, str]:
    """(a) 셀 평균 잔차 (diff 가 있으면 셀별 RMSE 차) — 발산형, 범위 ±v (v = |값| 의 q 백분위); (b) 셀 사운딩 수 — 순차형 log.
    stat_run: 제목의 전체 RMSE·편향이 어느 런 값인지 (diff 모드에서는 필수 — meta 가 한 런 값이라서, QA B2)."""
    if diff is not None and not stat_run:
        raise ValueError("diff 모드는 stat_run(제목 RMSE·편향의 출처 런) 을 지정해야 함")
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    fig = plt.figure(figsize=(10.5, 3.4), constrained_layout=True)
    edges_lon = GRID_LON0 + GRID_D * np.arange(NCELL_LON + 1); edges_lat = GRID_LAT0 + GRID_D * np.arange(NCELL_LAT + 1)
    div = LinearSegmentedColormap.from_list("div", [C_DIV_NEG, C_DIV_MID, C_DIV_POS])  # 보라(−) – 중립 – 적갈(+); 범주색(파랑 = 우리 물리, 주황 = CT)과 분리 (QA B3)
    val = _field(g, "mean_res") if diff is None else _field(diff, "diff"); v = float(np.nanpercentile(np.abs(val), q))
    ax, pc = _map_axes(fig, (1, 2, 1)); m = ax.pcolormesh(edges_lon, edges_lat, val, cmap=div, vmin=-v, vmax=v, transform=pc, zorder=1)
    cb = fig.colorbar(m, ax=ax, shrink=0.8, pad=0.02); cb.set_label(("셀 평균 잔차 = 예측 − 관측 (ppm)" if diff is None else f"{diff_label} (ppm)"), fontsize=8)
    ax.set_title(f"(a) {'셀 평균 잔차' if diff is None else diff_label} · 색 범위 ±{v:.2f} (|값| {q:g} 백분위)", fontsize=9)
    ax2, pc = _map_axes(fig, (1, 2, 2)); cnt = _field(g, "n"); m2 = ax2.pcolormesh(edges_lon, edges_lat, cnt, cmap="Blues", norm=LogNorm(1, np.nanmax(cnt)), transform=pc, zorder=1)
    cb2 = fig.colorbar(m2, ax=ax2, shrink=0.8, pad=0.02); cb2.set_label("셀 테스트 사운딩 수 (log)", fontsize=8)
    ax2.set_title(f"(b) 사운딩 수 · 셀 {meta['n_cells']:,} · 테스트 {meta['n_test']:,} (격자 밖 {meta['n_outside']:,})", fontsize=9)
    who = f" ({stat_run})" if stat_run else ""
    fig.suptitle(f"{title} — 0.25° 격자 · 전체 RMSE{who} {meta['rmse']:.3f} · 편향{who} {meta['bias']:+.3f} ppm", fontsize=10)
    table = g if diff is None else g.merge(diff[["i", "j", "diff"]], on=["i", "j"], how="left")
    return save(fig, table, name, date, out_dir)


def rmse_diff(g_a: pd.DataFrame, g_b: pd.DataFrame) -> pd.DataFrame:
    """같은 테스트 행에서 나온 두 격자의 셀별 RMSE 차 (b − a). 셀 n 이 같아야 한다 (같은 행 집합 검증)."""
    m = g_a.merge(g_b, on=["i", "j"], suffixes=("_a", "_b"))
    if len(m) != len(g_a) or len(m) != len(g_b) or not (m.n_a == m.n_b).all():
        raise ValueError("두 격자의 셀·n 이 다름 — 같은 테스트 행 집합이 아님")
    m["diff"] = m.rmse_b - m.rmse_a; return m[["i", "j", "n_a", "diff"]].rename(columns={"n_a": "n"})


# ---------------------------------------------------------------- 5. 결정 1 전후 월별 R² (time fold 2)
def _stats(r: np.ndarray, y: np.ndarray) -> dict:
    """SD = 테스트 행 전체 y 의 표준편차 (ct_compare monthly_time2.csv 의 SD 와 같은 정의, QA B9); R² 분모 = 유한 잔차 행 y 분산 (ct_compare stats 와 같은 정의)."""
    ok = np.isfinite(r); v = float(np.var(y[ok])); r = r[ok]
    return dict(n=int(ok.sum()), n_nan=int((~ok).sum()), SD=float(np.std(y)), rmse=float(np.sqrt((r ** 2).mean())), bias=float(r.mean()), r2=float(1 - (r ** 2).mean() / v))


def monthly_compare(runs: dict[str, str], idx_dir: str, scheme: str = "time", fold: int = 2, months: list[str] = MONTHS_2020) -> pd.DataFrame:
    """runs = {라벨: pred 경로}. 같은 테스트 행(row_idx 일치 검사)에서 월별·전체 n/SD/RMSE/bias/R² (ct_compare_2020.stats 와 같은 정의)."""
    dfs = {k: load_test_rows(p, idx_dir, scheme, fold, months) for k, p in runs.items()}; keys = list(dfs)
    base = dfs[keys[0]]
    for k in keys[1:]:
        if len(dfs[k]) != len(base) or not (dfs[k].row_idx.to_numpy() == base.row_idx.to_numpy()).all():
            raise ValueError(f"테스트 행 집합 불일치: {k} vs {keys[0]}")
    mon = pd.to_datetime(base.time).dt.month.to_numpy(); y = base.xco2.to_numpy(); rows = []
    for m in [*sorted(np.unique(mon)), 0]:
        sel = (mon == m) if m else np.ones(len(mon), bool)
        for k in keys:
            rows.append(dict(월=int(m), run=k, **_stats(dfs[k].res.to_numpy()[sel], y[sel])))
    return pd.DataFrame(rows)


@styled
def plot_monthly_r2(t: pd.DataFrame, labels: dict[str, str], colors: dict[str, str], title: str, name: str, date: str, out_dir: str | None = None,
                    r2_floor: float = -1.0) -> tuple[str, str]:
    """3단: R² · RMSE(+ 테스트 SD 막대) · 편향. t = monthly_compare 산출 (월 0 = 전체).
    R² 축은 r2_floor 아래를 잘라 표시하고(표시 범위이지 판정 기준 아님) 범위 밖 값은 수치로 적는다."""
    mt = t[t["월"] > 0]; keys = list(labels); m = np.array(sorted(mt["월"].unique()))
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.5, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 2], hspace=0.12))
    a1.axhline(0, color=C_TEXT2, lw=1, zorder=1); a3.axhline(0, color=C_TEXT2, lw=1, zorder=1)
    sd = mt[mt.run == keys[0]].set_index("월").reindex(m); a2.bar(m, sd.SD, width=0.6, color=C_GRID, label="테스트 XCO₂ SD", zorder=1)
    below = []
    for k in keys:
        s = mt[mt.run == k].set_index("월").reindex(m); r2 = s.r2.to_numpy()
        a1.plot(m, np.where(r2 >= r2_floor, r2, np.nan), color=colors[k], marker="o", label=labels[k], zorder=3)
        for mm, v in zip(m, r2):
            if v < r2_floor:
                a1.plot(mm, r2_floor, marker="v", color=colors[k], ls="none", zorder=4); below.append(f"{labels[k]} {int(mm)}월 {v:.2f}")  # 라벨 전체 (첫 단어만 쓰면 런 구분 안 됨, QA N1)
        a2.plot(m, s.rmse, color=colors[k], marker="o", label=labels[k], zorder=3); a3.plot(m, s.bias, color=colors[k], marker="o", label=labels[k], zorder=3)
    tot = t[t["월"] == 0].set_index("run")
    a1.set_ylabel("R² (월 내 분산 기준)"); a1.set_ylim(r2_floor - 0.1, max(1.0, mt.r2.max() + 0.15))
    a1.set_title(title + "\n전체(n {:,}): ".format(int(tot.n.iloc[0])) + " · ".join(f"{labels[k]} R² {tot.loc[k, 'r2']:.3f}, RMSE {tot.loc[k, 'rmse']:.3f}" for k in keys), fontsize=8.5)
    if below:
        a1.text(0.01, 0.96, f"▼ 표시 범위(R² ≥ {r2_floor:g}) 아래 (선 끊김): " + ", ".join(below), transform=a1.transAxes, ha="left", va="top", fontsize=7, color=C_TEXT2)
    a2.set_ylabel("테스트 RMSE (ppm)"); a2.set_ylim(0, None); a2.legend(loc="upper left", ncol=3)
    a3.set_ylabel("편향 (ppm)"); a3.set_xticks(m); a3.set_xticklabels([f"{int(mm)}월\nn {int(n):,}" for mm, n in zip(m, sd.n)], fontsize=7)
    return save(fig, t, name, date, out_dir)


# ---------------------------------------------------------------- 6. B1 forest (V-C, 승인 2026-09-24)
B1_THR_LEGACY = 0.064  # 구 train5 코드 고정 임계 = 0.1 × 2020-01 L_train xco2_uncertainty 중앙 0.640 (09-22 QA Q2 기록; train.py:85 구 파일럿 경로도 같은 식)
B1_THR_LEGACY_SRC = "구 train5 코드 고정 (09-22 QA Q2)"


def b1_table(rows: list[tuple]) -> pd.DataFrame:
    """rows = runs.B1_ROWS. summary.csv 의 physics=='B1' 행(delta·ci_lo·ci_hi·pass_B1)과 같은 scheme·fold·seed 의 물리/절제 test_clean 을 무변환으로 모은다.
    Δ = RMSE(절제) − RMSE(물리), CI = 일블록 부트스트랩 95 % (train.block_bootstrap). 행 순서 = rows 순서 × seed 오름차순."""
    out = []
    for label, d, scheme, fold, n_bg in rows:
        s = pd.read_csv(os.path.join(d, "summary.csv")); s = s[(s.scheme == scheme) & (s.fold == fold)]
        for seed in sorted(s.seed.dropna().astype(int).unique()):
            ss = s[s.seed == seed]; ph = ss.physics.astype(str)
            b, p, n = ss[ph == "B1"], ss[ph == "True"], ss[ph == "False"]
            if not (len(b) == len(p) == len(n) == 1):
                raise ValueError(f"{d}: {scheme}:{fold} seed {seed} 의 B1/물리/절제 행 {len(b)}/{len(p)}/{len(n)} (각 1이어야 함)")
            b, p, n = b.iloc[0], p.iloc[0], n.iloc[0]
            has_thr = "b1_thr" in b.index and pd.notna(b.get("b1_thr"))  # 09-22 승인 후 train5 런은 summary 에 기록, 구 런은 구 train5 코드 고정 0.064
            out.append(dict(label=label, run=os.path.basename(d), scheme=scheme, fold=fold, seed=seed, n_bg=n_bg, months=p.months,
                            rmse_phys=p.test_clean, rmse_nophys=n.test_clean, delta=b.delta, ci_lo=b.ci_lo, ci_hi=b.ci_hi, pass_B1=str(b.pass_B1) == "True",
                            b1_thr=float(b.b1_thr) if has_thr else B1_THR_LEGACY, b1_thr_src="summary" if has_thr else B1_THR_LEGACY_SRC))
    t = pd.DataFrame(out)
    t["pass_def"] = (t.ci_lo > 0) & (t.delta >= t.b1_thr)  # 판정 정의 검증용: B1 = CI 하한 > 0 및 Δ ≥ b1_thr (QA W2)
    return t


@styled
def plot_b1_forest(t: pd.DataFrame, date: str, out_dir: str | None = None) -> tuple[str, str]:
    """점 = Δ, 위스커 = 95 % CI. 채운 마커 = B1 통과(pass_B1), 빈 마커 = 미통과. 오른쪽 텍스트 = 물리/절제 test RMSE."""
    n = len(t); y = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * n + 1.8))
    ax.grid(axis="x"); ax.grid(axis="y", visible=False); ax.axvline(0, color=C_TEXT2, lw=1, zorder=1)
    for yi, r in zip(y, t.itertuples()):
        c = C_OURS if r.scheme == "space" else C_MUTED
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(r.delta, yi, marker="o", ms=8, color=c, mfc=c if r.pass_B1 else "white", mec=c if not r.pass_B1 else "white", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.label} · seed {r.seed} · 배경 {r.n_bg}변수" for r in t.itertuples()], fontsize=8)
    lo, hi = min(t.ci_lo.min(), 0), max(t.ci_hi.max(), 0); pad = 0.06 * (hi - lo); ax.set_xlim(lo - pad, hi + pad)
    for yi, r in zip(y, t.itertuples()):  # 수치는 축 밖 오른쪽 (축 범위를 늘리지 않음)
        ax.text(1.02, yi, f"물리 {r.rmse_phys:.3f} / 절제 {r.rmse_nophys:.3f} · Δ {r.delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=C_TEXT2, clip_on=False)
    grp = t.groupby(["b1_thr", "b1_thr_src"]).size()  # 값·출처별로 묶어 짧게
    thr = " · ".join(f"{v:.4f} ({src}) {k}행" for (v, src), k in grp.items())
    ax.set_xlabel("Δ = RMSE(절제) − RMSE(물리) (ppm) · + 이면 물리가 낮음")
    fig.text(0.5, -0.01, f"b1_thr = 0.1 × 평가 기간 L_train XCO₂ 불확도 중앙 — 이 그림의 판정 임계: {thr}", ha="center", va="top", fontsize=7, color=C_TEXT2)
    ax.plot([], [], marker="o", color=C_TEXT2, ls="none", label="B1 통과 = CI 하한 > 0 및 Δ ≥ b1_thr"); ax.plot([], [], marker="o", color=C_TEXT2, mfc="white", mec=C_TEXT2, ls="none", label="미통과")
    ax.plot([], [], color=C_OURS, lw=2, label="space fold 0"); ax.plot([], [], color=C_MUTED, lw=2, label="time fold 2")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=7, borderaxespad=0.3)
    ax.set_title(f"B1 판정 — 물리 주입 vs 절제 (일블록 부트스트랩 95 % CI) · 통과 {int(t.pass_B1.sum())}/{n}", fontsize=10, pad=24)
    return save(fig, t, "b1_forest", date, out_dir)


# ---------------------------------------------------------------- 7. 물리 자유전파 (V-D, 승인 2026-09-24)
def freerun_table(d: str) -> pd.DataFrame:
    """physics_rollout --mode free-run 산출 CSV(physics_freerun_202001_s{시작}.csv) 전부 → lag 별 커버리지·질량비(C합/C합₀)·정보량비(M합/M합₀). 계산 없이 비율만."""
    import glob
    fs = sorted(glob.glob(os.path.join(d, "physics_freerun_202001_s*.csv")))
    if not fs:
        raise FileNotFoundError(f"자유전파 CSV 없음: {d}")
    out = []
    for f in fs:
        import re
        mt = re.fullmatch(r"physics_freerun_202001_s(\d+)\.csv", os.path.basename(f))
        if mt is None:
            continue  # 규칙 밖 파일 무시 (QA W4)
        s0 = int(mt.group(1)); r = pd.read_csv(f); b = r[r.lag == 0].iloc[0]
        out.append(r.assign(start=s0, day=s0 // 24 + 1, utc_h=s0 % 24, mass_ratio=r.mass / b.mass, info_ratio=r.infosum / b.infosum))
    return pd.concat(out, ignore_index=True)


@styled
def plot_freerun(t: pd.DataFrame, ref_start: int, date: str, out_dir: str | None = None, D: float | None = None) -> tuple[str, str]:
    """3단 (x = lag 0–48 h): 커버리지(M > 0.01 셀 비율) · 질량비 · 정보량비. 기준 시작(ref_start) = 파랑 굵은 선, 나머지 = 회색 가는 선. lag 0·24·48 값 라벨(기준 선)."""
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True, gridspec_kw=dict(hspace=0.14))
    cols = [("coverage", "커버리지 (M > 0.01 셀 비율)"), ("mass_ratio", "질량비 C합 / C합₀"), ("info_ratio", "정보량비 M합 / M합₀")]
    others = [s for s in sorted(t.start.unique()) if s != ref_start]; dash = {s: ["-", (0, (4, 2)), (0, (1, 2))][i % 3] for i, s in enumerate(others)}  # 보조 시작일은 선 모양으로 구분
    for s0, g in t.groupby("start"):
        ref = s0 == ref_start; lab = f"시작 1월 {g.day.iloc[0]}일 {g.utc_h.iloc[0]:02d} UTC (스텝 {s0})" + (" · 기준" if ref else "")
        for ax, (c, _) in zip(axes, cols):
            ax.plot(g.lag, g[c], color=C_OURS if ref else C_MUTED, lw=2 if ref else 1.4, ls="-" if ref else dash[s0], marker=None, label=lab, zorder=3 if ref else 2)
            if ref:
                for k in (0, 24, 48):
                    v = g.loc[g.lag == k, c]
                    if len(v):
                        ax.text(k, float(v.iloc[0]), f"{float(v.iloc[0]):.3f}", ha="center", va="bottom", fontsize=7, color=C_TEXT2)
    for ax, (_, yl) in zip(axes, cols):
        ax.set_ylabel(yl, fontsize=8)
    axes[1].axhline(1, color=C_TEXT2, lw=1, zorder=1); axes[2].axhline(1, color=C_TEXT2, lw=1, zorder=1)
    axes[0].legend(loc="lower right", fontsize=7); axes[-1].set_xlabel("주입 후 경과 시간 lag (h)"); axes[-1].set_xticks(range(0, 49, 6))
    r = t[(t.start == ref_start) & (t.lag == 48)]
    head = f" · 기준 lag 48: 커버리지 {r.coverage.iloc[0]:.3f} · 질량비 {r.mass_ratio.iloc[0]:.3f}" if len(r) else ""
    dtxt = f"확산 D {D:,.0f} m²/s" if D is not None else "확산 D 미기재"  # 롤아웃 CSV 에 D 가 없어 호출자가 넘긴다 (QA Y4)
    axes[0].set_title(f"물리 자유전파 (학습 없음) — 2020-01 · 한 번 주입 후 48 h · ERA5 바람 이류 + {dtxt}" + head, fontsize=8.5)
    return save(fig, t, "physics_freerun", date, out_dir)


# ---------------------------------------------------------------- 8. 월별 R² vs CT (E, 09-28)
def r2_vs_ct_table(csv: str) -> pd.DataFrame:
    """ct_compare D-3 월별 CSV → 월별 R² = 1 − RMSE²/SD² (우리·CT 모두 공통 표본 RMSE, SD = SD_공통). 계산은 이 식 하나뿐."""
    mt = pd.read_csv(csv)
    if ct_format(mt) != "d3":
        raise ValueError(f"{csv}: D-3 형식(공통 표본 열)이 아님 — 공통 표본 R² 를 계산할 수 없음")
    sd2 = mt["SD_공통"] ** 2
    return pd.DataFrame({"월": mt["월"], "n_공통": mt["n_공통"], "SD_공통": mt["SD_공통"],
                         "rmse_ours": mt["ours_phys_rmse_공통"], "rmse_ct": mt["CT_debiased(train)_rmse"],
                         "r2_ours": 1 - mt["ours_phys_rmse_공통"] ** 2 / sd2, "r2_ct": 1 - mt["CT_debiased(train)_rmse"] ** 2 / sd2})


@styled
def plot_r2_vs_ct(t: pd.DataFrame, ours_label: str, date: str, out_dir: str | None = None, r2_floor: float = -1.0) -> tuple[str, str]:
    """월별 R² 선 2개(우리·CT_deb). r2_floor 아래는 ▼ 와 수치(표시 범위, 판정 기준 아님)."""
    m = t["월"].to_numpy(); fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.axhline(0, color=C_TEXT2, lw=1, zorder=1); below = []
    for col, c, lab in (("r2_ct", C_CT, "CT(편향 제거)"), ("r2_ours", C_OURS, f"우리 ({ours_label})")):
        v = t[col].to_numpy(); ax.plot(m, np.where(v >= r2_floor, v, np.nan), color=c, marker="o", label=lab, zorder=3)
        for mm, vv in zip(m, v):
            if vv < r2_floor:
                ax.plot(mm, r2_floor, marker="v", color=c, ls="none", zorder=4); below.append(f"{lab} {int(mm)}월 {vv:.2f}")
    ax.set_ylim(r2_floor - 0.1, 1.0); ax.set_ylabel("R² = 1 − RMSE² / SD²"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)  # 데이터 밖
    ax.set_xticks(m); ax.set_xticklabels([f"{int(mm)}월\nn {int(n):,}" for mm, n in zip(m, t["n_공통"])], fontsize=7)
    ax.set_xlabel("n = 공통 표본 (CT 보간 있는 테스트 행) · RMSE·SD 모두 공통 표본")
    sub = ("\n▼ 표시 범위(R² ≥ " + f"{r2_floor:g}) 아래: " + ", ".join(below)) if below else ""
    ax.set_title("월별 R² — 2020 · time fold 2 · seed 0 · 우리 vs CarbonTracker (편향 제거)" + sub, fontsize=9)
    return save(fig, t, "monthly_r2_vs_ct", date, out_dir)
