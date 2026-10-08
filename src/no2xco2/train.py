"""파일럿 결함 해소 실험: 중첩 검증 조기 종료 · 시드 반복 · 물리 절제 · 깨끗한 평가 · 일블록 부트스트랩(B1).

분할 (결정 10 보강): test = 폴드 f. 검증(val) = 훈련 사운딩 중 직교 블록 — time 테스트면 space 폴드 v 의 셀(버퍼 포함),
  space 테스트면 time 폴드 v 의 블록(버퍼 포함). train = 훈련 사운딩 − val − val 버퍼. 조기 종료는 val RMSE 최저 에폭 가중치.
평가: 최저 val 가중치를 로드해 opt=None (화소 전량 주입) 으로 한 번 더 롤아웃 → test RMSE(clean).
B1: 같은 시드의 물리 / 절제 예측을 같은 test 사운딩에서 짝지어 ΔRMSE = RMSE(절제) − RMSE(물리), UTC 일 블록 부트스트랩 1,000회 95% CI.
주의(구 파일럿 전용, QA 3차 T6): __main__ 의 μ 는 L_train 전체 평균(val·test 포함)이라 train5(훈련 행만) 와 수치 비교 불가. pilot2_seeds 기록 재현을 위해 그대로 둠.
"""
import argparse, os, time
import numpy as np, pandas as pd, torch
import no2xco2.model as M16; import no2xco2.data.splits as splits_mod

# 거리 층 경계(km)·라벨 — 원본 (QA B5 2026-09-24). import: scripts/ct_compare_2020.py.
# 사본(값 동일): viz.results2020 (Data 세션이 import 로 전환 예정) · baselines.kriging.STRATA (상한 np.inf, 구 파일럿, rung 1 재구성 시 정리).
STRATA = [0, 20, 67.5, 250, 500, 1000, 1e9]; SLAB = ["0–20", "20–67.5", "67.5–250", "250–500", "500–1000", ">1000"]


def nested_masks(df, scheme, fold, lab, vfold):
    tr_all = splits_mod.train_mask(df, scheme, fold) & lab
    other = "space" if scheme == "time" else "time"
    val = tr_all & (df[f"fold_{other}"].to_numpy() == vfold)
    tr = tr_all & splits_mod.train_mask(df, other, vfold)  # val 블록·버퍼 제외
    te_col = "year" if scheme == "year" else f"fold_{scheme}"  # year 스킴(leave-one-year-out)은 열 이름이 fold_year 가 아니라 year (QA 3차 T3)
    te = (df[te_col].to_numpy() == fold) & lab
    return tr, val, te


def fit_one(px, qry, df, U, V, T, y, mu, tr, val, te, seed, d, epochs, lr, chunk, px_frac, no_phys, dev, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    model = M16.Model(d, dev).to(dev)
    if no_phys: model.phys.advect = lambda h, u, v: h; model.phys.diffuse = lambda h: h
    opt = torch.optim.Adam(model.parameters(), lr); best = (np.inf, -1, None); hist = []
    for ep in range(epochs):
        t1 = time.time(); model.train(); pred = M16.run(model, px, qry, U, V, T, y - mu, tr, opt, chunk=chunk, px_frac=px_frac)
        r = pred - (y - mu); rt = np.sqrt(np.nanmean(r[tr] ** 2)); rv = np.sqrt(np.nanmean(r[val] ** 2)); rte = np.sqrt(np.nanmean(r[te] ** 2))
        hist.append((ep, rt, rv, rte)); el = time.time() - t1
        print(f"  [{tag}] ep {ep:2d} train {rt:.3f} val {rv:.3f} test {rte:.3f} β {model.beta.item():+.3f} {el:.0f}s", flush=True)
        if rv < best[0]: best = (rv, ep, {k: v.detach().clone() for k, v in model.named_parameters()})  # 파라미터만 (버퍼 ii/jj 는 expand 뷰라 deepcopy 불가)
        if el > 150: print("  에폭 150 s 초과 → 스왑 의심, 중단", flush=True); break
    if best[2] is None:  # val RMSE 가 전 에폭 NaN 이거나 epochs=0 (QA 3차 T4; train5.fit 과 같은 가드)
        raise RuntimeError(f"[{tag}] 유효한 val 에폭 없음 (epochs={epochs}, hist={len(hist)})")
    model.load_state_dict(best[2], strict=False); model.eval()
    with torch.no_grad(): pred = M16.run(model, px, qry, U, V, T, y - mu, tr, None, chunk=chunk)
    r = pred - (y - mu); D = model.phys.D().detach().cpu().numpy(); k = torch.exp(model.phys.logk).item()
    return dict(best_ep=best[1], val=best[0], train_clean=float(np.sqrt(np.nanmean(r[tr] ** 2))), test_clean=float(np.sqrt(np.nanmean(r[te] ** 2))),
                beta=model.beta.item(), D_med=float(np.median(D)), D0=float(D[0]), inv_k_h=1 / k / 3600, epochs_run=len(hist)), pred + mu


def day_blocks(t) -> np.ndarray:
    """UTC 일 블록 id (block_bootstrap 의 days) = 1970-01-01 기준 일수 (시각을 일 단위로 내림). t: Series·배열·리스트 (datetime 변환 가능). train5·ct_compare 와 정의 공유 (QA 2026-09-22).
    시각 해상도(ns/us/s)에 무관 — pandas 2.x 는 입력 단위를 보존하므로 asi8 // ns 상수 방식은 ns 가 아닌 입력에서 전 행을 블록 0 으로 뭉쳤음 (QA 3차 T2)."""
    return np.asarray(pd.to_datetime(np.asarray(t))).astype("datetime64[D]").astype(np.int64)


def block_bootstrap(res_a, res_b, days, n=1000, seed=0):
    """ΔRMSE = RMSE(b) − RMSE(a) 의 일블록 부트스트랩. res_*: 잔차, days: 블록 id."""
    rng = np.random.default_rng(seed); ud = np.unique(days); idx = {d: np.where(days == d)[0] for d in ud}; out = []
    for _ in range(n):
        pick = rng.choice(ud, len(ud), replace=True); ii = np.concatenate([idx[d] for d in pick])
        out.append(np.sqrt((res_b[ii] ** 2).mean()) - np.sqrt((res_a[ii] ** 2).mean()))
    return np.percentile(out, [2.5, 97.5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="time:2,space:0"); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--vfold", type=int, default=1)
    ap.add_argument("--d", type=int, default=8); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--chunk", type=int, default=48); ap.add_argument("--px-frac", type=float, default=0.15); ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/pilot2_seeds")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); px, qry, df, U, V, T = M16.prep(dev); print(f"준비 {time.time()-t0:.0f}s")
    y = df.xco2.to_numpy(np.float64); lab = (df.label == "L_train").to_numpy(); mu = y[lab].mean()
    days = day_blocks(df.time)
    rows = []; preds = {}
    for cfg in a.configs.split(","):
        scheme, fold = cfg.split(":"); fold = int(fold); tr, val, te = nested_masks(df, scheme, fold, lab, a.vfold)
        print(f"\n== {scheme} fold {fold} (val = {'space' if scheme=='time' else 'time'} fold {a.vfold}): train {tr.sum():,} / val {val.sum():,} / test {te.sum():,}", flush=True)
        for seed in [int(s) for s in a.seeds.split(",")]:
            for no_phys in (False, True):
                tag = f"{scheme}{fold}_s{seed}_{'nophys' if no_phys else 'phys'}"
                r, pred = fit_one(px, qry, df, U, V, T, y, mu, tr, val, te, seed, a.d, a.epochs, a.lr, a.chunk, a.px_frac, no_phys, dev, tag)
                r.update(scheme=scheme, fold=fold, seed=seed, physics=not no_phys); rows.append(r); preds[tag] = pred
                pd.DataFrame(dict(row_idx=df.row_idx, pred=pred, xco2=y, test=te)).to_parquet(f"{a.out}/pred_{tag}.parquet", index=False)
                print(f"  → {tag}: best ep {r['best_ep']} val {r['val']:.3f} test(clean) {r['test_clean']:.3f} β {r['beta']:+.3f} D0 {r['D0']:.0f} 1/k {r['inv_k_h']:.1f}h", flush=True)
            # B1: 같은 시드 물리 vs 절제, 같은 test 사운딩. 임계 0.064 = 0.1 × 0.640 (2020-01 단월 L_train xco2_uncertainty 중앙, 결정 12) — 구 파일럿 전용, 평가 기간 기준은 train5 --b1-frac (QA 2026-09-22)
            pa_, pb_ = preds[f"{scheme}{fold}_s{seed}_phys"], preds[f"{scheme}{fold}_s{seed}_nophys"]
            ra, rb = (pa_ - y)[te], (pb_ - y)[te]; ok = np.isfinite(ra) & np.isfinite(rb)
            d_pt = np.sqrt((rb[ok] ** 2).mean()) - np.sqrt((ra[ok] ** 2).mean()); lo, hi = block_bootstrap(ra[ok], rb[ok], days[te][ok])
            diff = pa_[te][ok] - pb_[te][ok]
            rows.append(dict(scheme=scheme, fold=fold, seed=seed, physics="B1", delta=d_pt, ci_lo=lo, ci_hi=hi,
                             pass_B1=bool(lo > 0 and d_pt >= 0.064), pred_rms_diff=float(np.sqrt((diff ** 2).mean())), pred_corr=float(np.corrcoef(pa_[te][ok], pb_[te][ok])[0, 1])))
            print(f"  B1 {scheme}{fold} s{seed}: ΔRMSE(절제−물리) {d_pt:+.3f} ppm, 95% CI [{lo:+.3f}, {hi:+.3f}], 예측 RMS차 {rows[-1]['pred_rms_diff']:.3f}, corr {rows[-1]['pred_corr']:.4f}", flush=True)
    res = pd.DataFrame(rows); res.to_csv(f"{a.out}/summary.csv", index=False); print(f"\n총 {time.time()-t0:.0f}s → {a.out}/summary.csv")
