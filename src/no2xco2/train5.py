"""여러 달 학습 · 평가.

- 입력: build_index 산출(idx_dir) + ERA5 경계층 바람(era5_dir).
- 분할: 전 개월 사운딩을 이어 붙여 train.nested_masks (시간 · 공간 버퍼가 달 경계를 넘어 적용된다).
- 표준화: NO₂ 는 전 기간 μ · σ (no2_stats.json), 배경 11변수는 훈련 행에서만 fit.
- 학습 롤아웃: 에폭 0 은 오름차순 연속 롤아웃(앞 달 말 상태를 이어받음)으로 달 시작 상태 캐시를 만든다.
  에폭 e ≥ 1 은 default_rng(seed·1000 + e) 순열로 달을 돈다. 달 m 의 시작 상태 = 가장 최근에 돈 m−1 달의 끝 상태
  (시간상 앞 달이 없으면 0 에서 시작).
- 에폭 평균: 청크 갱신 직후 파라미터를 누적해 에폭 말 균등 평균을 그 에폭의 후보로 삼는다 (학습은 원 파라미터 · Adam 상태로 계속).
- 선택: 후보 파라미터로 오름차순 롤아웃 1회(화소 전부, no_grad)의 val RMSE 가 최저인 에폭. patience 에폭 동안 갱신이 없으면 종료.
- 평가: 선택된 파라미터로 오름차순 롤아웃 → test RMSE.
- 수송 끔(--nophys-only, --ablation): 이류 · 확산을 항등으로 바꾼다 (감쇠 · 주입 · 디코드는 같음).
- 출력: <out>/summary.csv, state_<tag>.pt(선택된 파라미터), pred_<tag>.parquet (row_idx, pred, xco2, test, test_all).
- --ckpt-dir: 에폭 끝마다 이어하기 저장본(원 파라미터 · Adam 상태 · 달 시작 상태 캐시 · 최저 val 후보 · 난수 상태). --resume 이면 이어서.
"""
import argparse
import copy
import hashlib
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import no2xco2.model as M
from no2xco2.data import loader as L
from no2xco2.train import nested_masks


def parse_months(s: str) -> list[str]:
    if "-" in s and len(s) == 13:  # 202001-202412
        a, b = s.split("-"); out = []; y, m = int(a[:4]), int(a[4:])
        while f"{y}{m:02d}" <= b:
            out.append(f"{y}{m:02d}"); m += 1
            if m == 13: y, m = y + 1, 1
        return out
    return sorted(set(s.split(",")))


def _month_inputs(ym, df, idx_dir, era5_dir, mu_no2, sd_no2, bg_mean, bg_std, dev, qa_min):
    px = L.load_px(ym, idx_dir, mu_no2, sd_no2, dev, qa_min); qry = L.load_qry(df, dev); L.apply_bg_scaler(qry, bg_mean, bg_std)
    U, V, T = L.load_wind(ym, era5_dir, dev)
    return px, qry, U, V, T


def rollout_all(model, months, dfs, slices, idx_dir, era5_dir, mu_no2, sd_no2, bg_mean, bg_std, y, tr_mask, opt, chunk, px_frac, dev, qa_min, gen=None):
    """오름차순 연속 롤아웃 (앞 달 말 상태를 이어받음, 시간상 인접하지 않으면 0 에서 시작). 반환: 전체 예측 벡터."""
    pred = np.full(len(y), np.nan); h = None; h0_prev_end = None
    for ym, df, sl in zip(months, dfs, slices):
        if h0_prev_end is not None and L.month_h0(ym) != h0_prev_end:
            h = None
        px, qry, U, V, T = _month_inputs(ym, df, idx_dir, era5_dir, mu_no2, sd_no2, bg_mean, bg_std, dev, qa_min); h0_prev_end = L.month_h0(ym) + T
        p, h = M.run(model, px, qry, U, V, T, y[sl], tr_mask[sl], opt, chunk=chunk, px_frac=px_frac, h0=h, return_h=True, gen=gen)
        pred[sl] = p; del px, qry, U, V
    return pred


def rollout_random(model, months, dfs, slices, idx_dir, era5_dir, mu_no2, sd_no2, bg_mean, bg_std, y, tr_mask, opt, chunk, px_frac, dev, qa_min, order, cache, chain):
    """학습 롤아웃. order = 달 순서. chain=True: 오름차순 연속 (달 시작 = 같은 에폭 앞 달 끝 상태).
    chain=False: 달 시작 = cache.get(i) (가장 최근 값), 없으면 0. 반환: (예측, 갱신된 캐시 {i: i 달 시작 상태})."""
    pred = np.full(len(y), np.nan); new = dict(cache); h = None; prev_i = None
    for i in order:
        ym, df, sl = months[i], dfs[i], slices[i]
        h0 = (h if (prev_i is not None and prev_i + 1 == i and i in new) else None) if chain else new.get(i)
        px, qry, U, V, T = _month_inputs(ym, df, idx_dir, era5_dir, mu_no2, sd_no2, bg_mean, bg_std, dev, qa_min)
        p, h = M.run(model, px, qry, U, V, T, y[sl], tr_mask[sl], opt, chunk=chunk, px_frac=px_frac, h0=h0, return_h=True)
        pred[sl] = p; del px, qry, U, V
        j = i + 1  # 시간상 다음 달이 인접하면 그 달의 시작 상태로 저장
        if j < len(months) and L.month_h0(months[j]) == L.month_h0(ym) + T:
            new[j] = h.detach().clone()
        prev_i = i
    return pred, new


class AvgOpt:
    """옵티마이저를 감싸 청크 갱신(step) 직후 파라미터를 누적한다. 학습 파라미터 · Adam 상태는 바꾸지 않는다."""

    def __init__(self, opt, model):
        self.opt, self.model = opt, model; self.reset()

    def reset(self):
        self.sum = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}; self.n = 0

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.opt.step()
        with torch.no_grad():
            for k, v in self.model.named_parameters():
                self.sum[k] += v.detach()
        self.n += 1

    def average(self):
        return {k: (v / self.n).clone() for k, v in self.sum.items()}


class MonthFrames:
    """월 표 목록 대용 (메모리 절약): i 번째 = 이어 붙인 표의 slices[i] 행 사본."""

    def __init__(self, big: pd.DataFrame, slices):
        self.big, self.slices = big, list(slices)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return MonthFrames(self.big, self.slices[i])
        return self.big.iloc[self.slices[i]].reset_index(drop=True)

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def load_months(months, idx_dir, columns) -> pd.DataFrame:
    """월 parquet 를 pyarrow 로 이어 붙여 한 번만 pandas 로 바꾼다."""
    t = pa.concat_tables([pq.read_table(os.path.join(idx_dir, f"oco_{ym}.parquet"), columns=columns) for ym in months])
    return t.to_pandas(self_destruct=True)


def _mask_sha(*masks) -> str:
    return hashlib.sha256(b"".join(np.packbits(np.asarray(m, bool)).tobytes() for m in masks)).hexdigest()


def _replace_keep_prev(src, dst) -> None:
    """dst 가 있으면 .prev 로 보관하고 src 를 dst 로 교체 (어느 순간에 끊겨도 dst 또는 dst.prev 는 완전한 판)."""
    if os.path.exists(dst):
        shutil.copyfile(dst, dst + ".prev.tmp"); os.replace(dst + ".prev.tmp", dst + ".prev")
    os.replace(src, dst)


def _ckpt_path(ckpt, tag):
    return os.path.join(ckpt["dir"], f"ckpt_{tag}.pt")


def _ckpt_save(ckpt, tag, state) -> None:
    p = _ckpt_path(ckpt, tag); os.makedirs(ckpt["dir"], exist_ok=True)
    torch.save(state, p + ".tmp"); _replace_keep_prev(p + ".tmp", p)


def _ckpt_load(ckpt, tag):
    """저장본 · .prev 중 읽히는 것의 가장 늦은 에폭 (없으면 None). 파일이 있는데 모두 읽히지 않으면 중단."""
    p = _ckpt_path(ckpt, tag); best, bad = None, []
    for f in (p, p + ".prev"):
        if not os.path.exists(f):
            continue
        try:
            s = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            bad.append(f); print(f"  저장본 읽기 실패 {f}: {e!r}", flush=True); continue
        if best is None or s["ep"] > best["ep"]:
            best = s
    if bad and best is None:
        raise RuntimeError(f"[{tag}] 저장본이 모두 읽기 실패 {bad}")
    return best


def load_ckpt_params(model, params) -> None:
    """저장본에서 학습 파라미터만 복원한다. 버퍼(격자 상수)는 지금 모델과 같은지만 확인."""
    pn = {k for k, _ in model.named_parameters()}; bufs = dict(model.named_buffers())
    for k, v in bufs.items():
        assert k in params and torch.equal(params[k].detach().cpu(), v.detach().cpu()), f"저장본 버퍼 {k} 가 지금 모델과 다름"
    missing, unexpected = model.load_state_dict({k: v for k, v in params.items() if k in pn}, strict=False)
    assert set(missing) == set(bufs) and not unexpected, f"파라미터 복원 불일치: 빠짐 {sorted(set(missing) - set(bufs))} · 남음 {unexpected}"


def fit(months, dfs, slices, idx_dir, era5_dir, stats, bg_scaler, y, mu, tr, val, te, seed, d, epochs, lr, chunk, px_frac, no_phys, dev, tag, qa_min, out_dir, patience, ckpt=None):
    assert val.sum() > 0 and te.sum() > 0 and tr.sum() > 0, f"빈 마스크: train {tr.sum()} val {val.sum()} test {te.sum()}"
    torch.manual_seed(seed); np.random.seed(seed)
    model = M.Model(d, dev, n_bg=L.N_BG).to(dev)
    if no_phys: model.phys.advect = lambda h, u, v: h; model.phys.diffuse = lambda h: h
    opt = AvgOpt(torch.optim.Adam(model.parameters(), lr), model); best = (np.inf, -1, None, np.nan); hist = []
    cache = {}  # 달 시작 상태 캐시
    args = (months, dfs, slices, idx_dir, era5_dir, *stats, *bg_scaler, y - mu, tr)
    ep0, stop = 0, None
    if ckpt is not None:
        meta = dict(ckpt["meta"], tag=tag, seed=seed, n_rows=len(y), mu=float(mu), split_sha=_mask_sha(tr, val, te))
        s = _ckpt_load(ckpt, tag) if ckpt["resume"] else None
        if s is not None:
            bad = {k: (s["meta"].get(k), v) for k, v in meta.items() if s["meta"].get(k) != v}
            assert not bad, f"저장본 구성 불일치 {bad}"
            load_ckpt_params(model, s["params"]); opt.opt.load_state_dict(s["adam"])
            cache = {int(k): v.to(dev) for k, v in s["cache"].items()}; hist = s["hist"]; ep0, stop = s["ep"] + 1, s["stop"]
            b = s["best"]; best = (b[0], b[1], None if b[2] is None else {k: v.to(dev) for k, v in b[2].items()}, b[3])
            torch.set_rng_state(s["rng_cpu"]); np.random.set_state(s["rng_np"])  # 난수 복원은 모델 구성 뒤 맨 마지막
            print(f"  [{tag}] 이어하기: 에폭 {ep0} 부터 (best ep {best[1]} val {best[0]:.4f})", flush=True)
    for ep in range(ep0, epochs if stop is None else ep0):
        t1 = time.time(); model.train(); opt.reset()
        order = list(range(len(months))) if ep == 0 else [int(i) for i in np.random.default_rng(seed * 1000 + ep).permutation(len(months))]
        pred, cache = rollout_random(model, *args, opt, chunk, px_frac, dev, qa_min, order, cache, ep == 0)
        r = pred - (y - mu); rt = np.sqrt(np.nanmean(r[tr] ** 2)); rv = np.sqrt(np.nanmean(r[val] ** 2)); rte = np.sqrt(np.nanmean(r[te] ** 2))
        cand = opt.average() if opt.n > 0 else {k: v.detach().clone() for k, v in model.named_parameters()}  # 이 에폭의 후보 = 에폭 평균
        t2 = time.time(); mc = copy.deepcopy(model); mc.load_state_dict(cand, strict=False); mc.eval()
        with torch.no_grad():
            pv = rollout_all(mc, *args, None, chunk, 1.0, dev, qa_min)
        sel = float(np.sqrt(np.nanmean((pv - (y - mu))[val] ** 2))); del mc, pv
        el = time.time() - t1; hist.append((ep, rt, rv, rte, el))
        print(f"  [{tag}] ep {ep:2d} train {rt:.3f} val {rv:.3f} test {rte:.3f} β {model.beta.item():+.3f} {el:.0f}s · val(선택) {sel:.3f} ({time.time() - t2:.0f}s)", flush=True)
        if sel < best[0]:
            best = (sel, ep, cand, rv)
        if not np.isfinite(sel): stop = "nan"
        elif patience and ep - best[1] >= patience: stop = "patience"
        if ckpt is not None:
            _ckpt_save(ckpt, tag, dict(ep=ep, stop=stop, meta=meta, params=model.state_dict(), adam=opt.opt.state_dict(), cache=cache, best=best, hist=hist,
                                       rng_cpu=torch.get_rng_state(), rng_np=np.random.get_state()))
        if stop == "nan": print("  val RMSE NaN → 중단", flush=True); break
        if stop == "patience":
            print(f"  patience {patience}: ep {best[1]} 이후 val 갱신 없음 → 종료", flush=True); break
    if best[2] is None:
        raise RuntimeError(f"[{tag}] 유효한 val 에폭 없음 (epochs={epochs}, hist={len(hist)})")
    model.load_state_dict(best[2], strict=False); model.eval()
    torch.save({"params": {k: v.detach().cpu() for k, v in best[2].items()}, "tag": tag, "best_ep": best[1], "months": months, "n_bg": L.N_BG, "physics": not no_phys},
               os.path.join(out_dir, f"state_{tag}.pt"))
    with torch.no_grad(): pred = rollout_all(model, *args, None, chunk, 1.0, dev, qa_min)
    r = pred - (y - mu); k = torch.exp(model.phys.logk).item(); D = model.phys.D().detach().cpu().numpy()
    D_med, D0 = (np.nan, np.nan) if no_phys else (float(np.median(D)), float(D[0]))  # 수송 끔: 확산을 쓰지 않으므로 D 는 기록하지 않음
    return dict(best_ep=best[1], val_online=best[3], val_select=best[0], train_clean=float(np.sqrt(np.nanmean(r[tr] ** 2))), test_clean=float(np.sqrt(np.nanmean(r[te] ** 2))),
                test_bias=float(np.nanmean(r[te])), beta=float(model.beta.item()), val_clean=float(np.sqrt(np.nanmean(r[val] ** 2))), D_med=D_med, D0=D0, inv_k_h=1 / k / 3600,
                epochs_run=len(hist), epochs_max=epochs, patience=patience, sec_per_epoch=float(np.mean([h[4] for h in hist]))), pred + mu


def main():
    ap = argparse.ArgumentParser(description="물리 주입 잠재 격자 그래프 모델 학습 · 평가")
    ap.add_argument("--months", default="202001-202412", help="쉼표 목록 또는 YYYYMM-YYYYMM")
    ap.add_argument("--configs", default="space:0", help="분할 scheme:fold, 쉼표 구분 (scheme ∈ time · space · year)")
    ap.add_argument("--seeds", default="0"); ap.add_argument("--vfold", type=int, default=1)
    ap.add_argument("--d", type=int, default=8); ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--chunk", type=int, default=48); ap.add_argument("--px-frac", type=float, default=0.15)
    ap.add_argument("--qa-min", type=float, default=0.75); ap.add_argument("--device", default="cpu")
    ap.add_argument("--ablation", action="store_true", help="수송 켬 · 끔을 차례로 학습")
    ap.add_argument("--nophys-only", action="store_true", help="수송 끔만 학습")
    ap.add_argument("--idx-dir", default=L.DEFAULT_IDX); ap.add_argument("--era5-dir", default=L.DEFAULT_ERA5); ap.add_argument("--out", default="runs/gnn")
    ap.add_argument("--ckpt-dir", default=None, help="에폭 끝마다 이어하기 저장본을 둘 폴더 (구성 1개 · seed 1개일 때)")
    ap.add_argument("--resume", action="store_true", help="--ckpt-dir 저장본이 있으면 다음 에폭부터 이어서")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    assert not (a.nophys_only and a.ablation), "--nophys-only 와 --ablation 은 함께 쓰지 않음"
    assert not a.resume or a.ckpt_dir, "--resume 은 --ckpt-dir 와 함께"
    assert not a.ckpt_dir or (len(a.configs.split(",")) == 1 and len(a.seeds.split(",")) == 1 and not a.ablation), "--ckpt-dir 는 구성 1개 · seed 1개 · --ablation 없음"
    print(f"torch {torch.__version__} · 장치 {dev} · 스레드 {torch.get_num_threads()}", flush=True)
    months = parse_months(a.months); st = L.load_no2_stats(a.idx_dir); stats = (float(st["mean"]), float(st["std"]))
    big = load_months(months, a.idx_dir, L.OCO_COLS); n = np.cumsum([0] + [pq.ParquetFile(os.path.join(a.idx_dir, f"oco_{ym}.parquet")).metadata.num_rows for ym in months])
    assert n[-1] == len(big), f"월 행 수 합 {n[-1]} ≠ 이어 붙인 표 {len(big)}"
    slices = [slice(int(n[i]), int(n[i + 1])) for i in range(len(months))]; dfs = MonthFrames(big, slices)
    y = big.xco2.to_numpy(np.float64); lab = (big.label == "L_train").to_numpy()
    ckpt = None
    if a.ckpt_dir:  # 이어하기 구성 검사용: 결과를 바꾸는 인자 전부
        ckpt = dict(dir=a.ckpt_dir, resume=a.resume, meta=dict(months=",".join(months), configs=a.configs, vfold=a.vfold, d=a.d, epochs=a.epochs, lr=a.lr, chunk=a.chunk,
                                                               px_frac=a.px_frac, qa_min=a.qa_min, patience=a.patience, no2_mean=stats[0], no2_std=stats[1]))
    print(f"개월 {len(months)} ({months[0]}–{months[-1]}) · 사운딩 {len(big):,} (L_train {lab.sum():,}) · NO₂ μ {stats[0]:.3e} σ {stats[1]:.3e} · 준비 {time.time()-t0:.0f}s", flush=True)
    rows = []
    for cfg in a.configs.split(","):
        scheme, fold = cfg.split(":"); fold = int(fold)
        tr, val, te = nested_masks(big, scheme, fold, lab, a.vfold)
        mu = y[tr].mean(); bg_scaler = L.fit_bg_scaler(dfs, [tr[s] for s in slices])
        print(f"\n== {scheme} fold {fold} (val = {'space' if scheme == 'time' else 'time'} fold {a.vfold}): train {tr.sum():,} / val {val.sum():,} / test {te.sum():,} · μ_train {mu:.4f}", flush=True)
        for seed in [int(s) for s in a.seeds.split(",")]:
            for no_phys in ((False, True) if a.ablation else (True,) if a.nophys_only else (False,)):
                tag = f"{scheme}{fold}_s{seed}_{'nophys' if no_phys else 'phys'}"
                r, pred = fit(months, dfs, slices, a.idx_dir, a.era5_dir, stats, bg_scaler, y, mu, tr, val, te, seed, a.d, a.epochs, a.lr, a.chunk, a.px_frac, no_phys, dev, tag, a.qa_min, a.out, a.patience, ckpt)
                r.update(scheme=scheme, fold=fold, seed=seed, physics=not no_phys, months=f"{months[0]}-{months[-1]}", mu_train=float(mu), no2_mean=st["mean"], no2_std=st["std"]); rows.append(r)
                pd.DataFrame(dict(row_idx=big.row_idx, pred=pred, xco2=y, test=te, test_all=te)).to_parquet(f"{a.out}/pred_{tag}.parquet", index=False)
                print(f"  → {tag}: best ep {r['best_ep']} val {r['val_select']:.3f} test {r['test_clean']:.3f} (편향 {r['test_bias']:+.3f}) β {r['beta']:+.3f} 1/k {r['inv_k_h']:.1f} h · {r['sec_per_epoch']:.0f} s/에폭", flush=True)
                pd.DataFrame(rows).to_csv(f"{a.out}/summary.csv", index=False)
    print(f"\n총 {time.time()-t0:.0f}s → {a.out}/summary.csv")


if __name__ == "__main__":
    main()
