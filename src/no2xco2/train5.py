"""5년 학습 루프 (D2·D3·D4 적용 골격): 월 순차 warm-start 롤아웃 · 중첩 검증 조기 종료 · 깨끗한 평가 · (옵션) 절제 + B1.

- 입력: build_index 산출 (idx_dir) + ERA5 z100 (era5_dir). 월 단위로 로드해 메모리를 월 1개분으로 유지.
- 분할: 전 개월 사운딩을 이어 붙여 nested_masks (time/space 버퍼가 월 경계를 넘어 적용됨).
- 표준화: NO₂ 는 전 기간 μ·σ (no2_stats.json); 배경 loader.n_bg(harmonics) 변수(loader.BG_FEATS 11개 + --harmonics 의 2..N 차 sin/cos)는 훈련 행에서만 fit (D4).
- 롤아웃: 에폭마다 첫 달 h=0, 이후 달은 전월 말 h 를 이어받음 (D3). 청크 절단 BPTT 는 run() 그대로.
  이 설명은 기본 --month-order asc 학습 롤아웃과 깨끗한 평가(항상 asc) 경로다. desc·random 학습 롤아웃은 아래 진단 옵션·DR-2 참조.
- 옵션 (기본 = 현행): --patience (결정 3-a) · --beta-mode month (월별 β_m, 결정 7-a) · --weight scene (장면 균등 손실 가중, 결정 13-a) — 승인 09-24.
  --harmonics N (배경항 g 에 2..N 차 연주기 조화 sin/cos 추가, TS-1 승인 09-28; 기본 1 = 현행 11변수).
  진단 옵션 (09-28, 순차 학습 표류 판별): --month-order desc (학습 에폭만 월 역순 12→1; 달이 비인접이라 학습 롤아웃은 매월 콜드스타트,
  깨끗한 평가는 오름차순 그대로) · --init-state PATH (저장 state 의 파라미터로 시작; Adam 모멘트는 새로 시작).
- 순차 학습 표류 대응 (승인 09-29, docs/decision_drift_gate_2026-09-28.md):
  DR-1 --avg-epoch: 에폭마다 청크 갱신(opt.step) 직후 파라미터를 누적 → 에폭 말 균등 평균(첫 갱신 전 값은 불포함, 에폭마다 초기화).
    학습은 원 파라미터·Adam 상태로 계속. 평균 가중치를 state 저장·깨끗한 평가에 쓴다. 선택 지표 --avg-val:
      online (기본) = 현행 온라인 val(에폭 중 예측; 평균 가중치는 선택 시점에 채점되지 않음, 추가 롤아웃 없음)
      clean = 에폭마다 평균 가중치(--avg-epoch 없으면 에폭 말 가중치)의 사본으로 no_grad 오름차순 롤아웃 1회 → 그 val 로 선택
              (화소 --val-px-frac, 1 미만이면 --val-px-seed 고정 torch.Generator). 에폭당 롤아웃 1회 추가. val_online 은 계속 기록.
  DR-2 --month-order random: 에폭 0 은 오름차순 연속 롤아웃(현행과 같음)으로 달 시작 상태 캐시를 만들고, 에폭 e ≥ 1 은 12개월을
    np.random.default_rng(seed·1000 + e) 순열로 돈다. 달 m 의 시작 상태 = 캐시[m] = m−1월을 **가장 최근에 돈 직후의 끝 상태**
    (이번 에폭에 m−1월이 먼저 돌았으면 이번 값, 아니면 직전 에폭 값 — 연구책임자 수용 09-29, 설계 §7). 시간상 앞 달이 없는 달(2020-01 · 비인접)은 콜드스타트.
  깨끗한 평가는 두 옵션과 무관하게 항상 오름차순 연속 롤아웃.
  평가(test_clean · B1 · val_online)는 옵션과 무관하게 무가중 RMSE.
- 출력: <out>/summary.csv, state_<tag>.pt, pred_<tag>.parquet (row_idx, pred, xco2, test, test_all).
  pred 열 정의: test = summary 의 test_clean 표본(te; --eval-months 가 있으면 그 달로 한정 — viz.learning_curve_table 이 고정 테스트 집합으로 사용),
  test_all = 폴드 전체 테스트(te_all = summary 의 test_clean_all 표본). --eval-months 가 없으면 두 열이 같다 (QA A2 2026-09-24).
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch

import no2xco2.model as M
from no2xco2.data import loader as L
from no2xco2.train import block_bootstrap, day_blocks, nested_masks


def parse_months(s: str) -> list[str]:
    if "-" in s and len(s) == 13:  # 202001-202412
        a, b = s.split("-"); out = []; y, m = int(a[:4]), int(a[4:])
        while f"{y}{m:02d}" <= b:
            out.append(f"{y}{m:02d}"); m += 1
            if m == 13: y, m = y + 1, 1
        return out
    return sorted(set(s.split(",")))  # warm start 는 시간 순서를 전제


def scene_weights(df: pd.DataFrame, tr: np.ndarray) -> np.ndarray:
    """결정 13-a (승인 09-24): 훈련 사운딩 가중 = 1 / (같은 장면의 훈련 사운딩 수), 장면 = (snd_operation_mode, step_g, 0.25° 셀).
    훈련 행 평균이 1 이 되도록 정규화, 비훈련 행 = 0 (손실에 쓰이지 않음)."""
    ci = np.floor((df.latitude.to_numpy() - 20) / 0.25).astype(np.int64); cj = np.floor((df.longitude.to_numpy() - 100) / 0.25).astype(np.int64)
    key = pd.DataFrame(dict(m=df.snd_operation_mode.to_numpy(), s=df.step_g.to_numpy(), ci=ci, cj=cj))[tr]
    n = key.groupby(["m", "s", "ci", "cj"])["m"].transform("size").to_numpy()
    w = np.zeros(len(df)); w[tr] = 1.0 / n; w[tr] /= w[tr].mean()
    return w


def rollout_all(model, months, dfs, slices, idx_dir, era5_dir, mu_no2, sd_no2, bg_mean, bg_std, y, tr_mask, opt, chunk, px_frac, dev, qa_min, w=None, harmonics=1, gen=None):
    """전 개월 순차 롤아웃 (warm start). 반환: 전체 예측 벡터."""
    pred = np.full(len(y), np.nan); h = None; h0_prev_end = None
    for ym, df, sl in zip(months, dfs, slices):
        if h0_prev_end is not None and L.month_h0(ym) != h0_prev_end:  # 비인접 월: 공백을 1 h 로 취급하지 않고 콜드스타트 (QA T5)
            h = None
        px = L.load_px(ym, idx_dir, mu_no2, sd_no2, dev, qa_min); qry = L.load_qry(df, dev, harmonics); L.apply_bg_scaler(qry, bg_mean, bg_std)
        U, V, T = L.load_wind(ym, era5_dir, dev); h0_prev_end = L.month_h0(ym) + T
        p, h = M.run(model, px, qry, U, V, T, y[sl], tr_mask[sl], opt, chunk=chunk, px_frac=px_frac, h0=h, return_h=True, w=None if w is None else w[sl], gen=gen)
        pred[sl] = p; del px, qry, U, V
    return pred


def rollout_dr(model, months, dfs, slices, idx_dir, era5_dir, mu_no2, sd_no2, bg_mean, bg_std, y, tr_mask, opt, chunk, px_frac, dev, qa_min, w, harmonics, order, cache, chain):
    """DR-2 롤아웃. order = 달 인덱스 순서. chain=True: 오름차순 연속(달 시작 = 같은 에폭 앞 달 끝 상태, rollout_all 과 같음),
    chain=False: 달 시작 = cache.get(i) — 가장 최근 값(이번 에폭에 앞 달이 먼저 돌았으면 이번 값); 없으면 콜드스타트.
    반환: (예측, 갱신된 캐시 {i: i 달 시작 상태})."""
    pred = np.full(len(y), np.nan); new = dict(cache); h = None; prev_i = None
    for i in order:
        ym, df, sl = months[i], dfs[i], slices[i]
        if chain:
            h0 = h if (prev_i is not None and prev_i + 1 == i and i in new) else None
        else:
            h0 = new.get(i)
        px = L.load_px(ym, idx_dir, mu_no2, sd_no2, dev, qa_min); qry = L.load_qry(df, dev, harmonics); L.apply_bg_scaler(qry, bg_mean, bg_std)
        U, V, T = L.load_wind(ym, era5_dir, dev)
        p, h = M.run(model, px, qry, U, V, T, y[sl], tr_mask[sl], opt, chunk=chunk, px_frac=px_frac, h0=h0, return_h=True, w=None if w is None else w[sl])
        pred[sl] = p; del px, qry, U, V
        j = i + 1  # 시간상 다음 달이 인접이면 그 달의 시작 상태로 저장 (비인접 = 캐시 없음 → 콜드스타트, QA T5 와 같은 규칙)
        if j < len(months) and L.month_h0(months[j]) == L.month_h0(ym) + T:
            new[j] = h.detach().clone()
        prev_i = i
    return pred, new


class AvgOpt:
    """DR-1: 원 옵티마이저를 감싸 청크 갱신(step) 직후 파라미터를 누적한다. 학습 파라미터·Adam 상태는 바꾸지 않는다."""

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


def fit(months, dfs, slices, idx_dir, era5_dir, stats, bg_scaler, y, mu, tr, val, te, seed, d, epochs, lr, chunk, px_frac, no_phys, dev, tag, qa_min, max_epoch_s, te_all=None, out_dir=None, patience=None, beta_mode="scalar", w=None, harmonics=1, month_order="asc", init_state=None, avg_epoch=False, avg_val="online", val_px_frac=1.0, val_px_seed=0):
    te_all = te if te_all is None else te_all
    assert val.sum() > 0 and te.sum() > 0 and tr.sum() > 0, f"빈 마스크: train {tr.sum()} val {val.sum()} test {te.sum()}"  # QA R2
    torch.manual_seed(seed); np.random.seed(seed)
    model = M.Model(d, dev, n_bg=L.n_bg(harmonics), n_beta=12 if beta_mode == "month" else 1).to(dev)
    if init_state:  # 진단: 저장 state 에서 이어 학습 (구성 일치 검사)
        init_state = init_state.format(tag=tag)  # "{tag}" 자리표시 허용 → --ablation 시 물리·절제가 각자 state 에서 시작
        s0 = torch.load(init_state, weights_only=False)
        assert s0.get("harmonics", 1) == harmonics and s0.get("n_bg") == L.n_bg(harmonics) and s0.get("beta_mode", "scalar") == beta_mode and s0.get("physics", True) == (not no_phys), f"init_state 구성 불일치: {init_state}"
        model.load_state_dict(s0["params"], strict=False)
    if no_phys: model.phys.advect = lambda h, u, v: h; model.phys.diffuse = lambda h: h
    opt = torch.optim.Adam(model.parameters(), lr); best = (np.inf, -1, None); hist = []
    if avg_epoch:  # DR-1
        opt = AvgOpt(opt, model)
    cache = {}  # DR-2 달 시작 상태 캐시 (month_order random 에서만 사용)
    args = (months, dfs, slices, idx_dir, era5_dir, *stats, *bg_scaler, y - mu, tr)
    rest = (idx_dir, era5_dir, *stats, *bg_scaler, y - mu, tr)
    args_train = args if month_order == "asc" else (months[::-1], dfs[::-1], slices[::-1], *rest)  # 진단: 학습 에폭 월 순서
    for ep in range(epochs):
        t1 = time.time(); model.train()
        if avg_epoch:
            opt.reset()
        if month_order == "random":  # DR-2: 에폭 0 오름차순 연속(캐시 생성), 이후 시드 고정 순열 + 최근 캐시
            order = list(range(len(months))) if ep == 0 else [int(i) for i in np.random.default_rng(seed * 1000 + ep).permutation(len(months))]
            pred, cache = rollout_dr(model, *args, opt, chunk, px_frac, dev, qa_min, w, harmonics, order, cache, ep == 0)
        else:
            pred = rollout_all(model, *args_train, opt, chunk, px_frac, dev, qa_min, w, harmonics)
        r = pred - (y - mu); rt = np.sqrt(np.nanmean(r[tr] ** 2)); rv = np.sqrt(np.nanmean(r[val] ** 2)); rte = np.sqrt(np.nanmean(r[te] ** 2))
        cand = opt.average() if (avg_epoch and opt.n > 0) else {k: v.detach().clone() for k, v in model.named_parameters()}  # 이 에폭의 저장 후보 (DR-1 = 평균)
        sel, sel_s = rv, ""
        if avg_val == "clean":  # DR-1 깨끗한 val 선택: 후보 파라미터 사본으로 no_grad 오름차순 롤아웃
            t2 = time.time(); mc = copy.deepcopy(model); mc.load_state_dict(cand, strict=False); mc.eval()
            g = torch.Generator().manual_seed(val_px_seed) if val_px_frac < 1.0 else None
            with torch.no_grad():
                pv = rollout_all(mc, *args, None, chunk, val_px_frac, dev, qa_min, harmonics=harmonics, gen=g)
            sel = float(np.sqrt(np.nanmean((pv - (y - mu))[val] ** 2))); sel_s = f" · val_clean_sel {sel:.3f} ({time.time() - t2:.0f}s)"; del mc, pv
        el = time.time() - t1; hist.append((ep, rt, rv, rte, el))
        print(f"  [{tag}] ep {ep:2d} train {rt:.3f} val {rv:.3f} test {rte:.3f} β {model.beta.mean().item():+.3f}{'(월 평균)' if model.beta.numel() > 1 else ''} {el:.0f}s{sel_s}", flush=True)
        if sel < best[0]:  # 선택 지표: avg_val online = 온라인 val (현행), clean = 후보 파라미터의 깨끗한 val. 저장 = 후보
            best = (sel, ep, cand, rv)
        if not np.isfinite(sel): print("  val RMSE NaN → 중단 (예측 NaN 또는 val 결측)", flush=True); break  # QA R2
        if el > max_epoch_s: print(f"  에폭 {max_epoch_s} s 초과 → 중단", flush=True); break
        if patience and ep - best[1] >= patience:  # 결정 3-a (승인 09-24): 선택 지표(val_select = avg_val online 이면 val_online, clean 이면 깨끗한 val) 최저가 patience 에폭 동안 갱신 없으면 종료
            print(f"  patience {patience}: ep {best[1]} 이후 val_select({avg_val}) 갱신 없음 → 종료", flush=True); break
    if best[2] is None:  # QA R2: 유효 에폭 0 → 조용한 크래시 대신 명시적 실패 기록
        raise RuntimeError(f"[{tag}] 유효한 val 에폭 없음 (epochs={epochs}, hist={len(hist)})")
    model.load_state_dict(best[2], strict=False); model.eval()
    if out_dir is not None:  # 최저 val 파라미터 저장 (계획 A-4 사후 진단용; 수치 정의 변경 아님, d = 8 에서 ≈ 23 KB 측정). physics = 절제 여부 (QA A6)
        torch.save({"params": best[2], "tag": tag, "best_ep": best[1], "months": months, "n_bg": L.n_bg(harmonics), "harmonics": harmonics, "month_order": month_order, "avg_epoch": avg_epoch, "avg_val": avg_val, "val_px_frac": val_px_frac if avg_val == "clean" else None, "physics": not no_phys, "beta_mode": beta_mode, "weight": "none" if w is None else "scene"}, os.path.join(out_dir, f"state_{tag}.pt"))
    with torch.no_grad(): pred = rollout_all(model, *args, None, chunk, 1.0, dev, qa_min, harmonics=harmonics)
    r = pred - (y - mu); k = torch.exp(model.phys.logk).item(); D = model.phys.D().detach().cpu().numpy()  # 유효 D = D_cap·σ(θ_D) (승인 S1(b))
    cache_mb = sum(v.numel() * v.element_size() for v in cache.values()) / 1e6
    b = model.beta.detach().cpu().numpy(); bx = {}
    if b.size > 1:  # 월별 β: summary beta = 훈련 행 가중 평균 (정의 명시), 월별 값은 beta_m01–12 열
        mon = np.concatenate([pd.to_datetime(d.time).dt.month.to_numpy() - 1 for d in dfs]); cnt = np.bincount(mon[tr], minlength=12)
        bx = {f"beta_m{i + 1:02d}": float(b[i]) if cnt[i] > 0 else np.nan for i in range(12)}  # 훈련 행 0 인 달 = 학습 안 된 초기값 → NaN (QA X4)
        beta = float((b * cnt).sum() / cnt.sum())
    else:
        beta = float(b[0])
    D_med, D0 = (np.nan, np.nan) if no_phys else (float(np.median(D)), float(D[0]))  # 절제: 확산 미사용 → D 는 학습 안 된 초기값이라 기록하지 않음 (QA X3). k 는 절제에서도 적용
    return dict(best_ep=best[1], val_online=best[3], val_select=best[0], val_select_kind=avg_val, val_px_frac=val_px_frac if avg_val == "clean" else np.nan, train_clean=float(np.sqrt(np.nanmean(r[tr] ** 2))), test_clean=float(np.sqrt(np.nanmean(r[te] ** 2))), test_clean_all=float(np.sqrt(np.nanmean(r[te_all] ** 2))), test_bias=float(np.nanmean(r[te])),
                beta=beta, **bx, beta_mode=beta_mode, weight="none" if w is None else "scene", harmonics=harmonics, month_order=month_order, init_state=init_state or "", avg_epoch=avg_epoch, val_clean=float(np.sqrt(np.nanmean(r[val] ** 2))), cache_mb=cache_mb, D_med=D_med, D0=D0, inv_k_h=1 / k / 3600, epochs_run=len(hist), epochs_max=epochs, patience=patience or 0,
                sec_per_epoch=float(np.mean([h[4] for h in hist]))), pred + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="202001", help="쉼표 목록 또는 YYYYMM-YYYYMM")
    ap.add_argument("--configs", default="time:2"); ap.add_argument("--seeds", default="0"); ap.add_argument("--vfold", type=int, default=1)
    ap.add_argument("--d", type=int, default=8); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--chunk", type=int, default=48); ap.add_argument("--px-frac", type=float, default=0.15); ap.add_argument("--device", default="cpu")
    ap.add_argument("--qa-min", type=float, default=0.75); ap.add_argument("--ablation", action="store_true", help="절제(물리 없음)도 학습해 B1 산출")
    ap.add_argument("--idx-dir", default=L.DEFAULT_IDX); ap.add_argument("--era5-dir", default=L.DEFAULT_ERA5); ap.add_argument("--out", default="experiments/train5")
    ap.add_argument("--max-epoch-s", type=float, default=1e9)
    ap.add_argument("--beta-mode", choices=["scalar", "month"], default="scalar", help="결정 7-a: month = 월별 β_m 12개 (기본 scalar = 현행)")
    ap.add_argument("--weight", choices=["none", "scene"], default="none", help="결정 13-a: scene = 장면(모드·시각·0.25° 셀) 균등 손실 가중 (기본 none = 현행)")
    ap.add_argument("--harmonics", type=int, default=1, help="TS-1: 배경항 g 연주기 조화 최고 차수 (1 = 현행 11변수, 3 = 2·3차 sin/cos 추가 15변수)")
    ap.add_argument("--month-order", choices=["asc", "desc", "random"], default="asc", help="학습 에폭 월 순서 (기본 asc = 현행). desc = 진단, random = DR-2 (승인 09-29). 깨끗한 평가는 항상 asc")
    ap.add_argument("--avg-val", choices=["online", "clean"], default="online", help="DR-1 선택 지표: online (현행) / clean (에폭마다 후보 가중치로 깨끗한 val 롤아웃)")
    ap.add_argument("--val-px-frac", type=float, default=1.0, help="--avg-val clean 의 화소 비율 (1 미만이면 --val-px-seed 고정)")
    ap.add_argument("--val-px-seed", type=int, default=0)
    ap.add_argument("--avg-epoch", action="store_true", help="DR-1 (승인 09-29): 에폭 내 청크 갱신 후 파라미터 균등 평균을 저장·깨끗한 평가에 사용")
    ap.add_argument("--init-state", default=None, help="진단: 저장 state 파라미터로 시작 — 경로에 {tag} 자리표시 가능(예: experiments/x/state_{tag}.pt) (구성 일치 검사)")
    ap.add_argument("--patience", type=int, default=0, help="결정 3-a: 선택 지표 val_select(--avg-val online 이면 val_online, clean 이면 깨끗한 val) 최저가 이 에폭 수 동안 갱신 없으면 종료 (0 = 끔, 현행). 승인 조합 = --epochs 100 --patience 10 (09-24); DR 런(--avg-epoch·--month-order random)은 --epochs 200 --patience 10 (결정 3-a 재상정, 10-01)")
    ap.add_argument("--b1-frac", type=float, default=0.1, help="B1 점추정 임계 = b1_frac × 평가 기간 L_train xco2_uncertainty 중앙 (승인 2026-09-22; 60개월 중앙 0.576 → 0.0576)")
    ap.add_argument("--eval-months", default=None, help="테스트 RMSE 를 이 달들(쉼표)로 한정 — 학습 곡선용 고정 테스트 구간. test_clean_all 은 전체")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    months = parse_months(a.months); _st = L.load_no2_stats(a.idx_dir); stats = (float(_st["mean"]), float(_st["std"]))  # 한 번만 읽고 출처(n·months)도 summary 에 기록 (QA R10)
    dfs = [L.load_oco(ym, a.idx_dir, columns=L.OCO_COLS + ["xco2_uncertainty"] + (["snd_operation_mode"] if a.weight == "scene" else [])) for ym in months]
    n = np.cumsum([0] + [len(d) for d in dfs]); slices = [slice(int(n[i]), int(n[i + 1])) for i in range(len(dfs))]
    big = pd.concat(dfs, ignore_index=True); y = big.xco2.to_numpy(np.float64); lab = (big.label == "L_train").to_numpy()
    ym_row = np.concatenate([np.full(len(d), m) for d, m in zip(dfs, months)])
    ev = lab & (np.isin(ym_row, a.eval_months.split(",")) if a.eval_months else True)  # 평가 기간 = --eval-months(있으면) 아니면 로드된 전 개월 (QA S6)
    b1_thr = float(a.b1_frac * np.nanmedian(big.xco2_uncertainty.to_numpy()[ev])); print(f"B1 임계 {b1_thr:.4f} ppm = {a.b1_frac} × 평가 기간 L_train uncertainty 중앙 (n {int(ev.sum()):,})", flush=True)
    days = day_blocks(big.time)  # QA R9: train.py·ct_compare 와 정의 공유
    print(f"개월 {len(months)} ({months[0]}–{months[-1]}) · 사운딩 {len(big):,} (L_train {lab.sum():,}) · NO₂ μ {stats[0]:.3e} σ {stats[1]:.3e} · 준비 {time.time()-t0:.0f}s", flush=True)
    rows = []; preds = {}
    for cfg in a.configs.split(","):
        scheme, fold = cfg.split(":"); fold = int(fold); tr, val, te = nested_masks(big, scheme, fold, lab, a.vfold); te_all = te.copy()
        if a.eval_months:
            te = te & np.isin(ym_row, a.eval_months.split(","))
        mu = y[tr].mean(); bg_scaler = L.fit_bg_scaler(dfs, [tr[s] for s in slices], a.harmonics)
        w = scene_weights(big, tr) if a.weight == "scene" else None
        print(f"\n== {scheme} fold {fold} (val = {'space' if scheme=='time' else 'time'} fold {a.vfold}): train {tr.sum():,} / val {val.sum():,} / test {te.sum():,} · μ_train {mu:.2f}", flush=True)
        for seed in [int(s) for s in a.seeds.split(",")]:
            for no_phys in ((False, True) if a.ablation else (False,)):
                tag = f"{scheme}{fold}_s{seed}_{'nophys' if no_phys else 'phys'}"
                r, pred = fit(months, dfs, slices, a.idx_dir, a.era5_dir, stats, bg_scaler, y, mu, tr, val, te, seed, a.d, a.epochs, a.lr, a.chunk, a.px_frac, no_phys, dev, tag, a.qa_min, a.max_epoch_s, te_all, a.out, a.patience, a.beta_mode, w, a.harmonics, a.month_order, a.init_state, a.avg_epoch, a.avg_val, a.val_px_frac, a.val_px_seed)
                r.update(scheme=scheme, fold=fold, seed=seed, physics=not no_phys, months=f"{months[0]}-{months[-1]}", no2_n=_st["n"], no2_months=_st["months"], no2_mean=_st["mean"], no2_std=_st["std"]); rows.append(r); preds[tag] = pred
                pd.DataFrame(dict(row_idx=big.row_idx, pred=pred, xco2=y, test=te, test_all=te_all)).to_parquet(f"{a.out}/pred_{tag}.parquet", index=False)
                print(f"  → {tag}: best ep {r['best_ep']} val_select {r['val_select']:.3f}({r['val_select_kind']}) val_online {r['val_online']:.3f} test(clean) {r['test_clean']:.3f} (전체 {r['test_clean_all']:.3f}, 편향 {r['test_bias']:+.3f}) β {r['beta']:+.3f} D0 {r['D0']:.0f} 1/k {r['inv_k_h']:.1f}h · {r['sec_per_epoch']:.0f} s/ep", flush=True)
            if a.ablation:
                pa_, pb_ = preds[f"{scheme}{fold}_s{seed}_phys"], preds[f"{scheme}{fold}_s{seed}_nophys"]
                ra, rb = (pa_ - y)[te], (pb_ - y)[te]; ok = np.isfinite(ra) & np.isfinite(rb)
                d_pt = np.sqrt((rb[ok] ** 2).mean()) - np.sqrt((ra[ok] ** 2).mean()); lo, hi = block_bootstrap(ra[ok], rb[ok], days[te][ok])
                rows.append(dict(scheme=scheme, fold=fold, seed=seed, physics="B1", delta=d_pt, ci_lo=lo, ci_hi=hi, b1_thr=b1_thr, pass_B1=bool(lo > 0 and d_pt >= b1_thr)))
                print(f"  B1 {scheme}{fold} s{seed}: ΔRMSE(절제−물리) {d_pt:+.3f} ppm, 95% CI [{lo:+.3f}, {hi:+.3f}]", flush=True)
            pd.DataFrame(rows).to_csv(f"{a.out}/summary.csv", index=False)
    print(f"\n총 {time.time()-t0:.0f}s → {a.out}/summary.csv")


if __name__ == "__main__":
    main()
