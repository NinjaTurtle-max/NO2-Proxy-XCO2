"""물리 전용 롤아웃 (학습 없음): 결정 3의 이류–확산 연산자를 실제 ERA5 바람으로 돌려 안정성·보존성·갭 충전을 측정한다.

상태 = (C, M): C = 농도×존재도, M = 존재도(관측 정보량, 0–1). h = C/M (M>0). NaN 이 보간에서 번지는 것을 막기 위한 표준 처리.
  이류: 반라그랑주 — 출발점 x − u·Δt 를 이중선형 보간 (C, M 동일 연산). 영역 밖 출발점 = 유입 경계 → C=M=0 (정보 없음).
  확산: 명시적 5점 라플라시안 (C, M 동일 연산). 계량 Δx_lon = 0.25°·111.2 km·cos(lat), Δx_lat = 0.25°·111.2 km.
  주입(물리 전용 검사): qa≥0.75 화소를 4점 이중선형 가중으로 퇴적 → 관측 노드는 C=가중평균, M=1 로 대체 (학습 결합 아님).

모드:
  continuous  한 달 연속, 화소 있는 모든 스텝 주입 → 스텝별 커버리지·질량·최대/최소·NaN, 벽시계.
  free-run    첫 주입 스텝(--start 이후) 한 번만 주입 후 48 h 자유 전파 → 시차별 커버리지(M>0.01)·질량 유지율. 이류가 갭을 채우는 능력의 물리 상한.
D 는 검사용 설정값(--D, 기본 5,000 m²/s; 안정 조건 D·Δt/Δx² ≤ 1/4 → D ≤ 4.3×10⁴ @45°N). 모델에서는 학습 파라미터(결정 3).
"""
import argparse, os, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
from no2xco2.config import GRID_LAT0, GRID_D

NLAT, NLON = 121, 201; DT = 3600.0; KM = 111.2e3 * GRID_D
lat = GRID_LAT0 + GRID_D * np.arange(NLAT); coslat = np.cos(np.deg2rad(lat))[:, None]
II, JJ = np.meshgrid(np.arange(NLAT), np.arange(NLON), indexing="ij")
EPS = 1e-2


def deposit(enc: pd.DataFrame):
    num = np.zeros(NLAT * NLON); den = np.zeros(NLAT * NLON)
    for k in range(4):
        np.add.at(num, enc[f"n{k}"].to_numpy(), enc[f"w{k}"].to_numpy() * enc["no2_tvcd"].to_numpy())
        np.add.at(den, enc[f"n{k}"].to_numpy(), enc[f"w{k}"].to_numpy())
    obs = den > 0; val = np.zeros(NLAT * NLON); val[obs] = num[obs] / den[obs]
    return val.reshape(NLAT, NLON), obs.reshape(NLAT, NLON)


def semi_lagrangian(f, u, v):
    si = II - v * DT / KM; sj = JJ - u * DT / (KM * coslat)
    i0 = np.floor(si).astype(int); j0 = np.floor(sj).astype(int); ti = si - i0; tj = sj - j0
    inside = (i0 >= 0) & (i0 < NLAT - 1) & (j0 >= 0) & (j0 < NLON - 1)
    i0c = np.clip(i0, 0, NLAT - 2); j0c = np.clip(j0, 0, NLON - 2)
    out = (1 - ti) * (1 - tj) * f[i0c, j0c] + (1 - ti) * tj * f[i0c, j0c + 1] + ti * (1 - tj) * f[i0c + 1, j0c] + ti * tj * f[i0c + 1, j0c + 1]
    out[~inside] = 0.0
    return out


def diffuse(f, D):
    fp = np.pad(f, 1, mode="edge")  # 경계: 무플럭스
    c = fp[1:-1, 1:-1]; n = fp[:-2, 1:-1]; s = fp[2:, 1:-1]; w = fp[1:-1, :-2]; e = fp[1:-1, 2:]
    dy2 = KM ** 2; dx2 = (KM * coslat) ** 2
    return f + DT * D * ((n - c) / dy2 + (s - c) / dy2 + (w - c) / dx2 + (e - c) / dx2)


def step(C, M, u, v, D):
    C = diffuse(semi_lagrangian(C, u, v), D); M = diffuse(semi_lagrangian(M, u, v), D)
    return C, np.clip(M, 0, 1)


def stats(C, M):
    ok = M > EPS; h = np.where(ok, C / np.maximum(M, EPS), np.nan)
    return dict(coverage=ok.mean(), mass=C.sum(), infosum=M.sum(), hmin=np.nanmin(h) if ok.any() else np.nan, hmax=np.nanmax(h) if ok.any() else np.nan, nan=int(np.isnan(C).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["continuous", "free-run"], default="continuous")
    ap.add_argument("--enc", default="data/processed/index/trop_hi_202001.parquet", help="trop_hi_YYYYMM (step_g·no2) 또는 구 enc_tropomi (step_h·no2_tvcd)")
    ap.add_argument("--wind", default="data/raw/era5_wind/era5_wind_202001_z100.nc")
    ap.add_argument("--D", type=float, default=5000.0); ap.add_argument("--qa", type=float, default=0.75)
    ap.add_argument("--start", type=int, default=240, help="free-run: 이 스텝 이후 첫 주입 스텝에서 시작")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--save-fields", action="store_true", help="continuous: 스텝별 (C, M) float32 npz 저장 → readout 스모크 입력")
    a = ap.parse_args()
    _cols = pq.read_schema(a.enc).names
    import re as _re
    _m = _re.search(r"(\d{6})", os.path.basename(a.enc)); assert _m, f"--enc 파일명에 YYYYMM 없음: {a.enc}"
    _ym = _m.group(1)  # 출력 파일명·바람 월 검사에 사용 (QA 2026-09-24: 이전엔 202001 고정)
    if "step_g" in _cols:  # 새 인덱스(D3 전역 시각): 월 상대 step_h 로 환산, no2 → no2_tvcd
        _h0 = int((pd.Timestamp(f"{_ym[:4]}-{_ym[4:]}-01") - pd.Timestamp("2020-01-01")) / pd.Timedelta(hours=1))
        enc = pq.read_table(a.enc, columns=["step_g", "qa", "no2", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3"]).to_pandas().rename(columns={"no2": "no2_tvcd"})
        enc["step_h"] = (enc.pop("step_g") - _h0).astype("int32")
    else:
        enc = pq.read_table(a.enc, columns=["step_h", "qa", "no2_tvcd", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3"]).to_pandas()
    enc = enc[enc.qa >= a.qa]; groups = {s: g for s, g in enc.groupby("step_h")}
    from no2xco2.data.era5 import open_wind
    ds = open_wind(a.wind, _ym); U = ds["u_pbl"].values; V = ds["v_pbl"].values; T = U.shape[0]  # 위도 오름차순 (2026-09-22 정정) · ym 검사 = --wind 가 --enc 와 같은 달인지 (시간축 길이·시작 시각)
    print(f"[{a.mode}] 주입 스텝 {len(groups)} · 화소 {len(enc):,} (qa≥{a.qa}) · 바람 {T}스텝 · D={a.D:g} m²/s · D·Δt/Δx²(45°N)={a.D*DT/(KM*np.cos(np.deg2rad(45)))**2:.3f}")
    C = np.zeros((NLAT, NLON)); M = np.zeros((NLAT, NLON)); rows = []; t0 = time.time()
    if a.mode == "continuous":
        CF = np.zeros((T, NLAT, NLON), np.float32) if a.save_fields else None; MF = np.zeros_like(CF) if a.save_fields else None
        for t in range(T):
            C, M = step(C, M, U[t], V[t], a.D)
            if t in groups:
                val, obs = deposit(groups[t]); C[obs] = val[obs]; M[obs] = 1.0
            if a.save_fields: CF[t] = C; MF[t] = M
            rows.append(dict(step=t, injected=t in groups, **stats(C, M)))
        if a.save_fields:
            fp = os.path.join(a.out, f"physics_fields_{_ym}_D{int(a.D)}.npz"); np.savez(fp, C=CF, M=MF); print(f"필드 저장: {fp} ({os.path.getsize(fp)/1e6:.0f} MB)")
        df = pd.DataFrame(rows); out = os.path.join(a.out, f"physics_continuous_{_ym}.csv"); df.to_csv(out, index=False); el = time.time() - t0
        print(f"벽시계 {el:.1f}s ({el/T*1000:.0f} ms/스텝) · NaN 발생 스텝 {(df.nan>0).sum()} · h 최대 {df.hmax.max():.3e} (주입 최대 {enc.no2_tvcd.max():.3e}) · h 최소 {df.hmin.min():.3e} (주입 최소 {enc.no2_tvcd.min():.3e})")
        print(f"커버리지(M>{EPS}) 중앙 {df.coverage.median():.1%} · 최소 {df.coverage.min():.1%} · 최대 {df.coverage.max():.1%} · 월말 {df.coverage.iloc[-1]:.1%}")
        print(f"주입 직후 커버리지 중앙 {df[df.injected].coverage.median():.1%} · 주입 스텝의 관측 노드 비율 중앙 {np.median([deposit(g)[1].mean() for g in groups.values()]):.1%}")
    else:
        cand = [s for s in groups if a.start <= s < a.start + 24]
        assert cand, f"free-run: 스텝 [{a.start}, {a.start + 24}) 에 qa≥{a.qa} 주입 스텝 없음 — --start 조정"
        s0 = max(cand, key=lambda s: len(groups[s]))  # 하루 중 화소 최다 스텝 (동률이면 먼저 나온 스텝, 이전과 동일)
        val, obs = deposit(groups[s0]); C[obs] = val[obs]; M[obs] = 1.0
        base = stats(C, M); rows.append(dict(lag=0, **base))
        for k in range(1, 49):
            t = s0 + k
            if t >= T: break
            C, M = step(C, M, U[t], V[t], a.D); rows.append(dict(lag=k, **stats(C, M)))
        df = pd.DataFrame(rows); out = os.path.join(a.out, f"physics_freerun_{_ym}_s{s0}.csv"); df.to_csv(out, index=False)
        print(f"시작 스텝 {s0} (UTC {s0%24:02d}h, {int(_ym[4:])}월 {s0//24+1}일) · 주입 노드 {obs.sum():,} ({obs.mean():.1%})")
        print(f"{'lag h':>5s} {'coverage':>9s} {'mass/m0':>8s} {'info/i0':>8s} {'hmax':>10s}")
        for k in (0, 1, 3, 6, 12, 24, 36, 48):
            r = df[df.lag == k]
            if len(r): r = r.iloc[0]; print(f"{k:5d} {r.coverage:9.1%} {r.mass/base['mass']:8.3f} {r.infosum/base['infosum']:8.3f} {r.hmax:10.3e}")
    print(f"저장: {out}")
