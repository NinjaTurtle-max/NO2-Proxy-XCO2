"""결정 11 ① 합성 스모크 초안 (AI 작성, QA 확인 전 — tests/ 는 QA 소유라 tests/smoke/test_x11.py 로 옮기는 것은 QA 확인 뒤). 데이터 로드·학습 없음, 수 초.
검사: 기본 모델 키 보존 · D3 순서(정상/누출) · 상태 [N,d+1] 이어받기 · D₂ 확산 보존 · 장면 정의·xinj_table 달 경계 · scene_holdout 성질 · upstream_lag."""
import numpy as np
import pandas as pd
import torch

import no2xco2.model as M
from no2xco2.data import loader as L
from no2xco2.data import splits as S

N, NLAT, NLON, T, D = M.N, M.NLAT, M.NLON, 6, 8
Z = torch.zeros(T, NLAT, NLON); NODE = 60 * NLON + 100


def _q(row):
    return dict(node=torch.tensor([NODE, NODE + 1, NODE + NLON, NODE + NLON + 1]), edge=torch.tensor([[1.0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                seg=torch.zeros(4, dtype=torch.long), n=1, bg=torch.zeros(1, 11), month=torch.zeros(1, dtype=torch.long), rows=np.array([row]))


def _xi(vals):  # 합성 주입: 노드 NODE 에 가중 1 (나머지 가중 0), 값 vals
    k = len(vals)
    return dict(node=torch.tensor([NODE] * k + [NODE + 1] * k + [NODE + NLON] * k + [NODE + NLON + 1] * k), w=torch.tensor([1.0] * k + [0.0] * 3 * k), val=torch.tensor(vals * 4))


def _model():
    torch.manual_seed(0); mx = M.Model(D, torch.device("cpu"), n_bg=11, x11=True)
    with torch.no_grad():
        mx.phys.thetaD2.fill_(-40.0)  # D₂ ≈ 0 → 확산 없음, 바람 0 → 이류 없음
    return mx


def _run(mx, xinj, leak):
    qd = {1: _q(0), 2: _q(1), 3: _q(2)}
    with torch.no_grad():
        return M.run(mx, {}, qd, Z, Z, T, np.zeros(3), np.ones(3, bool), None, xinj=xinj, U2=Z, V2=Z, xleak=leak)


def test_default_model_has_no_x11_keys():
    k0 = set(M.Model(D, torch.device("cpu"), n_bg=11).state_dict()); k1 = set(M.Model(D, torch.device("cpu"), n_bg=11, x11=True).state_dict())
    assert not any("gamma" in k or "thetaD2" in k for k in k0) and k1 - k0 == {"gamma", "phys.thetaD2"} and k0 <= k1


def test_d3_order_normal_and_leak():
    mx = _model(); base = _run(mx, {}, False)
    dn = _run(mx, {2: _xi([3.0, 5.0])}, False) - base          # 정상: 장면 마지막 스텝(2)에 주입, 같은 스텝 디코드는 주입을 보지 않음
    dl = _run(mx, {1: _xi([3.0]), 2: _xi([5.0])}, True) - base  # 누출: 자기 스텝 주입 → 디코드
    assert np.allclose(dn, [0, 0, 4], atol=1e-5) and np.allclose(dl, [3, 5, 5], atol=1e-5)


def test_state_carries_hx():
    mx = _model()
    with torch.no_grad():
        _, hE = M.run(mx, {}, {1: _q(0)}, Z, Z, T, np.zeros(1), np.ones(1, bool), None, return_h=True, xinj={2: _xi([3.0, 5.0])}, U2=Z, V2=Z)
        nxt = M.run(mx, {}, {1: _q(0)}, Z, Z, T, np.zeros(1), np.ones(1, bool), None, h0=hE, xinj={}, U2=Z, V2=Z)
        nb = M.run(mx, {}, {1: _q(0)}, Z, Z, T, np.zeros(1), np.ones(1, bool), None, h0=torch.cat([hE[:, :D], torch.zeros(N, 1)], 1), xinj={}, U2=Z, V2=Z)
    assert hE.shape == (N, D + 1) and abs(float(nxt[0] - nb[0]) - 4) < 1e-5


def test_hx_diffusion_conserves():
    mx = M.Model(D, torch.device("cpu"), n_bg=11, x11=True); hx = torch.zeros(N); hx[NODE] = 1.0
    with torch.no_grad():
        h2 = mx.phys.forward_x(hx.reshape(NLAT, NLON), Z[0], Z[0]).reshape(N)
    assert abs(h2.sum().item() - 1) < 1e-4 and h2[NODE].item() < 1.0  # 감쇠 없음(k₂ = 0) · 퍼짐


def _big():
    t = pd.to_datetime(["2020-01-10T05:00", "2020-01-10T05:02", "2020-01-31T23:58", "2020-02-01T00:01", "2020-02-29T23:59", "2020-03-01T00:02"])
    sg = ((t - pd.Timestamp("2020-01-01")) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    return pd.DataFrame(dict(time=t, satellite=["OCO-2"] * 6, step_g=sg, label="L_train", latitude=20.5, longitude=100.5, **{f"n{j}": [j] * 6 for j in range(4)}, **{f"w{j}": [0.25] * 6 for j in range(4)}))


def test_scene_ids_and_xinj_table_month_boundary():
    big = _big(); lab = np.ones(6, bool); sid = S.scene_ids(big, lab)
    assert sid.tolist() == [1, 1, 2, 2, 3, 3]  # 300 s 초과 간격 = 새 장면
    tab, st = L.xinj_table(big, lab.copy(), sid, lab, np.arange(6.0), ["202001", "202002"])
    assert tab.ym.tolist() == ["202001", "202001", "202002", "202002"] and tab.step_h.tolist() == [221, 221, 0, 0]  # 달 경계 걸친 장면 → 뒤 달 첫 스텝
    assert st["n_cross_scenes"] == 1 and st["n_drop_scenes"] == 1  # 2월 말~3월 장면은 로드 범위 밖 → 버림
    tl, _ = L.xinj_table(big, lab.copy(), sid, lab, np.arange(6.0), ["202001", "202002"], leak=True)
    assert tl.step_h.tolist() == [221, 221, 743, 0, 695]  # 누출: 자기 스텝


def test_scene_holdout_properties():
    n_sc, per = 60, 20; t0 = pd.Timestamp("2020-01-01"); rows = []
    for s in range(n_sc):
        start = t0 + pd.Timedelta(days=int(s // 5) * 2 + (s % 5) * 0.2)  # 월 1~2 에 걸친 장면 60개
        rows += [dict(time=start + pd.Timedelta(seconds=float(k)), satellite="OCO-2" if s % 2 else "OCO-3", label="L_train") for k in range(per)]
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True); lab = np.ones(len(df), bool)
    role, sid = S.scene_holdout(df, lab, seed=0); role2, _ = S.scene_holdout(df, lab, seed=0)
    assert (role == role2).all() and set(role) <= {"train", "val", "test"}
    assert pd.DataFrame(dict(s=sid, r=role)).groupby("s").r.nunique().max() == 1  # 한 장면 한 역할
    assert (role == "test").any() and (role == "val").any() and (role == "train").sum() > (role == "test").sum()


def test_upstream_lag_past_only_other_scene():
    t = pd.to_datetime("2020-01-01") + pd.to_timedelta([1.0, 1.0, 3.0, 3.0, 3.2, 3.2], unit="D")
    df = pd.DataFrame(dict(time=t, latitude=20.5, longitude=100.5)); sid = np.array([1, 1, 2, 2, 3, 3]); m = np.ones(6, bool)
    for r in (0.0, 600.0):
        assert np.allclose(S.upstream_lag(df, sid, m, m, r), [np.inf, np.inf, 2.0, 2.0, 0.2, 0.2])
