"""그림 공통 스타일 · 시험 칸 · 글자 겹침 검사 (Fig. 3 이 import).
글자 겹침 검사: 글상자끼리 · 글상자 × 선분(축선 · 0선 · 오차막대) · 캔버스 밖 → 위반이 있으면 False."""
import glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from pathlib import Path

O = str(Path(__file__).resolve().parent / "out")  # 그림 출력 폴더
CM = 1 / 2.54; FS_T, FS_L, FS_K = 24, 19, 16  # 최종 인쇄 크기 기준 pt
C_ON, C_GREY, C_RED = "#1D5F99", "#9AA3AD", "#C0261A"
for f in glob.glob(os.path.expanduser("~/Library/Fonts/Pretendard-*.otf")):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Pretendard", "axes.unicode_minus": True, "font.size": FS_K, "axes.linewidth": 1.2,
                     "xtick.major.width": 1.2, "ytick.major.width": 1.2, "savefig.dpi": 300})

def test_cells():  # space fold 0 시험 칸 (남쪽 위도, 서쪽 경도) — splits.assign 과 같은 순열
    from no2xco2.data import splits as S
    cf = np.random.default_rng(S.SEED).permutation(np.arange(60) % S.K)
    return [(20 + 5 * (b // 10), 100 + 5 * (b % 10)) for b in np.where(cf == 0)[0]]


# ---------------- 겹침 검사 ----------------
def _segs(fig):
    out = []
    for a in fig.findobj(Line2D):
        if not a.get_visible() or a.get_linestyle() in ("None", "none", "") or a.get_gid() == "nochk":
            continue
        xy = a.get_transform().transform(np.asarray(a.get_xydata(), float))
        out += [(xy[i], xy[i + 1]) for i in range(len(xy) - 1) if np.all(np.isfinite(xy[i:i + 2]))]
    for c in fig.findobj(LineCollection):
        if c.get_visible() and c.get_gid() != "nochk":
            out += [(s[i], s[i + 1]) for s in (c.get_transform().transform(np.asarray(p, float)) for p in c.get_segments()) for i in range(len(s) - 1)]
    return out


def _hit(p, q, b):  # Liang–Barsky 선분 × 직사각형
    x0, y0, x1, y1 = b; dx, dy = q[0] - p[0], q[1] - p[1]; t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0: return False
        else:
            r = qq / pp
            if pp < 0: t0 = max(t0, r)
            else: t1 = min(t1, r)
            if t0 > t1: return False
    return True


def check(fig, name, pad=1.5):
    fig.canvas.draw(); rd = fig.canvas.get_renderer(); W, H = fig.canvas.get_width_height()
    skip = set()  # 보기 범위 밖이라 그려지지 않는 눈금 글자는 검사에서 뺀다
    for a in fig.axes:
        for ax_ in (a.xaxis, a.yaxis):
            drawn = {id(t) for t in ax_._update_ticks()}
            skip |= {id(x) for t in ax_.get_major_ticks() + ax_.get_minor_ticks() if id(t) not in drawn or a.get_gid() == "nochk" for x in (t.label1, t.label2, t.tick1line, t.tick2line)}
    tx = [(t, t.get_window_extent(rd)) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t.get_gid() != "nochk" and id(t) not in skip]
    bad = []
    for i, (t, b) in enumerate(tx):
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            bad.append(f"캔버스 밖: {t.get_text()[:30]!r}")
        for t2, b2 in tx[i + 1:]:
            if b.x0 < b2.x1 - pad and b2.x0 < b.x1 - pad and b.y0 < b2.y1 - pad and b2.y0 < b.y1 - pad:
                bad.append(f"글자×글자: {t.get_text()[:25]!r} / {t2.get_text()[:25]!r}")
    segs = _segs(fig); pts = []
    for a in fig.findobj(Line2D):  # 마커(오차막대 끝 · 점) — 지도 축(gid nochk) 제외
        if a.get_visible() and id(a) not in skip and a.get_marker() not in (None, "", "None", " ") and a.get_gid() != "nochk" and (a.axes is None or a.axes.get_gid() != "nochk"):
            pts += list(a.get_transform().transform(np.asarray(a.get_xydata(), float)))
    from matplotlib.collections import PathCollection
    for c in fig.findobj(PathCollection):
        if c.get_visible() and c.axes is not None and c.axes.get_gid() != "nochk":
            pts += list(c.get_offset_transform().transform(np.asarray(c.get_offsets(), float)))
    pts = np.asarray(pts) if pts else np.zeros((0, 2))
    for t, b in tx:
        bb = (b.x0 + pad, b.y0 + pad, b.x1 - pad, b.y1 - pad)
        if any(_hit(p, q, bb) for p, q in segs):
            bad.append(f"글자×선: {t.get_text()[:30]!r}")
        if len(pts) and np.any((pts[:, 0] > bb[0] - 6) & (pts[:, 0] < bb[2] + 6) & (pts[:, 1] > bb[1] - 6) & (pts[:, 1] < bb[3] + 6)):
            bad.append(f"글자×마커: {t.get_text()[:30]!r}")
    print(f"겹침 검사 {name}: 글상자 {len(tx)} · 선분 {len(segs)} · 위반 {len(bad)}")
    for x in bad[:20]:
        print("   ", x)
    return not bad
