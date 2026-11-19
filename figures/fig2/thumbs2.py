"""Fig. 2 보조 그림: (1) 2020 시험 지역 예측 썸네일 (12개월 수송 켬 모델) (2) 잠재 상태 큐브 (채널 0 = 주입된 NO₂, thumbs.py 의 th_no2_day.png 사용)
(3) 수송 연산 모식도 (합성 장에 이류 · 확산 · 감쇠를 적용). 실행: thumbs.py 다음에 python figures/fig2/thumbs2.py"""
import os
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2]); OUT = f"{ROOT}/figures/out/fig2"; os.makedirs(OUT, exist_ok=True); os.chdir(OUT)  # 그림 재료는 figures/out/fig2 에 쓴다
import glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.colors as mc
import matplotlib.image as mi; from matplotlib.transforms import Affine2D
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
import cartopy.crs as ccrs, cartopy.feature as cf
from scipy.ndimage import gaussian_filter, map_coordinates
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
R = ROOT; EXT = [100, 150, 20, 50]; DPI = 300
cells = np.load(f"{ROOT}/runs/region_pts.npz")["cells"]
cmX = mc.LinearSegmentedColormap.from_list("x", ["#EFE7F7", "#C6A9E2", "#8C5CC2", "#5A2A8F", "#2E1050"])
def base(w=5.0):
    fig = plt.figure(figsize=(w, w * 0.6)); ax = plt.axes([0, 0, 1, 1], projection=ccrs.PlateCarree()); ax.set_extent(EXT, ccrs.PlateCarree())
    ax.add_feature(cf.LAND.with_scale("50m"), facecolor="#F4F3EF", zorder=0); ax.add_feature(cf.OCEAN.with_scale("50m"), facecolor="#EEF3F7", zorder=0); return fig, ax
def finish(fig, ax, name):
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5, color="#3A3A3A", zorder=5); ax.spines["geo"].set_edgecolor("#9AA3AD"); ax.spines["geo"].set_linewidth(0.6)
    fig.savefig(name, dpi=DPI, facecolor="white"); plt.close(fig)
# (1) 2020 정답(학습 사운딩) · 예측(시험 지역)
o = pd.concat([pd.read_parquet(f, columns=["row_idx", "latitude", "longitude"]) for f in sorted(glob.glob(f"{R}/data/processed/index/oco_2020*.parquet"))])
p = pd.read_parquet(f"{R}/runs/gnn_m12_on/pred_space0_s0_phys.parquet", columns=["row_idx", "pred", "xco2", "test"]).merge(o, on="row_idx")
lo_, hi_ = np.nanpercentile(p.xco2, [2, 98]); tr = p[~p.test.astype(bool)].sample(frac=0.35, random_state=0); te = p[p.test.astype(bool)]
print("2020 행", len(p), "· 학습 쪽 표시", len(tr), "(35 % 표본) · 시험", len(te), "· 색 범위", round(lo_, 1), round(hi_, 1))
fig, ax = base(); ax.scatter(tr.longitude, tr.latitude, c=tr.xco2, cmap=cmX, vmin=lo_, vmax=hi_, s=0.25, lw=0, transform=ccrs.PlateCarree(), zorder=3, rasterized=True)
for i, j in cells: ax.add_patch(Rectangle((i * 5, j * 5), 5, 5, fc="white", ec="#C0261A", lw=1.0, ls=(0, (3, 2)), transform=ccrs.PlateCarree(), zorder=4))
finish(fig, ax, "th_target2020.png")
fig, ax = base()
for i, j in cells: ax.add_patch(Rectangle((i * 5, j * 5), 5, 5, fc="#FFFFFF", ec="none", transform=ccrs.PlateCarree(), zorder=2))
ax.scatter(te.longitude, te.latitude, c=te.pred, cmap=cmX, vmin=lo_, vmax=hi_, s=0.6, lw=0, transform=ccrs.PlateCarree(), zorder=3, rasterized=True)
for i, j in cells: ax.add_patch(Rectangle((i * 5, j * 5), 5, 5, fill=False, ec="#C0261A", lw=1.2, transform=ccrs.PlateCarree(), zorder=6))
finish(fig, ax, "th_pred2020.png")
# 색막대(공용, XCO₂)
fig = plt.figure(figsize=(2.4, 0.5)); cax = plt.axes([0.05, 0.55, 0.9, 0.3]); cb = plt.colorbar(plt.cm.ScalarMappable(mc.Normalize(lo_, hi_), cmX), cax=cax, orientation="horizontal")
cb.set_ticks([np.ceil(lo_), np.floor(hi_)]); cb.ax.tick_params(labelsize=8, length=2); cb.outline.set_linewidth(0.4); cb.set_label("XCO₂ (ppm)", fontsize=8, labelpad=1)
fig.savefig("cb_xco2.png", dpi=DPI, transparent=True); plt.close(fig)
# (2) 잠재 상태 큐브: 8 채널 평면(기울인 사선 투영), 맨 위 = 채널 0 (주입된 NO₂)
img = mi.imread("th_no2_day.png")
fig, ax = plt.subplots(figsize=(4.2, 3.4)); ax.set_aspect("equal"); ax.axis("off")
W, H, sx, sy, dz = 10.0, 6.0, 0.6, 0.42, 0.55   # 평면 크기 · 사선 · 층 간격
def P(x, y, z): return (x + sx * y, sy * y + z)
for c in range(8):  # 아래(채널 7)부터 위(채널 0)로 그림
    ch = 7 - c; z = c * dz; corners = np.array([P(0, 0, z), P(W, 0, z), P(W, H, z), P(0, H, z)])
    fc = "#FFFFFF" if ch else "#FFF4EA"
    ax.add_patch(Polygon(corners, closed=True, fc=fc, ec="#8C96A3" if ch else "#C2410C", lw=0.8 if ch else 1.4, zorder=c * 3))
    if ch == 0:
        tr_ = Affine2D().scale(W / img.shape[1], H / img.shape[0]).translate(0, 0) + Affine2D(np.array([[1, sx, 0], [0, sy, z], [0, 0, 1]])) + ax.transData
        im = ax.imshow(img[::-1], origin="lower", extent=None, zorder=c * 3 + 1, interpolation="bilinear"); im.set_transform(tr_); im.set_clip_path(Polygon(corners, closed=True, transform=ax.transData))
    for k in range(1, 12):  # 격자선
        x = W * k / 12; a, b = P(x, 0, z), P(x, H, z); ax.plot([a[0], b[0]], [a[1], b[1]], color="#C9D0D8" if ch else "#E8B48F", lw=0.35, zorder=c * 3 + 2)
    for k in range(1, 8):
        y = H * k / 8; a, b = P(0, y, z), P(W, y, z); ax.plot([a[0], b[0]], [a[1], b[1]], color="#C9D0D8" if ch else "#E8B48F", lw=0.35, zorder=c * 3 + 2)
ax.set_xlim(-0.3, W + sx * H + 0.3); ax.set_ylim(-0.3, sy * H + 7 * dz + 0.3)
fig.savefig("latent_cube.png", dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# (3) 수송 연산 모식도: 합성 기둥에 실제 연산(이류 = 반라그랑주 이동, 확산 = 가우스, 감쇠 = 곱) 적용
n = 40; yy, xx = np.mgrid[0:n, 0:n]; h0 = np.exp(-(((xx - 11) / 3.2) ** 2 + ((yy - 20) / 2.6) ** 2))
U, V = 9.0, 2.5                                    # 격자 칸 / 스텝 (모식도용 과장)
h1 = map_coordinates(h0, [yy - V, xx - U], order=1, mode="constant")   # 반라그랑주: 상류 지점 값을 가져옴
h2 = gaussian_filter(h1, 1.6); h3 = h2 * 0.72
cmN = mc.LinearSegmentedColormap.from_list("no2", ["#FFFFFF", "#FDD2A8", "#F59A4E", "#D9561C", "#8E2A07"])
for k, (h, name) in enumerate([(h0, "op_0.png"), (h1, "op_1.png"), (h2, "op_2.png"), (h3, "op_3.png")]):
    fig = plt.figure(figsize=(1.6, 1.6)); ax = plt.axes([0, 0, 1, 1]); ax.imshow(h, cmap=cmN, vmin=0, vmax=1, origin="lower", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, n, 4)); ax.set_yticks(np.arange(-0.5, n, 4)); ax.grid(color="#D5DBE2", lw=0.35); ax.tick_params(length=0, labelbottom=False, labelleft=False)
    for s_ in ax.spines.values(): s_.set_color("#8C96A3"); s_.set_linewidth(0.8)
    if k == 1:
        for (y0, x0) in [(8, 4), (8, 20), (31, 4), (31, 20), (20, 28)]:
            ax.add_patch(FancyArrowPatch((x0, y0), (x0 + 7, y0 + 2), arrowstyle="-|>", mutation_scale=9, color="#1D5F99", lw=1.6))
    fig.savefig(name, dpi=DPI, facecolor="white"); plt.close(fig)
print("완료")
