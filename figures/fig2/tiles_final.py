"""Fig. 2 최종 크기 보조 그림: 수송 연산 타일 (2.0 in, 400 dpi) · XCO₂ 색막대 막대 cb_bar.png (글자는 PowerPoint 글자).
실행: thumbs.py 다음에 python figures/fig2/tiles_final.py"""
import os
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.colors as mc
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter, map_coordinates
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
os.makedirs(Path(__file__).resolve().parents[1] / "out" / "fig2", exist_ok=True); os.chdir(Path(__file__).resolve().parents[1] / "out" / "fig2")
n = 40; yy, xx = np.mgrid[0:n, 0:n]; h0 = np.exp(-(((xx - 11) / 3.2) ** 2 + ((yy - 20) / 2.6) ** 2))
h1 = map_coordinates(h0, [yy - 2.5, xx - 9.0], order=1, mode="constant"); h2 = gaussian_filter(h1, 1.6); h3 = h2 * 0.72
cmN = mc.LinearSegmentedColormap.from_list("no2", ["#FFFFFF", "#FDD2A8", "#F59A4E", "#D9561C", "#8E2A07"])
for k, h in enumerate([h0, h1, h2, h3]):
    fig = plt.figure(figsize=(2.0, 2.0)); ax = plt.axes([0, 0, 1, 1]); ax.imshow(h, cmap=cmN, vmin=0, vmax=1, origin="lower", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, n, 4)); ax.set_yticks(np.arange(-0.5, n, 4)); ax.grid(color="#D5DBE2", lw=0.45); ax.tick_params(length=0, labelbottom=False, labelleft=False)
    for s_ in ax.spines.values(): s_.set_color("#8C96A3"); s_.set_linewidth(1.0)
    if k == 1:
        for (y0, x0) in [(7, 3), (7, 21), (32, 3), (32, 21), (19, 29)]:
            ax.add_patch(FancyArrowPatch((x0, y0), (x0 + 7, y0 + 2), arrowstyle="-|>", mutation_scale=12, color="#1D5F99", lw=2.0))
    fig.savefig(f"opF_{k}.png", dpi=400, facecolor="white"); plt.close(fig)
lo_, hi_ = 404.1, 418.5   # thumbs2.py 의 2020 XCO₂ 2–98 % 범위
cmX = mc.LinearSegmentedColormap.from_list("x", ["#EFE7F7", "#C6A9E2", "#8C5CC2", "#5A2A8F", "#2E1050"])
fig = plt.figure(figsize=(2.1, 0.62)); cax = plt.axes([0.06, 0.62, 0.88, 0.26])
cb = plt.colorbar(plt.cm.ScalarMappable(mc.Normalize(lo_, hi_), cmX), cax=cax, orientation="horizontal"); cb.set_ticks([405, 418])
cb.ax.tick_params(labelsize=15, length=3, pad=1); cb.outline.set_linewidth(0.6)
fig.text(0.5, 0.02, "XCO₂ (ppm)", ha="center", va="bottom", fontsize=15)
fig.savefig("cbF_xco2.png", dpi=400, transparent=True); plt.close(fig)
fig = plt.figure(figsize=(1.85, 0.16)); ax = plt.axes([0, 0, 1, 1]); ax.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap=cmX); ax.set_axis_off()
fig.savefig("cb_bar.png", dpi=400, facecolor="white"); plt.close(fig)
print("ok")
