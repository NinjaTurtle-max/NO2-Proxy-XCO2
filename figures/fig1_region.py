"""Fig. 1: 연구 영역(20–50°N, 100–150°E), 2020–2024 OCO-2/3 품질 통과 사운딩 위치(점), 공간 분할 시험 5° 칸.
입력: runs/region_pts.npz (region_points.py 산출). 실행: python figures/fig1_region.py runs/region_pts.npz figures/out/fig1_region.png"""
import sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cf
from matplotlib.patches import Rectangle
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
z = np.load(sys.argv[1]); lon = 100 + z["ix"] / 100; lat = 20 + z["iy"] / 100
m = (lon >= 100) & (lon <= 150) & (lat >= 20) & (lat <= 50); lon, lat = lon[m], lat[m]
fig = plt.figure(figsize=(12, 7.6)); ax = plt.axes(projection=ccrs.PlateCarree()); ax.set_extent([100, 150, 20, 50], ccrs.PlateCarree())
ax.add_feature(cf.OCEAN.with_scale("50m"), facecolor="#D3E4F1"); ax.add_feature(cf.LAND.with_scale("50m"), facecolor="#EEEADB")
ax.scatter(lon, lat, s=0.05, c="#1F5F8B", alpha=0.12, lw=0, rasterized=True, transform=ccrs.PlateCarree(), zorder=2)
ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.9, color="#222222", zorder=6); ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.4, color="#777777", zorder=6)
for i, j in z["cells"]:
    ax.add_patch(Rectangle((i * 5, j * 5), 5, 5, facecolor="#E34A33", alpha=0.18, edgecolor="none", transform=ccrs.PlateCarree(), zorder=3))
    ax.add_patch(Rectangle((i * 5, j * 5), 5, 5, fill=False, edgecolor="#C0261A", lw=2.6, transform=ccrs.PlateCarree(), zorder=7))
gl = ax.gridlines(draw_labels=True, xlocs=range(100, 151, 10), ylocs=range(20, 51, 10), lw=0.3, color="#999999"); gl.top_labels = gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = {"size": 14}
ax.add_patch(Rectangle((128.6, 23.4), 2.0, 2.0, facecolor="#F6D5D1", edgecolor="#C0261A", lw=2.6, transform=ccrs.PlateCarree(), zorder=8))
ax.text(131.2, 24.4, "시험 지역 (공간 분할 · 5° 칸)", fontsize=15, va="center", transform=ccrs.PlateCarree(), zorder=8, bbox=dict(fc="white", ec="none", alpha=0.85))
ax.scatter([129.6], [21.4], s=40, c="#1F5F8B", transform=ccrs.PlateCarree(), zorder=8)
ax.text(131.2, 21.4, "OCO-2/3 사운딩 위치 (2020–2024)", fontsize=15, va="center", transform=ccrs.PlateCarree(), zorder=8, bbox=dict(fc="white", ec="none", alpha=0.85))
fig.savefig(sys.argv[2], dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
print(f"점 {len(lon):,} · 사운딩 {int(z['n2']) + int(z['n3']):,} (OCO-2 {int(z['n2']):,} · OCO-3 {int(z['n3']):,}) · 시험 칸 {len(z['cells'])} · 시험 사운딩 {int(z['nt']):,}")
