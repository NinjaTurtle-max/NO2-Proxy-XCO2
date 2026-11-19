"""Fig. 2 정답 썸네일: 2020 학습 행(train.nested_masks("space", 0, lab, vfold=1))의 XCO₂. 실행: python figures/fig2/thumb_target_train.py"""
import os
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2]); OUT = f"{ROOT}/figures/out/fig2"; os.makedirs(OUT, exist_ok=True); os.chdir(OUT)  # 그림 재료는 figures/out/fig2 에 쓴다
import numpy as np, pandas as pd, matplotlib
from no2xco2.data import loader as L
from no2xco2.train import nested_masks
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.colors as mc
import cartopy.crs as ccrs, cartopy.feature as cf
from matplotlib.patches import Rectangle
IDX = f"{ROOT}/data/processed/index"
COLS = ["row_idx", "time", "label", "xco2", "latitude", "longitude", "fold_time", "fold_space", "space_block", "year"]
big = pd.concat([L.load_oco(f"2020{m:02d}", IDX, columns=COLS).assign(lab=lambda d: d.label.eq("L_train")).drop(columns="label") for m in range(1, 13)], ignore_index=True)
tr, va, te = nested_masks(big, "space", 0, big.lab.to_numpy(), 1)
print("행", len(big), "· train", int(tr.sum()), "· val", int(va.sum()), "· test", int(te.sum()))
cells = np.load(f"{ROOT}/runs/region_pts.npz")["cells"]
cmX = mc.LinearSegmentedColormap.from_list("x", ["#EFE7F7", "#C6A9E2", "#8C5CC2", "#5A2A8F", "#2E1050"])
t = big[tr]
fig = plt.figure(figsize=(5.0, 3.0)); ax = plt.axes([0, 0, 1, 1], projection=ccrs.PlateCarree()); ax.set_extent([100, 150, 20, 50], ccrs.PlateCarree())
ax.add_feature(cf.LAND.with_scale("50m"), facecolor="#F4F3EF", zorder=0); ax.add_feature(cf.OCEAN.with_scale("50m"), facecolor="#EEF3F7", zorder=0)
ax.scatter(t.longitude, t.latitude, c=t.xco2, cmap=cmX, vmin=404.1, vmax=418.5, s=0.2, lw=0, transform=ccrs.PlateCarree(), zorder=3, rasterized=True)
for i, j in cells: ax.add_patch(Rectangle((i * 5, j * 5), 5, 5, fc="white", ec="#C0261A", lw=1.0, ls=(0, (3, 2)), transform=ccrs.PlateCarree(), zorder=4))
ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5, color="#3A3A3A", zorder=5); ax.spines["geo"].set_edgecolor("#9AA3AD"); ax.spines["geo"].set_linewidth(0.6)
fig.savefig("th_target2020_train.png", dpi=300, facecolor="white")
