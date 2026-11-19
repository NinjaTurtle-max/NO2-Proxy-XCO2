"""Fig. 2 썸네일 (2020-01 자료, 20–50°N · 100–150°E): NO₂ 일별 · 월평균, ERA5 경계층 바람 · t2m · blh · sp, 시험 지역.
NO₂: encoder_index enc_tropomi_202001 (qa ≥ 0.75) 화소를 4노드 가중으로 0.25° 노드에 모은다. 바람 · 기상: era5_wind_202001_z100.nc (화소가 가장 많은 날 05 UTC).
실행: 저장소 루트에서 python figures/fig2/thumbs.py (출력 figures/out/fig2)"""
import os
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2]); OUT = f"{ROOT}/figures/out/fig2"; os.makedirs(OUT, exist_ok=True); os.chdir(OUT)  # 그림 재료는 figures/out/fig2 에 쓴다
import numpy as np, pandas as pd, pyarrow.parquet as pq, xarray as xr, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.colors as mc
import cartopy.crs as ccrs, cartopy.feature as cf
R = ROOT; NLAT, NLON = 121, 201; N = NLAT * NLON
EXT = [100, 150, 20, 50]; DPI = 300
def base(w=5.0):
    fig = plt.figure(figsize=(w, w * 0.6)); ax = plt.axes([0, 0, 1, 1], projection=ccrs.PlateCarree()); ax.set_extent(EXT, ccrs.PlateCarree())
    ax.add_feature(cf.LAND.with_scale("50m"), facecolor="#F4F3EF", zorder=0); ax.add_feature(cf.OCEAN.with_scale("50m"), facecolor="#EEF3F7", zorder=0)
    return fig, ax
def finish(fig, ax, name):
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5, color="#3A3A3A", zorder=5)
    ax.spines["geo"].set_edgecolor("#9AA3AD"); ax.spines["geo"].set_linewidth(0.6)
    fig.savefig(name, dpi=DPI, facecolor="white"); plt.close(fig)
# ---- NO₂ (일별 · 월평균, 노드 가중 평균) ----
pf = pq.ParquetFile(f"{R}/data/processed/encoder_index/enc_tropomi_202001.parquet")
num = np.zeros((31, N)); den = np.zeros((31, N))
for i in range(pf.num_row_groups):
    t = pf.read_row_group(i, columns=["step_h", "qa", "no2_tvcd", "n0", "n1", "n2", "n3", "w0", "w1", "w2", "w3"]).to_pandas()
    t = t[t.qa >= 0.75]; d = (t.step_h.to_numpy() // 24).astype(int).clip(0, 30)
    for k in range(4):
        idx = d * N + t[f"n{k}"].to_numpy(); w = t[f"w{k}"].to_numpy()
        num += np.bincount(idx, w * t.no2_tvcd.to_numpy(), minlength=31 * N).reshape(31, N); den += np.bincount(idx, w, minlength=31 * N).reshape(31, N)
cov = (den > 0).sum(1); day = int(np.argmax(cov)); print("일별 NO₂ 노드 수", cov.tolist(), "→ 고른 날", day + 1)
lon = 100 + 0.25 * np.arange(NLON); lat = 20 + 0.25 * np.arange(NLAT)
cmN = mc.LinearSegmentedColormap.from_list("no2", ["#FFF7EF", "#FDD2A8", "#F59A4E", "#D9561C", "#8E2A07"])
for tag, F in [("day", np.where(den[day] > 0, num[day] / np.maximum(den[day], 1e-12), np.nan)), ("month", np.where(den.sum(0) > 0, num.sum(0) / np.maximum(den.sum(0), 1e-12), np.nan))]:
    fig, ax = base(); A = F.reshape(NLAT, NLON) * 1e4 / 1e-4  # mol m⁻² → 상대 단위(색만)
    ax.pcolormesh(lon, lat, np.ma.masked_invalid(F.reshape(NLAT, NLON)), cmap=cmN, norm=mc.LogNorm(2e-5, 3e-4), shading="nearest", transform=ccrs.PlateCarree(), zorder=2, rasterized=True)
    finish(fig, ax, f"th_no2_{tag}.png")
# ---- ERA5 (같은 날 05 UTC ≈ 위성 통과) ----
E = xr.open_dataset(f"{R}/data/raw/era5_wind/era5_wind_202001_z100.nc").isel(time=day * 24 + 5).load()
la, lo = E.lat.values, E.lon.values; u, v = E.u_pbl.values, E.v_pbl.values; sp = np.hypot(u, v)
fig, ax = base(); ax.pcolormesh(lo, la, sp, cmap=mc.LinearSegmentedColormap.from_list("w", ["#F3F8FC", "#BBD7EE", "#5E9ACB", "#1D5F99"]), vmin=0, vmax=18, shading="nearest", transform=ccrs.PlateCarree(), zorder=2, rasterized=True)
s = 8; ax.quiver(lo[::s], la[::s], u[::s, ::s], v[::s, ::s], color="#123E66", scale=260, width=0.0035, headwidth=4, transform=ccrs.PlateCarree(), zorder=6)
finish(fig, ax, "th_wind.png")
for var, cmapn in [("t2m", "Greys"), ("blh", "Greys"), ("sp", "Greys")]:
    fig, ax = base(); Z = E[var].values
    ax.pcolormesh(lo, la, Z, cmap=mc.LinearSegmentedColormap.from_list("g", plt.get_cmap(cmapn)(np.linspace(0.12, 0.7, 256))), shading="nearest", transform=ccrs.PlateCarree(), zorder=2, rasterized=True)
    finish(fig, ax, f"th_bg_{var}.png")
# ---- OCO 사운딩 (정답) · 예측 (test) ----
o = pd.read_parquet(f"{R}/data/processed/index/oco_202001.parquet", columns=["row_idx", "latitude", "longitude", "xco2", "inside", "label"]); o = o[o.inside.astype(bool)]
p = pd.read_parquet(f"{R}/runs/gnn_m12_on/pred_space0_s0_phys.parquet", columns=["row_idx", "pred", "test"])
q = o.merge(p, on="row_idx"); lo_, hi_ = np.nanpercentile(o.xco2, [2, 98]); print("XCO₂ 색 범위", round(lo_, 1), round(hi_, 1), "· 1월 사운딩", len(o), "· 예측 결합", len(q), "· test", int(q.test.sum()))
cmX = mc.LinearSegmentedColormap.from_list("x", ["#EFE7F7", "#C6A9E2", "#8C5CC2", "#5A2A8F", "#2E1050"])
fig, ax = base(); ax.scatter(o.longitude, o.latitude, c=o.xco2, cmap=cmX, vmin=lo_, vmax=hi_, s=0.6, lw=0, transform=ccrs.PlateCarree(), zorder=3, rasterized=True); finish(fig, ax, "th_oco.png")
cells = np.load(f"{ROOT}/runs/region_pts.npz")["cells"]
from matplotlib.patches import Rectangle
t_ = q[q.test.astype(bool)]
fig, ax = base(); ax.scatter(t_.longitude, t_.latitude, c=t_.pred, cmap=cmX, vmin=lo_, vmax=hi_, s=1.2, lw=0, transform=ccrs.PlateCarree(), zorder=3, rasterized=True)
for i, j in cells: ax.add_patch(Rectangle((i * 5, j * 5), 5, 5, fill=False, ec="#C0261A", lw=1.2, transform=ccrs.PlateCarree(), zorder=7))
finish(fig, ax, "th_pred.png")
fig, ax = base()
for i, j in cells: ax.add_patch(Rectangle((i * 5, j * 5), 5, 5, fc="#F6C9C3", ec="#C0261A", lw=1.2, transform=ccrs.PlateCarree(), zorder=3))
finish(fig, ax, "th_split.png")
np.savez("thumbs_meta.npz", day=day + 1, xlo=lo_, xhi=hi_)
