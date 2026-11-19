"""Fig. 3: 사례일(2020-01-31) 연속 면 지도 5개(관측 + 4개 모델, GNN 지도에 ERA5 경계층 바람 화살표, 시험 사운딩 점은 같은 색지도)와
시험 1,204,908행 밀도 산점도 4개(N · R · R² · RMSE · MAE, 1:1 선). 40.2 × 35.5 cm.
입력: runs/maps (grid_map_gnn.py · grid_map_baselines.py 산출) · 각 모델 예측 parquet · data/processed/index/oco_202001.parquet.
출력: figures/out/fig_maps_scatter.png. 글자 겹침 검사(style.check)를 통과해야 종료 코드 0.
실행: 저장소 루트에서 python figures/fig3_maps.py"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.colors as mc
import style as M
from matplotlib.patches import Rectangle

D = "runs/maps"; R = "runs"; O = M.O; CM = M.CM
Z = np.load(f"{D}/grid_map_202001_m60.npz"); lat, lon = Z["lat"], Z["lon"]
F = {"on": Z["on"], "off": Z["off"], "krig": np.load(f"{D}/grid_202001_krig.npy"), "lgbm": np.load(f"{D}/grid_202001_lgbm.npy")}
NAME = {"on": "GNN, transport on", "off": "GNN, transport off", "krig": "Kriging", "lgbm": "LightGBM (same inputs)"}
# 시험 예측 (같은 1,204,908행)
on = pd.read_parquet(f"{R}/gnn_m60_on/pred_space0_s0_phys.parquet", columns=["row_idx", "pred", "xco2", "test"]); T = on[on.test.astype(bool)].rename(columns={"pred": "on"})
for k, f in (("off", f"{R}/gnn_m60_off/pred_space0_s0_nophys.parquet"), ("krig", f"{R}/kriging/pred_krig.parquet"), ("lgbm", f"{R}/lgbm_no2_wind/pred_lgbm_no2wind.parquet")):
    T = T.merge(pd.read_parquet(f, columns=["row_idx", "pred"]).rename(columns={"pred": k}), on="row_idx", validate="one_to_one")
assert len(T) == 1204908
y = T.xco2.to_numpy()


_j = pd.read_parquet("data/processed/index/oco_202001.parquet", columns=["time", "latitude", "longitude", "xco2", "label"])
OBS = _j[(pd.to_datetime(_j.time).dt.day == int(Z["day"])) & (_j.label == "L_train")]  # 그날 품질 통과 사운딩 전체 (모델 전)
_oi = np.clip(np.round((OBS.latitude.to_numpy() - 20) / 0.25).astype(int), 0, 120); _oj = np.clip(np.round((OBS.longitude.to_numpy() - 100) / 0.25).astype(int), 0, 200)
_s = np.zeros(121 * 201); _c = np.zeros(121 * 201); np.add.at(_s, _oi * 201 + _oj, OBS.xco2.to_numpy()); np.add.at(_c, _oi * 201 + _oj, 1)
OBSG = np.where(_c > 0, _s / np.maximum(_c, 1), np.nan).reshape(121, 201)  # 모델과 같은 0.25° 격자 칸 평균, 관측 없는 칸은 빈칸 (보간 없음)


def stats(p):
    e = p - y; return dict(R=np.corrcoef(p, y)[0, 1], R2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)))


def maps_scatter():
    import cartopy.crs as ccrs, cartopy.feature as cfe
    W, H = 40.2, 35.5; f = plt.figure(figsize=(W * CM, H * CM)); bx = lambda x, y_, w, h: [x / W, y_ / H, w / W, h / H]; pc = ccrs.PlateCarree()
    ext = [lon[0] - 0.125, lon[-1] + 0.125, lat[0] - 0.125, lat[-1] + 0.125]; cells = M.test_cells()
    norm = mc.Normalize(*np.nanpercentile(np.r_[np.concatenate([v.ravel() for v in F.values()]), Z["snd_obs"]], [2, 98]))
    MW, MH = 12.6, 7.56; POS = [(0.5, 24.8), (13.8, 24.8), (27.1, 24.8), (0.5, 15.8), (13.8, 15.8)]; keys = ["obs"] + list(F)  # 지도 3 + 2 배치
    for i, k in enumerate(keys):
        x, y0 = POS[i]; ax = f.add_axes(bx(x, y0, MW, MH), projection=pc); ax.set_gid("nochk"); ax.set_extent([100, 150, 20, 50], pc)
        ax.add_feature(cfe.OCEAN, facecolor="#E8EEF3", zorder=0); ax.add_feature(cfe.LAND, facecolor="#F2F0EA", zorder=0)
        if k == "obs":
            ax.imshow(np.ma.masked_invalid(OBSG), origin="lower", extent=ext, transform=pc, cmap="viridis", norm=norm, interpolation="nearest", zorder=2)
        else:
            im = ax.imshow(F[k], origin="lower", extent=ext, transform=pc, cmap="viridis", norm=norm, interpolation="bilinear", zorder=1)
        ax.coastlines(lw=0.5, color="#222222", zorder=3)
        for la, lo in cells:
            ax.add_patch(Rectangle((lo, la), 5, 5, fill=False, ec=M.C_RED, lw=0.9, transform=pc, zorder=5))
        if k != "obs":
            ax.scatter(Z["snd_lon"], Z["snd_lat"], c=Z["snd_obs"], cmap="viridis", norm=norm, s=4, lw=0.15, edgecolors="white", transform=pc, zorder=6)
        if k == "on":
            s_ = 10; LO, LA = np.meshgrid(lon[::s_], lat[::s_])
            q = ax.quiver(LO, LA, Z["u"][::s_, ::s_], Z["v"][::s_, ::s_], transform=pc, color="#F5F5F5", scale=300, width=0.004, zorder=4, edgecolor="#333333", linewidth=0.3)
            ax.quiverkey(q, 27.9 / W, 16.25 / H, 10, "10 m s⁻¹  arrows in b: ERA5 PBL wind", labelpos="E", fontproperties={"size": M.FS_K - 4}, coordinates="figure", color="#333333")
        ttl = "OCO-2/3 observations" if k == "obs" else NAME[k]
        f.text(x / W, (y0 + MH + 0.2) / H, f"{'abcde'[i]}  {ttl}", ha="left", va="bottom", fontsize=M.FS_K, fontweight="bold")
    ca = f.add_axes(bx(27.6, 21.1, 11.8, 0.4)); cb = plt.colorbar(im, cax=ca, orientation="horizontal"); ca.tick_params(labelsize=M.FS_K - 3, length=3)
    cb.set_label("XCO₂ (ppm), 2020-01-31", fontsize=M.FS_K - 2)
    f.text(27.6 / W, 19.1 / H, f"a: {len(OBS):,} OCO-2/3 soundings on the 0.25° grid\n    before modelling (blank = no observation)\nb–e: model fields; dots = held-out test soundings\nred boxes = held-out regions", ha="left", va="top", fontsize=M.FS_K - 4, linespacing=1.3)
    lo_, hi_ = np.percentile(y, [0.2, 99.8]); hbs = []
    for i, k in enumerate(F):
        x = 0.3 + i * 9.95; ax = f.add_axes(bx(x + 1.25, 2.7, 8.6, 10.5)); p = T[k].to_numpy(); st = stats(p)
        hb = ax.hexbin(y, p, gridsize=60, extent=(lo_, hi_, lo_, hi_), bins="log", cmap="magma_r", mincnt=1, linewidths=0, vmin=1, vmax=3e4); hbs.append(hb)
        ax.plot([lo_, hi_], [lo_, hi_], color=M.C_RED, lw=1.4, label="1:1"); ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
        ax.set_xlabel("Observed XCO₂ (ppm)", fontsize=M.FS_K - 3); ax.tick_params(labelsize=M.FS_K - 5, length=3)
        if i == 0:
            ax.set_ylabel("Predicted XCO₂ (ppm)", fontsize=M.FS_K - 3)
        ax.text(0.04, 0.97, f"N = {len(y):,}\nR = {st['R']:.3f}\nR² = {st['R2']:.3f}\nRMSE = {st['RMSE']:.3f}\nMAE = {st['MAE']:.3f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=M.FS_K - 5, linespacing=1.25, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        ax.legend(loc="lower right", frameon=False, fontsize=M.FS_K - 5, handlelength=1.2)
        f.text((x + 1.25) / W, 13.4 / H, f"{'fghi'[i]}  {NAME[k]}", ha="left", va="bottom", fontsize=M.FS_K - 1, fontweight="bold")
    ca2 = f.add_axes(bx(17.0, 0.95, 12.2, 0.3)); plt.colorbar(hbs[-1], cax=ca2, orientation="horizontal"); ca2.tick_params(labelsize=M.FS_K - 6, length=3)
    f.text(16.7 / W, 1.1 / H, "Soundings per hexagon bin (f–i)", ha="right", va="center", fontsize=M.FS_K - 5)
    f.text(0.3 / W, 35.3 / H, "Held-out regions: predicted XCO₂ fields (2020-01-31) and validation against OCO-2/3 (60-month models, 1,204,908 test soundings)",
           ha="left", va="top", fontsize=M.FS_L - 2, fontweight="bold")
    ok = M.check(f, "지도 · 산점도"); f.savefig(f"{O}/fig_maps_scatter.png", facecolor="white"); plt.close(f); return ok


if __name__ == "__main__":
    import os, sys
    os.makedirs(O, exist_ok=True); sys.exit(0 if maps_scatter() else 1)
