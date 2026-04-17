"""
10_reverse_calculate_xco2.py — Reverse Calculate XCO2 Concentration from SR-PINN
================================================================================
Objective: Use the trained SR-PINN model to predict XCO2 anomaly and 
           reverse calculate the total XCO2 concentration by adding background.
           Enhanced with Pearson r, R2, and RMSE for Total XCO2 validation.
"""

import os
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ─────────────────────────────────────────────────────────────────
# 1. 환경 및 장치 설정
# ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

POSSIBLE_PATHS = [
    "/Volumes/100.118.65.89/dataset/XCO2연구 데이터/03_split_output_025/anom_1d_balanced_025.parquet",
    "./anom_1d_balanced_025.parquet",
    "/Users/ganghyeon-u/Research/NO2-Proxy-XCO2/anom_1d_balanced_025.parquet"
]

DATA_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        DATA_PATH = p
        break

MODEL_PATH = "07_sr_pinn_v1_results/sr_pinn_highres_final.ckpt"
OUT_DIR = "10_inference_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 2. 모델 아키텍처 (07_sr_pinn_highres_engine.py와 동일)
# ─────────────────────────────────────────────────────────────────
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.SiLU(),
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        )
    def forward(self, x): 
        return x + self.conv(x)

class SR_PINN_AtmoNet(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 32, 3, padding=1)),
            nn.SiLU(),
            ResNetBlock(32)
        )
        self.lstm_cell = nn.Conv2d(32 + 32, 4 * 32, 3, padding=1)
        self.decoder = nn.Sequential(
            ResNetBlock(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.PixelShuffle(1),
            nn.Conv2d(64, 2, 1)
        )
        # PINN Specific Parameters (Required for loading checkpoint)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.kappa_net = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(),
            nn.Linear(16, 1), nn.Softplus()
        )

    def forward(self, x_seq):
        b, t, c, h, w = x_seq.shape
        h_t = torch.zeros(b, 32, h, w, device=x_seq.device)
        c_t = torch.zeros(b, 32, h, w, device=x_seq.device)
        for i in range(t):
            xt = self.encoder(x_seq[:, i])
            combined = torch.cat([xt, h_t], dim=1)
            gates = self.lstm_cell(combined)
            i_g, f_g, o_g, g_g = torch.split(gates, 32, dim=1)
            c_t = torch.sigmoid(f_g) * c_t + torch.sigmoid(i_g) * torch.tanh(g_g)
            h_t = torch.sigmoid(o_g) * torch.tanh(c_t)
        out = self.decoder(h_t)
        mu, log_var = torch.chunk(out, 2, dim=1)
        return mu, log_var

# ─────────────────────────────────────────────────────────────────
# 3. 데이터 로더
# ─────────────────────────────────────────────────────────────────
class InferenceDataset(Dataset):
    def __init__(self, df, seq_len=3):
        self.df = df.reset_index(drop=True)
        self.dates = sorted(df['date'].unique())
        self.seq_len = seq_len
        self.grid_h, self.grid_w = 120, 200 
        
    def __len__(self): 
        return len(self.dates) - self.seq_len
    
    def __getitem__(self, idx):
        seq_data = []
        for i in range(self.seq_len):
            day_df = self.df[self.df['date'] == self.dates[idx + i]]
            grid = np.zeros((6, self.grid_h, self.grid_w), dtype=np.float32)
            lats = ((day_df['latitude'] - 20.0) / 0.25).astype(int).clip(0, self.grid_h-1)
            lons = ((day_df['longitude'] - 100.0) / 0.25).astype(int).clip(0, self.grid_w-1)
            grid[0, lats, lons] = day_df['tropomi_no2'].values
            grid[1, lats, lons] = day_df['era5_u10'].values
            grid[2, lats, lons] = day_df['era5_v10'].values
            grid[3, lats, lons] = day_df['era5_blh'].values
            grid[4, lats, lons] = day_df['xco2_anomaly'].values
            grid[5, lats, lons] = day_df['xco2_background'].values
            seq_data.append(grid)
        seq_data = np.stack(seq_data)
        return (torch.from_numpy(seq_data[:, :4]), 
                torch.from_numpy(seq_data[-1, 4]), 
                torch.from_numpy(seq_data[-1, 5]),
                self.dates[idx + self.seq_len - 1].strftime('%Y-%m-%d'))

def plot_results(df, out_dir):
    """결과 시각화: SCI Journal Standard (300 DPI, High-Res Basemaps, Density Shading)"""
    if df.empty: return

    # Global Style Setup
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Inter', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'savefig.dpi': 300,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

    # 1. Total XCO2 Density Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Use hexbin for density representation (SCI Standard for large N)
    hb = ax.hexbin(df['true_xco2'], df['pred_xco2'], gridsize=40, cmap='Blues', 
                   mincnt=1, edgecolors='none', bins='log')
    cb = fig.colorbar(hb, ax=ax, label='$\log_{10}$(Count)', shrink=0.8)
    
    # Regression metrics
    r2 = r2_score(df['true_xco2'], df['pred_xco2'])
    pr, _ = pearsonr(df['true_xco2'], df['pred_xco2'])
    rmse = np.sqrt(mean_squared_error(df['true_xco2'], df['pred_xco2']))
    
    # 1:1 Reference Line
    min_val = min(df['true_xco2'].min(), df['pred_xco2'].min())
    max_val = max(df['true_xco2'].max(), df['pred_xco2'].max())
    ax.plot([min_val, max_val], [min_val, max_val], color='#D62728', lw=2, linestyle='--', label='1:1 Line', zorder=5)
    
    # Formatting
    ax.set_xlabel("Observed $XCO_2$ (Satellite, ppm)", labelpad=10)
    ax.set_ylabel("Reconstructed $XCO_2$ (SR-PINN, ppm)", labelpad=10)
    ax.set_title("SR-PINN Total $XCO_2$ Consistency Verification", pad=15, fontweight='bold')
    
    # Statistical Text Box
    stats_text = (f"$R^2 = {r2:.4f}$\n"
                  f"Pearson $r = {pr:.4f}$\n"
                  f"RMSE = {rmse:.4f} ppm")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='silver'))
    
    ax.grid(True, linestyle=(0, (5, 10)), alpha=0.3)
    ax.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Figure_XCO2_Scatter_Density.png"))
    plt.close()

    # 2. Time-series Plot (Daily Mean with Variance)
    plt.figure(figsize=(10, 5))
    df['date'] = pd.to_datetime(df['date'])
    daily_stats = df.groupby('date')['true_xco2'].agg(['mean', 'std']).reset_index()
    daily_pred = df.groupby('date')['pred_xco2'].mean().reset_index()
    
    plt.fill_between(daily_stats['date'], daily_stats['mean'] - daily_stats['std'], 
                     daily_stats['mean'] + daily_stats['std'], color='gray', alpha=0.2, label='Obs 1$\sigma$')
    plt.plot(daily_stats['date'], daily_stats['mean'], color='black', lw=1.5, label='Observed (Daily Avg)')
    plt.plot(daily_pred['date'], daily_pred['pred_xco2'], color='#D62728', lw=1.5, linestyle='--', label='Predicted (SR-PINN)')
    
    plt.title("XCO2 Temporal Variations in East Asia", pad=10, fontweight='bold')
    plt.ylabel("$XCO_2$ (ppm)")
    plt.grid(True, alpha=0.2)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Figure_XCO2_Temporal.png"))
    plt.close()

    # 3. High-Quality Spatial Mapping (Cartopy)
    if HAS_CARTOPY:
        spatial_avg = df.groupby(['latitude', 'longitude'])[['pred_xco2', 'true_xco2']].mean().reset_index()
        spatial_avg['bias'] = spatial_avg['pred_xco2'] - spatial_avg['true_xco2']
        
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 2, wspace=0.1)
        
        map_configs = [
            {'col': 'pred_xco2', 'cmap': 'YlOrRd', 'title': 'Mean Predicted $XCO_2$', 'label': 'ppm'},
            {'col': 'bias', 'cmap': 'RdBu_r', 'title': 'Prediction Bias (Model - Obs)', 'label': 'Bias (ppm)', 'center_zero': True}
        ]
        
        for i, config in enumerate(map_configs):
            ax = fig.add_subplot(gs[i], projection=ccrs.PlateCarree())
            ax.set_extent([100, 150, 20, 50], crs=ccrs.PlateCarree())
            
            # Use high-res features
            ax.add_feature(cfeature.LAND, facecolor='#F5F5F5')
            ax.add_feature(cfeature.OCEAN, facecolor='#E3F2FD')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            
            # Pivot data for interpolation/grid mapping
            lats = sorted(spatial_avg['latitude'].unique())
            lons = sorted(spatial_avg['longitude'].unique())
            grid_z = spatial_avg.pivot(index='latitude', columns='longitude', values=config['col']).values
            
            # Smoothing for 'ssambong' effect
            grid_z_smoothed = gaussian_filter(grid_z, sigma=0.5)
            
            # Set norm for bias
            norm = None
            if config.get('center_zero'):
                vmax = np.abs(spatial_avg['bias']).max()
                norm = plt.Normalize(-vmax, vmax)

            im = ax.pcolormesh(lons, lats, grid_z_smoothed, cmap=config['cmap'], 
                               norm=norm, shading='gouraud', transform=ccrs.PlateCarree())
            
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.6, pad=0.08)
            cbar.set_label(config['label'])
            
            ax.set_title(config['title'], fontweight='bold', pad=10)
            
            # Gridlines
            gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl.top_labels = False; gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            
        plt.savefig(os.path.join(out_dir, "Figure_XCO2_Spatial_Pro.png"), bbox_inches='tight')
        plt.close()

        # 4. Regional Hotspot Zoom-in Panels (NCP, KCR, JKT)
        HOTSPOTS = {
            'North China Plain': {'lat': (34.0, 41.0), 'lon': (113.0, 122.0)},
            'Seoul/Korea': {'lat': (35.0, 39.0), 'lon': (125.0, 130.0)},
            'Kanto/Japan': {'lat': (34.5, 37.5), 'lon': (138.5, 142.0)}
        }
        
        fig = plt.figure(figsize=(12, 16))
        gs = fig.add_gridspec(len(HOTSPOTS), 2, wspace=0.15, hspace=0.2)
        
        for r_idx, (r_name, spec) in enumerate(HOTSPOTS.items()):
            for c_idx, m_type in enumerate(['true_xco2', 'pred_xco2']):
                ax = fig.add_subplot(gs[r_idx, c_idx], projection=ccrs.PlateCarree())
                ax.set_extent([spec['lon'][0], spec['lon'][1], spec['lat'][0], spec['lat'][1]], crs=ccrs.PlateCarree())
                
                ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.0)
                ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', linewidth=0.7)
                
                # Filter data for this region
                reg_mask = (spatial_avg['latitude'] >= spec['lat'][0]) & (spatial_avg['latitude'] <= spec['lat'][1]) & \
                           (spatial_avg['longitude'] >= spec['lon'][0]) & (spatial_avg['longitude'] <= spec['lon'][1])
                reg_data = spatial_avg[reg_mask]
                
                if not reg_data.empty:
                    # Smoothing for zoom view
                    lats_z = sorted(reg_data['latitude'].unique())
                    lons_z = sorted(reg_data['longitude'].unique())
                    # Ensure it's a proper grid for pcolormesh
                    grid_v = reg_data.pivot(index='latitude', columns='longitude', values=m_type).values
                    grid_v = gaussian_filter(grid_v, sigma=0.3)
                    
                    im = ax.pcolormesh(lons_z, lats_z, grid_v, cmap='YlOrRd', 
                                       shading='gouraud', transform=ccrs.PlateCarree())
                    
                    if c_idx == 1: # Colorbar for Predicted only (or shared)
                        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
                        cbar.set_label('XCO2 (ppm)')
                
                type_label = "Observed" if c_idx == 0 else "SR-PINN"
                ax.set_title(f"{r_name}\n({type_label})", fontsize=11, fontweight='bold')
                
                gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
                gl.top_labels = False; gl.right_labels = False
                if c_idx == 1: gl.left_labels = False
                if r_idx < len(HOTSPOTS)-1: gl.bottom_labels = False
                gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}

        plt.suptitle("Regional $XCO_2$ Comparison: Satellite vs SR-PINN", fontsize=16, fontweight='bold', y=0.95)
        plt.savefig(os.path.join(out_dir, "Figure_XCO2_Hotspot_Zoom.png"), bbox_inches='tight', dpi=300)
        plt.close()

def reverse_calculate_xco2():
    print("🔮 Reverse Calculating XCO2 from PINN (East Asia Domain Validation)...")
    
    if DATA_PATH is None or not os.path.exists(DATA_PATH): return
    if not os.path.exists(MODEL_PATH): return

    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    LAT_MIN, LAT_MAX = 20.0, 50.0
    LON_MIN, LON_MAX = 100.0, 150.0
    df_sub = df[(df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) & 
                (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)].copy()
    
    dataset = InferenceDataset(df_sub)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = SR_PINN_AtmoNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    results = []
    with torch.no_grad():
        for x_seq, y_anom_obs, y_bg, date_str in tqdm(loader, desc="Inference"):
            x_seq = x_seq.to(DEVICE)
            mu, _ = model(x_seq)
            mu = mu.cpu().numpy().squeeze() 
            y_anom_obs = y_anom_obs.numpy().squeeze() 
            y_bg = y_bg.numpy().squeeze() 
            
            mask = y_anom_obs != 0
            if mask.any():
                pred_anom = mu[mask]; true_anom = y_anom_obs[mask]; background = y_bg[mask]
                pred_xco2 = pred_anom + background
                true_xco2 = true_anom + background
                
                lats_idx, lons_idx = np.where(mask)
                lats = lats_idx * 0.25 + LAT_MIN
                lons = lons_idx * 0.25 + LON_MIN
                
                temp_df = pd.DataFrame({
                    'date': date_str[0], 'latitude': lats, 'longitude': lons,
                    'pred_xco2': pred_xco2, 'true_xco2': true_xco2, 'background': background
                })
                results.append(temp_df)

    res_df = pd.concat(results, ignore_index=True)
    res_path = os.path.join(OUT_DIR, "xco2_reverse_calculated_east_asia.csv")
    res_df.to_csv(res_path, index=False)
    
    plot_results(res_df, OUT_DIR)
    
    if not res_df.empty:
        r2 = r2_score(res_df['true_xco2'], res_df['pred_xco2'])
        pearson_r, _ = pearsonr(res_df['true_xco2'], res_df['pred_xco2'])
        rmse = np.sqrt(mean_squared_error(res_df['true_xco2'], res_df['pred_xco2']))
        
        print("\n" + "="*50)
        print("📊 Final EA Metrics Summary (Total XCO2 Consistency)")
        print("="*50)
        print(f"  Total XCO2 R²       : {r2:.6f}")
        print(f"  Total XCO2 r (Pear) : {pearson_r:.6f}")
        print(f"  Total XCO2 RMSE     : {rmse:.4f} ppm")
        print("-" * 50)
        print(f"  Mean Observed XCO2  : {res_df['true_xco2'].mean():.4f} ppm")
        print(f"  Mean Predicted XCO2 : {res_df['pred_xco2'].mean():.4f} ppm")
        print(f"  Systematic Bias     : {res_df['pred_xco2'].mean() - res_df['true_xco2'].mean():.4f} ppm")
        print("="*50)

if __name__ == "__main__":
    reverse_calculate_xco2()
