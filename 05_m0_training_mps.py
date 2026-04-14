import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import r2_score
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
# 0. 경로 및 엄격한 하이퍼파라미터 설정
# ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps")
torch.set_default_dtype(torch.float32)

BASE_DIR = "/Volumes/100.118.65.89/dataset/XCO2연구 데이터"
PARQUET_IN = os.path.join(BASE_DIR, "ml_ready_dataset.parquet") 
OUT_DIR = os.path.join(BASE_DIR, "m0_results_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# Strict Constraints (Manual Hyperparameter Control)
BATCH_SIZE = 4
PATCH_SIZE = 32 # Adjusted to 32 to fit within the new 0.5 deg (60x100) grid
SEQ_LEN = 14
HIDDEN_DIMS = [32, 64, 32]
INIT_LR = 1e-4
MAX_LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
EPS = 1e-8

# ─────────────────────────────────────────────────────────────────
# 1. Model Architecture: ConvLSTM + Spatial Attention
# ─────────────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, gates.size(1) // 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class SpatialAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.scale = (in_dim // 8) ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h*w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h*w)
        v = self.value(x).view(b, -1, h*w)
        
        # Matrix multiplication: (B, HW, C') @ (B, C', HW) -> (B, HW, HW)
        # For PATCH_SIZE=64, HW=4096. Matrix size = 4 * 4096 * 4096 = 67M elements (256MB)
        # For PATCH_SIZE=128, HW=16384. Matrix size = 4 * 16384 * 16384 = 1G elements (4GB - CRASH)
        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return out, attn

class M0_Model(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cell1 = ConvLSTMCell(in_channels, HIDDEN_DIMS[0])
        self.cell2 = ConvLSTMCell(HIDDEN_DIMS[0], HIDDEN_DIMS[1])
        self.cell3 = ConvLSTMCell(HIDDEN_DIMS[1], HIDDEN_DIMS[2])
        self.attn = SpatialAttention(HIDDEN_DIMS[2])
        self.head = nn.Conv2d(HIDDEN_DIMS[2], 1, 1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        s1 = (torch.zeros(b, HIDDEN_DIMS[0], h, w, device=DEVICE), torch.zeros(b, HIDDEN_DIMS[0], h, w, device=DEVICE))
        s2 = (torch.zeros(b, HIDDEN_DIMS[1], h, w, device=DEVICE), torch.zeros(b, HIDDEN_DIMS[1], h, w, device=DEVICE))
        s3 = (torch.zeros(b, HIDDEN_DIMS[2], h, w, device=DEVICE), torch.zeros(b, HIDDEN_DIMS[2], h, w, device=DEVICE))
        
        for i in range(t):
            s1 = self.cell1(x[:, i], s1)
            s2 = self.cell2(s1[0], s2)
            s3 = self.cell3(s2[0], s3)
        
        feat, attn_map = self.attn(s3[0])
        return self.head(feat), attn_map

# ─────────────────────────────────────────────────────────────────
# 2. Data Pipeline: Large-scale Dataset with Flexible Anomaly
# ─────────────────────────────────────────────────────────────────
class LargeXCO2GridDataset(Dataset):
    def __init__(self, df, patch_size=64, seq_len=14, is_train=True):
        # Latitudinal Baseline Removal
        print(f"  [Data] Initializing Dataset (Patch Size: {patch_size})...")
        df['lat_bin'] = df['latitude'].round(0)
        df['time'] = pd.to_datetime(df['time'])
        df['month'] = df['time'].dt.month
        baseline = df.groupby(['lat_bin', 'month'])['xco2'].transform('mean')
        df['xco2_anomaly'] = df['xco2'] - baseline
        
        # Min-Max Scaling
        self.features = ['tropomi_no2', 'era5_u10', 'era5_v10', 'era5_blh', 'population_density', 'odiac_emission']
        for f in self.features:
            f_min, f_max = df[f].min(), df[f].max()
            df[f] = (df[f] - f_min) / (f_max - f_min + EPS)
            
        self.df = df
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.is_train = is_train
        self.dates = sorted(self.df['time'].unique())
        
        # Map lat/lon to 60x100 indices (0.5 deg resolution)
        self.df['lat_idx'] = ((self.df['latitude'] - 20.0) / 0.5).astype(int).clip(0, 59)
        self.df['lon_idx'] = ((self.df['longitude'] - 100.0) / 0.5).astype(int).clip(0, 99)
        self.grid_shape = (60, 100)

    def __len__(self):
        return len(self.dates) - self.seq_len

    def __getitem__(self, idx):
        seq_x = []
        for i in range(self.seq_len):
            date = self.dates[idx + i]
            day_df = self.df[self.df['time'] == date]
            
            grid = np.zeros((len(self.features) + 1, *self.grid_shape), dtype=np.float32)
            if not day_df.empty:
                lats, lons = day_df['lat_idx'].values, day_df['lon_idx'].values
                for f_idx, feat in enumerate(self.features):
                    grid[f_idx, lats, lons] = day_df[feat].values
                grid[-1, lats, lons] = day_df['xco2_anomaly'].values
            seq_x.append(grid)
        
        seq_x = np.stack(seq_x)
        
        # Patching
        h, w = self.grid_shape
        if self.is_train:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
        else:
            top, left = (h - self.patch_size)//2, (w - self.patch_size)//2
            
        patch = seq_x[:, :, top:top+self.patch_size, left:left+self.patch_size]
        x = torch.from_numpy(patch[:, :-1]).float()
        y = torch.from_numpy(patch[-1, -1:]).float()
        m = (torch.abs(y) > EPS).float()
        
        return x, y, m

# ─────────────────────────────────────────────────────────────────
# 3. Training Loop
# ─────────────────────────────────────────────────────────────────
def masked_mse_loss(pred, target, mask):
    diff = (pred - target)**2
    masked_diff = diff * mask
    return masked_diff.sum() / (mask.sum() + EPS)

def train():
    print("  [Data] Loading 7.5M row dataset...")
    df = pd.read_parquet(PARQUET_IN)
    
    dates = sorted(df['time'].unique())
    split_date = dates[int(len(dates)*0.8)]
    df_train = df[df['time'] < split_date].copy()
    df_val = df[df['time'] >= split_date].copy()

    # Pass PATCH_SIZE explicitly to ensure 64x64
    train_ds = LargeXCO2GridDataset(df_train, patch_size=PATCH_SIZE, is_train=True)
    val_ds = LargeXCO2GridDataset(df_val, patch_size=PATCH_SIZE, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = M0_Model(in_channels=6).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=INIT_LR)
    
    best_v_loss = float('inf')
    early_stop_cnt = 0

    print(f"  [Train] Starting Large-scale Training (Patch: {PATCH_SIZE}) on {DEVICE}...")
    for epoch in range(1, 101):
        model.train()
        train_loss = 0
        for x, y, m in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = masked_mse_loss(pred, y, m)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        v_loss, v_rmse, v_r2 = 0, 0, 0
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
                pred, _ = model(x)
                v_loss += masked_mse_loss(pred, y, m).item()
                mask_bool = m > 0
                if mask_bool.any():
                    p_f, t_f = pred[mask_bool].cpu().numpy(), y[mask_bool].cpu().numpy()
                    if len(t_f) > 1:
                        v_rmse += np.sqrt(np.mean((p_f - t_f)**2))
                        v_r2 += r2_score(t_f, p_f)
        
        avg_val = v_loss / len(val_loader)
        avg_rmse = v_rmse / len(val_loader)
        avg_r2 = v_r2 / len(val_loader)
        print(f"  [{epoch:03d}] T-Loss: {train_loss/len(train_loader):.6f} | V-Loss: {avg_val:.6f} | RMSE: {avg_rmse:.4f} | R2: {avg_r2:.4f}")
        
        if avg_val < best_v_loss:
            best_v_loss = avg_val
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "m0_baseline_large.ckpt"))
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        
        scheduler.step()
        torch.mps.empty_cache()
        if early_stop_cnt >= PATIENCE: break

if __name__ == "__main__":
    train()
