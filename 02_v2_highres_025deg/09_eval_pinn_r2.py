
"""
09_eval_pinn_r2.py — Evaluate PINN Model Performance (R2 & RMSE)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────
# 1. 모델 및 데이터셋 클래스 정의 (07/08번 스크립트와 동일)
# ─────────────────────────────────────────────────────────────────
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.SiLU(),
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        )
    def forward(self, x): return x + self.conv(x)

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
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.kappa_net = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1), nn.Softplus())

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

class AtmoGridDataset(Dataset):
    def __init__(self, df, seq_len=3):
        self.df = df.reset_index(drop=True)
        self.dates = sorted(df['date'].unique())
        self.seq_len = seq_len
        self.grid_h, self.grid_w = 120, 200 
        
    def __len__(self): return len(self.dates) - self.seq_len
    
    def __getitem__(self, idx):
        seq_data = []
        for i in range(self.seq_len):
            day_df = self.df[self.df['date'] == self.dates[idx + i]]
            grid = np.zeros((5, self.grid_h, self.grid_w), dtype=np.float32)
            lats = ((day_df['latitude'] - 20.0) / 0.25).astype(int).clip(0, self.grid_h-1)
            lons = ((day_df['longitude'] - 100.0) / 0.25).astype(int).clip(0, self.grid_w-1)
            grid[0, lats, lons] = day_df['tropomi_no2'].values
            grid[1, lats, lons] = day_df['era5_u10'].values
            grid[2, lats, lons] = day_df['era5_v10'].values
            grid[3, lats, lons] = day_df['era5_blh'].values
            grid[4, lats, lons] = day_df['xco2_anomaly'].values
            seq_data.append(grid)
        seq_data = np.stack(seq_data)
        return torch.from_numpy(seq_data[:, :4]), torch.from_numpy(seq_data[-1, 4])

# ─────────────────────────────────────────────────────────────────
# 2. 성능 평가 (Evaluation)
# ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
MODEL_PATH = "07_sr_pinn_v1_results/sr_pinn_highres_final.ckpt"
DATA_PATH = "/Volumes/100.118.65.89/dataset/XCO2연구 데이터/03_split_output_025/anom_1d_balanced_025.parquet"

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    print("📊 Evaluating PINN Model Performance...")
    model = SR_PINN_AtmoNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # NCP 등 주요 도메인 로드 (학습 시와 동일한 조건)
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df_sub = df[(df['latitude'] >= 30) & (df['latitude'] <= 42) & 
                (df['longitude'] >= 110) & (df['longitude'] <= 125)].copy()
    
    dataset = AtmoGridDataset(df_sub)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x_seq, y_obs in loader:
            x_seq = x_seq.to(DEVICE)
            y_obs = y_obs.cpu().numpy()
            
            mu, _ = model(x_seq)
            mu = mu.squeeze(1).cpu().numpy()
            
            # 관측값이 존재하는 격자(위성 관측점)만 마스킹하여 평가
            mask = y_obs != 0
            if mask.any():
                all_preds.extend(mu[mask].flatten())
                all_trues.extend(y_obs[mask].flatten())

    if len(all_trues) > 0:
        y_true = np.array(all_trues)
        y_pred = np.array(all_preds)
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print("\n" + "=" * 50)
        print("🎯 PINN Model Final Evaluation Metrics")
        print("=" * 50)
        print(f"  총 유효 평가 관측점 : {len(y_true):,} 개")
        print(f"  R² (결정계수)       : {r2:.4f}")
        print(f"  RMSE (평균제곱근오차): {rmse:.4f} ppm")
        print("=" * 50)
    else:
        print("❌ 유효한 관측값이 없습니다.")

if __name__ == "__main__":
    evaluate_model()
