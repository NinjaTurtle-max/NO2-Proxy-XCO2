
"""
07_sr_pinn_highres_engine.py — SR-Informed PINN for 0.25° High-Res XCO2
=====================================================================
Expert: Senior Computational Physicist & Lead AI Architect
Objective: Hybrid Physics-Deep Learning using Advection-Diffusion PDE.
Logic: L_total = L_data(Uncertainty) + L_pde + L_mass + L_sr_prior
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────────────────────────
# 1. 환경 및 장치 설정 (MPS/CUDA 지원)
# ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# 데이터 경로 후보군 (외장 볼륨 및 로컬 경로 탐색)
POSSIBLE_PATHS = [
    "/Volumes/100.118.65.89/dataset/XCO2연구 데이터/03_split_output_025/anom_1d_balanced_025.parquet",
    "./anom_1d_balanced_025.parquet",
    "../anom_1d_balanced_025.parquet",
    "/Users/ganghyeon-u/Research/NO2-Proxy-XCO2/anom_1d_balanced_025.parquet"
]

DATA_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        DATA_PATH = p
        print(f"✅ Data found at: {p}")
        break

if DATA_PATH is None:
    # 데이터가 정말 없을 경우를 대비한 가상 데이터 생성 (테스트용) 또는 중단
    print("❌ Critical Error: 0.25 deg parquet data not found in any path.")
    print("   Please ensure the external drive is mounted or data is in the local folder.")
    # 실제 운영 시에는 여기서 sys.exit() 하거나 더 적극적인 탐색 필요
    DATA_PATH = POSSIBLE_PATHS[0] # 우선 기본값 유지

OUT_DIR = "07_sr_pinn_v1_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 2. 모델 아키텍처: Encoder-Decoder with ConvLSTM
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
    def __init__(self, in_channels=4): # [no2, u, v, blh]
        super().__init__()
        # Encoder: 시공간 특징 추출 (Spectral Norm으로 수치 안정성 확보)
        self.encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 32, 3, padding=1)),
            nn.SiLU(),
            ResNetBlock(32)
        )
        # ConvLSTM Cell: 시간적 이류-확산 기억 (h_t)
        self.lstm_cell = nn.Conv2d(32 + 32, 4 * 32, 3, padding=1)
        
        # Decoder: 0.25 deg 해상도 복원 및 Uncertainty 예측
        self.decoder = nn.Sequential(
            ResNetBlock(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.PixelShuffle(1), # 이미 0.25도이므로 Refinement용으로 사용
            nn.Conv2d(64, 2, 1) # Output: [Mean_XCO2, Log_Variance]
        )
        
        # Physics Parameters (SR-Prior 기반 학습 가능 파라미터)
        # alpha(배출 계수), gamma(확산 지수)를 학습 가능하도록 설정
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        
        # Diffusion Sub-net: kappa = f(BLH) (기상 조건에 따른 확산 계수 모델링)
        self.kappa_net = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(),
            nn.Linear(16, 1), nn.Softplus() # 확산 계수는 반드시 양수
        )

    def forward(self, x_seq):
        # x_seq: (B, T, C, H, W)
        b, t, c, h, w = x_seq.shape
        h_t = torch.zeros(b, 32, h, w, device=x_seq.device)
        c_t = torch.zeros(b, 32, h, w, device=x_seq.device)
        
        # 시간적 흐름(Temporal Advection) 압축
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
# 3. Physics Engine: PDE Residual & Mass Conservation
# ─────────────────────────────────────────────────────────────────
def compute_physics_loss(mu, inputs, model, dx=25000):
    """
    지배 방정식: dC/dt + u*grad(C) - div(kappa * grad(C)) - S = 0
    mu: 예측된 XCO2 Anomaly 장
    inputs: 현재 시점의 [no2, u, v, blh]
    """
    no2, u, v, blh = [inputs[:, i] for i in range(4)]
    
    # 1. Advection Term: 이류 (u*dC/dx + v*dC/dy)
    dC_dx = (torch.roll(mu, -1, 3) - torch.roll(mu, 1, 3)) / (2 * dx)
    dC_dy = (torch.roll(mu, -1, 2) - torch.roll(mu, 1, 2)) / (2 * dx)
    advection = u * dC_dx + v * dC_dy
    
    # 2. Diffusion Term: 확산 (div(kappa * grad(C)))
    kappa = model.kappa_net(blh.unsqueeze(-1)).squeeze(-1)
    d2C_dx2 = (torch.roll(mu, -1, 3) - 2*mu + torch.roll(mu, 1, 3)) / (dx**2)
    d2C_dy2 = (torch.roll(mu, -1, 2) - 2*mu + torch.roll(mu, 1, 2)) / (dx**2)
    diffusion = kappa * (d2C_dx2 + d2C_dy2)
    
    # 3. Source Term: 배출원 (SR-Prior 기반)
    ws = torch.sqrt(u**2 + v**2 + 1e-6)
    source = (model.alpha * no2) / (torch.pow(ws, model.gamma) + 1e-4)
    
    # PDE Residual (Steady State Assumption)
    residual = advection - diffusion - source
    l_pde = torch.mean(residual**2)
    
    # Mass Conservation (권역 전체 질량 보존 제약)
    l_mass = torch.abs(torch.sum(residual))
    
    return l_pde, l_mass

# ─────────────────────────────────────────────────────────────────
# 4. Data Pipeline & Training Loop
# ─────────────────────────────────────────────────────────────────
class AtmoGridDataset(Dataset):
    def __init__(self, df, seq_len=3):
        self.df = df.reset_index(drop=True)
        self.dates = sorted(df['date'].unique())
        self.seq_len = seq_len
        # 동아시아 0.25도 격자 크기 (예시: 120x200)
        self.grid_h, self.grid_w = 120, 200 
        
    def __len__(self): 
        return len(self.dates) - self.seq_len
    
    def __getitem__(self, idx):
        # 4D 텐서 구성: (T, C, H, W)
        seq_data = []
        for i in range(self.seq_len):
            day_df = self.df[self.df['date'] == self.dates[idx + i]]
            grid = np.zeros((5, self.grid_h, self.grid_w), dtype=np.float32)
            
            # 위경도를 0.25도 격자 인덱스로 매핑
            lats = ((day_df['latitude'] - 20.0) / 0.25).astype(int).clip(0, self.grid_h-1)
            lons = ((day_df['longitude'] - 100.0) / 0.25).astype(int).clip(0, self.grid_w-1)
            
            grid[0, lats, lons] = day_df['tropomi_no2'].values
            grid[1, lats, lons] = day_df['era5_u10'].values
            grid[2, lats, lons] = day_df['era5_v10'].values
            grid[3, lats, lons] = day_df['era5_blh'].values
            grid[4, lats, lons] = day_df['xco2_anomaly'].values
            seq_data.append(grid)
            
        seq_data = np.stack(seq_data)
        # Input features: [no2, u, v, blh], Target: [xco2_anomaly]
        return torch.from_numpy(seq_data[:, :4]), torch.from_numpy(seq_data[-1, 4])

def train_sr_pinn():
    print("🚀 SR-PINN 0.25° Engine Starting...")
    
    # 1. 데이터 로드 및 도메인 분할 (예시: NCP 핫스팟 권역)
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # 특정 핫스팟 도메인으로 데이터 제한 (메모리 효율 및 집중 학습)
    df = df[(df['latitude'] >= 30) & (df['latitude'] <= 42) & 
            (df['longitude'] >= 110) & (df['longitude'] <= 125)]
    
    dataset = AtmoGridDataset(df)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. 모델 초기화 및 가속 설정
    model = SR_PINN_AtmoNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 3. 학습 루프
    print(f"  [Training] Device: {DEVICE}, Samples: {len(dataset)}")
    for epoch in range(1, 51):
        model.train()
        total_l, pde_l, mass_l, data_l = 0, 0, 0, 0
        
        for x_seq, y_obs in loader:
            x_seq, y_obs = x_seq.to(DEVICE), y_obs.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward: Mean 예측 및 불확실성(Uncertainty) 산출
            mu, log_var = model(x_seq)
            
            # [긴급 수정 1] Aleatoric Uncertainty 발산 방지 (Clamping)
            # log_var가 너무 작아지면 exp(-log_var)가 폭주하여 수치적 붕괴 발생
            log_var = torch.clamp(log_var, min=-5.0, max=3.0)
            
            # 1. Aleatoric Data Loss (불확실성 가중 MSE)
            mask = (y_obs != 0).float()
            # Loss: 0.5 * exp(-log_var) * (error^2) + 0.5 * log_var
            # 이 수식은 error가 클 때 sigma를 키워 loss를 낮추려는 경향이 있으므로 제약이 필수적임
            l_data = torch.sum(mask * (0.5 * torch.exp(-log_var) * (mu - y_obs)**2 + 0.5 * log_var)) / (mask.sum() + 1e-6)
            
            # 2. Physics Loss (PDE 잔차 및 질량 보존)
            l_pde, l_mass = compute_physics_loss(mu, x_seq[:, -1], model)
            
            # [긴급 수정 2] PDE 가중치 상향 (0.1 -> 1.0)
            # 모델이 물리 법칙을 '옵션'이 아닌 '필수'로 인지하게 함
            loss = l_data + 1.0 * l_pde + 0.1 * l_mass
            
            loss.backward()
            optimizer.step()
            
            total_l += loss.item()
            pde_l += l_pde.item()
            mass_l += l_mass.item()
            data_l += l_data.item()
            
        print(f"Epoch {epoch:02d} | Loss: {total_l/len(loader):.4f} | Data: {data_l/len(loader):.4f} | PDE: {pde_l/len(loader):.4f} | Mass: {mass_l/len(loader):.4f}")
        
    # 4. 결과 저장
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "sr_pinn_highres_final.ckpt"))
    print(f"\n✅ Training Complete. Models saved to {OUT_DIR}")
    print(f"Optimized Physics: Alpha={model.alpha.item():.6f}, Gamma={model.gamma.item():.4f}")

if __name__ == "__main__":
    train_sr_pinn()
