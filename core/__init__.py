# core: 길 B(XCO₂ 이상장 재구성) PINN 모델 뼈대
# ─────────────────────────────────────────────────────────────
# 공유 물리/손실:
#   pde_residual.py  - 등방성 Laplacian 확산 + source + anti-escape (변동도 정당화)
#   losses.py        - adaptive physics loss · 3-phase curriculum
#   base_pinn.py     - 공유 베이스(헤드·SR source β=0.0081·게이트·PDE)
# 6종 인코더 (forward 계약 공유: → c_hat,log_var,pde_res,s_anthro,gate):
#   arch_str.py            [V1] Spatiotemporal Transformer (local attention)
#   arch_convlstm_attn.py  [V2] ConvLSTM + Spatial Attention
#   arch_cnn_lstm.py       [V3] CNN + ConvGRU
#   arch_cnn3d.py          [V4] 3D CNN
#   arch_unet_pconv.py     [V5] U-Net + Partial Convolution
#   arch_mlp.py            [V6] MLP point-wise baseline
#
# 주의: arch_b_convlstm.py 는 소실된 4개 모듈(sr_prior 등)에 의존 → import 불가(레거시).

from .base_pinn import BaseSpatioTemporalPINN
from .arch_str import STRPINN
from .arch_convlstm_attn import ConvLSTMAttnPINN
from .arch_cnn_lstm import CNNLSTMPINN
from .arch_cnn3d import CNN3DPINN
from .arch_unet_pconv import UNetPConvPINN
from .arch_mlp import MLPPINN

ARCHITECTURES = {
    "str": STRPINN, "convlstm_attn": ConvLSTMAttnPINN, "cnn_lstm": CNNLSTMPINN,
    "cnn3d": CNN3DPINN, "unet_pconv": UNetPConvPINN, "mlp": MLPPINN,
}
