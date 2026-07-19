"""
paper2_chlorophyll_multimodal_transformer.py
============================================
Reference implementation of Paper 2's model (Tea-Physio-FL):
a physiology-aware multimodal transformer that fuses 4 modalities
(RGB, chlorophyll, VOC, microclimate) with PHYSIOLOGY-CONSTRAINED attention,
infers a latent biophysical state Z, and regresses a pest-RISK score Y in [0,1].

Key difference from Paper 3: this predicts a *continuous risk* (forecasting)
rather than a class (detection), and the chemistry gates the attention (⊙ P).

Runs on tiny synthetic data. (Fittingly, Paper 2 itself used synthetic
chlorophyll/VOC proxies -- see docs/04 §0.)

Paper mapping
-------------
- 4 modality encoders ................ Section IV-B
- Physiology-constrained attention ... A = softmax((QKᵀ/√d) ⊙ P) V, Section IV-C
- Latent state Z, risk Y=h(Z) ........ Section IV-D, (X)->Z->Y
- Federated training ................. see shared_concepts/federated_learning_demo.py

Run:  python paper2_chlorophyll_multimodal_transformer.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

D = 48                      # shared latent width
IMG_SIZE = 32               # tiny for demo
CHL_DIM = 4                 # chlorophyll index vector
VOC_DIM = 16                # VOC spectrum vector
ENV_DIM = 3                 # microclimate: temp/humidity/light
Z_DIM = 4                   # latent state coords: Z1..Z4 (photo-stability, VOC, thermal, hydric)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Modality-specific encoders -- each stream mapped into the same latent width D.
# ----------------------------------------------------------------------------
class ModalityEncoders(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB -> a few patch tokens (simplified ViT tokeniser).
        self.rgb_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, D, 3, 2, 1), nn.ReLU(),
        )
        self.chl = nn.Sequential(nn.Linear(CHL_DIM, D), nn.GELU())
        self.voc = nn.Sequential(nn.Linear(VOC_DIM, D), nn.GELU())
        self.env = nn.Sequential(nn.Linear(ENV_DIM, D), nn.GELU())

    def forward(self, rgb, chl, voc, env):
        B = rgb.size(0)
        vis = self.rgb_cnn(rgb).flatten(2).transpose(1, 2)   # [B, n_tokens, D]
        e_chl = self.chl(chl)                                 # [B, D]
        e_voc = self.voc(voc)                                 # [B, D]
        e_env = self.env(env)                                 # [B, D]
        # Build a token sequence: visual patch tokens + one token per other modality.
        tokens = torch.cat([vis, e_chl[:, None], e_voc[:, None], e_env[:, None]], dim=1)
        return tokens, e_chl, e_voc, e_env


# ----------------------------------------------------------------------------
# Physiology-constrained cross-modal attention (Paper 2's signature mechanism).
# A biology-derived gate P element-wise multiplies the attention scores.
# ----------------------------------------------------------------------------
class PhysiologyConstrainedAttention(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.Wq, self.Wk, self.Wv = (nn.Linear(d, d) for _ in range(3))
        self.to_gate = nn.Linear(2 * d, d)     # P built from chlorophyll + VOC
        self.d = d

    def forward(self, tokens, e_chl, e_voc):
        Q, K, V = self.Wq(tokens), self.Wk(tokens), self.Wv(tokens)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d)         # [B, N, N]
        gate = torch.sigmoid(self.to_gate(torch.cat([e_chl, e_voc], dim=-1)))  # [B, D] in (0,1)
        p_key = gate.mean(-1, keepdim=True).unsqueeze(1)             # [B, 1, 1] broadcast over keys
        scores = scores * p_key                                     # ⊙ P : chemistry reshapes attention
        weights = F.softmax(scores, dim=-1)
        return weights @ V


# ----------------------------------------------------------------------------
# Full model: encoders -> constrained attention -> latent Z -> risk Y.
# ----------------------------------------------------------------------------
class PhysioMultimodalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = ModalityEncoders()
        self.attn = PhysiologyConstrainedAttention()
        self.norm = nn.LayerNorm(D)
        self.to_Z = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, Z_DIM))  # g(H) -> Z
        self.to_Y = nn.Sequential(nn.Linear(Z_DIM, 1))                              # h(Z) -> risk

    def forward(self, rgb, chl, voc, env, return_Z=False):
        tokens, e_chl, e_voc, e_env = self.encoders(rgb, chl, voc, env)
        attended = self.norm(self.attn(tokens, e_chl, e_voc) + tokens)   # residual + norm
        H = attended.mean(dim=1)                                          # fuse tokens -> H
        Z = self.to_Z(H)                                                  # latent biophysical state
        Y = torch.sigmoid(self.to_Y(Z)).squeeze(-1)                       # monotone risk in [0,1]
        return (Y, Z) if return_Z else Y


# ----------------------------------------------------------------------------
# Synthetic data where risk genuinely depends on chlorophyll & VOC, so the
# model has a real signal to learn (mirroring the paper's ablation finding
# that chlorophyll/VOC matter most).
# ----------------------------------------------------------------------------
def make_fake_physio(n=512, batch=32):
    rgb = torch.rand(n, 3, IMG_SIZE, IMG_SIZE)
    chl = torch.rand(n, CHL_DIM)
    voc = torch.rand(n, VOC_DIM)
    env = torch.rand(n, ENV_DIM)
    # Ground-truth latent stress: low chlorophyll + high VOC => high risk.
    stress = (1 - chl.mean(1)) * 0.6 + voc.mean(1) * 0.4 + 0.1 * env[:, 0]
    risk = torch.sigmoid(4 * (stress - stress.mean()))                   # target Y in (0,1)
    ds = TensorDataset(rgb, chl, voc, env, risk)
    n_tr = int(0.8 * n)
    tr = DataLoader(TensorDataset(rgb[:n_tr], chl[:n_tr], voc[:n_tr], env[:n_tr], risk[:n_tr]),
                    batch_size=batch, shuffle=True)
    te = DataLoader(TensorDataset(rgb[n_tr:], chl[n_tr:], voc[n_tr:], env[n_tr:], risk[n_tr:]),
                    batch_size=batch)
    return tr, te


def main():
    print(f"Device: {DEVICE}")
    tr, te = make_fake_physio()
    model = PhysioMultimodalTransformer().to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    crit = nn.MSELoss()                       # risk regression (continuous target)

    print("\n=== Training physiology-aware risk forecaster ===")
    for ep in range(6):
        model.train(); lsum, tot = 0.0, 0
        for rgb, chl, voc, env, y in tr:
            rgb, chl, voc, env, y = (t.to(DEVICE) for t in (rgb, chl, voc, env, y))
            pred = model(rgb, chl, voc, env)
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            lsum += loss.item() * rgb.size(0); tot += rgb.size(0)
        print(f"  epoch {ep+1}: train MSE={lsum/tot:.4f}")

    # Evaluate + peek at the learned latent state Z.
    model.eval(); mse, tot = 0.0, 0
    with torch.no_grad():
        for rgb, chl, voc, env, y in te:
            rgb, chl, voc, env, y = (t.to(DEVICE) for t in (rgb, chl, voc, env, y))
            pred, Z = model(rgb, chl, voc, env, return_Z=True)
            mse += F.mse_loss(pred, y, reduction="sum").item(); tot += rgb.size(0)
    print(f"\nTest MSE: {mse/tot:.4f}  (lower = better risk prediction)")
    print("Example latent state Z (Z1=photo-stability, Z2=VOC, Z3=thermal, Z4=hydric):")
    print("  ", Z[0].cpu().round(decimals=3).tolist())
    print("\nThe model infers an UNSEEN biophysical state Z, then maps Z -> risk.")


if __name__ == "__main__":
    main()
