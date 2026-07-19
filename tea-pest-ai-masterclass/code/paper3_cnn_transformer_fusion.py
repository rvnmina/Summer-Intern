"""
paper3_cnn_transformer_fusion.py
================================
Reference implementation of Paper 3's model (Edge-Multimodal, tea pests):
a hybrid CNN-Transformer visual backbone whose features are conditioned on
environmental sensors (temperature, humidity, illumination) via cross-modal
attention -- i.e. it learns P(y | x, e), not just P(y | x).

Runs on tiny synthetic data (5 pest classes + 3 env variables) in seconds.

Paper mapping
-------------
- CNN branch (local texture) ........... Eq. (3), Section 4.2
- Transformer branch (global context) .. Eqs. (4)-(5), patch tokens + self-attention
- Feature blend F = alpha*z_c+(1-a)*z_t . Eq. (6)
- Cross-modal fusion F*(x,e) ........... Eqs. (7)-(8), Section 4.3
- Classifier head -> 5 classes ......... Section 4.4

Run:  python paper3_cnn_transformer_fusion.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

NUM_CLASSES = 5          # aphid, red spider mite, tea mosquito bug, looper, thrips
NUM_ENV = 3              # temperature, humidity, illumination
IMG_SIZE = 64            # small for a fast demo (paper uses 224)
D_MODEL = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# CNN branch: local texture features (lesion boundaries, insect morphology).
# Uses depthwise-separable-style efficiency + a residual connection.
# ----------------------------------------------------------------------------
class CNNBranch(nn.Module):
    def __init__(self, d_out=D_MODEL):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.Hardswish(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.Hardswish(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling -> one vector
        self.proj = nn.Linear(64, d_out)

    def forward(self, x):
        f = self.stem(x)                       # [B, 64, H/4, W/4]
        g = self.pool(f).flatten(1)            # [B, 64]
        return self.proj(g)                    # z_c : [B, d_out]


# ----------------------------------------------------------------------------
# Transformer branch: split image into patch tokens, run self-attention.
# ----------------------------------------------------------------------------
class TransformerBranch(nn.Module):
    def __init__(self, d_out=D_MODEL, patch=8, img_size=IMG_SIZE):
        super().__init__()
        self.patch = patch
        n_patches = (img_size // patch) ** 2
        patch_dim = 3 * patch * patch
        self.embed = nn.Linear(patch_dim, d_out)                 # linear patch projection
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_out) * 0.02)  # positional encoding
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_out, nhead=4, dim_feedforward=2 * d_out, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

    def _to_patches(self, x):
        # [B,3,H,W] -> [B, n_patches, 3*patch*patch]  (the ViT tokenisation trick)
        B, C, H, W = x.shape
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)                    # [B,C,H/p,W/p,p,p]
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)
        return x

    def forward(self, x):
        tokens = self.embed(self._to_patches(x)) + self.pos      # [B, N, d]
        enc = self.encoder(tokens)                               # self-attention over patches
        return enc.mean(dim=1)                                   # z_t : [B, d] (pool tokens)


# ----------------------------------------------------------------------------
# Cross-modal fusion: environment (Query) modulates visual features (Key/Value).
# Implements F*(x,e) = sigma( attention(Q=e, K=V=vis) + z_v )   [Eqs. 7-8]
# ----------------------------------------------------------------------------
class CrossModalFusion(nn.Module):
    def __init__(self, d=D_MODEL, n_env=NUM_ENV):
        super().__init__()
        self.Wv = nn.Linear(d, d)        # project visual feature
        self.We = nn.Linear(n_env, d)    # project environment vector
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wvv = nn.Linear(d, d)
        self.d = d

    def forward(self, vis, env):
        z_v = self.Wv(vis)               # [B, d]
        z_e = self.We(env)               # [B, d]
        Q, K, V = self.Wq(z_e), self.Wk(z_v), self.Wvv(z_v)
        # Single-token attention: similarity between env-query and vis-key.
        score = (Q * K).sum(-1, keepdim=True) / math.sqrt(self.d)   # [B, 1]
        attn = torch.sigmoid(score)                                 # env-derived gate in (0,1)
        return F.relu(attn * V + z_v)                               # residual fusion F*(x,e)


# ----------------------------------------------------------------------------
# Full multimodal model.
# ----------------------------------------------------------------------------
class MultimodalPestNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = CNNBranch()
        self.vit = TransformerBranch()
        self.alpha = nn.Parameter(torch.tensor(0.5))     # learnable blend weight (Eq. 6)
        self.fusion = CrossModalFusion()
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 128), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, env):
        z_c = self.cnn(image)
        z_t = self.vit(image)
        a = torch.sigmoid(self.alpha)                    # keep alpha in (0,1)
        F_vis = a * z_c + (1 - a) * z_t                  # hybrid visual feature (Eq. 6)
        F_star = self.fusion(F_vis, env)                 # environment-conditioned (Eqs. 7-8)
        return self.head(F_star)


# ----------------------------------------------------------------------------
# Demo on synthetic multimodal data.
# ----------------------------------------------------------------------------
def make_fake_multimodal(n=256, batch=32):
    images = torch.rand(n, 3, IMG_SIZE, IMG_SIZE)
    env = torch.rand(n, NUM_ENV)                          # normalised temp/humidity/light
    labels = torch.randint(0, NUM_CLASSES, (n,))
    ds = TensorDataset(images, env, labels)
    n_tr = int(0.8 * n)
    tr = DataLoader(TensorDataset(images[:n_tr], env[:n_tr], labels[:n_tr]),
                    batch_size=batch, shuffle=True)
    te = DataLoader(TensorDataset(images[n_tr:], env[n_tr:], labels[n_tr:]), batch_size=batch)
    return tr, te


def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader = make_fake_multimodal()
    model = MultimodalPestNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)   # Paper 3 uses Adam, lr=1e-4
    crit = nn.CrossEntropyLoss()

    print("\n=== Training the multimodal (image + environment) model ===")
    for ep in range(3):
        model.train()
        tot, corr, lsum = 0, 0, 0.0
        for img, env, y in train_loader:
            img, env, y = img.to(DEVICE), env.to(DEVICE), y.to(DEVICE)
            logits = model(img, env)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            lsum += loss.item() * img.size(0)
            corr += (logits.argmax(1) == y).sum().item(); tot += img.size(0)
        print(f"  epoch {ep+1}: loss={lsum/tot:.3f} acc={corr/tot:.3f}")

    # Show the model reacts to the environment: same image, different env -> can differ.
    model.eval()
    with torch.no_grad():
        img = torch.rand(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        cold = model(img, torch.tensor([[0.1, 0.9, 0.2]]).to(DEVICE)).softmax(-1)
        hot = model(img, torch.tensor([[0.9, 0.3, 0.9]]).to(DEVICE)).softmax(-1)
    print("\nSame leaf image, different weather -> different class probabilities:")
    print("  cold/humid env:", cold.squeeze().cpu().round(decimals=3).tolist())
    print("  hot/dry env   :", hot.squeeze().cpu().round(decimals=3).tolist())
    print("\nThat difference IS the point of P(y | x, e): environment conditions the prediction.")


if __name__ == "__main__":
    main()
