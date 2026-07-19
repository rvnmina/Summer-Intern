"""
paper1_mobilenetv3_transfer_learning.py
=======================================
Reference implementation of Paper 1's core recipe (SoC-DPU paper):
  MobileNetV3 + transfer learning + two-phase fine-tuning + honest metrics.

It reproduces the *method*, not the accuracy numbers, on tiny SYNTHETIC data so
it runs in seconds with no dataset download.

Paper mapping
-------------
- MobileNetV3 backbone ............ Section 4.1 / Table 5
- Transfer learning (freeze base) . Section 6 ("freeze all layers ... feature extractor")
- Custom head + 20% Dropout ....... Section 6 ("Dropout Layer with a dropout rate of 20%")
- Phase 1: train head only ........ 10 epochs, Adam, lr=2e-4
- Phase 2: unfreeze + fine-tune ... 20 epochs, RMSProp, lr=2e-5
- Metrics: precision/recall/F1 .... Section 5.1, Eqs. (2)-(4)

Run:  python paper1_mobilenetv3_transfer_learning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

NUM_CLASSES = 11          # e.g. mung bean: 10 pests/diseases + healthy
IMG_SIZE = 160            # Paper 1 resizes to 160x160
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Model: MobileNetV3 with a custom classification head (transfer learning).
# ----------------------------------------------------------------------------
def build_model(num_classes=NUM_CLASSES, pretrained=True, freeze_base=True):
    """Load MobileNetV3-Large, optionally freeze its feature extractor, and
    replace the classifier head for our own classes."""
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v3_large(weights=weights)

    # TRANSFER LEARNING: freeze the pre-trained feature extractor so its good
    # ImageNet weights are not destroyed by the randomly-initialised head early on.
    if freeze_base:
        for p in net.features.parameters():
            p.requires_grad = False

    # Custom head: Linear -> h-swish -> Dropout(0.2) -> Linear (logits).
    in_feats = net.classifier[0].in_features
    net.classifier = nn.Sequential(
        nn.Linear(in_feats, 640),
        nn.Hardswish(),          # h-swish activation, as in MobileNetV3 (docs/02 §3.4)
        nn.Dropout(0.2),         # 20% dropout = regularisation (Paper 1, Section 6)
        nn.Linear(640, num_classes),
    )
    return net.to(DEVICE)


def set_base_trainable(net, trainable: bool):
    """Freeze/unfreeze the backbone for the two-phase schedule.
    Batch-Norm layers are kept in eval mode for stability (Paper 1, Section 6)."""
    for p in net.features.parameters():
        p.requires_grad = trainable
    for m in net.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()  # freeze BN running stats even when unfrozen


# ----------------------------------------------------------------------------
# Training / evaluation loops.
# ----------------------------------------------------------------------------
def run_epoch(net, loader, criterion, optimizer=None):
    train = optimizer is not None
    net.train() if train else net.eval()
    total, correct, loss_sum = 0, 0, 0.0
    torch.set_grad_enabled(train)
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = net(xb)
        loss = criterion(logits, yb)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    torch.set_grad_enabled(True)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_full(net, loader):
    """Honest per-class metrics + confusion matrix (docs/02 §3.2-3.3)."""
    net.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        logits = net(xb.to(DEVICE))
        y_pred += logits.argmax(1).cpu().tolist()
        y_true += yb.tolist()
    print("\nPer-class precision / recall / F1:")
    print(classification_report(y_true, y_pred, zero_division=0, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))


# ----------------------------------------------------------------------------
# Tiny synthetic dataset so the script runs immediately.
# Replace `make_fake_loaders` with a real ImageFolder loader for actual data.
# ----------------------------------------------------------------------------
def make_fake_loaders(n=256, batch=32):
    x = torch.rand(n, 3, IMG_SIZE, IMG_SIZE)         # fake RGB images
    y = torch.randint(0, NUM_CLASSES, (n,))          # random labels
    ds = TensorDataset(x, y)
    n_tr = int(0.8 * n)
    tr = DataLoader(TensorDataset(x[:n_tr], y[:n_tr]), batch_size=batch, shuffle=True)
    te = DataLoader(TensorDataset(x[n_tr:], y[n_tr:]), batch_size=batch)
    return tr, te


# To use REAL data instead, organise folders as:
#   data/train/<class_name>/*.jpg  and  data/val/<class_name>/*.jpg
# then:
#   from torchvision import datasets, transforms
#   tf = transforms.Compose([transforms.Resize((160,160)), transforms.ToTensor()])
#   train_ds = datasets.ImageFolder("data/train", tf)
#   train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader = make_fake_loaders()
    net = build_model(freeze_base=True)
    criterion = nn.CrossEntropyLoss()

    # ---- PHASE 1: train ONLY the new head (backbone frozen) ----
    # Paper 1: 10 epochs, Adam, lr=2e-4. We use 2 epochs here for a fast demo.
    print("\n=== PHASE 1: train classifier head only (backbone frozen) ===")
    opt1 = torch.optim.Adam(net.classifier.parameters(), lr=2e-4)
    for ep in range(2):
        tl, ta = run_epoch(net, train_loader, criterion, opt1)
        print(f"  epoch {ep+1}: train_loss={tl:.3f} train_acc={ta:.3f}")

    # ---- PHASE 2: unfreeze backbone, fine-tune everything gently ----
    # Paper 1: 20 epochs, RMSProp, lr=2e-5, BN frozen.
    print("\n=== PHASE 2: unfreeze backbone, fine-tune (RMSProp, tiny lr) ===")
    set_base_trainable(net, True)
    opt2 = torch.optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=2e-5)
    for ep in range(2):
        tl, ta = run_epoch(net, train_loader, criterion, opt2)
        print(f"  epoch {ep+1}: train_loss={tl:.3f} train_acc={ta:.3f}")

    # ---- Evaluate with honest per-class metrics ----
    print("\n=== EVALUATION (metrics are meaningless on random data; shows the pipeline) ===")
    evaluate_full(net, test_loader)
    print("\nDone. Swap `make_fake_loaders` for a real ImageFolder loader to train for real.")


if __name__ == "__main__":
    main()
