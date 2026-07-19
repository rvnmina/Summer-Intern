# Phase 4 — Future Work, Novel Ideas, and Skills to Learn

This phase turns the three papers into a **project you can actually build**. It has two parts: (A) a
menu of novel research directions ranked by difficulty, and (B) the concrete skills each requires,
taught from theory to code.

---

## Part A — Novel ideas to build on top of these papers

Each idea explicitly patches a *stated limitation* of one or more papers, which makes it defensible
research (reviewers love "you addressed a limitation the original authors admitted").

### Idea 1 (Beginner) — Replace synthetic physiology with *real* proxy signals
**Gap it fixes:** Paper 2's chlorophyll/VOC are synthetic. **Do:** derive a *real* chlorophyll-proxy
from the RGB image itself using vegetation indices (e.g. a visible-band index like `ExG = 2G−R−B`, or
approximate GLI/VARI), and feed that as the "chlorophyll" channel. Now the physiology signal is
grounded in the actual pixels, not simulated. **Deliverable:** show accuracy with image-derived vs
synthetic proxy. **Skills:** image processing, vegetation indices (Part B-1).

### Idea 2 (Beginner→Intermediate) — Unified benchmark + honest test protocol
**Gap it fixes:** P1 augmented its test set; each paper uses its own private data. **Do:** build one
clean pipeline that trains on all three tasks with a *strict* real-only test set and reports
confusion matrices + per-class F1. **Deliverable:** a reproducible benchmark repo. **Skills:** PyTorch
data pipelines, evaluation (Part B-2).

### Idea 3 (Intermediate) — Add explainability that combines Grad-CAM + attention rollout
**Gap it fixes:** P3 calls for better field explainability. **Do:** produce, for every prediction, a
Grad-CAM map (CNN branch) *and* an attention-rollout map (transformer branch), and overlay them so a
farmer sees *why* the model flagged a leaf. **Deliverable:** an explainability dashboard.
**Skills:** hooks/gradients in PyTorch, attention rollout (Part B-3).

### Idea 4 (Intermediate) — Few-shot / continual adaptation to new pests
**Gap it fixes:** P2 and P3 both admit new pests need full retraining. **Do:** add a **prototypical-
network** head so the system can learn a *new* pest from just 5–10 example photos without retraining
the backbone. **Deliverable:** "add a new pest in 10 shots" demo. **Skills:** metric/few-shot learning
(Part B-4).

### Idea 5 (Intermediate→Advanced) — Robust federated learning under non-IID
**Gap it fixes:** P2's non-IID instability. **Do:** replace plain FedAvg with **FedProx** (adds a
"stay close to global" penalty) or **SCAFFOLD** (corrects client drift), and measure convergence vs
FedAvg on deliberately non-IID splits. **Deliverable:** an FL-algorithm comparison. **Skills:**
federated optimization (Part B-5).

### Idea 6 (Advanced) — True 4-modality pipeline with a real chlorophyll sensor
**Gap it fixes:** the synthetic-physiology critique at its root. **Do:** buy a cheap SPAD-style
chlorophyll meter + DHT22 temp/humidity + light sensor on a Raspberry Pi, collect a small real
4-modality dataset, and validate P2's framework on *real* biochemical inputs. **Deliverable:** the
first (small) real physiology-multimodal tea dataset. **Skills:** IoT sensing, embedded Python.

### Idea 7 (Advanced) — Deploy for real on a Jetson / Coral and measure energy
**Gap it fixes:** P3's silicon isn't fabricated. **Do:** take the multimodal model, quantize with
TFLite/ONNX, run it on a Jetson Nano or Google Coral, and measure *actual* latency/energy with a
USB power meter — reproducing (or contesting) the paper's numbers on commodity hardware.
**Deliverable:** an independent efficiency benchmark. **Skills:** model export + edge runtimes
(Part B-6).

**Recommended starter project for a beginner:** Ideas **1 + 2 + 3** combined = "A reproducible,
explainable, image-derived-physiology multimodal pest classifier." It touches all three papers, is
achievable in a semester, and directly answers their limitations.

---

## Part B — Skills to learn (theory → math → code), from basics up

> Learn these roughly in order. Each block: *why you need it → the theory/math → runnable code.*
> Everything here uses PyTorch (the framework Paper 3 used). Install: `pip install torch torchvision
> numpy matplotlib scikit-learn opencv-python`.

### B-0. Prerequisite: PyTorch tensors & autograd (1–2 days)
**Why:** it's the language all three models are written in. **Theory:** a *tensor* is an
n-dimensional array; **autograd** automatically computes gradients (derivatives of the loss w.r.t.
each weight) so you can train by gradient descent. **Gradient descent** = "nudge each weight a little
in the direction that reduces the loss," repeated.

```python
import torch
# A tensor that "remembers" operations so gradients can flow back
w = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0])
loss = (w * x - 12.0) ** 2      # we want w*x to equal 12 -> ideal w = 4
loss.backward()                  # autograd computes d(loss)/dw
print("gradient:", w.grad.item())        # negative -> increase w
with torch.no_grad():
    w -= 0.01 * w.grad           # one gradient-descent step
print("updated w:", w.item())    # moved toward 4
```

### B-1. Image processing & vegetation indices (2–3 days) — enables Idea 1
**Why:** to build a *real* chlorophyll proxy from RGB and to do the augmentations all three papers use.
**Theory:** an image is a `H×W×3` array of Red/Green/Blue values 0–255. Healthy, chlorophyll-rich
leaves reflect **more green, less red**. A **vegetation index** exploits this. Excess Green:
`ExG = 2G − R − B` (high where vegetation is greenest/healthiest).

```python
import cv2, numpy as np
img = cv2.imread("leaf.jpg").astype(np.float32) / 255.0   # BGR in OpenCV
B, G, R = img[..., 0], img[..., 1], img[..., 2]
exg = 2 * G - R - B                       # Excess Green index ~ chlorophyll proxy
chl_proxy = exg.mean()                     # one scalar "chlorophyll" feature per image
print("image-derived chlorophyll proxy:", chl_proxy)

# The augmentations from the papers, via torchvision:
from torchvision import transforms
aug = transforms.Compose([
    transforms.RandomRotation(25),                 # ±25° (Paper 3, Table 3)
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # illumination/humidity
    transforms.GaussianBlur(3),                    # fog/raindrops
    transforms.Resize((224, 224)),                 # Paper 3 input size
    transforms.ToTensor(),
])
```

### B-2. Building & evaluating a classifier properly (3–4 days) — enables Idea 2
**Why:** every paper is, at heart, a classifier; you must train and *honestly* evaluate one.
**Theory:** split data **before** augmentation into train/val/test so no augmented copy of a training
image leaks into test (Paper 3 stresses this). **Cross-entropy loss** measures how wrong the predicted
probabilities are: `L = −Σ y·log(ŷ)`. Report **per-class** precision/recall/F1 + a confusion matrix,
never just overall accuracy (recall Paper 1's hidden 34% downy-mildew class).

```python
from sklearn.metrics import classification_report, confusion_matrix
import torch, torch.nn as nn

# ... after training `model` ...
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        y_pred += logits.argmax(1).tolist()
        y_true += yb.tolist()
print(classification_report(y_true, y_pred, digits=4))  # per-class precision/recall/F1
print(confusion_matrix(y_true, y_pred))                 # find the weak classes
```

### B-3. Explainability: Grad-CAM + attention rollout (4–5 days) — enables Idea 3
**Why:** trust in the field; both P3 and P2 emphasize it. **Theory (Grad-CAM):** take the gradient of
the predicted class score w.r.t. the last conv feature maps; average those gradients per channel to
get "importance weights"; weight-sum the feature maps and ReLU → a heatmap of *where* the model
looked. **Attention rollout:** multiply the attention matrices across transformer layers to trace how
information flows from patches to the output.

```python
import torch, torch.nn.functional as F

def grad_cam(model, x, target_layer, class_idx=None):
    feats, grads = {}, {}
    target_layer.register_forward_hook(lambda m, i, o: feats.__setitem__('v', o))
    target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__('v', go[0]))
    logits = model(x)                              # forward
    if class_idx is None: class_idx = logits.argmax(1)
    model.zero_grad()
    logits[0, class_idx].backward()                # backward from the chosen class
    weights = grads['v'].mean(dim=(2, 3), keepdim=True)   # global-average-pool the gradients
    cam = F.relu((weights * feats['v']).sum(1))    # weighted sum of feature maps, keep positives
    return (cam - cam.min()) / (cam.max() + 1e-8)  # normalize to [0,1] heatmap
```

### B-4. Few-shot learning with prototypical networks (5–7 days) — enables Idea 4
**Why:** adapt to new pests from a handful of photos (P2/P3 limitation). **Theory:** embed every image
into a vector with the backbone. For each class, average its few support examples into a **prototype**
vector. Classify a new image by which prototype it's **closest** to (smallest distance). No retraining
— you just add a new prototype for a new pest.

```python
import torch
def prototypical_predict(support_embeds, support_labels, query_embeds, n_classes):
    # 1) Build one prototype (mean vector) per class from the few support examples
    protos = torch.stack([support_embeds[support_labels == c].mean(0) for c in range(n_classes)])
    # 2) Classify each query by nearest prototype (negative squared Euclidean distance)
    dists = torch.cdist(query_embeds, protos)      # [n_query, n_classes]
    return dists.argmin(1)                          # closest prototype wins
```

### B-5. Federated optimization: FedAvg → FedProx (5–7 days) — enables Idea 5
**Why:** the core of Paper 2, and its stability under non-IID is an open problem.
**Theory (FedProx):** identical to FedAvg but each client adds a **proximal penalty** `(μ/2)·‖θ −
θ_global‖²` to its local loss — punishing local models for straying too far from the global one, which
tames non-IID client drift.

```python
import torch
def fedprox_local_loss(base_loss, local_params, global_params, mu=0.01):
    # base_loss: the usual task loss; add a "stay near global" penalty
    prox = sum(((lp - gp) ** 2).sum() for lp, gp in zip(local_params, global_params))
    return base_loss + (mu / 2) * prox
```
See `../code/shared_concepts/federated_learning_demo.py` for a full FedAvg loop over non-IID clients.

### B-6. Model compression & edge export (1 week) — enables Idea 7
**Why:** every paper deploys to edge; this is how you actually do it.
**Theory:** (1) **quantization** (Paper 1 doc §3.5) — FP32→INT8; (2) **export** to a portable format
(ONNX or TorchScript) that edge runtimes (TensorRT, TFLite, ONNX Runtime) can accelerate.

```python
import torch
# Dynamic INT8 quantization of a trained model's linear layers (CPU, one line)
q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
# Export to ONNX for edge runtimes (Jetson/Coral)
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(q_model, dummy, "pest_model.onnx", opset_version=13,
                  input_names=["image"], output_names=["logits"])
```

### B-7 (Optional, advanced). Hardware for the curious — FPGA/HLS
Only if you want P1/P3's hardware angle. **Theory:** **High-Level Synthesis (HLS)** lets you write C/
C++ that a tool (Xilinx Vitis HLS) turns into an FPGA circuit; **Vitis AI** quantizes and compiles a
neural net into a DPU-runnable `.xmodel`. You don't need to design silicon — you *use* the DPU. Start
with the free Vitis AI tutorials and a Zynq/ZCU104 board (or the QEMU emulator). This is a
multi-month skill; treat it as optional depth, not a project blocker.

---

## Suggested 8-week learning + build plan

| Week | Focus |
|------|-------|
| 1 | B-0 PyTorch + B-1 image processing; reproduce Paper 1's transfer-learning demo (`code/paper1_*`). |
| 2 | B-2 proper training/eval; build the honest benchmark (Idea 2). |
| 3 | Attention from scratch (`code/shared_concepts/attention_from_scratch.py`); understand Paper 3 math. |
| 4 | Build the hybrid CNN–Transformer + fusion (`code/paper3_*`); add image-derived chlorophyll (Idea 1). |
| 5 | B-3 explainability; add Grad-CAM + attention overlay (Idea 3). |
| 6 | B-5 federated learning (`code/shared_concepts/federated_learning_demo.py`); FedAvg vs FedProx (Idea 5). |
| 7 | B-4 few-shot head (Idea 4); B-6 quantize + export. |
| 8 | Integrate into one repo, write the README, record results, prepare the presentation (Phase 5). |

Continue to `07_github_guide.md`.
