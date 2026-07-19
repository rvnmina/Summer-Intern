# Phase 2 — Paper 3 Deep Dive: Edge-Multimodal (read second)

> **Full title:** Transformative Energy-Efficient Edge-Optimised Multimodal Deep Learning Framework
> for Pest Management and Severity Analysis in Tea Plants
> **Authors:** MD Tausif Mallick, Himadri Saha, Amlan Chakrabarti · *ACM Trans. Embedded Computing
> Systems 25(4), Article 63, 2026*

> **Why you're reading this second:** It reuses Paper 1's ideas (CNN, quantization, edge hardware,
> the same metrics) and introduces the two big concepts Paper 2 needs — **the transformer /
> self-attention** and **multimodal fusion**. This doc teaches both from zero.

---

## 0. The 60-second version

Tea plantations get hit by 5 major pests. Photo-only classifiers work great in a lab but **fall
apart in the field** because of low light, humidity, and leaves hiding each other. Key insight: the
*weather* (temperature, humidity, light) strongly affects both how pests behave *and* how the photo
looks — so **feed the weather into the model too**. This paper builds a **hybrid CNN–Transformer**
that fuses the pest image with live environmental sensor readings via **attention**, then compresses
it (pruning + INT8 quantization + distillation) and runs it on **custom low-power edge hardware**:
**25 ms** per photo, **0.12 J** per inference, **60% less energy** than a Jetson Nano. Multimodal
fusion keeps accuracy at **85.5%** under combined field distortions where an image-only model drops
to **77.9%**.

---

## 1. Theory and Concept

### 1.1 The core problem
A pest classifier trained on clean images learns `P(y | x)` — "probability of pest class `y` given
image `x`." In the field, the *distribution* of images changes (dark, blurry, occluded). This is
**domain shift**: train and test data no longer look alike, so accuracy collapses. Also, standard
edge devices (drones, handheld scanners) can't run big models within their power/latency budget.

### 1.2 The proposed solution
Two reframings:

1. **Condition on the environment.** Instead of `P(y | x)`, learn `P(y | x, e)` where `e` = [temp,
   humidity, illumination]. The same leaf photo means different things at 32 °C vs 20 °C. Environment
   `e` is fed in and, through **cross-modal attention**, *modulates* the visual features. This is
   **multimodal learning**.
2. **Bake deployment limits into the objective.** They don't just maximize accuracy; they minimize
   loss **subject to** hardware constraints (latency ≤ 25 ms, energy ≤ 0.12 J, memory ≤ 110 MB) — see
   §3.1. Then they co-design custom hardware to hit those numbers.

### 1.3 The methodology, end to end
```
[Image x] → CNN branch (local texture) ┐
[Image x] → Transformer branch (global context) ┘→ fuse → visual feature F(x)
[Weather e] → tiny encoder → z_e
                    │
        cross-modal attention: e modulates F(x) → F*(x,e)
                    │
             classifier head → one of 5 pests
                    │
   compress (prune + INT8 + distill) → deploy on custom dual-mode edge chip
                    │
   Edge mode: solar-powered offline inference   |   Cloud mode: periodic model updates
```
The authors are explicit: **the novelty is the system-level integration**, not any single new
component. CNNs, transformers, attention, quantization, distillation are all borrowed — the
contribution is fusing them under joint accuracy + hardware constraints for real tea fields.

---

## 2. Technical Vocabulary — zero to advanced

Terms already covered in Paper 1 (CNN, convolution, ReLU, softmax, quantization, precision/recall/F1,
FPGA, DPU-style accelerators) are assumed. New terms below.

### 2.1 The Transformer & Self-Attention (the single most important new idea)
- **What:** A neural architecture (from the 2017 paper *"Attention Is All You Need"*) that lets every
  part of the input look at **every other part** and decide what's relevant — "attention."
- **The problem it solves:** A CNN only looks at a small local window at a time. To relate a lesion on
  the top-left of a leaf to one on the bottom-right, a CNN needs many stacked layers. A transformer
  relates *any two positions directly* in one step — this is **long-range/global context**.
- **How self-attention works (intuition):** Each input element creates three vectors:
  - a **Query (Q)** — "what am I looking for?"
  - a **Key (K)** — "what do I contain?"
  - a **Value (V)** — "what do I pass on if attended to?"
  You compare every Query against every Key (dot product = similarity), softmax those into weights,
  and take a weighted sum of the Values. Elements that match get more weight.
- **Why the √d:** dot products grow with vector dimension `d`; dividing by `√d` keeps them in a
  stable range so softmax doesn't saturate. (Math in §3.2.)
- **Multi-head attention:** run several attention operations ("heads") in parallel, each learning a
  different kind of relationship (one head tracks color, another tracks shape), then concatenate.
- **When:** Images (Vision Transformers), text, audio, and — here — fusing image + sensors.

### 2.2 Patches & tokens (how a transformer eats an image)
- **What:** A transformer works on a *sequence* of vectors ("tokens"), but an image is a 2-D grid.
- **How:** Chop the image into fixed **patches** (e.g. 16×16 pixels), flatten each patch into a
  vector, and treat the sequence of patch-vectors as tokens. The paper's formula: an `H×W` image with
  patch size `p` gives `N = HW/p²` tokens. This is the **Vision Transformer (ViT)** recipe.
- **Positional encoding:** since attention is order-agnostic, you add a "position tag" to each token
  so the model knows where each patch was.

### 2.3 Hybrid CNN–Transformer backbone
- **What:** Use a CNN branch *and* a transformer branch together.
- **Why:** CNNs are great at **local** texture (lesion edges, insect legs) and are efficient;
  transformers are great at **global** context (overall leaf layout). Combining them gives the best of
  both — and, as §3.5 shows, **reduces variance** (error) versus either alone.
- **How (this paper):** CNN branch → `f_c(x)`; transformer branch → `f_t(x)`; project both to a shared
  size and blend: `F(x) = α·z_c + (1−α)·z_t`, where `α∈[0,1]` balances the two.

### 2.4 Multimodal learning & cross-modal attention
- **Modality:** a *type* of data. Here: the **visual** modality (image) and the **environmental**
  modality (temp/humidity/light numbers). "Multimodal" = using more than one type.
- **Fusion:** how you combine modalities. Naïve = **concatenation** (just glue the vectors). Better =
  **cross-modal attention**, where one modality forms the Queries and the other forms Keys/Values, so
  the environment can *reweight* which visual features matter. (Math in §3.3.)
- **Why:** Environment carries information the pixels don't — humidity predicts certain infestations;
  temperature changes pest activity. Conditioning on it gives robustness to field variability.

### 2.5 The three model-compression techniques
- **Structured pruning:** delete whole redundant filters/attention-heads (not scattered individual
  weights), then fine-tune. Here it removes ~28–30% of parameters/FLOPs with tiny accuracy loss.
  "Structured" = removes chunks that map to real hardware savings.
- **Post-training quantization (INT8):** same as Paper 1 — FP32 → INT8, ~75% smaller, faster. (Math
  §3.4.)
- **Knowledge distillation:** train a small **student** network to imitate a big **teacher** network's
  *soft* outputs (its full probability distribution, not just the final label). The student learns the
  teacher's "dark knowledge" (e.g. "this is aphids, but 10% looks like thrips") and reaches near-
  teacher accuracy at a fraction of the size. (Math §3.6.)
- **FLOPs / MACs:** FLOPs = floating-point operations; MACs = multiply-accumulate operations (1 MAC ≈
  2 FLOPs). They measure compute cost. Fewer MACs → faster, lower energy.

### 2.6 Edge / Cloud / Dual-mode deployment
- **Edge (on-device) inference:** run the model *on the device in the field*. Pros: works offline,
  low latency, private. Cons: limited compute.
- **Cloud inference:** send data to a powerful server. Pros: big models. Cons: needs connectivity,
  higher latency/power, privacy concerns.
- **Dual-mode (this paper):** run inference at the edge always; *periodically* sync compact feature
  embeddings (not raw images, saving >80% bandwidth) to the cloud for retraining/updates. Best of
  both. Under intermittent networks, inference continuity stays **100%** (it never stops), and updates
  simply queue up ("backlog") and drain when connectivity returns.

### 2.7 The custom hardware vocabulary
- **Systolic array:** a grid of multiply-accumulate units that pump data through rhythmically (like a
  heart — "systolic") — the standard efficient way to do convolution/matrix-multiply in hardware.
- **Winograd optimization:** a math trick that computes small convolutions with fewer multiplications.
- **SRAM tiling / hierarchical buffers:** keep data in tiny fast on-chip memory (SRAM) instead of
  fetching from big slow off-chip memory (DRAM). Moving 1 bit from DRAM costs ~10 pJ; from SRAM
  ~0.05 pJ → the paper's **~200×** memory-energy reduction.
- **Dynamic power gating:** switch off the CNN/Transformer/fusion units when idle to save energy.
- **FPGA prototype (Xilinx ZCU104) + ASIC post-synthesis:** they *validate* on an FPGA (real
  measurements: 200 MHz, 25 ms, 0.12 J) and *estimate* a future custom ASIC chip (800 MHz, 12 ms,
  0.05 J). ASIC = a chip manufactured for exactly this design (fastest/most efficient, but expensive
  to fabricate).

### 2.8 Explainability (XAI): Grad-CAM vs attention
- **Grad-CAM:** a method to make a heatmap over the image showing *which pixels* drove a CNN's
  decision — for trust/debugging. Weakness: blurry for tiny lesions.
- **Attention rollout / token attribution:** the transformer analog — use the attention weights to
  highlight important regions, giving sharper maps. The paper combines both for reliable explanations.

---

## 3. Mathematical Foundations — step by step

### 3.1 The constrained learning objective (the paper's heart)

Ordinary training just minimizes a loss `L`. This paper minimizes loss **subject to hardware limits**:

$$\min_{\theta}\; \mathcal{L}\big(f_\theta(x,e),\, y\big) \quad \text{s.t.} \quad
\mathcal{C}_{\text{latency}}(\theta)\le \tau,\;\;
\mathcal{C}_{\text{memory}}(\theta)\le M,\;\;
\mathcal{C}_{\text{energy}}(\theta)\le E_{\max}.$$

Plain English: "Find model parameters `θ` that make the fewest mistakes, **but only among models that
run within 25 ms, fit in 110 MB, and use ≤ 0.12 J.**" The constraints `τ, M, E_max` are the
deployment budgets. This is what "deployment-aware" means, and it's the paper's core framing.

### 3.2 Scaled dot-product attention (self-attention)

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V$$

Walk through it with tiny numbers. Say we have 2 tokens, each turned into 2-D Q, K, V:
- `Q = [[1,0],[0,1]]`, `K = [[1,0],[0,1]]`, `V = [[10,0],[0,10]]`, and `d = 2`.
1. **Scores** `QKᵀ = [[1,0],[0,1]]` (each token's similarity to each other token).
2. **Scale** by `√d = √2 ≈ 1.414`: `[[0.707,0],[0,0.707]]`.
3. **Softmax each row**: row 1 `softmax([0.707,0]) ≈ [0.67,0.33]`; row 2 `≈ [0.33,0.67]`.
4. **Weighted sum of V**: token 1 output `= 0.67·[10,0]+0.33·[0,10] = [6.7,3.3]`; token 2 `=
   [3.3,6.7]`. Each token became a *mix* of all values, weighted by relevance. That's attention.

### 3.3 Cross-modal fusion (environment modulates vision)

Project image feature `F(x)` and environment `e` into a shared space, then attend with **environment
as the Query**:

$$z_v = W_v F(x),\quad z_e = W_e e,\qquad Q=W_Q z_e,\; K=W_K z_v,\; V=W_V z_v$$
$$F^{*}(x,e) = \sigma\!\Big(\text{softmax}\big(\tfrac{QK^{\top}}{\sqrt{d}}\big)V + z_v\Big)$$

Because `Q` comes from the **environment** and `K,V` come from the **image**, the weather decides
*which visual features to amplify*. The `+ z_v` is a **residual connection** (keep the original visual
info so nothing is lost); `σ` is a non-linearity. This is "environment-conditioned feature
modulation" — smarter than gluing vectors together.

### 3.4 INT8 quantization (same as Paper 1, restated)

$$w_{int8} = \text{round}\!\Big(\frac{w_{fp32}}{s}\Big),\qquad s = \frac{\max(w)-\min(w)}{2^{b}-1},\; b=8$$

~75% memory saved, negligible accuracy change. Chosen because lower precisions (INT4) hurt stability.

### 3.5 Why hybrids win: the bias–variance decomposition

For any model, expected squared error splits into three parts:

$$\mathbb{E}\big[(y-\hat y)^2\big] = \underbrace{\text{Bias}[\hat y]^2}_{\text{too-simple error}} +
\underbrace{\text{Var}[\hat y]}_{\text{too-sensitive error}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$

- **Bias:** error from wrong assumptions (model too simple). CNNs have useful bias (locality) but can
  miss global structure.
- **Variance:** error from over-sensitivity to the particular training data. Transformers are
  flexible → higher variance on small data.
The hybrid blends them: `F(x)=α·z_c+(1−α)·z_t`, and the paper shows the blended variance is bounded:

$$\text{Var}[F(x)] \le \alpha^2\,\text{Var}[z_c] + (1-\alpha)^2\,\text{Var}[z_t].$$

Because `α² + (1−α)² < 1` for `α∈(0,1)`, mixing **reduces variance** — the mathematical reason a
hybrid is more robust than either branch alone.

### 3.6 Knowledge distillation loss

$$\mathcal{L}_{KD} = \tau^{2}\, \mathrm{KL}\!\big(p^{T}_{\tau}(x)\,\|\,p^{S}_{\tau}(x)\big)$$

- `p^T_τ`, `p^S_τ` are the teacher's and student's **softened** probability outputs, softened by a
  **temperature** `τ` (divide logits by `τ` before softmax — larger `τ` = softer, more informative
  distribution).
- `KL(·‖·)` is **KL divergence**, a measure of how different two probability distributions are; the
  student minimizes it to match the teacher.
- The `τ²` rescales gradients. Result: a small student ≈ big teacher's accuracy.

### 3.7 Latency and energy models (for the hardware)

$$T = \frac{N_{\text{MAC}}}{f_{\text{clk}}\cdot P_{\text{par}}}, \qquad E = \alpha_e N_{\text{MAC}} + \beta_e M_{\text{mem}}$$

- **Latency `T`** = (number of MAC operations) ÷ (clock speed × parallelism). With `N_MAC ≈ 1.3
  billion`, `f_clk = 200 MHz`, `P_par = 256` → **≈ 25 ms**. This is *why* pruning (fewer MACs) and
  more parallel hardware make it faster.
- **Energy `E`** = compute cost (`α_e·N_MAC`) + memory-access cost (`β_e·M_mem`). SRAM tiling shrinks
  `M_mem` dramatically → the ~200× memory-energy win.

---

## 4. Code Implementation

A runnable hybrid CNN–Transformer with environment-conditioned cross-modal fusion is in
[`../code/paper3_cnn_transformer_fusion.py`](../code/paper3_cnn_transformer_fusion.py). Attention
built by hand is in
[`../code/shared_concepts/attention_from_scratch.py`](../code/shared_concepts/attention_from_scratch.py),
and pruning/quantization/distillation in
[`../code/shared_concepts/quantization_and_distillation.py`](../code/shared_concepts/quantization_and_distillation.py).

Conceptual skeleton of the fusion:

```python
import torch, torch.nn as nn

class CrossModalFusion(nn.Module):
    """Environment (e) queries the visual features (F) so weather can reweight what matters."""
    def __init__(self, d):
        super().__init__()
        self.Wv, self.We = nn.Linear(d, d), nn.Linear(3, d)     # 3 env vars -> d
        self.Wq, self.Wk, self.Wvv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.d = d
    def forward(self, F_vis, e):                                # F_vis:[B,d]  e:[B,3]
        z_v, z_e = self.Wv(F_vis), self.We(e)
        Q, K, V = self.Wq(z_e), self.Wk(z_v), self.Wvv(z_v)
        scores = (Q * K).sum(-1, keepdim=True) / (self.d ** 0.5)  # simple 1-token attention
        attn = torch.sigmoid(scores)                              # environment gating weight
        return torch.relu(attn * V + z_v)                         # residual fusion F*(x,e)
```

---

## 5. Results to quote

- **Accuracy:** 95.1% (proposed multimodal) vs 93.2% hybrid-image-only vs 89.6% CNN-only.
- **Robustness (the headline):** under *combined* field distortions, multimodal holds **85.5%** while
  the hybrid image-only drops to **77.9%** and CNN-only to **69.2%** — that's the "~15% error
  reduction" claim.
- **Efficiency:** custom hardware = **25 ms**, **0.12 J**, **40 FPS**, **60% less energy** than Jetson
  Nano; SRAM tiling gives ~200× memory-energy saving.
- **Ablations:** removing the environment modality drops accuracy 95.1→93.2% and IoU 87.5→84.1% —
  proving the environment genuinely helps.
- **Generalization:** on unseen datasets (IP102, PlantVillage) with no retraining, multimodal degrades
  least (79.1% / 82.0%) — more robust to domain shift than unimodal baselines.

---

## 6. Limitations & critique (for Q&A)

- **Single region (Darjeeling):** no Assam/Nilgiri validation — regional pests may differ.
- **Small dataset:** 1,520 real images (augmented to 7,600) — modest vs big benchmarks.
- **Hardware not fabricated:** results are FPGA-measured + ASIC-*estimated*; no real silicon or long-
  term field durability yet.
- **Sensor drift:** cheap temp/humidity sensors need recalibration; the model relies on them.
- **New pests need retraining:** no few-shot/continual adaptation yet.

**One-sentence novelty:** *"A deployment-aware system that conditions tea-pest recognition on live
weather via cross-modal attention and co-designs custom dual-mode edge hardware, delivering field-
robust accuracy at 25 ms / 0.12 J — 60% less energy than a Jetson Nano."*

Continue to `04_paper2_tea_physio_fl.md` (the most advanced — now you're ready for it).
