# Phase 2 — Paper 1 Deep Dive: SoC-DPU

> **Full title:** High-speed system-on-chip-based platform for real-time crop disease and pest
> detection using deep learning techniques
> **Authors:** MD Tausif Mallick, D Omkar Murty, Ranita Pal, Swagata Mandal, Himadri Nath Saha,
> Amlan Chakrabarti · *Computers & Electrical Engineering 123 (2025) 110182*

---

## 0. The 60-second version (read this first)

Farmers of **mustard** and **mung bean** lose crops to diseases and pests. Spotting them by eye is
slow and subjective. This paper builds an automatic photo classifier: you take a phone photo of a
leaf, and a neural network tells you which of ~9–11 diseases/pests it is (or "healthy"). Two things
make it special:

1. **The brain (software):** a **MobileNetV3** CNN, trained with **transfer learning**, hitting
   **96.14%** accuracy on mung bean and **93.25%** on mustard.
2. **The body (hardware):** instead of running on a slow CPU, the trained network runs on a **Xilinx
   ZCU104 System-on-Chip** using a **Deep-Learning Processor Unit (DPU)**, giving a **24× speed-up**,
   **~29% higher throughput**, and **~19% lower power** than software alternatives.

A phone app captures the photo, cleans it up (de-noising), sends it to the chip, and shows the result.

---

## 1. Theory and Concept

### 1.1 The core problem

Crop diseases cause ~20% global crop loss and push farmers to over-use pesticides (polluting soil and
water). Traditional detection is either (a) a human expert eyeballing leaves — subjective, doesn't
scale; or (b) expensive spectroscopy equipment — bulky and costly. Because most infestations show
**visible** patterns on leaves, a **camera + machine learning** approach is attractive and cheap.

But there's a deployment catch. GPUs/TPUs (the usual hardware for deep learning) need a host computer,
aren't field-programmable, and are power-hungry. Real farms need something **fast, low-power, and
self-contained**.

### 1.2 The proposed solution (two halves)

**Half A — the model.** Rather than train a neural network from zero (which needs millions of images
they don't have), they take **MobileNetV3**, a network *already trained* on 1.2 million general
images (ImageNet), and **re-purpose** it for leaves. This is **transfer learning**. They add a small
custom "classification head" and **fine-tune** on their own datasets (5,507 mung-bean and 6,283
mustard augmented images they collected themselves in West Bengal, India).

**Half B — the hardware.** They deploy the trained model on an **FPGA-based System-on-Chip (SoC)**.
Think of the SoC as two chips glued together: a normal CPU part (the "Processing System", PS) and a
reconfigurable-logic part (the "Processing Logic", PL). Into the PL they load a **DPU** — a circuit
purpose-built to run neural networks extremely fast and efficiently. The DPU does the heavy math; the
CPU orchestrates.

### 1.3 The methodology, end to end

```
[Collect leaf photos] → [Clean + label + resize to 160×160] → [Augment: rotate/flip/zoom/Photoshop]
      → [Transfer-learn MobileNetV3: freeze base, train head, then fine-tune]
      → [Save trained model as .h5] → [Quantize to INT8 with Vitis AI] → [Compile to .xmodel]
      → [Load onto ZCU104 SoC's DPU] → [Phone app sends photo → SoC runs inference → returns label]
```

The clever framing: the **same hardware** can run *any* future crop model with **no hardware change**
— you just load a new `.xmodel`. That is the "reconfigurable" superpower of FPGAs.

---

## 2. Technical Vocabulary — from zero to advanced

Each term below: **what it is → how it works → why it's needed → when it's used**, with a tiny code
illustration where useful.

### 2.1 Machine Learning (ML)
- **What:** Programming a computer to learn patterns from *examples* (data) instead of you writing
  explicit rules.
- **How:** You show it many (input, correct-answer) pairs; it adjusts internal numbers ("parameters")
  to reduce its mistakes.
- **Why:** For messy tasks like "is this leaf diseased?" nobody can write down all the rules by hand.
- **When:** Any time you have data and want predictions.

### 2.2 Deep Learning & Neural Networks
- **What:** ML using **neural networks** — layers of simple math units ("neurons") stacked deep.
- **How:** Each neuron computes `output = activation(weights · inputs + bias)`. Stacking layers lets
  the network learn simple features (edges) → complex features (leaf spots) → decisions (disease X).
- **Why:** Deep nets automatically *discover* useful features from raw pixels; you don't hand-craft
  them.
- **When:** Images, audio, text — anything high-dimensional.

```python
# One artificial neuron, from scratch (no libraries)
def neuron(inputs, weights, bias):
    z = sum(i * w for i, w in zip(inputs, weights)) + bias   # weighted sum
    return max(0, z)   # ReLU activation: keep positives, zero out negatives

print(neuron([0.5, 0.2], [0.8, -0.4], 0.1))   # -> 0.42
```

### 2.3 Convolutional Neural Network (CNN)
- **What:** A neural network specialized for images.
- **How:** Instead of connecting every pixel to every neuron (impossibly many weights), a small
  **filter/kernel** (e.g. 3×3 numbers) slides across the image computing dot products, producing a
  **feature map** that highlights a pattern (an edge, a texture, a spot). Early layers catch edges;
  deeper layers combine them into "leaf lesion" detectors.
- **Why:** Massively fewer parameters + built-in assumption that nearby pixels are related
  (**spatial locality**). This is called an **inductive bias**.
- **When:** Almost all image tasks.

```python
import numpy as np
def convolve2d(image, kernel):
    kh, kw = kernel.shape
    H, W = image.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            patch = image[y:y+kh, x:x+kw]      # small window of the image
            out[y, x] = np.sum(patch * kernel) # dot product = "how much this pattern matches here"
    return out

img = np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]], float)
edge = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], float)  # an edge-detector kernel
print(convolve2d(img, edge))
```

Key CNN sub-terms you'll see in the paper:
- **Kernel / filter:** the small sliding weight grid.
- **Stride:** how many pixels the filter jumps each step (bigger stride → smaller output).
- **Padding:** adding a border of zeros so the output isn't shrunk.
- **Feature map:** the output image after a convolution.
- **Pooling (max/average):** shrinks a feature map by summarizing each little region (e.g. take the
  max). Reduces compute and adds robustness to small shifts.
- **ReLU:** `max(0, x)` — the standard **activation function** that adds non-linearity so the network
  can learn complex shapes.

### 2.4 MobileNetV3 (the specific model used)
- **What:** A **lightweight** CNN designed by Google (2019) to run on phones — high accuracy, few
  parameters (this paper's variant: **3.6 million** parameters vs InceptionV3's 23.1M and VGG19's
  20.5M).
- **How:** It uses three efficiency tricks:
  - **Depthwise separable convolution:** splits a normal convolution into two cheaper steps (filter
    each channel separately, then mix channels with 1×1 convolutions). Big compute savings.
  - **Inverted residual block:** expand channels → do a cheap depthwise conv → project back down,
    with a **skip/residual connection** that adds the input back to the output (helps gradients flow
    in deep nets).
  - **Squeeze-and-Excite (SE) block:** learns to weight each channel by importance — a cheap form of
    "attention over channels."
  - **h-swish activation:** a fast approximation of the smooth "swish" function, defined as
    `h-swish(x) = x · ReLU6(x+3) / 6` (see §3.4).
- **Why:** It gives near state-of-the-art accuracy at a fraction of the size — perfect for edge
  devices.
- **When:** Whenever you need good vision accuracy under tight compute/energy budgets.

### 2.5 Transfer Learning
- **What:** Reusing a model trained on one big task as the starting point for a new, smaller task.
- **How:** Take MobileNetV3 pre-trained on ImageNet (1.2M photos). Chop off its final classifier.
  Keep the **feature extractor** (it already knows edges, textures, shapes). Attach a *new* small
  classifier for your 9/11 leaf classes. Train.
- **Why:** You only have a few thousand leaf images — not enough to learn "what an edge is" from
  scratch. The pre-trained network already knows that.
- **When:** Small dataset + a related big pre-trained model exists (almost always, for images).

### 2.6 Fine-tuning (and the two-phase schedule this paper uses)
- **What:** After transfer learning, *gently* updating the reused layers too.
- **How (this paper's exact recipe):**
  - **Phase 1 (10 epochs):** *Freeze* the whole MobileNetV3 base (don't change its weights). Train
    *only* the new head. Learning rate `0.0002`, **Adam** optimizer, batch size 32. (Freezing avoids
    the random new head's large early errors from wrecking the good pre-trained weights.)
  - **Phase 2 (20 more epochs):** *Unfreeze* the base and train everything together, but at a **much
    smaller** learning rate `0.00002` with the **RMSProp** optimizer. Keep Batch-Norm layers frozen
    for stability. A 20% **Dropout** layer regularizes the head.
- **Why:** Two speeds — first adapt the head safely, then delicately tune the deep features.
- **When:** Standard practice whenever you transfer-learn.

### 2.7 Data Augmentation
- **What:** Artificially enlarging your dataset by transforming existing images.
- **How:** Rotate, flip, shift, zoom (via TensorFlow's `ImageDataGenerator`), plus brightness/
  contrast/color tweaks in Photoshop. 223 original mung-bean images → 5,507; 245 mustard → 6,283.
- **Why:** More variety → the model **generalizes** better and **overfits** less; also fixes **class
  imbalance** (some classes had far fewer photos).
- **When:** Almost always with small image datasets.

### 2.8 Overfitting, Generalization, Regularization
- **Overfitting:** the model memorizes training photos (including noise) instead of learning the real
  pattern — great on training data, bad on new data.
- **Generalization:** the opposite — performing well on *unseen* data (the real goal).
- **Regularization:** techniques that fight overfitting. Here: **Dropout** (randomly zero out 20% of
  neurons each step so the net can't over-rely on any one) and augmentation.

### 2.9 Softmax classification head
- **What:** The final layer that turns raw scores into probabilities over the classes.
- **How:** `softmax` exponentiates each score and normalizes so they sum to 1 (see §3.1). The class
  with the highest probability is the prediction.
- **Why:** Gives interpretable per-class confidence (the app shows "Aphids — 97.68%").
- **When:** Any multi-class classification.

### 2.10 Evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
- **Confusion matrix:** a grid: rows = true class, columns = predicted class. The diagonal = correct.
  Off-diagonal cells reveal *which* classes get confused (e.g. downy mildew mistaken for Alternaria
  blight). See §3.3 and the paper's Figs. 19–20.
- **Precision, Recall, F1:** see the math in §3.2. Precision = "when it says disease X, how often
  right?"; Recall = "of all real disease-X cases, how many did it catch?"; F1 = balance of the two.

### 2.11 The hardware vocabulary (the paper's unique contribution)
- **FPGA (Field-Programmable Gate Array):** a chip whose internal digital circuit you can
  *reconfigure* after manufacturing — like LEGO for logic gates. You describe a circuit; the FPGA
  becomes it.
- **SoC (System-on-Chip):** a single chip combining a CPU + other subsystems. The **Xilinx ZCU104**
  here has a **Processing System (PS)** = ARM CPUs, and **Processing Logic (PL)** = the FPGA fabric.
- **DPU (Deep-Learning Processor Unit):** a ready-made circuit design (an "IP core") from Xilinx that
  you drop into the PL to execute CNNs efficiently. It contains processing elements (each a
  multiply-accumulate unit), a scheduler, and on-chip memory.
- **AXI bus:** the high-speed "highway" connecting the CPU (PS) and the DPU (PL) so they can exchange
  data and instructions.
- **BRAM vs UltraRAM (URAM):** two kinds of on-chip memory. The paper compares them: URAM version
  uses less power/LUTs but more DSP; BRAM version the opposite. They chose URAM (lower power).
- **LUT (Look-Up Table), DSP slice, Slice Register:** the raw building blocks of an FPGA. LUTs
  implement logic; DSP slices do fast multiply/accumulate; registers store bits. The paper reports
  its design uses **31,901 LUTs, 47,797 registers, 44 URAM, 222 DSPs, 3.2 W** (URAM variant).
- **Vitis AI / Vivado / PetaLinux:** Xilinx's software toolchain — Vivado builds the hardware
  "bitstream," PetaLinux is the on-chip Linux OS, and Vitis AI quantizes and compiles the neural
  network into an `.xmodel` the DPU can run.
- **Quantization (INT8):** converting the model's 32-bit floating-point weights to 8-bit integers
  (see §3.5). Makes it ~4× smaller and much faster/cheaper to run, with tiny accuracy loss.
- **Throughput (GOPS):** Giga (billion) Operations Per Second — how much math the chip does per
  second. Their design hits **~140 GOPS**.
- **DPU configurations (B512…B4096):** preset DPU sizes trading speed for chip resources. They use
  **B1152** (peak 1152 operations/cycle).

---

## 3. Mathematical Foundations — explained like you're in high school

### 3.1 Softmax (turning scores into probabilities)

Suppose the network outputs three raw scores (called **logits**) for three classes: `z = [2.0, 1.0,
0.1]`. Softmax converts them to probabilities:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Step by step:
1. Exponentiate each: `e^2.0=7.39`, `e^1.0=2.72`, `e^0.1=1.11`.
2. Sum them: `7.39+2.72+1.11 = 11.22`.
3. Divide each by the sum: `[0.659, 0.242, 0.099]`. They now sum to 1 → probabilities.
Prediction = class 0 (highest, 65.9%). Exponentiation guarantees positivity and exaggerates the gap
between the top score and the rest.

### 3.2 Precision, Recall, F1 (the core metrics)

Define, for one class:
- **TP** (true positive): predicted diseased *and* actually diseased.
- **FP** (false positive): predicted diseased but actually healthy (false alarm).
- **FN** (false negative): predicted healthy but actually diseased (a miss).

$$\text{Precision} = \frac{TP}{TP+FP}, \qquad \text{Recall} = \frac{TP}{TP+FN}$$

$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision}+\text{Recall}}$$

**Worked example.** A model flags 10 leaves as "aphids." 8 truly are (TP=8), 2 aren't (FP=2). It also
missed 2 real aphid leaves (FN=2).
- Precision = 8/(8+2) = **0.80** (80% of alarms were correct).
- Recall = 8/(8+2) = **0.80** (caught 80% of real cases).
- F1 = 2·0.8·0.8 / (0.8+0.8) = **0.80**.

F1 is the **harmonic mean** — it only stays high if *both* precision and recall are high, which is
why it's preferred for imbalanced data. (In the paper, MobileNetV3 got F1 = 0.9590 on mung bean.)

### 3.3 Reading a confusion matrix

The paper's mung-bean matrix (Fig. 19) shows, e.g., aphids recognized only **86.95%** of the time,
often confused with **bruchids** (because black bean aphids look like bruchids from a distance). For
mustard, **downy mildew** scored just **34%** — too few samples + it looks like Alternaria/bacterial
blight. **Lesson to present:** high overall accuracy can hide specific weak classes; always inspect
the confusion matrix.

### 3.4 The h-swish activation

$$\text{h-swish}(x) = x \cdot \frac{\text{ReLU6}(x+3)}{6}, \qquad \text{ReLU6}(y)=\min(\max(0,y),6)$$

It approximates the smooth "swish" function but uses only cheap operations (add, clamp, multiply) —
important for phones/edge chips. Example: at `x=0`, h-swish = `0·(3/6)=0`; at `x=3`, `3·(6/6)=3`; at
`x=-3`, `-3·0=0`.

### 3.5 INT8 quantization math

To store a float weight `w` as an 8-bit integer:

$$w_{int8} = \text{round}\!\left(\frac{w}{s}\right), \qquad s = \frac{\max(w)-\min(w)}{2^{b}-1}, \quad b=8$$

`s` is the **scale** (how much real value each integer step represents). Example: weights range from
−0.5 to +0.5, so `s = 1.0/255 ≈ 0.00392`. A weight `w=0.1` becomes `round(0.1/0.00392)=round(25.5)=
26`. At inference you multiply back by `s` to approximate the original. Result: 4× smaller model,
faster integer math, ~negligible accuracy loss.

### 3.6 The 24× acceleration

$$\text{Acceleration} = \frac{\text{CPU inference time}}{\text{DPU inference time}}$$

Mung bean: `0.217 ms / 0.009 ms = 24.11×`. Mustard: `0.216 / 0.009 = 24×`. That's the paper's
headline hardware number.

---

## 4. Code Implementation

A complete, runnable transfer-learning + fine-tuning script mirroring this paper's recipe is in
[`../code/paper1_mobilenetv3_transfer_learning.py`](../code/paper1_mobilenetv3_transfer_learning.py).
It uses PyTorch's built-in MobileNetV3, freezes the base, trains a custom head, then fine-tunes — and
runs on synthetic data so you can execute it immediately. The INT8 quantization idea is demonstrated
in [`../code/shared_concepts/quantization_and_distillation.py`](../code/shared_concepts/quantization_and_distillation.py).

Here is the conceptual skeleton (the full file adds data loading, training loop, and metrics):

```python
import torch, torch.nn as nn
from torchvision import models

def build_model(num_classes=11, freeze_base=True):
    # 1) Load MobileNetV3 pre-trained on ImageNet = TRANSFER LEARNING
    net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    # 2) Optionally FREEZE the feature extractor (Phase 1 of fine-tuning)
    if freeze_base:
        for p in net.features.parameters():
            p.requires_grad = False
    # 3) Replace the classifier head for OUR classes, with 20% Dropout (regularization)
    in_feats = net.classifier[0].in_features
    net.classifier = nn.Sequential(
        nn.Linear(in_feats, 640), nn.Hardswish(),   # h-swish, as in MobileNetV3
        nn.Dropout(0.2),
        nn.Linear(640, num_classes),                # final layer -> logits; softmax applied in loss
    )
    return net
```

---

## 5. Results you should quote in your presentation

- **Accuracy:** 96.14% (mung bean), 93.25% (mustard) — beating InceptionV3 and VGG19, at **6× fewer
  parameters** (3.6M vs ~20–23M).
- **Speed:** DPU is **24×** faster than CPU (0.009 ms vs ~0.217 ms per inference).
- **Efficiency:** ~140 GOPS throughput; **3.2 W** (URAM) vs 4.8 W (BRAM); lower LUT/power than prior
  MobileNet-on-FPGA works (Table 16).
- **Statistical check:** a **Friedman test** (a non-parametric significance test) confirmed the model
  differences are real (p=0.034 mung bean, p=0.031 mustard, below the 0.05 threshold).

---

## 6. Limitations & how to critique it (great for Q&A)

- **Single region, single sensor:** data from West Bengal with one DSLR — real deployment across
  regions/cameras may shift performance (**domain shift**, a concept Papers 2 & 3 tackle head-on).
- **Weak classes:** downy mildew at 34% recognition — small-sample classes drag reliability; the
  headline accuracy hides this.
- **Test-set augmentation:** they augmented the *test* set too (due to few images). Purists prefer a
  purely real test set to avoid optimistic bias (Paper 3 explicitly fixes this).
- **Detection, not prevention:** it recognizes *visible* symptoms only — by then damage is done. This
  is exactly the gap **Paper 2** attacks with pre-symptomatic forecasting.

**One-sentence novelty (for your professor):** *"The first crop pest/disease classifier that runs
MobileNetV3 on an FPGA-SoC DPU for a 24× real-time speed-up, packaged with a farmer-facing phone
app, and reconfigurable to new crops without any hardware change."*

Continue to `03_paper3_edge_multimodal.md` (read second — it teaches transformers and multimodal
fusion, which Paper 2 assumes you already know).
