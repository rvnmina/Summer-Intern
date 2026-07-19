# Phase 3 — Synthesis: How the Three Papers Connect

## 1. The overarching theme

All three papers are chapters of one research program: **"Make crop/tea pest detection work in the
real world — accurate, cheap, low-power, and deployable in the actual field."** They come from the
same core group (Mallick, Saha, Chakrabarti recur on all three). Read together, they trace a clear
intellectual arc:

> **From recognizing visible damage on custom hardware (P1) → to fusing images with the environment
> and squeezing it onto energy-efficient edge chips (P3) → to forecasting invisible, pre-symptomatic
> risk across many farms with privacy-preserving federated learning (P2).**

Every paper is driven by the *same frustration with lab-only AI*: models that ace clean benchmarks
but fail under field lighting, humidity, occlusion, poor connectivity, and tight power budgets.

## 2. The three axes of progression

Think of the trilogy moving along three axes simultaneously:

| Axis | Paper 1 (SoC-DPU) | Paper 3 (Edge-Multimodal) | Paper 2 (Tea-Physio-FL) |
|------|-------------------|---------------------------|--------------------------|
| **What it predicts** | Visible disease/pest class | Visible pest class (robustly) | *Future* pest **risk** (pre-symptom) |
| **Model** | CNN (MobileNetV3) | Hybrid CNN–Transformer | Physiology-constrained multimodal Transformer |
| **Modalities** | Image only | Image + microclimate (2) | Image + chlorophyll + VOC + climate (4) |
| **Nature** | Reactive | Reactive, environment-aware | **Predictive/anticipatory** |
| **Hardware** | FPGA-SoC + DPU | Custom dual-mode edge chip | Edge nodes (ARM) |
| **Training scope** | Single central model | Single central model + cloud sync | **Federated** across many estates |
| **Deployment concern** | Speed (24×) | Energy/latency (25 ms, 0.12 J) | Privacy + non-IID scalability |
| **Maturity of evidence** | Real hardware, real images | FPGA-measured + real images | Simulated hardware + **synthetic** physiology |

## 3. How they build on each other (the concept ladder)

- **P1 lays the foundation:** the CNN, transfer learning, the same evaluation metrics
  (accuracy/precision/recall/F1/confusion matrix), INT8 quantization, and the idea that *hardware
  matters* (DPU, FPGA-SoC). Every later paper silently assumes all of this.
- **P3 adds two rungs:** the **transformer/self-attention** and **multimodal fusion**. It keeps P1's
  quantization and edge-hardware philosophy but generalizes "just an image" into "image + sensors,"
  and adds distillation + pruning to the compression toolkit. It also introduces the deployment-*aware*
  learning objective (constraints inside the loss).
- **P2 adds the final, hardest rungs:** it takes P3's multimodal transformer and (a) generalizes 2
  modalities to 4, (b) makes the attention **physiology-constrained**, (c) reframes the task from
  *detection* to *latent-state forecasting*, and (d) distributes training via **federated learning**.

You literally cannot understand P2's "physiology-constrained cross-modal attention" without first
understanding P3's "cross-modal attention," which itself needs P1's CNN foundation. That's why the
reading order is P1 → P3 → P2.

## 4. Shared DNA (motifs that recur)

1. **Attention as the workhorse.** P1 hints at it (Squeeze-and-Excite = channel attention); P3 makes
   it central (self- and cross-modal attention); P2 constrains it with biology. Attention is the
   through-line.
2. **"Deployment is a first-class citizen."** Not an afterthought — P1 (24× on a chip), P3
   (constraints *in* the objective + custom silicon), P2 (federated + edge-latency budget).
3. **Lightweight-by-design.** MobileNetV3 (P1) → pruned/quantized/distilled hybrid (P3) → 58 MB
   federated model (P2). All obsess over fitting real edge devices.
4. **The field-robustness problem = domain shift.** P1 collects multi-condition data; P3 *conditions*
   on the environment to fight it; P2 forecasts under non-IID cross-estate shift. Same enemy, escalating
   sophistication.
5. **Same team, same crops-and-agriculture mission**, moving from mustard/mung bean (P1) to tea (P3,
   P2).

## 5. Combined significance — the story to tell your professor

> "These three papers together propose a **full stack for sustainable, real-world agricultural pest
> AI**. Paper 1 proves you can run an accurate classifier on cheap, fast, reconfigurable field
> hardware. Paper 3 shows that fusing the image with the *environment* makes it robust to messy field
> conditions, and co-designs the silicon to run it on solar power. Paper 2 pushes the frontier from
> *detecting* damage to *forecasting* it before it happens, using plant physiology and federated
> learning so many farms can collaborate without surrendering their data. The arc is **reactive →
> robust → predictive**, and the unifying obsession is **deployability**: accuracy that survives
> contact with a real plantation."

A good critical closing note: the evidence base *weakens* as ambition *grows* — P1 has real hardware
and real images; P3 has FPGA measurements + estimated ASIC; P2 relies on synthetic physiology and
simulated federation. So the trilogy is best framed as **"a validated foundation (P1) supporting an
increasingly ambitious but progressively more speculative vision (P3, then P2)."** Saying this shows
mature judgment.

---

## 6. Consolidated Glossary (every term, one place)

**Activation function** — a non-linear function (ReLU, h-swish) applied to a neuron's output so the
network can learn complex, non-straight-line patterns.

**ASIC** — Application-Specific Integrated Circuit; a chip manufactured for one design. Fastest/most
efficient but expensive to fabricate. (P3 estimates one.)

**Attention (self-)** — mechanism letting each input element weight the importance of every other
element via Query·Key similarity, then take a weighted sum of Values.

**AUC** — Area Under the ROC Curve; threshold-independent measure of class separation (0.5 random, 1.0
perfect).

**AXI bus** — high-speed on-chip highway connecting the CPU (PS) and accelerator (PL) on a Xilinx SoC.

**Backbone** — the main feature-extracting body of a network (before the task-specific head).

**Bias–variance tradeoff** — total error = bias² (too-simple) + variance (too-sensitive) + irreducible
noise; hybrids reduce variance.

**BRAM / UltraRAM (URAM)** — two kinds of on-chip FPGA memory with different power/resource tradeoffs.

**Class imbalance** — some classes have far more samples than others; fixed via augmentation.

**CNN** — Convolutional Neural Network; image model using sliding filters to detect local patterns.

**Confusion matrix** — grid of true vs predicted classes; diagonal = correct, off-diagonal = specific
confusions.

**Convolution / kernel / filter / stride / padding** — the sliding-window dot-product operation and
its knobs (see Paper 1 doc §2.3).

**Cross-modal attention** — attention where Queries come from one modality and Keys/Values from
another, letting one modality modulate the other.

**Depthwise separable convolution** — a cheaper convolution factorized into per-channel filtering +
1×1 channel mixing (MobileNet efficiency trick).

**Distillation (knowledge)** — training a small "student" model to imitate a big "teacher's" soft
outputs, transferring accuracy cheaply.

**Domain shift** — train and deployment data differ statistically, hurting accuracy; the field-
robustness enemy.

**DPU** — Deep-learning Processor Unit; Xilinx IP core that runs CNNs efficiently on an FPGA.

**Dropout** — regularization that randomly zeroes a fraction of neurons during training to prevent
overfitting.

**Edge inference** — running the model on the field device itself (offline, low-latency, private).

**Epoch** — one full pass over the training data.

**Federated learning (FedAvg)** — training one model across many sites by averaging their locally-
trained weights, without centralizing raw data.

**Feature map** — the output image produced by applying a convolution filter.

**Fine-tuning** — gently updating a pre-trained model on a new task (often after transfer learning).

**FLOPs / MACs** — floating-point operations / multiply-accumulate operations; measures of compute
cost (1 MAC ≈ 2 FLOPs).

**FPGA** — Field-Programmable Gate Array; a chip whose logic circuit is reconfigurable after
manufacture.

**F1-score** — harmonic mean of precision and recall; robust single number for imbalanced data.

**Grad-CAM** — heatmap explaining which pixels drove a CNN's decision.

**h-swish** — fast activation `x·ReLU6(x+3)/6` used in MobileNetV3.

**Inductive bias** — built-in assumptions of a model (CNNs assume locality; helps learn from less
data).

**IoU (Intersection over Union)** — overlap between predicted and true regions; localization metric.

**Latent variable (`Z`)** — an unobserved quantity inferred from observable clues; here the plant's
hidden biophysical state.

**Logits** — raw pre-softmax scores.

**Lipschitz-continuous** — a function whose output changes are bounded relative to input changes;
implies stability.

**LUT / DSP slice / register** — raw FPGA building blocks (logic / fast multiply / bit storage).

**MobileNetV3** — efficient lightweight CNN (inverted residuals + SE blocks + h-swish).

**Modality** — a type of data (image, chlorophyll, VOC, climate). Multimodal = several types.

**Non-IID** — data across sites is *not* identically distributed; the central challenge of federated
learning.

**Overfitting / generalization / regularization** — memorizing vs performing on unseen data vs
techniques to fight memorization.

**Positional encoding** — position tags added to transformer tokens so order/location is known.

**Precision / recall** — correctness of positive predictions / fraction of real positives caught.

**Pruning (structured)** — removing whole redundant filters/heads to shrink the model.

**PS / PL** — Processing System (CPU) / Processing Logic (FPGA fabric) halves of a Xilinx SoC.

**Quantization (INT8)** — converting FP32 weights to 8-bit integers; ~4× smaller, faster.

**Residual/skip connection** — adding a layer's input to its output to ease training of deep nets.

**Softmax** — converts scores into a probability distribution summing to 1.

**SoC** — System-on-Chip; CPU + accelerators + memory on one chip.

**SRAM tiling** — keeping data in tiny fast on-chip memory to avoid costly off-chip (DRAM) access.

**Squeeze-and-Excite (SE)** — channel-attention block that reweights feature channels by importance.

**Systolic array** — rhythmic grid of MAC units for efficient matrix/convolution hardware.

**Temperature (τ)** — softmax-softening knob used in distillation.

**Transfer learning** — reusing a model pre-trained on a big task as a starting point for a new task.

**Transformer** — architecture built on self-attention; models global/long-range relationships.

**Vision Transformer (ViT)** — a transformer applied to images by splitting them into patch tokens.

**VOC (Volatile Organic Compounds)** — airborne chemicals plants emit; insects use them to pick hosts;
shift before visible stress.

**Winograd** — algorithm reducing the multiplications needed for small convolutions.

Continue to `06_future_work_and_skills.md`.
