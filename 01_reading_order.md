# Phase 1 — Reading Order and Orientation

## The one-paragraph summary of all three papers

All three papers try to answer the same real-world question: **"How do we automatically detect pests
and diseases on crops, accurately, cheaply, and in the actual field — not just in a lab?"** They
differ in *how much* of that problem they bite off:

- **Paper 1 (SoC-DPU)** takes ordinary photos of mustard and mung-bean leaves and classifies the
  disease/pest using a lightweight neural network, and — the novel part — runs that network on a
  special reconfigurable chip (an FPGA System-on-Chip) so it is **24× faster** than a normal CPU.
- **Paper 2 (Tea-Physio-FL)** argues that by the time you can *see* damage in a photo, it is already
  too late. It fuses the photo with **plant-physiology signals** (chlorophyll level, the smells the
  plant emits, and weather) inside a **transformer**, and trains it across many tea estates without
  sharing raw data (**federated learning**) to *predict* pest risk *before* symptoms appear.
- **Paper 3 (Edge-Multimodal)** is the engineering bridge: it fuses photos with live weather sensors
  in a **hybrid CNN–Transformer**, then squeezes the whole system onto **custom low-power hardware**
  that can run on a solar panel in a remote plantation with 25 ms latency and 0.12 J per photo.

## Recommended reading order: **P1 → P3 → P2**

> Read them in order of *conceptual difficulty and prerequisites*, not publication date.

### 1st: Paper 1 (SoC-DPU) — the foundation

**Why first:** It uses the most classical, most-taught building blocks — a Convolutional Neural
Network (CNN), transfer learning, accuracy/precision/recall. If you understand this paper you will
understand ~60% of the vocabulary the other two papers assume you already know. Its only "hard" part
(the FPGA/DPU hardware) is self-contained and does not require the other papers.

**Prerequisites it introduces (that the others reuse):** what a CNN is, convolution, transfer
learning, fine-tuning, softmax classification, confusion matrix, precision/recall/F1, quantization
(INT8), and the idea of hardware acceleration.

### 2nd: Paper 3 (Edge-Multimodal) — the bridge

**Why second:** It reuses *everything* from Paper 1 (CNN, quantization, edge hardware, the same
metrics) and then adds exactly **two** genuinely new ideas: (a) the **transformer / self-attention**
mechanism, and (b) **multimodal fusion** (combining an image with non-image sensor data). It teaches
these two ideas gently and with clear math, so it is the perfect place to meet them for the first
time. It also shares Paper 2's application domain (tea) and its "combine images with environment"
philosophy, so it warms you up for Paper 2.

**New prerequisites it introduces:** self-attention, multi-head attention, Vision Transformer (ViT),
cross-modal attention, knowledge distillation, structured pruning, the bias–variance tradeoff, and
domain shift.

### 3rd: Paper 2 (Tea-Physio-FL) — the most advanced

**Why last:** It assumes you are already fluent in transformers and cross-modal attention (from
Paper 3) and then stacks the two hardest new concepts on top: (a) **federated learning** (training
one model across many sites without centralizing data, under "non-IID" conditions), and (b) a
**latent-variable, physiology-constrained** formulation (modeling an *unobservable* biological state
`Z` that you never measure directly). It is also the most abstract paper — heavy on
conceptual/mathematical framing and lighter on concrete implementation detail — so you want maximum
background before you tackle it.

**New prerequisites it introduces:** latent variables, federated averaging (FedAvg), non-IID data,
physiology-guided attention constraints, and stochastic temporal modeling.

## A dependency map (what you must understand before what)

```
                 ┌─────────────────────────────────────────────┐
                 │  CNN · convolution · transfer learning ·    │
   PAPER 1  ───▶ │  softmax · precision/recall/F1 · INT8 quant │
   (easiest)     │  · hardware acceleration (FPGA/DPU)         │
                 └───────────────────┬─────────────────────────┘
                                     │ (reused wholesale)
                                     ▼
                 ┌─────────────────────────────────────────────┐
   PAPER 3  ───▶ │  + self-attention · multi-head attention ·  │
   (bridge)      │    ViT · cross-modal fusion · distillation ·│
                 │    pruning · domain shift · bias–variance   │
                 └───────────────────┬─────────────────────────┘
                                     │ (reused + generalized)
                                     ▼
                 ┌─────────────────────────────────────────────┐
   PAPER 2  ───▶ │  + latent biophysical state Z · physiology- │
   (hardest)     │    constrained attention · federated        │
                 │    learning (FedAvg) · non-IID · temporal   │
                 └─────────────────────────────────────────────┘
```

## How to read each paper (a 4-pass technique for beginners)

1. **Pass 1 — Map (10 min):** Read only the title, abstract, all figure/table captions, section
   headings, and the conclusion. Goal: know *what problem* and *what result*, nothing else.
2. **Pass 2 — Story (45 min):** Read Introduction + Related Work + the first paragraph of each method
   subsection. Goal: understand *why* they did it and *how it differs* from prior work. Skip every
   equation on this pass.
3. **Pass 3 — Machinery (2–3 hrs):** Now read the Method and Results sections slowly, and every time
   you hit a term you don't know, look it up in the corresponding doc here. Work through the math
   with pen and paper.
4. **Pass 4 — Teach (1 hr):** Close the paper and explain it out loud (or write it) as if teaching a
   friend. Whatever you *can't* explain is exactly what you must reread. This is the single best
   preparation for presenting to your professor.

## What your professor will most likely ask (prepare these)

- "In one sentence, what is the *novel* contribution of each paper?" (Every paper explicitly states
  it — see the summary boxes in each doc.)
- "Why not just use a bigger, more accurate model?" (Answer: field deployment — energy, latency,
  cost, connectivity. This is the thread linking all three papers.)
- "What is the difference between *detection* and *forecasting*?" (P1/P3 detect visible symptoms;
  P2 forecasts risk before symptoms — this is P2's whole point.)
- "What are the limitations?" (Each doc has a dedicated *Limitations & how to critique it* section.)

Move on to `02_paper1_soc_dpu.md`.
