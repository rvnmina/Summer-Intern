# Summer-Intern
# Crop & Tea-Plant Pest AI — A Complete Masterclass

A beginner-to-advanced study companion for three research papers on AI-driven crop and tea-plant
pest/disease detection, written so that someone with **no prior background** can present all three
papers with full confidence.

All three papers share a research group (MD Tausif Mallick, Himadri Nath Saha, Amlan Chakrabarti and
collaborators) and form a natural progression from **classical deep learning on custom hardware** →
**physiology-aware multimodal transformers with federated learning** → **energy-efficient edge
hardware co-design**.

---

## The three papers

| # | Short name | Full title | Venue / Year |
|---|------------|------------|--------------|
| **P1** | **SoC-DPU** | High-speed system-on-chip-based platform for real-time crop disease and pest detection using deep learning techniques | Computers & Electrical Engineering, 2025 |
| **P2** | **Tea-Physio-FL** | Physiology-Aware Chlorophyll-Guided Multimodal Transformer With Federated Learning for Early Pest-Risk Forecasting in Tea Plantations | IEEE Access, 2026 |
| **P3** | **Edge-Multimodal** | Transformative Energy-Efficient Edge-Optimised Multimodal Deep Learning Framework for Pest Management and Severity Analysis in Tea Plants | ACM Trans. Embedded Computing Systems, 2026 |

---

## How to use this repository

Read the docs **in this order** (this is also the recommended reading order for the papers — see
`docs/01_reading_order.md` for the reasoning):

1. `docs/01_reading_order.md` — Phase 1: which paper to read first and why.
2. `docs/02_paper1_soc_dpu.md` — Phase 2: full breakdown of Paper 1 (read **first**).
3. `docs/03_paper3_edge_multimodal.md` — Phase 2: full breakdown of Paper 3 (read **second**; teaches transformers + fusion).
4. `docs/04_paper2_tea_physio_fl.md` — Phase 2: full breakdown of Paper 2 (read **third**; most advanced).
5. `docs/05_synthesis.md` — Phase 3: how the three papers connect (+ full glossary).
6. `docs/06_future_work_and_skills.md` — Phase 4: new ideas + skills to learn (with theory & code).
7. `docs/07_github_guide.md` — Phase 5: how to structure and publish this on GitHub.

> **Note on numbering:** docs are numbered in **reading order**, so `03_` covers Paper *3* and `04_`
> covers Paper *2*. This is deliberate — Paper 3 is easier and teaches the transformer/attention
> machinery that Paper 2 depends on.

Every technical term is explained from scratch the first time it appears, and a consolidated
**glossary** lives at the bottom of `docs/05_synthesis.md`.

## Runnable code

The `code/` folder contains **runnable, heavily commented PyTorch** reference implementations of the
core ideas. Each file runs standalone on tiny synthetic data so you can see the concept work without
downloading any dataset.

| File | Demonstrates | Paper |
|------|--------------|-------|
| `code/paper1_mobilenetv3_transfer_learning.py` | Transfer learning + fine-tuning a MobileNetV3 classifier | P1 |
| `code/paper2_chlorophyll_multimodal_transformer.py` | Physiology-constrained cross-modal attention + latent risk head | P2 |
| `code/paper3_cnn_transformer_fusion.py` | Hybrid CNN–Transformer backbone + environment-conditioned fusion | P3 |
| `code/shared_concepts/attention_from_scratch.py` | Scaled dot-product & multi-head attention, built by hand | P2, P3 |
| `code/shared_concepts/quantization_and_distillation.py` | INT8 quantization + knowledge distillation | P1, P3 |
| `code/shared_concepts/federated_learning_demo.py` | FedAvg across simulated "plantations" | P2 |

### Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt
python code/shared_concepts/attention_from_scratch.py
```

> These are **teaching implementations**: correct and runnable, but small. They are meant to make the
> math tangible, not to reproduce the papers' exact accuracy numbers.

### Verification status

All six Python files are **syntax-verified** (`python -m py_compile`), and the core math (INT8
quantization, FedAvg weighted averaging, scaled-dot-product attention) has been **numerically
checked in NumPy** against the worked examples in the docs. The full PyTorch models require `torch`
installed via `pip install -r code/requirements.txt`; `paper1_...py` downloads pretrained MobileNetV3
weights on first run (needs internet once). Run any file directly, e.g.:

```bash
python code/shared_concepts/quantization_and_distillation.py
python code/paper3_cnn_transformer_fusion.py
```
