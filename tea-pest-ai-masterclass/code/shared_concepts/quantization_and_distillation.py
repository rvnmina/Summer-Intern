"""
quantization_and_distillation.py
================================
The two edge-compression tricks shared by Paper 1 and Paper 3:
  1. INT8 quantization  (FP32 weights -> 8-bit ints; ~4x smaller)
  2. Knowledge distillation (small "student" imitates a big "teacher")

Both are shown from scratch on tiny data so the math is transparent.

Run:  python quantization_and_distillation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# PART 1 -- INT8 QUANTIZATION (Paper 1 §3.5, Paper 3 §4.7)
#   w_int8 = round(w / s),   s = (max - min) / (2^b - 1),   b = 8
# ============================================================================
def quantize_int8(weights: torch.Tensor):
    """Quantize a float tensor to signed 8-bit integers, return ints + scale."""
    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / (2 ** 8 - 1)          # value each integer step represents
    zero_point = torch.round(-w_min / scale)         # maps real 0 to an integer
    q = torch.round(weights / scale + zero_point)
    q = q.clamp(0, 255).to(torch.uint8)              # 8-bit storage
    return q, scale, zero_point


def dequantize_int8(q, scale, zero_point):
    """Reconstruct the approximate float value from the int8 representation."""
    return (q.float() - zero_point) * scale


def demo_quantization():
    print("=" * 70)
    print("PART 1: INT8 QUANTIZATION")
    print("=" * 70)
    torch.manual_seed(0)
    w = torch.randn(8) * 0.3                          # some float weights
    q, scale, zp = quantize_int8(w)
    w_hat = dequantize_int8(q, scale, zp)
    print("original float weights :", w.round(decimals=3).tolist())
    print("int8 codes (0..255)    :", q.tolist())
    print("scale (per-step value) :", round(scale.item(), 5))
    print("reconstructed weights  :", w_hat.round(decimals=3).tolist())
    print("max abs error          :", (w - w_hat).abs().max().item())
    fp32_bytes, int8_bytes = w.numel() * 4, q.numel() * 1
    print(f"memory: FP32={fp32_bytes} bytes -> INT8={int8_bytes} bytes "
          f"({fp32_bytes / int8_bytes:.0f}x smaller)")

    # PyTorch's built-in one-liner for a whole model (dynamic quantization):
    net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 5))
    qnet = torch.quantization.quantize_dynamic(net, {nn.Linear}, dtype=torch.qint8)
    print("\nPyTorch dynamic-quantized model ready:", type(qnet).__name__)


# ============================================================================
# PART 2 -- KNOWLEDGE DISTILLATION (Paper 3 §3.6)
#   L_KD = tau^2 * KL( softmax(teacher/tau) || softmax(student/tau) )
# ============================================================================
class Teacher(nn.Module):          # bigger, more accurate
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 5))
    def forward(self, x): return self.net(x)


class Student(nn.Module):          # smaller, cheaper -- what we deploy
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 5))
    def forward(self, x): return self.net(x)


def distillation_loss(student_logits, teacher_logits, targets, tau=4.0, alpha=0.7):
    """Blend of (a) matching the teacher's soft outputs and (b) the true labels."""
    # (a) Soft loss: KL divergence between softened distributions.
    p_teacher = F.softmax(teacher_logits / tau, dim=1)
    log_p_student = F.log_softmax(student_logits / tau, dim=1)
    soft = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (tau ** 2)
    # (b) Hard loss: ordinary cross-entropy with the real labels.
    hard = F.cross_entropy(student_logits, targets)
    return alpha * soft + (1 - alpha) * hard


def demo_distillation():
    print("\n" + "=" * 70)
    print("PART 2: KNOWLEDGE DISTILLATION")
    print("=" * 70)
    torch.manual_seed(0)
    X = torch.randn(400, 20)
    y = (X[:, :5].sum(1) > 0).long() % 5              # a learnable synthetic rule

    teacher, student = Teacher(), Student()
    # 1) Train the teacher normally.
    optT = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    for _ in range(200):
        optT.zero_grad(); F.cross_entropy(teacher(X), y).backward(); optT.step()
    teacher.eval()
    t_acc = (teacher(X).argmax(1) == y).float().mean().item()

    # 2) Train the student to imitate the teacher (distillation).
    optS = torch.optim.Adam(student.parameters(), lr=1e-3)
    for _ in range(200):
        with torch.no_grad():
            t_logits = teacher(X)
        loss = distillation_loss(student(X), t_logits, y, tau=4.0, alpha=0.7)
        optS.zero_grad(); loss.backward(); optS.step()
    s_acc = (student(X).argmax(1) == y).float().mean().item()

    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    print(f"teacher accuracy: {t_acc:.3f}  ({n_teacher:,} params)")
    print(f"student accuracy: {s_acc:.3f}  ({n_student:,} params, "
          f"{n_teacher / n_student:.1f}x smaller)")
    print("The tiny student recovers most of the teacher's skill -> deployable at the edge.")


if __name__ == "__main__":
    demo_quantization()
    demo_distillation()
