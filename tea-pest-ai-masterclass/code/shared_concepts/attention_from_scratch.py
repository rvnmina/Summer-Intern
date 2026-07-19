"""
attention_from_scratch.py
==========================
Self-attention and multi-head attention, built BY HAND so you can see exactly
what the transformer math in Paper 3 (Eq. 5/7) and Paper 2 (Eq. for A) does.

Run:  python attention_from_scratch.py

Nothing to download. It prints small tensors you can verify with a calculator,
then shows a physiology-constrained variant (Paper 2's signature `⊙ P` trick).

Concepts demonstrated
---------------------
1. Scaled dot-product attention:  softmax(Q Kᵀ / √d) V
2. Multi-head attention:          several attentions in parallel, concatenated
3. Physiology-constrained attention: softmax((Q Kᵀ / √d) ⊙ P) V   (Paper 2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1) Scaled dot-product attention -- the single most important transformer op.
# ----------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V : tensors of shape [..., seq_len, d]
    Returns : attended values [..., seq_len, d]  and the attention weights.

    Intuition:
      - Q ("queries")  = what each token is looking for.
      - K ("keys")     = what each token advertises about itself.
      - V ("values")   = what each token actually passes on.
      Q · Kᵀ measures how well every query matches every key (similarity).
      We divide by √d so the numbers stay in a sane range for softmax.
      softmax turns similarities into weights that sum to 1.
      Then we take a weighted sum of the values.
    """
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)   # [..., seq, seq]
    weights = F.softmax(scores, dim=-1)               # each row sums to 1
    output = weights @ V                              # weighted sum of values
    return output, weights


# ----------------------------------------------------------------------------
# 2) Multi-head attention -- run several attentions in parallel.
#    Each "head" can learn a different kind of relationship (color, shape, ...).
# ----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide evenly among heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        # One big linear layer produces Q, K, V for all heads at once.
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)  # final mix after concatenation

    def _split_heads(self, x):
        # [B, seq, d_model] -> [B, num_heads, seq, d_head]
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_head).transpose(1, 2)

    def forward(self, x):
        Q = self._split_heads(self.Wq(x))
        K = self._split_heads(self.Wk(x))
        V = self._split_heads(self.Wv(x))
        out, weights = scaled_dot_product_attention(Q, K, V)  # per-head attention
        # Concatenate heads back: [B, heads, seq, d_head] -> [B, seq, d_model]
        B, H, S, Dh = out.shape
        out = out.transpose(1, 2).contiguous().view(B, S, H * Dh)
        return self.Wo(out), weights


# ----------------------------------------------------------------------------
# 3) Physiology-constrained attention -- Paper 2's key mechanism.
#    A biology-derived gate P element-wise multiplies the raw scores, so the
#    chemistry can suppress/boost interactions BEFORE softmax.
# ----------------------------------------------------------------------------
class PhysiologyConstrainedAttention(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        # Build the constraint gate P from chlorophyll + VOC embeddings.
        self.to_gate = nn.Linear(2 * d_model, d_model)
        self.d = d_model

    def forward(self, tokens, e_chl, e_voc):
        # tokens: [B, seq, d];  e_chl, e_voc: [B, d]
        Q, K, V = self.Wq(tokens), self.Wk(tokens), self.Wv(tokens)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d)      # [B, seq, seq]
        gate = torch.sigmoid(self.to_gate(torch.cat([e_chl, e_voc], dim=-1)))  # [B, d] in (0,1)
        # Turn the per-feature gate into a per-key scalar and apply it (⊙ P).
        p_key = gate.mean(dim=-1, keepdim=True).unsqueeze(1)       # [B, 1, 1] broadcast
        scores = scores * p_key                                    # chemistry reshapes attention
        weights = F.softmax(scores, dim=-1)
        return weights @ V, weights


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
def demo():
    torch.manual_seed(0)
    print("=" * 70)
    print("1) SCALED DOT-PRODUCT ATTENTION (hand-checkable example)")
    print("=" * 70)
    # Two tokens, 2-D Q/K/V, matching the worked example in docs/03 §3.2.
    Q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    K = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    V = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
    out, w = scaled_dot_product_attention(Q, K, V)
    print("attention weights (each row sums to 1):\n", w.squeeze(0))
    print("attended output (mix of values):\n", out.squeeze(0))

    print("\n" + "=" * 70)
    print("2) MULTI-HEAD ATTENTION on a fake sequence of image patch-tokens")
    print("=" * 70)
    x = torch.randn(1, 5, 32)          # batch=1, 5 tokens, d_model=32
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    y, w = mha(x)
    print("input  shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape), "(same shape, contextually enriched)")
    print("per-head attention weights shape:", tuple(w.shape), "= [batch, heads, seq, seq]")

    print("\n" + "=" * 70)
    print("3) PHYSIOLOGY-CONSTRAINED ATTENTION (Paper 2): chemistry gates attention")
    print("=" * 70)
    tokens = torch.randn(1, 5, 32)
    e_chl = torch.randn(1, 32)         # chlorophyll embedding
    e_voc = torch.randn(1, 32)         # VOC embedding
    pca = PhysiologyConstrainedAttention(d_model=32)
    out_healthy, w_healthy = pca(tokens, e_chl * 0.1, e_voc * 0.1)   # weak stress signal
    out_stress, w_stress = pca(tokens, e_chl * 3.0, e_voc * 3.0)     # strong stress signal
    print("Same image, DIFFERENT physiology -> different attention sharpness.")
    print("healthy-signal attention entropy:", _entropy(w_healthy).item())
    print("stressed-signal attention entropy:", _entropy(w_stress).item())
    print("(Lower entropy = more focused attention; the biology changed the focus.)")


def _entropy(weights):
    """Average Shannon entropy of the attention distributions (a focus measure)."""
    p = weights.clamp_min(1e-9)
    return (-(p * p.log()).sum(-1)).mean()


if __name__ == "__main__":
    demo()
