"""The factored path must match attention over the dense compressed cache.

The compressor returns both the dense compressed tensors and the factors from
the same call, so that dense result is an exact oracle: any divergence is a bug
in the factored operator, not a property of compression.
"""

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.compressor import SVDCompressor          # noqa: E402
from core.factored_attention import factored_attention  # noqa: E402


def dense_attention(q, k, v, scaling, mask=None):
    logits = (q @ k.transpose(-1, -2)) * scaling
    if mask is not None:
        logits = logits + mask
    p = torch.softmax(logits.float(), dim=-1).to(q.dtype)
    return p @ v


def run_case(n_heads, n_q_heads, seq_len, head_dim, preserve_last_n,
             residual_bits, tau=0.9, seed=0, dtype=torch.float32, key_scale=1.0):
    torch.manual_seed(seed)
    # key_scale emulates the massive activations of deep layers: the factored
    # intermediate q (Sigma_r Psi_r^T)^T is ~sigma_1 |q|, which overflows fp16
    # long before any final logit does.
    key = key_scale * torch.randn(1, n_heads, seq_len, head_dim, dtype=dtype)
    value = torch.randn(1, n_heads, seq_len, head_dim, dtype=dtype)

    def compress(factored):
        return SVDCompressor.compress_layer_shared_basis_eigh(
            key, value, [tau] * n_heads,
            preserve_last_n=preserve_last_n,
            preserve_first_n=0,
            value_residual_bits=residual_bits,
            qjl_seed=7,
            return_factors=factored,
            eigh_compute_device="cpu",
        )

    # Dense oracle with the switch off; the factored path returns no dense tensors.
    ck, cv, _, _, _ = compress(False)
    _, _, _, _, factors = compress(True)
    assert factors is not None, "return_factors=True should produce factors"

    q = torch.randn(1, n_q_heads, 3, head_dim, dtype=dtype)
    scaling = 1.0 / math.sqrt(head_dim)

    g = n_q_heads // n_heads
    k_rep = ck.repeat_interleave(g, dim=1)
    v_rep = cv.repeat_interleave(g, dim=1)
    ref = dense_attention(q, k_rep, v_rep, scaling)

    got = factored_attention(q, factors, mask=None, scaling=scaling)

    rel = (got - ref).norm() / ref.norm().clamp_min(1e-12)
    return float(rel)


def main():
    cases = [
        ("MHA, no tail, no sketch", dict(n_heads=4, n_q_heads=4, seq_len=64,
                                         head_dim=16, preserve_last_n=0,
                                         residual_bits=0)),
        ("MHA, tail=8, no sketch", dict(n_heads=4, n_q_heads=4, seq_len=64,
                                        head_dim=16, preserve_last_n=8,
                                        residual_bits=0)),
        ("MHA, tail=8, 1-bit sketch", dict(n_heads=4, n_q_heads=4, seq_len=64,
                                           head_dim=16, preserve_last_n=8,
                                           residual_bits=1)),
        ("GQA 8q/2kv, tail=8, sketch", dict(n_heads=2, n_q_heads=8, seq_len=64,
                                            head_dim=16, preserve_last_n=8,
                                            residual_bits=1)),
        ("FP16, tail=8, sketch", dict(n_heads=4, n_q_heads=4, seq_len=64,
                                      head_dim=16, preserve_last_n=8,
                                      residual_bits=1, dtype=torch.float16)),
        ("FP16 massive activation", dict(n_heads=4, n_q_heads=4, seq_len=64,
                                         head_dim=16, preserve_last_n=8,
                                         residual_bits=1, dtype=torch.float16,
                                         key_scale=400.0)),
    ]
    tol = 1e-2
    failures = 0
    for name, kw in cases:
        rel = run_case(**kw)
        ok = rel < tol
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name:32s} rel={rel:.3e}")
    print("\nall passed" if failures == 0 else f"\n{failures} case(s) failed")
    return 1 if failures else 0


class FactoredAttentionEquivalenceTest(unittest.TestCase):
    def test_existing_cases(self):
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    raise SystemExit(main())
