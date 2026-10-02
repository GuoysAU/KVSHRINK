"""FactoredCache holds only factors, and its bookkeeping matches a dense cache.

Checks, against a DynamicCache carrying the dense compressed tensors from the
same compression call:
  1. attention output agrees after compression (no tokens generated yet);
  2. attention output still agrees after appending generated tokens;
  3. reported sequence length matches;
  4. nothing dense-prefix-sized is stored.
"""

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.cache_utils import DynamicCache          # noqa: E402
from core.compressor import SVDCompressor                  # noqa: E402
from core.factored_cache import (                         # noqa: E402
    FactoredCache, get_active, use_factored_cache,
)
from core.factored_attention import factored_attention     # noqa: E402


def dense_attn(q, k, v, scaling):
    p = torch.softmax(((q @ k.transpose(-1, -2)) * scaling).float(), dim=-1).to(q.dtype)
    return p @ v


def main():
    torch.manual_seed(0)
    H, HQ, N, D, TAIL = 2, 8, 64, 16, 8
    key = torch.randn(1, H, N, D)
    value = torch.randn(1, H, N, D)

    def compress(factored):
        return SVDCompressor.compress_layer_shared_basis_eigh(
            key, value, [0.9] * H, preserve_last_n=TAIL, preserve_first_n=0,
            value_residual_bits=1, qjl_seed=7,
            return_factors=factored,
            eigh_compute_device="cpu",
        )

    # Dense oracle from the same inputs (switch off), factors from the same call
    # shape with the switch on: the factored path returns no dense tensors.
    dense_k, dense_v, _, _, _ = compress(False)
    ck, cv, _, _, factors = compress(True)

    dense = DynamicCache()
    dense.update(dense_k, dense_v, 0)
    fac = FactoredCache()
    fac.add_layer(0, factors)

    scaling = 1.0 / math.sqrt(D)
    g = HQ // H
    ok = True

    # 1. right after compression
    q = torch.randn(1, HQ, 1, D)
    ref = dense_attn(q, dense_k.repeat_interleave(g, 1), dense_v.repeat_interleave(g, 1), scaling)
    got = factored_attention(q, factors, scaling=scaling,
                             tail_k=fac.tail_k[0], tail_v=fac.tail_v[0])
    r1 = float((got - ref).norm() / ref.norm())
    ok &= r1 < 1e-4
    print(f"[{'PASS' if r1 < 1e-4 else 'FAIL'}] post-compression        rel={r1:.3e}")

    # 2. after three generated tokens are appended
    for _ in range(3):
        nk, nv = torch.randn(1, H, 1, D), torch.randn(1, H, 1, D)
        dk, dv = dense.update(nk, nv, 0)
        tk, tv = fac.update(nk, nv, 0)
        q = torch.randn(1, HQ, 1, D)
        ref = dense_attn(q, dk.repeat_interleave(g, 1), dv.repeat_interleave(g, 1), scaling)
        got = factored_attention(q, factors, scaling=scaling, tail_k=tk, tail_v=tv)
    r2 = float((got - ref).norm() / ref.norm())
    ok &= r2 < 1e-4
    print(f"[{'PASS' if r2 < 1e-4 else 'FAIL'}] after 3 generated toks  rel={r2:.3e}")

    # 3. length bookkeeping
    same_len = fac.get_seq_length(0) == dense.get_seq_length(0)
    ok &= same_len
    print(f"[{'PASS' if same_len else 'FAIL'}] seq_length               "
          f"{fac.get_seq_length(0)} vs {dense.get_seq_length(0)}")

    # 4. the compressor must not hand back any dense tensor at all: counting only
    #    the FactoredCache would miss dense tensors still held by the caller.
    no_dense_returned = ck is None and cv is None
    ok &= no_dense_returned
    print(f"[{'PASS' if no_dense_returned else 'FAIL'}] no dense returned        "
          f"{'None, None' if no_dense_returned else 'dense tensors leaked'}")

    # 5. the tail must own its storage, not view the original key/value tensor
    tail = fac.tail_k[0]
    owns = tail.untyped_storage().nbytes() == tail.numel() * tail.element_size()
    ok &= owns
    print(f"[{'PASS' if owns else 'FAIL'}] tail owns storage        "
          f"{tail.untyped_storage().nbytes()} B backing "
          f"{tail.numel() * tail.element_size()} B logical")

    # 6. what is actually held, against the uncompressed cache it replaces
    orig_bytes = 2 * key.numel() * key.element_size()
    fac_bytes = fac.stored_bytes()
    print(f"[INFO] held {fac_bytes} B vs uncompressed {orig_bytes} B "
          f"({orig_bytes / max(fac_bytes, 1):.2f}x)")

    print("\nall passed" if ok else "\nFAILED")
    return 0 if ok else 1


class FactoredCacheTest(unittest.TestCase):
    def test_existing_checks(self):
        self.assertEqual(main(), 0)

    def test_activation_scope_restores_nested_and_exception(self):
        outer = FactoredCache()
        inner = FactoredCache()
        self.assertIsNone(get_active())

        with use_factored_cache(outer):
            self.assertIs(get_active(), outer)
            with use_factored_cache(inner):
                self.assertIs(get_active(), inner)
            self.assertIs(get_active(), outer)

        self.assertIsNone(get_active())
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with use_factored_cache(outer):
                self.assertIs(get_active(), outer)
                raise RuntimeError("boom")
        self.assertIsNone(get_active())


if __name__ == "__main__":
    raise SystemExit(main())
