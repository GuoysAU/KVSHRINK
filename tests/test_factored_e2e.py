"""End-to-end check on a tiny real model: factored path vs dense path.

Mirrors experiment.py's scoring path (prefill -> compress -> score an option with
a cache) on a small randomly initialised Mistral, so the whole HuggingFace
plumbing is exercised: mask construction, cache protocol, attention dispatch,
GQA. Any divergence here is a bug in the factored path.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import MistralConfig, MistralForCausalLM, DynamicCache  # noqa: E402
from strategies.uniform import UniformStrategy                            # noqa: E402
from core.factored_cache import FactoredCache, register, use_factored_cache  # noqa: E402


def build_model(seed=0):
    torch.manual_seed(seed)
    cfg = MistralConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
        max_position_embeddings=256, sliding_window=None,
    )
    model = MistralForCausalLM(cfg).eval()
    return model


def compress_all(strategy, cache):
    """Compress every layer with the strategy's explicit cache backend.

    The factored path returns no dense tensors, so the dense oracle has to come
    from a separate strategy configured with the dense backend.
    """
    dense, factors = [], []
    for li in range(len(cache)):
        layer = cache.layers[li]
        result = strategy.compress_layer_kv((layer.keys, layer.values), li)
        dense.append(result.dense_kv)
        factors.append(result.factors)
    return dense, factors


def score(model, cache, ctx_len, option_ids, device):
    opt = torch.tensor([option_ids], device=device)
    cache_position = torch.arange(ctx_len, ctx_len + len(option_ids),
                                  device=device, dtype=torch.long)
    attention_mask = torch.ones((1, ctx_len + len(option_ids)),
                                device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(opt, past_key_values=cache, use_cache=True,
                    cache_position=cache_position,
                    attention_mask=attention_mask, return_dict=True)
    return out.logits


def run(option_ids, preserve_last_n, n_kv_heads=2, verbose=True):
    device = "cpu"
    model = build_model().to(device)
    if n_kv_heads != model.config.num_key_value_heads:
        raise ValueError
    torch.manual_seed(1)
    ctx = torch.randint(0, 64, (1, 40), device=device)

    with torch.no_grad():
        pre = model(ctx, use_cache=True, return_dict=True)
    uncompressed = pre.past_key_values

    strategy_kwargs = dict(
        preserve_last_n=preserve_last_n,
        preserve_first_n=0,
        compression_method="shared_basis",
        value_residual_bits=1,
        qjl_seed=0,
        decomposition_backend="batched_eigh",
        eigh_compute_device="cpu",
    )
    dense_strategy = UniformStrategy(0.9, cache_backend="dense", **strategy_kwargs)
    factored_strategy = UniformStrategy(0.9, cache_backend="factored", **strategy_kwargs)

    dense_layers = compress_all(dense_strategy, uncompressed)[0]
    factors = compress_all(factored_strategy, uncompressed)[1]
    assert all(f is not None for f in factors), "factors missing"

    # --- dense reference: stock attention over the reconstructed cache -------
    model.config._attn_implementation = "eager"
    dense_cache = DynamicCache()
    for i, kv in enumerate(dense_layers):
        dense_cache.update(kv[0], kv[1], i)
    ref = score(model, dense_cache, ctx.shape[1], option_ids, device)

    # --- factored: attention consumes the factors ---------------------------
    register(model)
    fac = FactoredCache()
    for i, f in enumerate(factors):
        fac.add_layer(i, f)
    with use_factored_cache(fac):
        got = score(model, fac, ctx.shape[1], option_ids, device)

    rel = float((got - ref).norm() / ref.norm())
    return rel


def run_two_samples():
    """A second sample's prefill must not pick up the previous factored cache."""
    device = "cpu"
    model = build_model().to(device)
    register(model)
    torch.manual_seed(1)
    strategy_kwargs = dict(
        preserve_last_n=8,
        preserve_first_n=0,
        compression_method="shared_basis",
        value_residual_bits=1,
        qjl_seed=0,
        decomposition_backend="batched_eigh",
        eigh_compute_device="cpu",
    )
    dense_strategy = UniformStrategy(0.9, cache_backend="dense", **strategy_kwargs)
    factored_strategy = UniformStrategy(0.9, cache_backend="factored", **strategy_kwargs)
    rels = []
    for s in range(2):
        ctx = torch.randint(0, 64, (1, 40 + 7 * s), device=device)
        with torch.no_grad():
            pre = model(ctx, use_cache=True, return_dict=True)   # prefill
        unc = pre.past_key_values
        dense_layers = compress_all(dense_strategy, unc)[0]
        factors = compress_all(factored_strategy, unc)[1]
        dc = DynamicCache()
        for i, kv in enumerate(dense_layers):
            dc.update(kv[0], kv[1], i)
        ref = score(model, dc, ctx.shape[1], [7, 11, 3], device)
        fc = FactoredCache()
        for i, f in enumerate(factors):
            fc.add_layer(i, f)
        with use_factored_cache(fc):
            got = score(model, fc, ctx.shape[1], [7, 11, 3], device)
        rels.append(float((got - ref).norm() / ref.norm()))
    return rels


def main():
    cases = [
        ("q_len=1, tail=0", dict(option_ids=[7], preserve_last_n=0)),
        ("q_len=1, tail=8", dict(option_ids=[7], preserve_last_n=8)),
        ("q_len=3, tail=0", dict(option_ids=[7, 11, 3], preserve_last_n=0)),
        ("q_len=3, tail=8", dict(option_ids=[7, 11, 3], preserve_last_n=8)),
    ]
    tol = 1e-4
    bad = 0
    for name, kw in cases:
        rel = run(**kw)
        ok = rel < tol
        bad += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name:16s} rel={rel:.3e}")

    for i, rel in enumerate(run_two_samples()):
        ok = rel < tol
        bad += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] sample {i} of 2      rel={rel:.3e}")

    print("\nall passed" if bad == 0 else f"\n{bad} failed")
    return 1 if bad else 0


class FactoredEndToEndTest(unittest.TestCase):
    def test_existing_cases(self):
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    raise SystemExit(main())
