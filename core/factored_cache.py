"""A KV cache that holds only the compressed factors.

Unlike the default path, nothing dense is persisted for the compressed prefix:
each layer keeps ``(U_r, Sigma_r Psi_r^T, C_V, signs, rho)`` plus the small
uncompressed tail (``preserve_last_n`` tokens, extended by whatever the model
generates). Attention is computed by ``core.factored_attention``, which consumes
the factors directly, so ``K_hat``/``V_hat``/``R_hat`` are never formed.

``update()`` returns only the dense tail; the registered attention function
supplies the factored prefix itself. Sequence-length accessors report the full
prefix + tail so causal masks are built at the right width.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from .factored_attention import factored_attention

# Attention registration is global, but the cache in use is local to one
# forward/generate scope. ContextVar also isolates concurrent execution contexts.
_ACTIVE: ContextVar[Optional["FactoredCache"]] = ContextVar(
    "kvshrink_factored_cache",
    default=None,
)


@contextmanager
def use_factored_cache(cache):
    """Activate a factored cache only for the enclosed model call."""
    active = cache if isinstance(cache, FactoredCache) else None
    token = _ACTIVE.set(active)
    try:
        yield cache
    finally:
        _ACTIVE.reset(token)


def get_active() -> Optional["FactoredCache"]:
    return _ACTIVE.get()


class FactoredCache(DynamicCache):
    """Per-layer factors + dense tail. No dense prefix is ever stored."""

    def __init__(self) -> None:
        super().__init__()
        self.factors: Dict[int, Dict[str, Any]] = {}
        self.tail_k: Dict[int, Optional[torch.Tensor]] = {}
        self.tail_v: Dict[int, Optional[torch.Tensor]] = {}

    # -- construction -----------------------------------------------------
    def add_layer(self, layer_idx: int, factors: Dict[str, Any]) -> None:
        if factors.get("k_pre") is not None:
            raise NotImplementedError("preserve_first_n > 0 is not supported")
        self.factors[layer_idx] = factors
        self.tail_k[layer_idx] = factors.get("k_post")
        self.tail_v[layer_idx] = factors.get("v_post")

    @classmethod
    def from_factors(cls, layer_factors) -> "FactoredCache":
        """Build a cache from one bundle per layer, in layer order."""
        cache = cls()
        for layer_idx, factors in enumerate(layer_factors):
            cache.add_layer(layer_idx, factors)
        return cache

    def has_factors(self, layer_idx: int) -> bool:
        return layer_idx in self.factors

    # -- HF cache protocol ------------------------------------------------
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx not in self.factors:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        tk, tv = self.tail_k[layer_idx], self.tail_v[layer_idx]
        tk = key_states if tk is None else torch.cat([tk, key_states], dim=2)
        tv = value_states if tv is None else torch.cat([tv, value_states], dim=2)
        self.tail_k[layer_idx], self.tail_v[layer_idx] = tk, tv
        # Only the dense tail travels back to the attention module; the factored
        # prefix is read from this cache by the registered attention function.
        return tk, tv

    def _length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.factors:
            return super().get_seq_length(layer_idx)
        tail = self.tail_k[layer_idx]
        return self.factors[layer_idx]["mid_seq"] + (0 if tail is None else tail.shape[2])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._length(layer_idx)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        if layer_idx not in self.factors:
            return super().get_mask_sizes(cache_position, layer_idx)
        # Called before the layers run update(), so the incoming query length is
        # added here, matching DynamicLayer.get_mask_sizes.
        return cache_position.shape[0] + self._length(layer_idx), 0

    # -- storage accounting ----------------------------------------------
    def stored_bytes(self) -> int:
        """Bytes actually resident, measured from the tensors themselves."""
        total = 0
        for layer_idx, f in self.factors.items():
            for name in ("U_r", "sigma_psi_t", "C_V"):
                for t in f[name]:
                    total += t.numel() * t.element_size()
            sk = f.get("sketch")
            if sk is not None:
                packed, norms = sk
                total += packed.numel() * packed.element_size()
                total += norms.numel() * norms.element_size()
            for t in (self.tail_k[layer_idx], self.tail_v[layer_idx]):
                if t is not None:
                    total += t.numel() * t.element_size()
        return total


def kvshrink_factored_attention(module, query, key, value, attention_mask,
                                scaling=None, dropout=0.0, **kwargs):
    """Registered attention: factored when the active cache has factors."""
    cache = get_active()
    layer_idx = getattr(module, "layer_idx", None)
    # Identity check rather than a lifecycle flag: the factored branch is taken
    # only when `key` is the very tail tensor this cache just returned from
    # update(). A prefill (or any forward on another cache) fails it and falls
    # back, so a stale active handle cannot leak into an unrelated forward.
    in_use = (
        cache is not None
        and layer_idx is not None
        and cache.has_factors(layer_idx)
        and key is cache.tail_k.get(layer_idx)
    )
    if not in_use:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)

    out = factored_attention(
        query, cache.factors[layer_idx],
        tail_k=key, tail_v=value,
        mask=attention_mask, scaling=scaling,
    )
    return out.transpose(1, 2).contiguous(), None


def register(model=None) -> None:
    """Register the attention function (and its mask fn) and select it."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

    # Must go through register(): it writes to the class-level _global_mapping,
    # which is what _preprocess_mask_arguments consults. Plain item assignment
    # lands in _local_mapping, create_causal_mask then returns None and the
    # causal mask silently disappears (multi-token options score bidirectionally).
    ALL_ATTENTION_FUNCTIONS.register("kvshrink_factored", kvshrink_factored_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register("kvshrink_factored", eager_mask)
    if model is not None:
        model.config._attn_implementation = "kvshrink_factored"
