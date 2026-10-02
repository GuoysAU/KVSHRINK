"""Attention computed directly on the compressed factors.

The compressor produces, per layer, a factored representation of the cached
prefix plus a small uncompressed tail (``preserve_last_n``):

    K_mid ~ U_r (Sigma_r Psi_r^T)          U_r: [H, n, r], Sigma_r Psi_r^T: [r, d]
    V_mid ~ U_r C_V + R_hat                C_V: [H, r, d]
    R_hat  = c * diag(rho) B G             B: [H*n, d] signs, rho: row scales

This module consumes those factors without ever forming K_hat, V_hat or R_hat:

    logits = concat( (q Psi_r Sigma_r) U_r^T , q K_post^T ) / sqrt(d)
    p      = softmax(logits)
    out    = (p_mid U_r) C_V + c * ((p_mid * rho) B) G + p_post V_post

``preserve_first_n`` is 0 for every evaluated task (all are 0-shot), so only the
two-segment layout is implemented; a non-zero value is rejected rather than
silently mis-sliced.
"""

import math
from typing import Any, Dict, Optional

import torch

from .compressor import SVDCompressor


def _pad_ragged(factors: Dict[str, Any], dt: torch.dtype, device):
    """Batch the per-head factors, which are stored at each head's own rank.

    Storage is ragged so it occupies exactly the r_h that Eq. (60) charges;
    padding to the layer maximum happens here and lives only for this call.
    """
    H, D, mid = factors["num_heads"], factors["head_dim"], factors["mid_seq"]
    ranks = factors["rank"]
    r_max = max(ranks)
    U_r = torch.zeros(H, mid, r_max, dtype=dt, device=device)
    sigma_psi_t = torch.zeros(H, r_max, D, dtype=dt, device=device)
    C_V = torch.zeros(H, r_max, D, dtype=dt, device=device)
    for h, rh in enumerate(ranks):
        U_r[h, :, :rh] = factors["U_r"][h].to(dt)
        sigma_psi_t[h, :rh, :] = factors["sigma_psi_t"][h].to(dt)
        C_V[h, :rh, :] = factors["C_V"][h].to(dt)
    return U_r, sigma_psi_t, C_V


def _sketch_apply(p_mid: torch.Tensor, sketch, head_dim: int, seed: int,
                  num_heads: int, mid_seq: int) -> Optional[torch.Tensor]:
    """Return ``p_mid @ R_hat`` without materializing ``R_hat``.

    ``p_mid`` is [H, Q, n]; the stored code covers all heads as [H*n, d].
    """
    if sketch is None:
        return None
    packed, norms = sketch                                   # [H*n, d/8] uint8, [H*n, 1]
    projection = SVDCompressor._get_qjl_projection(
        dimension=head_dim, device=p_mid.device, seed=seed
    )                                                        # [d, d]
    # Unpacked transiently; only the packed code is persisted.
    signs = SVDCompressor.unpack_signs(packed, head_dim, p_mid.dtype)
    signs = signs.reshape(num_heads, mid_seq, head_dim)
    norms = norms.reshape(num_heads, 1, mid_seq).to(p_mid.dtype)
    weighted = (p_mid * norms) @ signs                       # [H, Q, d]
    scale = math.sqrt(math.pi / 2.0) / head_dim
    return scale * (weighted @ projection.to(p_mid.dtype))


def factored_attention(
    query: torch.Tensor,
    factors: Dict[str, Any],
    mask: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    tail_k: Optional[torch.Tensor] = None,
    tail_v: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention over [factored prefix | uncompressed tail].

    Args:
        query: [B, Hq, Q, d]. With GQA, ``Hq`` is a multiple of the KV head count.
        factors: the bundle produced by ``compress_layer_shared_basis_eigh``.
        mask: additive mask broadcastable to [B, Hq, Q, n_total], or None.
        scaling: defaults to ``1/sqrt(d)``.

    Returns:
        [B, Hq, Q, d]
    """
    if factors.get("k_pre") is not None:
        raise NotImplementedError(
            "preserve_first_n > 0 is not supported by the factored path; "
            "the evaluated tasks are 0-shot so this segment is always empty."
        )

    batch = factors["batch"]
    if batch != 1 or query.shape[0] != 1:
        raise NotImplementedError("factored attention assumes batch size 1")

    H = factors["num_heads"]
    D = factors["head_dim"]
    mid_seq = factors["mid_seq"]
    out_dtype = query.dtype
    scaling = (1.0 / math.sqrt(D)) if scaling is None else scaling

    # Arithmetic in fp32 while storage stays fp16. The factored form computes an
    # intermediate q (Sigma_r Psi_r^T)^T whose magnitude is ~sigma_1 |q|, far
    # larger than any final logit: deep layers have Key activations in the
    # hundreds, so this overflows fp16 (max 65504) and softmax then yields NaN.
    # The dense path never sees it because SDPA accumulates in fp32 internally.
    dt = torch.float32

    U_r, sigma_psi_t, C_V = _pad_ragged(factors, dt, query.device)

    # Fold the query-head groups of GQA into the query axis so the factors are
    # never expanded: [1, Hq, Q, d] -> [H, g*Q, d].
    hq, q_len = query.shape[1], query.shape[2]
    if hq % H != 0:
        raise ValueError(f"query heads {hq} not a multiple of kv heads {H}")
    g = hq // H
    q = query[0].to(dt).reshape(H, g * q_len, D)

    # Logits over the factored prefix, never forming K_hat.
    s_mid = (q @ sigma_psi_t.transpose(-1, -2)) @ U_r.transpose(-1, -2)  # [H, gQ, n]

    # The tail grows during generation, so the cache passes it in explicitly;
    # the bundle's own copy is the post-compression starting point.
    k_post = factors.get("k_post") if tail_k is None else tail_k
    v_post = factors.get("v_post") if tail_v is None else tail_v
    if k_post is not None:
        k_post = k_post[0].to(dt)                   # [H, m, d]
        v_post = v_post[0].to(dt)
        s_post = q @ k_post.transpose(-1, -2)       # [H, gQ, m]
        logits = torch.cat([s_mid, s_post], dim=-1)
    else:
        logits = s_mid
    logits = logits * scaling

    if mask is not None:
        m = mask
        if m.dim() == 4:
            m = m[0]                                # [Hq or 1, Q, n_total]
            if m.shape[0] == 1:
                m = m.expand(hq, -1, -1)
            m = m.reshape(H, g * q_len, -1)
        logits = logits + m.to(logits.dtype)

    p = torch.softmax(logits, dim=-1)

    p_mid = p[..., :mid_seq]
    out = (p_mid @ U_r) @ C_V                       # [H, gQ, d], no V_hat
    sk = _sketch_apply(p_mid, factors.get("sketch"), D,
                       factors.get("qjl_seed", 0), H, mid_seq)
    if sk is not None:
        out = out + sk
    if k_post is not None:
        out = out + p[..., mid_seq:] @ v_post

    return out.reshape(1, hq, q_len, D).to(out_dtype)
