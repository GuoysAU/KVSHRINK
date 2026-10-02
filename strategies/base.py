from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Callable, List
import torch
from core.compression_types import CompressionResult
from core.compressor import SVDCompressor


class CompressionStrategy(ABC):
    def __init__(
        self,
        decomposition_backend: str = "batched_eigh",
        cache_backend: str = "factored",
        eigh_compute_device: str = "cpu",
    ) -> None:
        if decomposition_backend not in ("reference_svd", "batched_eigh"):
            raise ValueError(
                "decomposition_backend must be 'reference_svd' or 'batched_eigh'"
            )
        if cache_backend not in ("dense", "factored"):
            raise ValueError("cache_backend must be 'dense' or 'factored'")
        if eigh_compute_device not in ("cpu", "input"):
            raise ValueError("eigh_compute_device must be 'cpu' or 'input'")
        if cache_backend == "factored" and decomposition_backend != "batched_eigh":
            raise ValueError(
                "factored cache requires decomposition_backend='batched_eigh'"
            )
        self.decomposition_backend = decomposition_backend
        self.cache_backend = cache_backend
        self.eigh_compute_device = eigh_compute_device

    def _validate_cache_compatibility(
        self, compression_method: str, preserve_first_n: int
    ) -> None:
        if compression_method not in ("shared_basis", "independent"):
            raise ValueError(
                "compression_method must be 'shared_basis' or 'independent'"
            )
        if self.cache_backend != "factored":
            return
        if compression_method != "shared_basis":
            raise ValueError("factored cache requires compression_method='shared_basis'")
        if preserve_first_n:
            raise ValueError("factored cache does not support preserve_first_n > 0")

    @abstractmethod
    def compress_layer_kv(
        self,
        layer_kv: Tuple[torch.Tensor, torch.Tensor],
        layer_idx: int
    ) -> CompressionResult:
        """
        Compress a single layer's KV cache.

        Returns:
            One CompressionResult containing either dense KV tensors or a
            factored representation, together with per-layer statistics.
        """
        pass

    def _compress_layer_with_tau_getter(
        self,
        layer_kv: Tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        tau_getter: Callable[[int], float],
        compression_method: str,
        preserve_last_n: int,
        preserve_first_n: int,
        value_residual_bits: int = 0,
        qjl_seed: int = 0,
    ) -> CompressionResult:
        """
        通用的 layer 级别压缩模板：
        - 逐 head 获取 tau
        - 调用指定压缩方法
        - 回写压缩后的 K/V
        - 汇总 layer/head 统计信息
        """
        key, value = layer_kv
        _, num_heads, _, _ = key.shape

        # Fast path: batched eigendecomposition of K^T K over all heads at once
        # (mathematically equivalent to the per-head SVD loop).
        if self.decomposition_backend == "batched_eigh" and compression_method == "shared_basis":
            taus = [tau_getter(h) for h in range(num_heads)]
            comp_key, comp_value, heads_stats, layer_svd_time, factors = (
                SVDCompressor.compress_layer_shared_basis_eigh(
                    key, value, taus,
                    preserve_last_n=preserve_last_n,
                    preserve_first_n=preserve_first_n,
                    value_residual_bits=value_residual_bits,
                    qjl_seed=qjl_seed,
                    return_factors=self.cache_backend == "factored",
                    eigh_compute_device=self.eigh_compute_device,
                )
            )
            layer_stats = {
                "layer": layer_idx,
                "layer_svd_time": layer_svd_time,
                "heads": heads_stats,
            }
            return CompressionResult(
                dense_kv=None if comp_key is None else (comp_key, comp_value),
                stats=layer_stats,
                factors=factors,
            )

        if self.decomposition_backend == "batched_eigh" and compression_method == "independent":
            taus = [tau_getter(h) for h in range(num_heads)]
            comp_key, comp_value, heads_stats, layer_svd_time = (
                SVDCompressor.compress_layer_independent_eigh(
                    key, value, taus,
                    preserve_last_n=preserve_last_n,
                    preserve_first_n=preserve_first_n,
                    eigh_compute_device=self.eigh_compute_device,
                )
            )
            layer_stats = {
                "layer": layer_idx,
                "layer_svd_time": layer_svd_time,
                "heads": heads_stats,
            }
            return CompressionResult((comp_key, comp_value), layer_stats)

        total_svd_time = 0.0
        heads_stats: List[Dict[str, Any]] = []

        for head_idx in range(num_heads):
            tau = tau_getter(head_idx)
            k_head = key[:, head_idx, :, :]
            v_head = value[:, head_idx, :, :]

            if compression_method == "shared_basis":
                comp_k, comp_v, head_stats = SVDCompressor.compress_head_shared_basis(
                    k_head,
                    v_head,
                    tau,
                    preserve_last_n,
                    preserve_first_n,
                    value_residual_bits=value_residual_bits,
                    qjl_seed=qjl_seed,
                )
            else:
                comp_k, comp_v, head_stats = SVDCompressor.compress_head(
                    k_head, v_head, tau, preserve_last_n, preserve_first_n
                )

            key[:, head_idx, :, :] = comp_k
            value[:, head_idx, :, :] = comp_v
            total_svd_time += head_stats["total_svd_time"]

            heads_stats.append({
                "head": head_idx,
                "tau": tau,
                "K": head_stats["K"],
                "V": head_stats["V"],
            })

        layer_stats = {
            "layer": layer_idx,
            "layer_svd_time": total_svd_time,
            "heads": heads_stats,
        }

        return CompressionResult((key, value), layer_stats)
