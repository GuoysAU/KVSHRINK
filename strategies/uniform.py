import torch
from typing import Tuple
from .base import CompressionStrategy
from core.compression_types import CompressionResult


class UniformStrategy(CompressionStrategy):
    """All heads in all layers use same tau

    支持两种压缩方法:
    - independent: K和V独立做SVD（原方法）
    - shared_basis: 用K的SVD基压缩V（共享基，Attention-aware）
    """

    def __init__(self, tau: float, preserve_last_n: int = 0, preserve_first_n: int = 0,
                 compression_method: str = "shared_basis", value_residual_bits: int = 0,
                 qjl_seed: int = 0, decomposition_backend: str = "batched_eigh",
                 cache_backend: str = "factored",
                 eigh_compute_device: str = "cpu"):
        """
        Args:
            tau: 能量保留阈值
            preserve_last_n: 保留最后N个token不压缩（保护RoPE几何）
            preserve_first_n: 保留前N个token不压缩（保护few-shot示例）
            compression_method: 压缩方法
                     "independent": K和V独立做SVD（原方法）
                     "shared_basis": 用K的SVD基压缩V（默认，Attention-aware）
        """
        super().__init__(
            decomposition_backend=decomposition_backend,
            cache_backend=cache_backend,
            eigh_compute_device=eigh_compute_device,
        )
        self._validate_cache_compatibility(compression_method, preserve_first_n)
        self.tau = tau
        self.preserve_last_n = preserve_last_n
        self.preserve_first_n = preserve_first_n
        self.compression_method = compression_method
        self.value_residual_bits = value_residual_bits
        self.qjl_seed = qjl_seed

    def compress_layer_kv(
        self,
        layer_kv: Tuple[torch.Tensor, torch.Tensor],
        layer_idx: int
    ) -> CompressionResult:
        """
        Compress all heads in a layer with same tau.

        Args:
            layer_kv: (key, value), each [batch, num_heads, seq_len, head_dim]
            layer_idx: layer index (unused for uniform)

        Returns:
            CompressionResult with the layer representation and statistics.
        """
        return self._compress_layer_with_tau_getter(
            layer_kv=layer_kv,
            layer_idx=layer_idx,
            tau_getter=lambda _head_idx: self.tau,
            compression_method=self.compression_method,
            preserve_last_n=self.preserve_last_n,
            preserve_first_n=self.preserve_first_n,
            value_residual_bits=self.value_residual_bits,
            qjl_seed=self.qjl_seed,
        )
