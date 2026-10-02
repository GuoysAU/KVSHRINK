import torch
from typing import Tuple, Dict
from .base import CompressionStrategy
from core.compression_types import CompressionResult


class AdaptiveStrategy(CompressionStrategy):
    """Each (layer, head) uses different tau based on tau_map

    支持两种模式:
    - per-head: 每个head有独立的tau，tau_map格式为 {(layer_idx, head_idx): tau}
    - layer-level: 每个layer的所有head共享同一个tau，tau_map格式为 {(layer_idx, -1): tau}

    支持两种压缩方法:
    - independent: K和V独立做SVD（原方法）
    - shared_basis: 用K的SVD基压缩V（共享基，Attention-aware）
    """

    def __init__(self, tau_map: Dict[Tuple[int, int], float] = None, preserve_last_n: int = 0,
                 preserve_first_n: int = 0, compression_method: str = "shared_basis",
                 value_residual_bits: int = 0, qjl_seed: int = 0,
                 decomposition_backend: str = "batched_eigh", cache_backend: str = "factored",
                 eigh_compute_device: str = "cpu"):
        """
        Args:
            tau_map: {(layer_idx, head_idx): tau} 或 {(layer_idx, -1): tau}
                     per-head模式: {(0,0): 0.95, (0,1): 0.93, ..., (39,39): 0.88}
                     layer-level模式: {(0,-1): 0.95, (1,-1): 0.90, ...}
                     Can be None initially and updated later with update_tau()
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
        self.tau_map = tau_map or {}
        self.preserve_last_n = preserve_last_n
        self.preserve_first_n = preserve_first_n
        self.compression_method = compression_method
        self.value_residual_bits = value_residual_bits
        self.qjl_seed = qjl_seed
        self._is_layer_level = None  # 延迟检测

    def _check_layer_level(self) -> bool:
        """检查是否为layer-level模式"""
        if self._is_layer_level is None:
            self._is_layer_level = any(head_idx == -1 for (_, head_idx) in self.tau_map.keys())
        return self._is_layer_level

    def update_tau(self, tau_map: Dict[Tuple[int, int], float]) -> None:
        """
        Update tau_map for a new sample.

        Args:
            tau_map: {(layer_idx, head_idx): tau} for the current sample
        """
        self.tau_map = tau_map
        self._is_layer_level = None  # 重置检测


    def _get_tau(self, layer_idx: int, head_idx: int) -> float:
        """获取指定layer和head的tau值"""
        if self._check_layer_level():
            # layer-level模式：所有head使用同一个tau
            tau = self.tau_map.get((layer_idx, -1), None)
            if tau is None:
                raise ValueError(f"Missing layer-level tau for layer {layer_idx}")
        else:
            # per-head模式
            tau = self.tau_map.get((layer_idx, head_idx), None)
            if tau is None:
                raise ValueError(f"Missing tau for layer {layer_idx}, head {head_idx}")
        return tau

    def compress_layer_kv(
        self,
        layer_kv: Tuple[torch.Tensor, torch.Tensor],
        layer_idx: int
    ) -> CompressionResult:
        """
        Compress a layer, each head with its own tau (or shared tau in layer-level mode).

        Args:
            layer_kv: (key, value), each [batch, num_heads, seq_len, head_dim]
            layer_idx: current layer index

        Returns:
            CompressionResult with the layer representation and statistics.
        """
        # layer-level模式下，只查询一次tau
        is_layer_level = self._check_layer_level()
        layer_tau = self._get_tau(layer_idx, 0) if is_layer_level else None

        def tau_getter(head_idx: int) -> float:
            if is_layer_level:
                return layer_tau
            return self._get_tau(layer_idx, head_idx)

        return self._compress_layer_with_tau_getter(
            layer_kv=layer_kv,
            layer_idx=layer_idx,
            tau_getter=tau_getter,
            compression_method=self.compression_method,
            preserve_last_n=self.preserve_last_n,
            preserve_first_n=self.preserve_first_n,
            value_residual_bits=self.value_residual_bits,
            qjl_seed=self.qjl_seed,
        )
