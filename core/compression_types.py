"""Shared return types for KV-cache compression."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


LayerStats = Dict[str, Any]
LayerFactors = Dict[str, Any]

DenseKV = Tuple[torch.Tensor, torch.Tensor]


# No slots=True: it needs Python 3.10+, and the saving here is one instance
# dict per layer -- negligible next to the tensors this object points at.
@dataclass
class CompressionResult:
    """All outputs produced by one layer-compression call."""

    dense_kv: Optional[DenseKV]
    stats: LayerStats
    factors: Optional[LayerFactors] = None

    def __post_init__(self) -> None:
        if (self.dense_kv is None) == (self.factors is None):
            raise ValueError(
                "compression result must contain exactly one cache representation"
            )
