"""Contract tests for the public Strategy compression result."""

import json
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.compression_types import CompressionResult  # noqa: E402
from strategies.uniform import UniformStrategy  # noqa: E402


class CompressionResultTest(unittest.TestCase):
    @staticmethod
    def _layer(seq_len):
        torch.manual_seed(seq_len)
        return (
            torch.randn(1, 2, seq_len, 16),
            torch.randn(1, 2, seq_len, 16),
        )

    def test_reference_svd_returns_dense_kv(self):
        strategy = UniformStrategy(
            0.9,
            compression_method="shared_basis",
            value_residual_bits=0,
            decomposition_backend="reference_svd",
            cache_backend="dense",
        )
        result = strategy.compress_layer_kv(self._layer(16), layer_idx=0)

        self.assertIsNotNone(result.dense_kv)
        self.assertIsNone(result.factors)
        self.assertEqual(result.dense_kv[0].shape, (1, 2, 16, 16))
        json.dumps(result.stats)

    def test_factored_result_does_not_leak_into_later_skip(self):
        strategy = UniformStrategy(
            0.9,
            preserve_last_n=8,
            compression_method="shared_basis",
            value_residual_bits=0,
            decomposition_backend="batched_eigh",
            cache_backend="factored",
            eigh_compute_device="cpu",
        )

        factored = strategy.compress_layer_kv(self._layer(16), layer_idx=0)
        short_kv = self._layer(8)
        skipped = strategy.compress_layer_kv(short_kv, layer_idx=1)

        self.assertIsNone(factored.dense_kv)
        self.assertIsNotNone(factored.factors)
        self.assertIsNotNone(skipped.dense_kv)
        self.assertIsNone(skipped.factors)
        self.assertTrue(torch.equal(skipped.dense_kv[0], short_kv[0]))
        self.assertTrue(torch.equal(skipped.dense_kv[1], short_kv[1]))
        self.assertEqual(skipped.stats["heads"][0]["K"]["status"], "skipped")
        self.assertFalse(hasattr(strategy, "_last_factors"))

    def test_independent_batched_eigh_returns_dense_kv(self):
        strategy = UniformStrategy(
            0.9,
            compression_method="independent",
            value_residual_bits=0,
            decomposition_backend="batched_eigh",
            cache_backend="dense",
            eigh_compute_device="cpu",
        )
        result = strategy.compress_layer_kv(self._layer(16), layer_idx=0)
        self.assertIsNotNone(result.dense_kv)
        self.assertIsNone(result.factors)

    def test_empty_result_is_rejected(self):
        with self.assertRaises(ValueError):
            CompressionResult(dense_kv=None, stats={}, factors=None)

    def test_two_representations_are_rejected(self):
        dense_kv = self._layer(4)
        with self.assertRaises(ValueError):
            CompressionResult(dense_kv=dense_kv, stats={}, factors={})

    def test_unknown_compression_method_is_rejected(self):
        with self.assertRaises(ValueError):
            UniformStrategy(0.9, compression_method="typo")


if __name__ == "__main__":
    unittest.main()
