import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.compressor import SVDCompressor  # noqa: E402


class QJLResidualTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.key = torch.randn(1, 32, 16, dtype=torch.float32)
        self.value = torch.randn(1, 32, 16, dtype=torch.float32)

    def test_qjl_residual_is_deterministic_and_accounted_for(self):
        _, value_a, stats_a = SVDCompressor.compress_head_shared_basis(
            self.key,
            self.value,
            tau=0.5,
            value_residual_bits=1,
            qjl_seed=11,
        )
        _, value_b, stats_b = SVDCompressor.compress_head_shared_basis(
            self.key,
            self.value,
            tau=0.5,
            value_residual_bits=1,
            qjl_seed=11,
        )

        self.assertTrue(torch.equal(value_a, value_b))
        residual = stats_a["V"]["residual_correction"]
        expected_code_bytes = (32 * 16 + 7) // 8
        expected_norm_bytes = 32 * 4
        self.assertEqual(residual["code_bytes"], expected_code_bytes)
        self.assertEqual(residual["norm_bytes"], expected_norm_bytes)
        self.assertEqual(
            stats_a["V"]["comp_bytes"],
            stats_a["V"]["rank"] * 16 * 4
            + expected_code_bytes
            + expected_norm_bytes,
        )
        self.assertEqual(stats_a["V"]["comp_bytes"], stats_b["V"]["comp_bytes"])

    def test_qjl_residual_lowers_compression_ratio(self):
        _, _, base_stats = SVDCompressor.compress_head_shared_basis(
            self.key,
            self.value,
            tau=0.5,
            value_residual_bits=0,
        )
        _, corrected_value, corrected_stats = SVDCompressor.compress_head_shared_basis(
            self.key,
            self.value,
            tau=0.5,
            value_residual_bits=1,
        )

        self.assertEqual(corrected_value.shape, self.value.shape)
        self.assertTrue(torch.isfinite(corrected_value).all())
        self.assertLess(
            corrected_stats["avg_compression_ratio"],
            base_stats["avg_compression_ratio"],
        )
        self.assertTrue(corrected_stats["V"]["residual_correction"]["enabled"])

    def test_invalid_residual_bit_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            SVDCompressor.compress_head_shared_basis(
                self.key,
                self.value,
                tau=0.5,
                value_residual_bits=2,
            )


if __name__ == "__main__":
    unittest.main()
