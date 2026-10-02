"""Tests for adapting and releasing supported Hugging Face cache layouts."""

from contextlib import closing
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_wrapper import iter_cache_layers  # noqa: E402


class _Layer:
    def __init__(self, index):
        self.keys = f"k{index}"
        self.values = f"v{index}"


class _LayerCache:
    def __init__(self, count):
        self.layers = [_Layer(index) for index in range(count)]


class _LegacyCache:
    def __init__(self, count):
        self.key_cache = [f"k{index}" for index in range(count)]
        self.value_cache = [f"v{index}" for index in range(count)]


class CacheLayerIteratorTest(unittest.TestCase):
    expected = [(index, (f"k{index}", f"v{index}")) for index in range(3)]

    def test_supported_layouts_have_the_same_order(self):
        layer_cache = _LayerCache(3)
        legacy_cache = _LegacyCache(3)
        tuple_cache = tuple(pair for _, pair in self.expected)

        self.assertEqual(list(iter_cache_layers(layer_cache)), self.expected)
        self.assertEqual(list(iter_cache_layers(legacy_cache)), self.expected)
        self.assertEqual(list(iter_cache_layers(tuple_cache)), self.expected)
        self.assertTrue(all(layer.keys is None for layer in layer_cache.layers))
        self.assertTrue(all(key is None for key in legacy_cache.key_cache))

    def test_layer_reference_is_released_after_consumption(self):
        cache = _LayerCache(3)
        iterator = iter_cache_layers(cache)

        with closing(iterator):
            self.assertEqual(next(iterator), self.expected[0])
            self.assertEqual(cache.layers[0].keys, "k0")
            self.assertEqual(next(iterator), self.expected[1])
            self.assertIsNone(cache.layers[0].keys)

        self.assertIsNone(cache.layers[1].keys)
        self.assertEqual(cache.layers[2].keys, "k2")

    def test_closing_releases_current_layer_on_error(self):
        cache = _LayerCache(2)
        iterator = iter_cache_layers(cache)

        with self.assertRaisesRegex(RuntimeError, "compression failed"):
            with closing(iterator):
                next(iterator)
                raise RuntimeError("compression failed")

        self.assertIsNone(cache.layers[0].keys)
        self.assertEqual(cache.layers[1].keys, "k1")


if __name__ == "__main__":
    unittest.main()
