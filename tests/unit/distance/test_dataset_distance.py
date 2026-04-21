"""Tests for ``themap.distance.dataset_distance`` helpers."""

from unittest.mock import patch

import pytest

from themap.distance.dataset_distance import _resolve_device


class TestResolveDevice:
    """Tests for the ``_resolve_device`` helper."""

    def test_explicit_cpu_passthrough(self):
        assert _resolve_device("cpu") == "cpu"

    def test_explicit_cuda_passthrough(self):
        assert _resolve_device("cuda") == "cuda"

    def test_explicit_indexed_cuda_passthrough(self):
        assert _resolve_device("cuda:1") == "cuda:1"

    def test_auto_resolves_to_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert _resolve_device("auto") == "cuda"

    def test_auto_resolves_to_cpu_when_no_gpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert _resolve_device("auto") == "cpu"

    def test_auto_falls_back_to_cpu_when_torch_missing(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert _resolve_device("auto") == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
