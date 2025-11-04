from __future__ import annotations

import importlib.util
import os
import sys

import torch


def test_omnicoder_dml_native_optional_present_or_skip() -> None:
    # Presence is optional; just ensure the optional module can be resolved when built
    spec = importlib.util.find_spec('omnicoder_dml_native')
    if spec is None:
        # Not built in CI/Docker CPU; acceptable
        return
    mod = importlib.import_module('omnicoder_dml_native')  # type: ignore
    assert mod is not None


def test_ops_registered_when_module_present() -> None:
    spec = importlib.util.find_spec('omnicoder_dml_native')
    if spec is None:
        return
    import omnicoder_dml_native  # type: ignore  # noqa: F401
    fused = getattr(torch.ops, 'omnicoder_dml', None)
    assert fused is not None and hasattr(fused, 'mla') and hasattr(fused, 'matmul_int4')


