import os
import json

from omnicoder.inference.runtimes.onnx_provider_profiles import get_provider_options, get_ptq_op_types_preset
from omnicoder.inference.memory_estimator import estimate_kv_cache_bytes
from omnicoder.config import MobilePreset


def test_provider_options_cpu():
    prov, opts = get_provider_options('CPUExecutionProvider')
    assert prov == ['CPUExecutionProvider']
    assert isinstance(opts, list)


def test_provider_options_nnapi():
    prov, opts = get_provider_options('NNAPIExecutionProvider', nnapi_accel='NnapiAccelerator.qnn')
    assert prov == ['NNAPIExecutionProvider']
    assert isinstance(opts[0], dict)
    assert opts[0].get('nnapi_accelerator_name') == 'NnapiAccelerator.qnn'


def test_ptq_presets():
    for p in ['generic','nnapi','coreml','dml']:
        ops = get_ptq_op_types_preset(p)
        assert 'MatMul' in ops
        assert isinstance(ops, list)


def test_kv_estimator_quant_modes():
    preset = MobilePreset()
    fp16 = estimate_kv_cache_bytes(preset, 1024, 1, kvq='fp16')
    u8 = estimate_kv_cache_bytes(preset, 1024, 1, kvq='u8')
    nf4 = estimate_kv_cache_bytes(preset, 1024, 1, kvq='nf4')
    assert u8 < fp16
    assert nf4 <= u8


