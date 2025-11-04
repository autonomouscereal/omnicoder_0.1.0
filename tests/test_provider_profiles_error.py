import os
from pathlib import Path


def test_get_provider_options_invalid_profile_json(tmp_path, monkeypatch):
    prof = tmp_path / 'bad_profile.json'
    prof.write_text('{not: valid json', encoding='utf-8')
    monkeypatch.setenv('OMNICODER_PROVIDER_PROFILE', str(prof))
    from omnicoder.inference.runtimes.onnx_provider_profiles import get_provider_options
    provs, opts = get_provider_options('CPUExecutionProvider')
    assert isinstance(provs, list) and isinstance(opts, list)
    assert len(provs) == 1 and len(opts) == 1
    # Falls back to requested provider on error
    assert provs[0] in ('CPUExecutionProvider','NNAPIExecutionProvider','CoreMLExecutionProvider','DmlExecutionProvider')


def test_get_provider_options_valid_profile_json(tmp_path, monkeypatch):
    prof = tmp_path / 'ok_profile.json'
    prof.write_text('{"provider": "DmlExecutionProvider", "provider_options": {"queue": 0}}', encoding='utf-8')
    monkeypatch.setenv('OMNICODER_PROVIDER_PROFILE', str(prof))
    from omnicoder.inference.runtimes.onnx_provider_profiles import get_provider_options
    provs, opts = get_provider_options('CPUExecutionProvider')
    assert provs == ['DmlExecutionProvider']
    assert isinstance(opts, list) and isinstance(opts[0], dict)

