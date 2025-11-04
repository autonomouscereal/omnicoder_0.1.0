from pathlib import Path
import json

from omnicoder.inference.runtimes.onnx_provider_profiles import get_provider_options


def test_profiles_json_loadable():
    for name in ['pixel7_nnapi.json','iphone15_coreml_ane.json','windows_dml.json']:
        p = Path('profiles') / name
        data = json.loads(p.read_text(encoding='utf-8'))
        assert 'provider' in data
        assert 'intra_op_num_threads' in data
        # Ensure the provider options function returns a matching provider name
        providers, _ = get_provider_options(data['provider'])
        assert isinstance(providers, list) and providers


