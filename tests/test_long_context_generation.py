import os
import torch

from omnicoder.inference.generate import build_mobile_model_by_name, generate
from omnicoder.config import get_mobile_preset, get_rope_scale_for_target_ctx, get_rope_interp_base


def test_long_context_generation_canary():
    os.environ.setdefault('OMNICODER_EXPORT_TINY', '1')
    preset_name = 'mobile_4gb'
    preset = get_mobile_preset(preset_name)
    target_ctx = 32768
    rope_scale = get_rope_scale_for_target_ctx(preset.max_seq_len, target_ctx)
    rope_base = get_rope_interp_base(10000.0, rope_scale)
    model = build_mobile_model_by_name(preset_name, rope_scale=rope_scale, rope_base=rope_base, multi_token=2, mem_slots=4)
    model.eval()
    tok = torch.randint(0, getattr(model, 'vocab_size', 32000), (1, 8), dtype=torch.long)
    out = generate(model, tok, max_new_tokens=1, window_size=512, return_stats=False)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 1
import torch

from omnicoder.inference.generate import build_mobile_model_by_name
from omnicoder.config import get_rope_scale_for_target_ctx


def test_long_context_forward_small_cpu():
    preset_name = 'mobile_4gb'
    model = build_mobile_model_by_name(preset_name, rope_scale=get_rope_scale_for_target_ctx(8192, 32768))
    model.eval()
    tok = 64
    with torch.inference_mode():
        ids = torch.randint(0, model.vocab_size, (1, tok), dtype=torch.long)
        out = model(ids, past_kv=None, use_cache=False)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        assert logits.shape[0] == 1 and logits.shape[1] == tok and logits.shape[2] == model.vocab_size


def test_long_context_onxx_roundtrip_names(tmp_path):
    # Export long-context variants and ensure outputs exist (deterministic and bounded)
    import sys, os, pathlib, json
    from omnicoder.export.onnx_export import main as onnx_export_main
    out = tmp_path / 'decode.onnx'
    # Prepare deterministic argv for legacy exporter and 32k emission only
    argv_orig = sys.argv
    sys.argv = [
        'onnx_export',
        '--output', str(out),
        '--seq_len', '1',
        '--mobile_preset', 'mobile_4gb',
        '--decode_step',
        '--emit_longctx_variants',
        '--no_dynamo',
        '--opset', '17',
    ]
    # Force only 32k emission in-process
    env_orig = os.environ.copy()
    os.environ['OMNICODER_EXPORT_ALL_LONGCTX'] = '0'
    # Diagnostics
    try:
        logs_dir = pathlib.Path('tests_logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logs_dir = pathlib.Path('.')
    diag_path = logs_dir / 'longctx_diag.json'
    diag = {'stage': 'pre', 'argv': sys.argv}
    try:
        onnx_export_main()
        diag.update({'stage': 'post_main', 'out_exists': out.exists()})
        ctx32 = str(out).replace('.onnx', '_ctx32k.onnx')
        diag.update({'ctx32_path': ctx32, 'ctx32_exists': os.path.exists(ctx32)})
    except Exception as e:
        diag.update({'stage': 'exception', 'error': str(e)})
        try:
            diag_path.write_text(json.dumps(diag, indent=2))
        finally:
            # restore and re-raise to fail the test
            sys.argv = argv_orig
            os.environ.clear(); os.environ.update(env_orig)
        raise
    finally:
        try:
            diag_path.write_text(json.dumps(diag, indent=2))
        except Exception:
            pass
        # restore
        sys.argv = argv_orig
        os.environ.clear(); os.environ.update(env_orig)
    # Check base exists
    assert out.exists()
    ctx32 = str(out).replace('.onnx', '_ctx32k.onnx')
    # 32k is required; 128k is optional and can be enabled via env
    assert os.path.exists(ctx32)


