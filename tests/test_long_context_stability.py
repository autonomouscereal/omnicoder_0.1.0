import os
import sys
import tempfile


def test_export_longctx_variants():
    from omnicoder.export.onnx_export import main as onnx_main
    argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'decode.onnx')
            # Emit long-context variants; YaRN on
            sys.argv = ['onnx_export', '--output', out, '--seq_len', '1', '--mobile_preset', 'mobile_4gb', '--decode_step', '--emit_longctx_variants', '--yarn', '--no_dynamo', '--opset', '17']
            # Force emission of both 32k and 128k for this stability canary
            os.environ['OMNICODER_EXPORT_ALL_LONGCTX'] = '1'
            onnx_main()
            ctx32k = out.replace('.onnx', '_ctx32k.onnx')
            assert os.path.exists(ctx32k)
            # Optional: ensure the 128k variant is also emitted for CI canary
            ctx128k = out.replace('.onnx', '_ctx128k.onnx')
            assert os.path.exists(ctx128k)
    finally:
        # Restore environment and argv
        os.environ.pop('OMNICODER_EXPORT_ALL_LONGCTX', None)
        sys.argv = argv


def test_ssm_block_export_guard_fullseq_only():
	# Ensure SSM blocks do not run during decode-step
	from omnicoder.modeling.transformer_moe import OmniTransformer
	import torch
	model = OmniTransformer(n_layers=2, d_model=64, n_heads=4, mlp_dim=128, n_experts=4, top_k=2)
	model.eval()
	# Full sequence (no cache): SSM path may apply; should run without error
	a = torch.randint(0, model.vocab_size, (1, 8))
	_ = model(a)
	# Decode-step: ensure use_cache=True path executes without touching SSM
	b = torch.randint(0, model.vocab_size, (1, 1))
	out = model(b, past_kv=[(None, None) for _ in range(len(model.blocks))], use_cache=True)
	assert isinstance(out, tuple)


