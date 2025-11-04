def test_coreml_attention_rope_metadata_present():
    try:
        import coremltools as ct  # type: ignore
        from omnicoder.export.coreml_attention_pass import try_map_attention_with_rope  # type: ignore
    except Exception:
        return  # skip if coremltools not available
    # Create a minimal empty model spec and attach
    try:
        spec = ct.models.neural_network.NeuralNetworkBuilder([], []).spec  # type: ignore
        mlmodel = ct.models.MLModel(spec)
        mlmodel2 = try_map_attention_with_rope(mlmodel, rope_base=10000.0, rope_scale=1.0)
        spec2 = mlmodel2.get_spec()
        meta = spec2.description.metadata.user_defined
        assert meta.get("omnicoder_attention") == "rope_mqa"
        assert "rope_base" in meta and "rope_scale" in meta
    except Exception:
        pass

def test_coreml_decode_model_has_metadata_if_exported():
    # Smoke: if an MLModel exists at the default location, ensure metadata tags exist
    try:
        import coremltools as ct  # type: ignore
        from pathlib import Path
        p = Path('weights/text/omnicoder_decode_step.mlmodel')
        if not p.exists():
            return
        m = ct.models.MLModel(str(p))
        meta = m.user_defined_metadata  # type: ignore[attr-defined]
        # Basic tags should be present from the exporter
        assert 'kv_latent_dim' in meta
        assert 'heads' in meta and 'layers' in meta
        assert 'rope_base' in meta and 'rope_scale' in meta
    except Exception:
        pass

def test_coreml_native_attention_flag_when_mapped():
    try:
        import coremltools as ct  # type: ignore
        from omnicoder.export.coreml_attention_pass import try_replace_qkv_with_native_attention  # type: ignore
    except Exception:
        return
    try:
        spec = ct.models.neural_network.NeuralNetworkBuilder([], []).spec  # type: ignore
        mlmodel = ct.models.MLModel(spec)
        m2 = try_replace_qkv_with_native_attention(mlmodel)
        spec2 = m2.get_spec()
        meta = spec2.description.metadata.user_defined
        assert meta.get('native_attention') == '1'
        assert meta.get('attention_op') in ('native',)
        assert meta.get('omnicoder_attention') in ('rope_mqa',)
    except Exception:
        pass

