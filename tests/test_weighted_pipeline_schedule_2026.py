from __future__ import annotations

import torch

from omnicoder.modeling.omnicoder2026 import OmniCoder2026, OmniCoder2026Config


def _tiny_config() -> dict:
    return {
        "vocab_size": 96,
        "n_layers": 2,
        "d_model": 32,
        "n_heads": 4,
        "head_dim": 8,
        "num_key_value_heads": 1,
        "mlp_dim": 64,
        "max_seq_len": 32,
        "local_window": 16,
        "csa_block_size": 16,
        "csa_top_k_blocks": 2,
        "hca_block_size": 16,
        "latent_dim": 8,
        "rope_dim": 8,
        "sink_tokens": 1,
        "q_lora_rank": 8,
        "o_lora_rank": 8,
        "o_groups": 1,
        "index_head_dim": 8,
        "flow_latent_dim": 8,
        "layer_pattern": ("kda", "kda"),
    }


def _weighted_tiny_model() -> OmniCoder2026:
    torch.manual_seed(7)
    model = OmniCoder2026(OmniCoder2026Config(**_tiny_config()))
    summary = model.apply_weighted_device_map(
        [torch.device("cpu"), torch.device("cpu")],
        embed_device=torch.device("cpu"),
        head_device=torch.device("cpu"),
    )
    assert summary["pipeline_stages"] == [{"device": "cpu", "start": 0, "end": 2}]
    return model


def test_weighted_pipeline_loss_matches_standard_loss_on_cpu():
    model = _weighted_tiny_model()
    batch = torch.tensor(
        [
            [2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30, 31, 32, 33],
        ],
        dtype=torch.long,
    )

    standard = model(batch, labels=batch, return_logits=False, return_hidden=False)["loss"]
    piped = model.forward_weighted_pipeline_loss(batch, batch, microbatches=4, async_streams=False)

    assert standard is not None
    assert torch.allclose(piped, standard, atol=1e-6, rtol=1e-6)


def test_weighted_pipeline_backward_preserves_state_dict_contract():
    model = _weighted_tiny_model()
    before_keys = tuple(model.state_dict().keys())
    batch = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=torch.long)

    loss = model.forward_weighted_pipeline_loss(batch, batch, microbatches=2, async_streams=False)
    loss.backward()

    assert tuple(model.state_dict().keys()) == before_keys
    assert any(parameter.grad is not None for parameter in model.parameters())
