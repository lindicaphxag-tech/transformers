"""Zamba2 cached single-token SSM must respect the prefill scan dt limit."""
from unittest.mock import patch

import torch

from transformers import DynamicCache, Zamba2Config, Zamba2Model
from transformers.models.zamba2 import modeling_zamba2


def test_zamba2_recurrent_dt_obeys_configured_positive_minimum():
    torch.manual_seed(20261014)
    config = Zamba2Config(
        vocab_size=67, hidden_size=16, mamba_dt_rank="auto", mamba_d_state=2,
        num_hidden_layers=3, num_attention_heads=2, n_mamba_heads=8,
        mamba_ngroups=8, intermediate_size=4, chunk_size=8,
        layers_block_type=["mamba", "hybrid", "hybrid"],
        num_mem_blocks=1, use_mem_rope=False, time_step_min=0.1,
    )
    model = Zamba2Model(config).eval()
    ids = torch.arange(1, 12).reshape(1, -1)
    cache = DynamicCache(config=config)
    real_update = modeling_zamba2.mamba2_selective_state_update
    seen = []

    def spy_update(state, hidden, dt, A, B, C, D=None, **kwargs):
        seen.append((dt.detach().clone(), kwargs))
        return real_update(state, hidden, dt, A, B, C, D, **kwargs)

    with torch.no_grad():
        model(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True)
        with patch.object(modeling_zamba2, "mamba2_selective_state_update", spy_update):
            output = model(input_ids=ids[:, -1:], past_key_values=cache, use_cache=True)
    assert output.last_hidden_state.shape == (1, 1, config.hidden_size)
    assert len(seen) >= 1
    for dt, kwargs in seen:
        assert kwargs["dt_softplus"] is False
        assert float(dt.float().min()) >= 0.1 - 1e-6
        torch.testing.assert_close(kwargs["dt_bias"], torch.zeros_like(kwargs["dt_bias"]))
