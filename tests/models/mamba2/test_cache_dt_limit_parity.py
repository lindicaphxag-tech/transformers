"""Bounded continuous-time SSM caching must obey the same dt contract as scans.

These tests construct a tiny Mamba2 model from config. They specifically
verify the recurrent-path kernel arguments so a near-zero random layer's
output cannot accidentally conceal a missing runtime time-step clamp.
"""
from unittest.mock import patch

import pytest
import torch

from transformers import Mamba2Config, Mamba2Model
from transformers.models.mamba2 import modeling_mamba2


def tiny_model(bounds):
    torch.manual_seed(20261014)
    cfg = Mamba2Config(
        hidden_size=16, expand=2, head_dim=8, num_heads=4, n_groups=2,
        state_size=4, num_hidden_layers=2, chunk_size=8, conv_kernel=3,
        vocab_size=79, time_step_limit=bounds, use_cache=True,
    )
    return Mamba2Model(cfg).eval()


@pytest.mark.parametrize("bounds", [(0.0, 0.0001), (0.1, float("inf")), (0.002, 0.03)])
def test_cached_mamba2_one_token_uses_runtime_scan_time_step_limits(bounds):
    model = tiny_model(bounds)
    ids = torch.arange(1, 13)[None, :]
    real_update = modeling_mamba2.mamba2_selective_state_update
    calls = []

    def spy_update(state, hidden, dt, A, B, C, D=None, **kwargs):
        # The cached mixer, not merely our standalone arithmetic reference,
        # must send preprocessed/bounded dt to the kernel exactly once.
        calls.append((dt.detach().cpu().float().clone(), kwargs.copy()))
        return real_update(state, hidden, dt, A, B, C, D, **kwargs)

    with torch.no_grad():
        whole = model(ids, use_cache=False).last_hidden_state
        prefix = model(ids[:, :-1], use_cache=True)
        with patch.object(modeling_mamba2, "mamba2_selective_state_update", spy_update):
            continuation = model(ids[:, -1:], cache_params=prefix.cache_params, use_cache=True)

    assert len(calls) == model.config.num_hidden_layers
    for dt, kwargs in calls:
        assert kwargs["dt_softplus"] is False
        assert float(dt.min()) >= bounds[0] - 1e-6
        assert float(dt.max()) <= bounds[1] + 1e-6
        torch.testing.assert_close(kwargs["dt_bias"], torch.zeros_like(kwargs["dt_bias"]))
    got = torch.cat([prefix.last_hidden_state, continuation.last_hidden_state], dim=1)
    torch.testing.assert_close(got, whole, rtol=3e-3, atol=3e-3)


def test_default_unbounded_case_preserves_kernel_preprocessing_contract():
    model = tiny_model((0.0, float("inf")))
    ids = torch.arange(1, 13)[None, :]
    real_update = modeling_mamba2.mamba2_selective_state_update
    flags = []

    def spy_update(state, hidden, dt, A, B, C, D=None, **kwargs):
        flags.append(kwargs["dt_softplus"])
        return real_update(state, hidden, dt, A, B, C, D, **kwargs)

    with torch.no_grad():
        prefix = model(ids[:, :-1], use_cache=True)
        with patch.object(modeling_mamba2, "mamba2_selective_state_update", spy_update):
            model(ids[:, -1:], cache_params=prefix.cache_params, use_cache=True)
    assert flags == [True] * model.config.num_hidden_layers
