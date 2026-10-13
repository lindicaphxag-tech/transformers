"""Regression for #49483: chunked-prefill MUST NOT replay cached tokens.

Small deterministic CPU Llama; no Hub model weights or GPU. The external issue
author reported the bug. This test only validates an independent fix candidate.
"""
from copy import deepcopy

import pytest
import torch

from transformers import LlamaConfig, LlamaForCausalLM, StaticCache


@pytest.mark.parametrize("cache_kind", ["dynamic", "static"])
@pytest.mark.parametrize("chunk_size", [None, 4, 128])
def test_cached_prefix_chunked_prefill_matches_uncached_reference(cache_kind, chunk_size):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
            attn_implementation="eager",
        )
    ).eval()
    input_ids = torch.arange(1, 18).unsqueeze(0)
    with torch.no_grad():
        if cache_kind == "dynamic":
            prefix_cache = model(
                input_ids[:, :10], use_cache=True
            ).past_key_values
        else:
            prefix_cache = StaticCache(
                config=model.config, max_cache_len=17
            )
            model(
                input_ids[:, :10], past_key_values=prefix_cache,
                use_cache=True,
            )
        reference = model(input_ids, use_cache=False).logits[:, -1]
        result = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=deepcopy(prefix_cache),
            prefill_chunk_size=chunk_size,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            disable_compile=True,
        )
    assert int(result.past_key_values.get_seq_length()) == 17
    torch.testing.assert_close(
        result.scores[0], reference, rtol=1e-5, atol=1e-5
    )


def test_fresh_chunked_prefill_preserves_default_logits_and_cache_length():
    """Also cover fresh prompts, so a resumed-prefix fix cannot regress them."""
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=97, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
            bos_token_id=1, eos_token_id=None, pad_token_id=0,
            attn_implementation="eager",
        )
    ).eval()
    input_ids = torch.arange(1, 18).unsqueeze(0)
    results = []
    with torch.no_grad():
        for chunk_size in (None, 4, 128):
            out = model.generate(
                input_ids,
                prefill_chunk_size=chunk_size,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                disable_compile=True,
            )
            assert int(out.past_key_values.get_seq_length()) == 17
            results.append(out.scores[0])
    for logits in results[1:]:
        torch.testing.assert_close(logits, results[0], rtol=1e-5, atol=1e-5)
