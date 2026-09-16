from pathlib import Path

from transformers import LukeTokenizer


FIXTURES = Path(__file__).parents[2] / "fixtures"


def make_tokenizer():
    return LukeTokenizer(
        vocab=str(FIXTURES / "vocab.json"),
        merges=str(FIXTURES / "merges.txt"),
        entity_vocab_file=str(FIXTURES / "test_entity_vocab.json"),
    )


def test_plain_text_adds_special_tokens():
    tokenizer = make_tokenizer()

    single = tokenizer("Hello world", return_special_tokens_mask=True)
    assert single["input_ids"][0] == tokenizer.cls_token_id
    assert single["input_ids"][-1] == tokenizer.sep_token_id
    assert single["special_tokens_mask"] == [1] + [0] * (len(single["input_ids"]) - 2) + [1]

    pair = tokenizer("Hello world", "Second", return_special_tokens_mask=True)
    assert pair["input_ids"][0] == tokenizer.cls_token_id
    assert pair["input_ids"][-1] == tokenizer.sep_token_id
    assert pair["special_tokens_mask"][0] == 1
    assert pair["special_tokens_mask"][-1] == 1
