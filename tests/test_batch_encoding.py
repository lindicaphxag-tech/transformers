"""Regression tests for BatchEncoding tensor conversion."""

import numpy as np
import pytest

from transformers import BatchEncoding


@pytest.mark.parametrize(
    "value",
    [
        [],
        [[], []],
        [[], [1, 2]],
        [[1, 2], []],
        [[[]], [[]]],
    ],
)
def test_batch_encoding_empty_check_matches_flatten_semantics(value):
    batch = BatchEncoding({"input_ids": value})
    batch.convert_to_tensors("np")

    if value == [] or value == [[], []] or value == [[[]], [[]]]:
        assert batch["input_ids"].dtype == np.int64
    else:
        assert batch["input_ids"].size > 0


def test_batch_encoding_large_batch_conversion():
    values = [[i] for i in range(100_000)]
    batch = BatchEncoding({"input_ids": values})
    batch.convert_to_tensors("np")
    np.testing.assert_array_equal(batch["input_ids"].ravel(), np.arange(100_000))
