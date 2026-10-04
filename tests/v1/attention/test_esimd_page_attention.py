# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.utils import (
    is_valid_esimd_page_attention_input,
)


def _inputs(
    *,
    page_size: int = 64,
    query_dtype: torch.dtype = torch.float16,
    output_dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype = torch.float16,
):
    query = torch.empty((2, 4, 256), dtype=query_dtype)
    output = torch.empty((2, 4, 256), dtype=output_dtype)
    key_cache = torch.empty((4, page_size, 2, 256), dtype=kv_dtype)
    value_cache = torch.empty_like(key_cache)
    return query, output, key_cache, value_cache


@pytest.mark.parametrize(
    "kv_dtype",
    [torch.float16, torch.float8_e4m3fn, torch.float8_e5m2],
)
@pytest.mark.parametrize("page_size", [64, 128, 1024])
def test_esimd_page_attention_accepts_kernel_abi(page_size, kv_dtype):
    assert is_valid_esimd_page_attention_input(
        *_inputs(page_size=page_size, kv_dtype=kv_dtype)
    )


@pytest.mark.parametrize("page_size", [16, 80, 2048])
def test_esimd_page_attention_rejects_unsupported_page_size(page_size):
    assert not is_valid_esimd_page_attention_input(
        *_inputs(page_size=page_size)
    )


@pytest.mark.parametrize(
    ("query_dtype", "output_dtype"),
    [
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bfloat16),
    ],
)
def test_esimd_page_attention_requires_fp16_query_and_output(
    query_dtype, output_dtype
):
    assert not is_valid_esimd_page_attention_input(
        *_inputs(query_dtype=query_dtype, output_dtype=output_dtype)
    )


def test_esimd_page_attention_rejects_mismatched_output_shape():
    query, output, key_cache, value_cache = _inputs()
    assert not is_valid_esimd_page_attention_input(
        query, output[:, :2], key_cache, value_cache
    )
