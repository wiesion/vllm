# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for metadata passed to the native XPU GDN kernel."""

from types import SimpleNamespace

import torch

from vllm import _xpu_ops


def test_xpu_gdn_index_metadata_is_contiguous_int32():
    storage = torch.arange(32, dtype=torch.int64)
    metadata = SimpleNamespace(
        num_accepted_tokens=storage[::2],
        non_spec_query_start_loc=storage[1::2],
        non_spec_token_indx=storage[::2],
        non_spec_state_indices_tensor=storage[1::2],
        spec_query_start_loc=storage[::2],
        spec_token_indx=storage[1::2],
        spec_state_indices_tensor=storage.reshape(4, 8)[:, ::2],
    )

    prepared = _xpu_ops._prepare_xpu_gdn_index_tensors(metadata)

    assert prepared.keys() == {
        "num_accepted_tokens",
        "non_spec_query_start_loc",
        "non_spec_token_indx",
        "non_spec_state_indices_tensor",
        "spec_query_start_loc",
        "spec_token_indx",
        "spec_state_indices_tensor",
    }
    for name, tensor in prepared.items():
        assert tensor.dtype == torch.int32
        assert tensor.is_contiguous()
        torch.testing.assert_close(
            tensor, getattr(metadata, name).to(torch.int32).contiguous()
        )


def test_xpu_gdn_index_metadata_preserves_compliant_tensors_and_none():
    compliant = torch.arange(4, dtype=torch.int32)
    metadata = SimpleNamespace(
        num_accepted_tokens=None,
        non_spec_query_start_loc=compliant,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=None,
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
    )

    prepared = _xpu_ops._prepare_xpu_gdn_index_tensors(metadata)

    assert prepared["non_spec_query_start_loc"] is compliant
    assert prepared["num_accepted_tokens"] is None
    assert prepared["spec_state_indices_tensor"] is None
