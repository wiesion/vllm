# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock


def test_esimd_router_handles_dflash_verify_batch() -> None:
    block = object.__new__(Qwen3NextSparseMoeBlock)
    torch.nn.Module.__init__(block)
    router_op = Mock()
    block._esimd_router_plan = None
    block._use_spec_fp8_router = True
    block._esimd_moe_ops = SimpleNamespace(esimd_gemm_fp8_pert=router_op)
    block._batch_router_buffers = {}
    block.n_routed_experts = 256

    gate = Mock()
    gate.weight = torch.empty(256, 2048, dtype=torch.float16)
    gate.bias = None
    fp8_weight = torch.empty(256, 2048, dtype=torch.float8_e4m3fn)
    weight_scale = torch.ones(1)

    hidden_states = torch.empty(17, 2048, dtype=torch.float16)
    with patch.object(
        Qwen3NextSparseMoeBlock,
        "_get_fp8_router_weight",
        return_value=(fp8_weight, weight_scale),
    ):
        output = block._esimd_router_logits(hidden_states, 17, gate)

    assert output.shape == (17, 256)
    router_op.assert_called_once_with(
        hidden_states,
        fp8_weight,
        weight_scale,
        output,
    )
    gate.assert_not_called()


def test_esimd_router_preserves_regular_qwen_fp16_path() -> None:
    block = object.__new__(Qwen3NextSparseMoeBlock)
    torch.nn.Module.__init__(block)
    router_op = Mock()
    block._esimd_router_plan = None
    block._use_spec_fp8_router = False
    block._esimd_moe_ops = SimpleNamespace(esimd_gemv_fp16=router_op)
    block._batch_router_buffers = {}
    block.n_routed_experts = 256

    gate = Mock()
    gate.weight = torch.empty(256, 2048, dtype=torch.float16)
    gate.bias = None
    hidden_states = torch.empty(8, 2048, dtype=torch.float16)

    output = block._esimd_router_logits(hidden_states, 8, gate)

    router_op.assert_called_once_with(hidden_states, gate.weight, output)
    gate.assert_not_called()


def test_esimd_router_does_not_expand_regular_qwen_batch_range() -> None:
    block = object.__new__(Qwen3NextSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block._esimd_router_plan = None
    block._use_spec_fp8_router = False
    block._esimd_moe_ops = SimpleNamespace(esimd_gemv_fp16=Mock())
    block._batch_router_buffers = {}
    block.n_routed_experts = 256

    gate = Mock(return_value=(torch.empty(17, 256), None))
    gate.weight = torch.empty(256, 2048, dtype=torch.float16)
    gate.bias = None
    hidden_states = torch.empty(17, 2048, dtype=torch.float16)

    block._esimd_router_logits(hidden_states, 17, gate)

    block._esimd_moe_ops.esimd_gemv_fp16.assert_not_called()
    gate.assert_called_once_with(hidden_states)
