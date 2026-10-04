# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the measured XPU runtime-FP8 ESIMD adapter."""

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

from vllm.model_executor.layers import layernorm
from vllm.model_executor.kernels.linear.scaled_mm import xpu


def test_prepare_xpu_w8a16_weight_restores_contiguous_out_in_layout():
    checkpoint = torch.arange(24, dtype=torch.float16).reshape(6, 4)
    loader_view = checkpoint.t()

    prepared = xpu._prepare_xpu_w8a16_weight(loader_view, 6, 4)

    assert prepared.shape == (6, 4)
    assert prepared.is_contiguous()
    torch.testing.assert_close(prepared, checkpoint)


def test_prepare_xpu_w8a16_square_weight_uses_stride_to_resolve_layout():
    checkpoint = torch.arange(16, dtype=torch.float16).reshape(4, 4)

    prepared = xpu._prepare_xpu_w8a16_weight(checkpoint.t(), 4, 4)

    assert prepared.is_contiguous()
    torch.testing.assert_close(prepared, checkpoint)


def test_measured_esimd_fp8_batch_path_dispatches_for_eligible_35b_shape():
    inputs = torch.zeros(12, 2048, dtype=torch.float16)
    weight = torch.zeros(2048, 2048, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, dtype=torch.float32)
    op = Mock(side_effect=lambda _x, _w, _s, out: out.fill_(3))

    with (
        patch.object(xpu, "_is_xpu_tensor", return_value=True),
        patch.object(xpu, "_load_esimd_op", return_value=op),
    ):
        result = xpu._try_esimd_fp8_linear(inputs, weight, scale, None)

    assert result is not None
    assert result.shape == (12, 2048)
    assert torch.all(result == 3)
    op.assert_called_once_with(inputs, weight, scale, result)


def test_measured_esimd_fp8_path_dispatches_for_wide_decode_projection():
    inputs = torch.zeros(1, 2048, dtype=torch.float16)
    weight = torch.zeros(4608, 2048, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, dtype=torch.float32)
    op = Mock(side_effect=lambda _x, _w, _s, out: out.fill_(3))

    with (
        patch.object(xpu, "_is_xpu_tensor", return_value=True),
        patch.object(xpu, "_load_esimd_op", return_value=op),
    ):
        result = xpu._try_esimd_fp8_linear(inputs, weight, scale, None)

    assert result is not None
    assert result.shape == (1, 4608)
    assert torch.all(result == 3)
    op.assert_called_once_with(inputs, weight, scale, result)


def test_measured_esimd_fp8_gemm_path_dispatches_for_wide_projection():
    inputs = torch.zeros(4, 2048, dtype=torch.float16)
    weight = torch.zeros(4608, 2048, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, dtype=torch.float32)
    op = Mock(side_effect=lambda _x, _w, _s, out: out.fill_(3))

    with (
        patch.object(xpu, "_is_xpu_tensor", return_value=True),
        patch.object(xpu, "_load_esimd_op", return_value=op),
    ):
        result = xpu._try_esimd_fp8_linear(inputs, weight, scale, None)

    assert result is not None
    assert result.shape == (4, 4608)
    assert torch.all(result == 3)
    op.assert_called_once_with(inputs, weight, scale, result)


def test_measured_esimd_fp8_gemm_path_dispatches_for_narrow_projection():
    inputs = torch.zeros(5, 5120, dtype=torch.float16)
    weight = torch.zeros(48, 5120, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, dtype=torch.float32)
    op = Mock(side_effect=lambda _x, _w, _s, out: out.fill_(3))

    with (
        patch.object(xpu, "_is_xpu_tensor", return_value=True),
        patch.object(xpu, "_load_esimd_op", return_value=op),
    ):
        result = xpu._try_esimd_fp8_linear(
            inputs, weight, scale, None, allow_qwen_spec_n48=True
        )

    assert result is not None
    assert result.shape == (5, 48)
    assert torch.all(result == 3)
    op.assert_called_once_with(inputs, weight, scale, result)


def test_narrow_projection_falls_back_without_qwen_spec_guard():
    inputs = torch.zeros(5, 5120, dtype=torch.float16)
    weight = torch.zeros(48, 5120, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, dtype=torch.float32)

    with (
        patch.object(xpu, "_is_xpu_tensor", return_value=True),
        patch.object(xpu, "_load_esimd_op") as load,
    ):
        result = xpu._try_esimd_fp8_linear(inputs, weight, scale, None)

    assert result is None
    load.assert_not_called()


@pytest.mark.parametrize(
    "n,k",
    [
        (7168, 5120),
        (8192, 5120),
        (17408, 5120),
        (5120, 8704),
        (5120, 3072),
        (4608, 2048),
    ],
)
def test_measured_esimd_fp8_m1_path_dispatches_for_qwen36_tp2_shapes(n, k):
    inputs = torch.zeros(1, k, dtype=torch.float16)
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, dtype=torch.float32)
    op = Mock(side_effect=lambda _x, _w, _s, out: out.fill_(3))

    with (
        patch.object(xpu, "_is_xpu_tensor", return_value=True),
        patch.object(xpu, "_load_esimd_op", return_value=op),
    ):
        result = xpu._try_esimd_fp8_linear(inputs, weight, scale, None)

    assert result is not None
    assert result.shape == (1, n)
    assert torch.all(result == 3)
    op.assert_called_once_with(inputs, weight, scale, result)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (2, 5120, 3072),
        (4, 5120, 3072),
        (8, 5120, 3072),
        (12, 5120, 3072),
        (16, 5120, 3072),
        (32, 5120, 3072),
        (64, 5120, 3072),
    ],
)
def test_measured_esimd_fp8_batch_path_dispatches_for_winning_regions(m, n, k):
    inputs = torch.zeros(m, k, dtype=torch.float16)
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, dtype=torch.float32)
    op = Mock()

    with (
        patch.object(xpu, "_is_xpu_tensor", return_value=True),
        patch.object(xpu, "_load_esimd_op", return_value=op),
    ):
        result = xpu._try_esimd_fp8_linear(inputs, weight, scale, None)

    assert result is not None
    op.assert_called_once_with(inputs, weight, scale, result)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (33, 7168, 5120),
        (64, 7168, 5120),
        (33, 17408, 5120),
        (64, 17408, 5120),
        (33, 5120, 8704),
        (64, 5120, 8704),
    ],
)
def test_measured_esimd_fp8_batch_path_rejects_losing_regions(m, n, k):
    inputs = torch.zeros(m, k, dtype=torch.float16)
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, dtype=torch.float32)

    with (
        patch.object(xpu, "_is_xpu_tensor", return_value=True),
        patch.object(xpu, "_load_esimd_op") as load,
    ):
        result = xpu._try_esimd_fp8_linear(inputs, weight, scale, None)

    assert result is None
    load.assert_not_called()


def test_measured_esimd_fp8_batch_gate_preserves_m1():
    batch_inputs = torch.zeros(12, 3072, dtype=torch.float16)
    m1_input = batch_inputs[:1]
    weight = torch.zeros(5120, 3072, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, dtype=torch.float32)
    gemv = Mock()
    gemm = Mock()

    def load_op(is_decode):
        return gemv if is_decode else gemm

    with (
        patch.object(xpu, "_DISABLE_ESIMD_FP8_BATCH", True),
        patch.object(xpu, "_is_xpu_tensor", return_value=True),
        patch.object(xpu, "_load_esimd_op", side_effect=load_op),
    ):
        assert xpu._try_esimd_fp8_linear(
            batch_inputs, weight, scale, None
        ) is None
        result = xpu._try_esimd_fp8_linear(m1_input, weight, scale, None)

    assert result is not None
    gemv.assert_called_once_with(m1_input, weight, scale, result)
    gemm.assert_not_called()


@pytest.mark.parametrize(
    "m,n,k,scale_elements,bias,dtype",
    [
        (33, 2048, 2048, 1, False, torch.float16),
        (65, 2048, 2048, 1, False, torch.float16),
        (1, 32769, 512, 1, False, torch.float16),
        (1, 48, 5120, 1, False, torch.float16),
        (1, 2048, 2048, 2, False, torch.float16),
        (1, 2048, 2048, 1, True, torch.float16),
        (1, 2048, 2048, 1, False, torch.bfloat16),
    ],
)
def test_measured_esimd_fp8_path_rejects_unqualified_inputs(
    m, n, k, scale_elements, bias, dtype
):
    inputs = torch.zeros(m, k, dtype=dtype)
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    scale = torch.ones(scale_elements, dtype=torch.float32)

    with (
        patch.object(xpu, "_is_xpu_tensor", return_value=True),
        patch.object(xpu, "_load_esimd_op") as load,
    ):
        result = xpu._try_esimd_fp8_linear(
            inputs,
            weight,
            scale,
            torch.zeros(n, dtype=torch.float16) if bias else None,
        )

    assert result is None
    load.assert_not_called()


def test_load_esimd_fp8_op_caches_exact_callable():
    op = Mock()
    xpu._load_esimd_op.cache_clear()

    try:
        with patch.object(xpu, "get_esimd_op", return_value=op) as get_op:
            assert xpu._load_esimd_op(True) is op
            assert xpu._load_esimd_op(True) is op
            assert xpu._load_esimd_op(False) is op
            assert xpu._load_esimd_op(False) is op

        assert get_op.call_args_list == [
            call("esimd_gemv_fp8_pert"),
            call("esimd_gemm_fp8_pert"),
        ]
    finally:
        xpu._load_esimd_op.cache_clear()


@pytest.mark.parametrize("rows", [1, 4, 8, 12, 16, 128])
def test_gemma_rms_norm_xpu_uses_split_esimd_decode_path(
    default_vllm_config, rows
):
    norm = layernorm.GemmaRMSNorm(128, eps=1e-6)
    norm.weight.data.fill_(0.25)
    inputs = torch.zeros(rows, 128, dtype=torch.float16)
    residual = torch.ones_like(inputs)
    op = Mock()
    module = SimpleNamespace(esimd_fused_add_rms_norm_batched=op)

    with (
        patch.object(layernorm, "_DISABLE_ESIMD_NORM", False),
        patch.object(layernorm.torch.compiler, "is_compiling", return_value=False),
        patch.object(layernorm.importlib, "import_module", return_value=module),
        patch.object(layernorm, "_is_xpu_tensor", return_value=True),
    ):
        result = norm.forward_xpu(inputs, residual)

    assert result[0] is inputs
    assert result[1] is residual
    op.assert_called_once()
    call_args = op.call_args.args
    assert call_args[0] is inputs
    assert call_args[1] is residual
    torch.testing.assert_close(
        call_args[2], torch.full((128,), 1.25, dtype=torch.float16)
    )
    assert call_args[3] == 1e-6
