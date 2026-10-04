# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm import xpu
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
    prepare_fp8_moe_layer_for_xpu,
)


def test_xpu_block_fp8_moe_preserves_checkpoint_native_layout() -> None:
    """Fused decode and grouped GEMM share canonical [E, N, K] weights."""
    w13 = (torch.arange(4 * 256 * 128).reshape(4, 256, 128).remainder(16)).to(
        torch.float8_e4m3fn
    )
    w2 = w13.transpose(-1, -2).contiguous()
    w13_scale = torch.empty(4, 2, 1, dtype=torch.float32)
    w2_scale = torch.empty(4, 1, 2, dtype=torch.float32)

    actual_w13, actual_s13, actual_w2, actual_s2 = prepare_fp8_moe_layer_for_xpu(
        w13, w13_scale, w2, w2_scale
    )

    assert actual_w13.shape == w13.shape
    assert actual_w2.shape == w2.shape
    assert actual_w13.is_contiguous()
    assert actual_w2.is_contiguous()
    assert torch.equal(actual_w13.float(), w13.float())
    assert torch.equal(actual_w2.float(), w2.float())
    assert actual_s13 is w13_scale
    assert actual_s2 is w2_scale


def test_xpu_block_fp8_moe_preserves_ambiguous_square_weights() -> None:
    weight = (torch.arange(4 * 128 * 128).reshape(4, 128, 128).remainder(16)).to(
        torch.float8_e4m3fn
    )
    weight = weight.transpose(-1, -2)
    scale = torch.empty(4, 1, 1, dtype=torch.float32)

    actual_w13, _, actual_w2, _ = prepare_fp8_moe_layer_for_xpu(
        weight, scale, weight, scale
    )

    assert actual_w13.is_contiguous()
    assert actual_w2.is_contiguous()
    assert torch.equal(actual_w13.float(), weight.float())
    assert torch.equal(actual_w2.float(), weight.float())


def test_xpu_tensor_fp8_moe_transposes_to_grouped_gemm_layout() -> None:
    """Per-tensor grouped GEMM retains the established [E, K, N] contract."""
    w13 = torch.empty(4, 1024, 512, dtype=torch.float8_e4m3fn)
    w2 = torch.empty(4, 512, 512, dtype=torch.float8_e4m3fn)
    w13_scale = torch.empty(4, dtype=torch.float32)
    w2_scale = torch.empty(4, dtype=torch.float32)

    actual_w13, actual_s13, actual_w2, actual_s2 = prepare_fp8_moe_layer_for_xpu(
        w13, w13_scale, w2, w2_scale
    )

    assert actual_w13.shape == (4, 512, 1024)
    assert actual_w2.shape == (4, 512, 512)
    assert actual_w13.is_contiguous()
    assert actual_w2.is_contiguous()
    assert actual_s13 is w13_scale
    assert actual_s2 is w2_scale


def test_xpu_block_fp8_detection_requires_matching_scale_layouts() -> None:
    """A single 3-D scale tensor must not silently select a mixed layout."""
    w13 = torch.empty(4, 1024, 512, dtype=torch.float8_e4m3fn)
    w2 = torch.empty(4, 512, 512, dtype=torch.float8_e4m3fn)
    block_scale = torch.empty(4, 8, 4, dtype=torch.float32)
    tensor_scale = torch.empty(4, dtype=torch.float32)

    with pytest.raises(ValueError, match="matching block-scale layouts"):
        prepare_fp8_moe_layer_for_xpu(w13, block_scale, w2, tensor_scale)


def test_xpu_block_fp8_rejects_tail_blocks_before_native_dispatch() -> None:
    w13 = torch.empty(4, 1024, 513, dtype=torch.float8_e4m3fn)
    w2 = torch.empty(4, 512, 512, dtype=torch.float8_e4m3fn)
    w13_scale = torch.empty(4, 8, 5, dtype=torch.float32)
    w2_scale = torch.empty(4, 4, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match="dimensions divisible by 128"):
        prepare_fp8_moe_layer_for_xpu(w13, w13_scale, w2, w2_scale)


def test_xpu_block_fp8_rejects_non_float32_scales() -> None:
    w13 = torch.empty(4, 1024, 512, dtype=torch.float8_e4m3fn)
    w2 = torch.empty(4, 512, 512, dtype=torch.float8_e4m3fn)
    w13_scale = torch.empty(4, 8, 4, dtype=torch.float16)
    w2_scale = torch.empty(4, 4, 4, dtype=torch.float16)

    with pytest.raises(ValueError, match="float32 block scales"):
        prepare_fp8_moe_layer_for_xpu(w13, w13_scale, w2, w2_scale)


def _make_dense_kernel(
    dtype: torch.dtype,
    block_shape: tuple[int, int] = (128, 128),
) -> xpu.XPUFp8BlockScaledMMKernel:
    kernel = xpu.XPUFp8BlockScaledMMKernel.__new__(xpu.XPUFp8BlockScaledMMKernel)
    kernel.config = SimpleNamespace(out_dtype=dtype)
    kernel.weight_group_shape = block_shape
    return kernel


def _make_dense_layer(n: int, k: int) -> SimpleNamespace:
    return SimpleNamespace(
        weight=torch.empty(n, k, dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.arange(
            1,
            (n // 128) * (k // 128) + 1,
            dtype=torch.float32,
        ).view(n // 128, k // 128),
    )


def test_xpu_block_fp8_onednn_crossover_uses_exact_scales() -> None:
    dtype = torch.float16
    m, n, k = xpu._XPU_BLOCK_ONEDNN_MIN_M, 256, 384
    inputs = torch.empty(2, m // 2, k, dtype=dtype)
    layer = _make_dense_layer(n, k)
    bias = torch.arange(n, dtype=dtype)
    native = Mock(
        side_effect=lambda a, b, scale, bias: torch.zeros(
            a.shape[0], b.shape[1], dtype=dtype
        )
    )

    with patch.object(
        torch.ops._xpu_C,
        "fp8_gemm_w8a16",
        native,
        create=True,
    ):
        output = _make_dense_kernel(dtype).apply_weights(layer, inputs, bias)

    assert output.shape == (2, m // 2, n)
    torch.testing.assert_close(output, bias.expand_as(output))
    actual_input, actual_weight, actual_scale, actual_bias = native.call_args.args
    assert actual_input.data_ptr() == inputs.data_ptr()
    assert actual_weight.data_ptr() == layer.weight.data_ptr()
    assert actual_scale.is_contiguous()
    torch.testing.assert_close(actual_scale, layer.weight_scale_inv.t())
    assert actual_bias is None


def test_xpu_small_m_block_fp8_uses_esimd_without_input_quantization() -> None:
    m, n, k = 4, 256, 512
    inputs = torch.empty(m, k, dtype=torch.float16)
    layer = _make_dense_layer(n, k)
    expected = torch.zeros(m, n, dtype=torch.float16)
    esimd = Mock(side_effect=lambda _a, _b, _s, out, _bn, _bk: out.copy_(expected))

    with (
        patch.object(xpu, "_load_esimd_block_op", return_value=esimd),
        patch.object(Fp8BlockScaledMMLinearKernel, "apply_weights") as fallback,
    ):
        output = _make_dense_kernel(torch.float16).apply_weights(layer, inputs)

    torch.testing.assert_close(output, expected)
    esimd.assert_called_once()
    assert esimd.call_args.args[0].data_ptr() == inputs.data_ptr()
    assert esimd.call_args.args[1] is layer.weight
    assert esimd.call_args.args[2] is layer.weight_scale_inv
    assert esimd.call_args.args[4:] == (128, 128)
    fallback.assert_not_called()
