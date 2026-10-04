# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.model_executor.layers.quantization import _sym_int4_esimd

SymInt4EsimdAdapter = _sym_int4_esimd.SymInt4EsimdAdapter


def test_single_token_linear_lazily_uses_esimd_gemv():
    loads = []
    calls = []

    def gemv(input, weight, scale, output):
        calls.append((input, weight, scale, output))
        return output

    module = SimpleNamespace(esimd_gemv_int4=gemv)

    def load_module():
        loads.append(module)
        return module

    adapter = SymInt4EsimdAdapter(load_module)
    with FakeTensorMode():
        inputs = torch.empty((1, 128), dtype=torch.float16, device="xpu")
        weight = torch.empty((16, 64), dtype=torch.uint8, device="xpu")
        scale = torch.empty((16, 1), dtype=torch.float16, device="xpu")

        output = adapter.try_linear(inputs, weight, scale, enabled=True)
        second_output = adapter.try_linear(inputs, weight, scale, enabled=True)

    assert output is calls[0][-1]
    assert output.shape == (1, 16)
    assert output.dtype == torch.float16
    assert second_output is calls[1][-1]
    assert len(loads) == 1


def test_small_batch_linear_uses_esimd_gemm_at_m64():
    calls = []

    def gemm(input, weight, scale, output):
        calls.append((input, weight, scale, output))
        return output

    adapter = SymInt4EsimdAdapter(lambda: SimpleNamespace(esimd_gemm_int4_pgrp=gemm))
    with FakeTensorMode():
        inputs = torch.empty((64, 128), dtype=torch.float16, device="xpu")
        weight = torch.empty((16, 64), dtype=torch.uint8, device="xpu")
        scale = torch.empty((16, 1), dtype=torch.float16, device="xpu")

        output = adapter.try_linear(inputs, weight, scale, enabled=True)

    assert output is calls[0][-1]
    assert output.shape == (64, 16)


def test_linear_default_gate_is_enabled_and_uses_available_op():
    calls = []
    adapter = SymInt4EsimdAdapter(
        lambda: SimpleNamespace(esimd_gemv_int4=lambda *args: calls.append(args))
    )
    with FakeTensorMode():
        inputs = torch.empty((1, 128), dtype=torch.float16, device="xpu")
        weight = torch.empty((16, 64), dtype=torch.uint8, device="xpu")
        scale = torch.empty((16, 1), dtype=torch.float16, device="xpu")

        output = adapter.try_linear(inputs, weight, scale)

    assert output is not None
    assert calls


@pytest.mark.parametrize(
    "env_name", ["DISABLE_ESIMD_INT4", "VLLM_XPU_DISABLE_ESIMD_INT4"]
)
def test_linear_disable_switch_falls_back_without_loading_ops(env_name):
    loads = []
    adapter = SymInt4EsimdAdapter(lambda: loads.append(object()))
    with (
        FakeTensorMode(),
        patch.dict(_sym_int4_esimd.os.environ, {env_name: "1"}, clear=False),
    ):
        inputs = torch.empty((1, 128), dtype=torch.float16, device="xpu")
        weight = torch.empty((16, 64), dtype=torch.uint8, device="xpu")
        scale = torch.empty((16, 1), dtype=torch.float16, device="xpu")

        assert adapter.try_linear(inputs, weight, scale) is None

    assert loads == []


@pytest.mark.parametrize(("rows", "out_features"), [(65, 16), (2, 15)])
def test_ineligible_batch_shape_falls_back_without_loading_ops(rows, out_features):
    loads = []
    adapter = SymInt4EsimdAdapter(lambda: loads.append(object()))

    with FakeTensorMode():
        inputs = torch.empty((rows, 128), dtype=torch.float16, device="xpu")
        weight = torch.empty((out_features, 64), dtype=torch.uint8, device="xpu")
        scale = torch.empty((out_features, 1), dtype=torch.float16, device="xpu")

        assert adapter.try_linear(inputs, weight, scale, enabled=True) is None

    assert loads == []


def test_missing_or_empty_weight_falls_back_without_loading_ops():
    loads = []
    adapter = SymInt4EsimdAdapter(lambda: loads.append(object()))

    with FakeTensorMode():
        inputs = torch.empty((1, 128), dtype=torch.float16, device="xpu")
        scale = torch.empty((16, 1), dtype=torch.float16, device="xpu")
        empty_weight = torch.empty((0, 64), dtype=torch.uint8, device="xpu")

        assert adapter.try_linear(inputs, None, scale, enabled=True) is None
        assert adapter.try_linear(inputs, empty_weight, scale, enabled=True) is None

    assert loads == []


def test_unavailable_esimd_package_falls_back_and_caches_failure():
    loads = []

    def unavailable():
        loads.append(None)
        raise ImportError("optional ESIMD package is absent")

    adapter = SymInt4EsimdAdapter(unavailable)
    with FakeTensorMode():
        inputs = torch.empty((1, 128), dtype=torch.float16, device="xpu")
        weight = torch.empty((16, 64), dtype=torch.uint8, device="xpu")
        scale = torch.empty((16, 1), dtype=torch.float16, device="xpu")

        assert adapter.try_linear(inputs, weight, scale, enabled=True) is None
        assert adapter.try_linear(inputs, weight, scale, enabled=True) is None

    assert len(loads) == 1


@pytest.mark.parametrize("error", [RuntimeError("bad ABI"), AttributeError("bad op")])
def test_broken_extension_load_falls_back_and_caches_failure(error):
    loads = []

    def unavailable():
        loads.append(None)
        raise error

    adapter = SymInt4EsimdAdapter(unavailable)
    with FakeTensorMode():
        inputs = torch.empty((1, 128), dtype=torch.float16, device="xpu")
        weight = torch.empty((16, 64), dtype=torch.uint8, device="xpu")
        scale = torch.empty((16, 1), dtype=torch.float16, device="xpu")

        assert adapter.try_linear(inputs, weight, scale, enabled=True) is None
        assert adapter.try_linear(inputs, weight, scale, enabled=True) is None

    assert len(loads) == 1


def test_compiling_falls_back_without_loading_ops():
    loads = []
    adapter = SymInt4EsimdAdapter(lambda: loads.append(object()))
    with (
        FakeTensorMode(),
        patch.object(torch.compiler, "is_compiling", return_value=True),
        patch.dict(
            _sym_int4_esimd.os.environ,
            {"ENABLE_ESIMD_INT4_LINEAR": "1"},
        ),
    ):
        inputs = torch.empty((1, 128), dtype=torch.float16, device="xpu")
        weight = torch.empty((16, 64), dtype=torch.uint8, device="xpu")
        scale = torch.empty((16, 1), dtype=torch.float16, device="xpu")

        assert adapter.try_linear(inputs, weight, scale, enabled=True) is None
        assert adapter.try_fused2(inputs, weight, scale, weight, scale) is None

    assert loads == []


def test_invalid_linear_abi_falls_back_without_loading_ops():
    with FakeTensorMode():
        valid_input = torch.empty((2, 128), dtype=torch.float16, device="xpu:0")
        valid_weight = torch.empty((16, 64), dtype=torch.uint8, device="xpu:0")
        valid_scale = torch.empty((16, 1), dtype=torch.float16, device="xpu:0")
        invalid_cases = [
            (
                torch.empty((2, 128), dtype=torch.float16, device="cpu"),
                valid_weight,
                valid_scale,
            ),
            (
                torch.empty((2, 128), dtype=torch.bfloat16, device="xpu:0"),
                valid_weight,
                valid_scale,
            ),
            (valid_input.t(), valid_weight, valid_scale),
            (
                valid_input,
                torch.empty((16, 64), dtype=torch.int32, device="xpu:0"),
                valid_scale,
            ),
            (
                valid_input,
                torch.empty((16, 63), dtype=torch.uint8, device="xpu:0"),
                valid_scale,
            ),
            (
                valid_input,
                valid_weight,
                torch.empty((16, 1), dtype=torch.float32, device="xpu:0"),
            ),
            (
                valid_input,
                valid_weight,
                torch.empty((16, 2), dtype=torch.float16, device="xpu:0"),
            ),
            (
                valid_input,
                torch.empty((16, 64), dtype=torch.uint8, device="xpu:1"),
                valid_scale,
            ),
        ]

        for inputs, weight, scale in invalid_cases:
            loads = []
            adapter = SymInt4EsimdAdapter(lambda loads=loads: loads.append(object()))
            assert adapter.try_linear(inputs, weight, scale, enabled=True) is None
            assert loads == []


def test_single_token_fused2_reuses_caller_output_buffers():
    calls = []

    def fused2(input, w0, s0, o0, w1, s1, o1):
        calls.append((input, w0, s0, o0, w1, s1, o1))
        return o0

    adapter = SymInt4EsimdAdapter(
        lambda: SimpleNamespace(esimd_gemv_int4_fused2=fused2)
    )
    with (
        FakeTensorMode(),
        patch.dict(
            _sym_int4_esimd.os.environ,
            {"ENABLE_ESIMD_INT4_LINEAR": "1"},
        ),
    ):
        inputs = torch.empty((1, 128), dtype=torch.float16, device="xpu")
        w0 = torch.empty((16, 64), dtype=torch.uint8, device="xpu")
        s0 = torch.empty((16, 1), dtype=torch.float16, device="xpu")
        w1 = torch.empty((24, 64), dtype=torch.uint8, device="xpu")
        s1 = torch.empty((24, 1), dtype=torch.float16, device="xpu")
        o0 = torch.empty((1, 16), dtype=torch.float16, device="xpu")
        o1 = torch.empty((1, 24), dtype=torch.float16, device="xpu")

        output = adapter.try_fused2(inputs, w0, s0, w1, s1, output0=o0, output1=o1)

    assert output == (o0, o1)
    assert calls == [(inputs, w0, s0, o0, w1, s1, o1)]


def test_invalid_caller_output_buffer_falls_back_before_loading_ops():
    loads = []
    adapter = SymInt4EsimdAdapter(lambda: loads.append(object()))

    with FakeTensorMode():
        inputs = torch.empty((1, 128), dtype=torch.float16, device="xpu")
        weight = torch.empty((16, 64), dtype=torch.uint8, device="xpu")
        scale = torch.empty((16, 1), dtype=torch.float16, device="xpu")
        wrong_output = torch.empty((1, 15), dtype=torch.float16, device="xpu")

        assert (
            adapter.try_linear(
                inputs,
                weight,
                scale,
                output=wrong_output,
                enabled=True,
            )
            is None
        )

    assert loads == []


def test_fused2_layer_helper_preserves_shapes_and_uses_one_launch():
    calls = []

    def fused2(input, w0, s0, o0, w1, s1, o1):
        calls.append((input, w0, s0, o0, w1, s1, o1))

    adapter = SymInt4EsimdAdapter(
        lambda: SimpleNamespace(esimd_gemv_int4_fused2=fused2)
    )
    method = SimpleNamespace(is_sym_int4=True)
    with (
        FakeTensorMode(),
        patch.object(_sym_int4_esimd, "_ESIMD_INT4_ADAPTER", adapter),
        patch.dict(
            _sym_int4_esimd.os.environ,
            {"ENABLE_ESIMD_INT4_LINEAR": "1"},
        ),
    ):
        inputs = torch.empty((1, 120), dtype=torch.float16, device="xpu")
        layer0 = SimpleNamespace(
            quant_method=method,
            bias=None,
            gather_output=False,
            _int4_in_features_padded=128,
            weight_esimd=torch.empty((16, 64), dtype=torch.uint8, device="xpu"),
            scale_esimd=torch.empty((16, 1), dtype=torch.float16, device="xpu"),
        )
        layer1 = SimpleNamespace(
            quant_method=method,
            bias=None,
            gather_output=False,
            _int4_in_features_padded=128,
            weight_esimd=torch.empty((24, 64), dtype=torch.uint8, device="xpu"),
            scale_esimd=torch.empty((24, 1), dtype=torch.float16, device="xpu"),
        )
        outputs = _sym_int4_esimd.try_esimd_int4_fused2_layers(inputs, layer0, layer1)

    assert outputs is not None
    assert outputs[0].shape == (1, 16)
    assert outputs[1].shape == (1, 24)
    assert calls[0][0].shape == (1, 128)


@pytest.mark.parametrize(
    ("active_lora", "expected"),
    [(False, True), (True, False)],
    ids=["inactive-lora", "active-lora"],
)
def test_fused2_layer_helper_handles_lora_wrapper(active_lora, expected):
    calls = []
    adapter = SymInt4EsimdAdapter(
        lambda: SimpleNamespace(esimd_gemv_int4_fused2=lambda *args: calls.append(args))
    )
    method = SimpleNamespace(is_sym_int4=True)

    with (
        FakeTensorMode(),
        patch.object(_sym_int4_esimd, "_ESIMD_INT4_ADAPTER", adapter),
        patch.dict(
            _sym_int4_esimd.os.environ,
            {"ENABLE_ESIMD_INT4_LINEAR": "1"},
        ),
    ):
        inputs = torch.empty((1, 128), dtype=torch.float16, device="xpu")
        base_layers = [
            SimpleNamespace(
                quant_method=method,
                bias=None,
                gather_output=False,
                _int4_in_features_padded=128,
                weight_esimd=torch.empty((size, 64), dtype=torch.uint8, device="xpu"),
                scale_esimd=torch.empty((size, 1), dtype=torch.float16, device="xpu"),
            )
            for size in (16, 24)
        ]
        layers = [
            SimpleNamespace(
                base_layer=base_layer,
                punica_wrapper=SimpleNamespace(no_lora=not active_lora),
                is_lora_active=Mock(return_value=active_lora),
            )
            for base_layer in base_layers
        ]
        outputs = _sym_int4_esimd.try_esimd_int4_fused2_layers(inputs, *layers)

    assert (outputs is not None) is expected
    assert bool(calls) is expected
    for layer in layers:
        layer.is_lora_active.assert_not_called()
