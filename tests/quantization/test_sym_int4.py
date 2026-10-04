# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization import get_quantization_config, sym_int4
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info


@pytest.fixture(autouse=True)
def _pretend_xpu_platform(monkeypatch):
    monkeypatch.setattr(
        sym_int4, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )


def test_sym_int4_is_registered():
    assert get_quantization_config("sym_int4") is sym_int4.SymInt4Config


def test_sym_int4_config_rejects_non_xpu(monkeypatch):
    monkeypatch.setattr(
        sym_int4, "current_platform", SimpleNamespace(is_xpu=lambda: False)
    )
    with pytest.raises(ValueError, match="only supported on XPU"):
        sym_int4.SymInt4Config()


@pytest.mark.parametrize(
    ("backend", "available", "expected", "expected_probes"),
    [
        ("cpu", True, False, 0),
        ("xpu_esimd", True, True, 1),
        ("xpu_esimd", False, False, 1),
    ],
)
def test_weight_quant_backend_selection(
    monkeypatch,
    backend,
    available,
    expected,
    expected_probes,
):
    probes: list[None] = []

    def fake_available():
        probes.append(None)
        return available

    monkeypatch.setattr(
        sym_int4.envs,
        "VLLM_SYM_INT4_WEIGHT_QUANT_BACKEND",
        backend,
    )
    monkeypatch.setattr(sym_int4, "_esimd_q40_available", fake_available)

    assert sym_int4._use_xpu_esimd_q40() is expected
    assert len(probes) == expected_probes


def test_linear_layout_reuses_packed_weight_storage():
    layer = nn.Module()
    qweight = torch.zeros((4, 16), dtype=torch.int32)
    scales = torch.zeros((4, 2), dtype=torch.float16)

    sym_int4._register_linear_int4_layouts(layer, qweight, scales)

    assert layer.weight_esimd.shape == (4, 64)
    assert layer.weight_esimd.dtype == torch.uint8
    assert layer.weight_esimd.is_contiguous()
    assert layer.weight_esimd.untyped_storage().data_ptr() == (
        qweight.untyped_storage().data_ptr()
    )
    assert layer.scale_esimd.untyped_storage().data_ptr() == (
        scales.untyped_storage().data_ptr()
    )
    assert layer.qweight.shape == (16, 4)
    assert layer.qweight.stride() == (1, 16)
    assert layer.qweight.untyped_storage().data_ptr() == (
        qweight.untyped_storage().data_ptr()
    )
    assert layer.scales.shape == (2, 4)
    assert layer.scales.is_contiguous()
    assert layer.scales.untyped_storage().data_ptr() != (
        scales.untyped_storage().data_ptr()
    )
    assert layer.qzeros.tolist() == [8]


def test_linear_apply_uses_guarded_esimd_before_native(monkeypatch):
    observed: dict[str, Any] = {}

    class FakeAdapter:
        def try_linear(self, inputs, weight, scale, *, bias, output, allow_batch):
            observed["args"] = (
                inputs,
                weight,
                scale,
                bias,
                output,
                allow_batch,
            )
            return torch.full(
                (inputs.shape[0], weight.shape[0]), 3.0, dtype=inputs.dtype
            )

    monkeypatch.setattr(sym_int4, "_ESIMD_INT4_ADAPTER", FakeAdapter())
    layer = nn.Module()
    layer._int4_in_features_padded = 128
    layer.weight_esimd = nn.Parameter(
        torch.empty((16, 64), dtype=torch.uint8), requires_grad=False
    )
    layer.scale_esimd = nn.Parameter(
        torch.empty((16, 1), dtype=torch.float16), requires_grad=False
    )
    layer._sym_int4_allow_batch_esimd = True
    method = sym_int4.SymInt4LinearMethod(sym_int4.SymInt4Config())
    inputs = torch.ones((2, 128), dtype=torch.float16)

    output = method.apply(layer, inputs)

    assert output.shape == (2, 16)
    assert output.tolist() == [[3.0] * 16] * 2
    assert observed["args"][0].untyped_storage().data_ptr() == (
        inputs.untyped_storage().data_ptr()
    )
    assert observed["args"][1] is layer.weight_esimd
    assert observed["args"][2] is layer.scale_esimd
    assert observed["args"][3] is None
    assert observed["args"][4] is None
    assert observed["args"][5] is True


def test_linear_apply_disables_batch_esimd_by_default(monkeypatch):
    observed: dict[str, Any] = {}

    class FakeAdapter:
        def try_linear(self, inputs, weight, scale, *, bias, output, allow_batch):
            observed["output"] = output
            observed["allow_batch"] = allow_batch
            return torch.empty((inputs.shape[0], weight.shape[0]), dtype=inputs.dtype)

    monkeypatch.setattr(sym_int4, "_ESIMD_INT4_ADAPTER", FakeAdapter())
    layer = nn.Module()
    layer._int4_in_features_padded = 128
    layer.weight_esimd = nn.Parameter(
        torch.empty((16, 64), dtype=torch.uint8), requires_grad=False
    )
    layer.scale_esimd = nn.Parameter(
        torch.empty((16, 1), dtype=torch.float16), requires_grad=False
    )
    method = sym_int4.SymInt4LinearMethod(sym_int4.SymInt4Config())

    method.apply(layer, torch.ones((2, 128), dtype=torch.float16))

    assert observed == {"output": None, "allow_batch": False}


def test_linear_post_load_quantization_zero_pads_k(monkeypatch):
    observed: dict[str, Any] = {}

    def fake_quantize(weight):
        observed["calls"] = observed.get("calls", 0) + 1
        observed["weight"] = weight.clone()
        rows, cols = weight.shape
        return (
            torch.zeros((rows, cols // 8), dtype=torch.int32),
            torch.ones((rows, cols // 128), dtype=torch.float16),
        )

    monkeypatch.setattr(sym_int4, "_runtime_quantize_q4_0", fake_quantize)
    layer = nn.Module()
    layer.register_parameter(
        "weight",
        nn.Parameter(torch.ones((3, 130), dtype=torch.bfloat16)),
    )
    method = sym_int4.SymInt4LinearMethod(sym_int4.SymInt4Config())

    method.process_weights_after_loading(layer)

    assert observed["weight"].shape == (3, 256)
    assert torch.count_nonzero(observed["weight"][:, 130:]) == 0
    assert layer.weight is None
    assert layer.qweight.shape == (32, 3)
    assert layer.scales.shape == (2, 3)
    assert layer._int4_in_features == 130
    assert layer._int4_in_features_padded == 256

    method.process_weights_after_loading(layer)
    assert observed["calls"] == 1


def test_linear_post_load_rejects_unmaterialized_weight():
    layer = nn.Module()
    layer.register_parameter(
        "weight",
        nn.Parameter(torch.empty((3, 128), dtype=torch.bfloat16, device="meta")),
    )
    method = sym_int4.SymInt4LinearMethod(sym_int4.SymInt4Config())

    with pytest.raises(RuntimeError, match="was not materialized"):
        method.process_weights_after_loading(layer)


def test_linear_post_load_rejects_partial_weight():
    layer = nn.Module()
    layer.register_parameter(
        "weight",
        nn.Parameter(torch.ones((3, 128), dtype=torch.bfloat16)),
    )
    info = get_layerwise_info(layer)
    info.load_numel_total = layer.weight.numel()
    info.load_numel = info.load_numel_total - 1
    method = sym_int4.SymInt4LinearMethod(sym_int4.SymInt4Config())

    with pytest.raises(RuntimeError, match="complete checkpoint load"):
        method.process_weights_after_loading(layer)


def test_moe_quantization_pads_each_gate_up_half_and_down_k(monkeypatch):
    observed = []

    def fake_quantize(weight):
        observed.append(weight.clone())
        rows, cols = weight.shape
        return (
            torch.zeros((rows, cols // 8), dtype=torch.int32),
            torch.ones((rows, cols // 128), dtype=torch.float16),
        )

    monkeypatch.setattr(sym_int4, "_runtime_quantize_q4_0", fake_quantize)
    w13 = torch.ones((2, 6, 128), dtype=torch.bfloat16)
    w2 = torch.ones((2, 128, 3), dtype=torch.bfloat16)

    q13, s13, padded_i13 = sym_int4._quantize_moe_w13(w13)
    q2, s2, padded_i2 = sym_int4._quantize_moe_w2(w2)

    padded_w13 = observed[0].view(2, 2, 128, 128)
    assert torch.count_nonzero(padded_w13[:, :, 3:]) == 0
    assert torch.count_nonzero(observed[1].view(2, 128, 128)[:, :, 3:]) == 0
    assert padded_i13 == padded_i2 == 128
    assert q13.shape == (2, 256, 64)
    assert s13.shape == (2, 256, 1)
    assert q2.shape == (2, 128, 64)
    assert s2.shape == (2, 128, 1)


def test_moe_w13_bias_padding_matches_each_weight_half():
    bias = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float16)

    padded = sym_int4._pad_moe_w13_bias(bias, 128).view(1, 2, 128)

    torch.testing.assert_close(padded[0, 0, :3], bias[0, :3])
    torch.testing.assert_close(padded[0, 1, :3], bias[0, 3:])
    assert torch.count_nonzero(padded[:, :, 3:]) == 0


def test_moe_setup_uses_current_wna16_modular_oracle(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        XPUExpertsWNA16,
    )
    from vllm.model_executor.layers.fused_moe.oracle import int_wna16

    kernel = object()
    observed: dict[str, Any] = {}

    def fake_select(moe_config, weight_key):
        observed["select"] = (moe_config, weight_key)
        return "XPU", XPUExpertsWNA16

    def fake_make(**kwargs):
        observed["make"] = kwargs
        return kernel

    monkeypatch.setattr(int_wna16, "select_wna16_moe_backend", fake_select)
    monkeypatch.setattr(int_wna16, "make_wna16_moe_kernel", fake_make)
    moe_config = SimpleNamespace(has_bias=False)
    method = sym_int4.SymInt4MoEMethod(sym_int4.SymInt4Config(), moe_config)
    layer = nn.Module()
    layer.w13_scales = nn.Parameter(torch.ones((2, 4, 1), dtype=torch.float16))
    layer.w2_scales = nn.Parameter(torch.ones((2, 4, 1), dtype=torch.float16))
    layer._expert_routing_tables = lambda: ("g2p", "p2g", "local")

    method._setup_moe_kernel(layer)

    assert method.moe_kernel is kernel
    assert observed["select"][0] is moe_config
    assert observed["make"]["backend"] == "XPU"
    assert observed["make"]["experts_cls"] is XPUExpertsWNA16
    assert observed["make"]["routing_tables"] == ("g2p", "p2g", "local")
