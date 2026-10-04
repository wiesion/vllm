# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy adapter for the optional sym-INT4 ESIMD execution ABI."""

import importlib
import os
from collections.abc import Callable
from types import ModuleType

import torch
import torch.nn.functional as F


def _int4_esimd_disabled() -> bool:
    values = (
        os.environ.get("DISABLE_ESIMD_INT4", "0"),
        os.environ.get("VLLM_XPU_DISABLE_ESIMD_INT4", "0"),
    )
    return any(
        value.strip().upper() in {"1", "ON", "TRUE", "YES", "Y"} for value in values
    )


def _load_esimd_module() -> ModuleType:
    return importlib.import_module("custom_esimd_kernels_vllm")


def _get_base_layer_for_no_lora(layer: object) -> object | None:
    base_layer = getattr(layer, "base_layer", None)
    if base_layer is None:
        return layer
    punica_wrapper = getattr(layer, "punica_wrapper", None)
    if not getattr(punica_wrapper, "no_lora", False):
        return None
    return base_layer


def _has_linear_abi_layout(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    scale: torch.Tensor | None,
) -> bool:
    if weight is None or scale is None:
        return False
    if input.ndim != 2 or weight.ndim != 2 or scale.ndim != 2:
        return False
    if input.device.type != "xpu":
        return False
    if input.device != weight.device or input.device != scale.device:
        return False
    if (
        input.dtype != torch.float16
        or weight.dtype != torch.uint8
        or scale.dtype != torch.float16
    ):
        return False
    if (
        not input.is_contiguous()
        or not weight.is_contiguous()
        or not scale.is_contiguous()
    ):
        return False

    rows, in_features = input.shape
    out_features, packed_in_features = weight.shape
    if rows == 0 or in_features == 0 or out_features == 0:
        return False
    if in_features % 128 != 0 or packed_in_features != in_features // 2:
        return False
    return scale.shape == (out_features, in_features // 128)


def _has_output_layout(
    output: torch.Tensor | None,
    input: torch.Tensor,
    out_features: int,
) -> bool:
    return output is None or (
        output.shape == (input.shape[0], out_features)
        and output.dtype == torch.float16
        and output.device == input.device
        and output.is_contiguous()
    )


class SymInt4EsimdAdapter:
    """Own optional ESIMD imports and translate their output-buffer ABI."""

    def __init__(
        self,
        module_loader: Callable[[], object] = _load_esimd_module,
    ) -> None:
        self._module_loader = module_loader
        self._module: object | None = None
        self._loaded = False

    def _get_module(self) -> object | None:
        if not self._loaded:
            self._loaded = True
            try:
                self._module = self._module_loader()
            except (AttributeError, ImportError, OSError, RuntimeError):
                self._module = None
        return self._module

    def try_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor | None,
        scale: torch.Tensor | None,
        *,
        bias: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        enabled: bool | None = None,
        allow_batch: bool = True,
    ) -> torch.Tensor | None:
        if enabled is None:
            enabled = not _int4_esimd_disabled()
        if not enabled:
            return None
        if getattr(torch.compiler, "is_compiling", lambda: False)():
            return None
        if bias is not None or not _has_linear_abi_layout(input, weight, scale):
            return None
        assert weight is not None
        if not _has_output_layout(output, input, weight.shape[0]):
            return None

        rows = input.shape[0]
        if rows == 1:
            op_name = "esimd_gemv_int4"
        elif allow_batch and 2 <= rows <= 64 and weight.shape[0] % 16 == 0:
            op_name = "esimd_gemm_int4_pgrp"
        else:
            return None
        module = self._get_module()
        op = getattr(module, op_name, None)
        if not callable(op):
            return None

        if output is None:
            output = torch.empty(
                (input.shape[0], weight.shape[0]),
                dtype=torch.float16,
                device=input.device,
            )
        op(input, weight, scale, output)
        return output

    def try_fused2(
        self,
        input: torch.Tensor,
        w0: torch.Tensor | None,
        s0: torch.Tensor | None,
        w1: torch.Tensor | None,
        s1: torch.Tensor | None,
        *,
        output0: torch.Tensor | None = None,
        output1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if _int4_esimd_disabled():
            return None
        if getattr(torch.compiler, "is_compiling", lambda: False)():
            return None
        if (
            input.shape[0] != 1
            or not _has_linear_abi_layout(input, w0, s0)
            or not _has_linear_abi_layout(input, w1, s1)
        ):
            return None
        assert w0 is not None and w1 is not None
        if not _has_output_layout(output0, input, w0.shape[0]) or not (
            _has_output_layout(output1, input, w1.shape[0])
        ):
            return None

        module = self._get_module()
        op = getattr(module, "esimd_gemv_int4_fused2", None)
        if not callable(op):
            return None

        if output0 is None:
            output0 = torch.empty(
                (1, w0.shape[0]), dtype=torch.float16, device=input.device
            )
        if output1 is None:
            output1 = torch.empty(
                (1, w1.shape[0]), dtype=torch.float16, device=input.device
            )
        op(input, w0, s0, output0, w1, s1, output1)
        return output0, output1


_ESIMD_INT4_ADAPTER = SymInt4EsimdAdapter()


def try_esimd_int4_linear(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    scale: torch.Tensor | None,
    *,
    output: torch.Tensor | None = None,
    allow_batch: bool = True,
) -> torch.Tensor | None:
    """Run the validated INT4 GEMV/GEMM ABI with guarded fallback."""
    return _ESIMD_INT4_ADAPTER.try_linear(
        input,
        weight,
        scale,
        output=output,
        allow_batch=allow_batch,
    )


def try_esimd_int4_fused2_layers(
    input: torch.Tensor,
    layer0: object,
    layer1: object,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse two sym-INT4 projections without bypassing linear semantics."""
    layer0 = _get_base_layer_for_no_lora(layer0)
    layer1 = _get_base_layer_for_no_lora(layer1)
    if layer0 is None or layer1 is None:
        return None
    method0 = getattr(layer0, "quant_method", None)
    method1 = getattr(layer1, "quant_method", None)
    if not (
        getattr(method0, "is_sym_int4", False)
        and getattr(method1, "is_sym_int4", False)
        and getattr(layer0, "bias", None) is None
        and getattr(layer1, "bias", None) is None
        and not getattr(layer0, "gather_output", False)
        and not getattr(layer1, "gather_output", False)
    ):
        return None

    padded0 = getattr(layer0, "_int4_in_features_padded", None)
    padded1 = getattr(layer1, "_int4_in_features_padded", None)
    if not isinstance(padded0, int) or padded0 != padded1:
        return None

    input_2d = input.reshape(-1, input.shape[-1])
    if input_2d.shape[-1] > padded0:
        return None
    if input_2d.shape[-1] < padded0:
        input_2d = F.pad(input_2d, (0, padded0 - input_2d.shape[-1]))

    buffers = getattr(layer0, "_esimd_int4_fused2_buffers", None)
    expected_shapes = (
        (1, layer0.weight_esimd.shape[0]),
        (1, layer1.weight_esimd.shape[0]),
    )
    if not (
        isinstance(buffers, tuple)
        and len(buffers) == 2
        and tuple(buffers[0].shape) == expected_shapes[0]
        and tuple(buffers[1].shape) == expected_shapes[1]
        and buffers[0].device == input_2d.device
        and buffers[1].device == input_2d.device
    ):
        buffers = (
            torch.empty(
                expected_shapes[0], dtype=torch.float16, device=input_2d.device
            ),
            torch.empty(
                expected_shapes[1], dtype=torch.float16, device=input_2d.device
            ),
        )
        layer0._esimd_int4_fused2_buffers = buffers

    outputs = _ESIMD_INT4_ADAPTER.try_fused2(
        input_2d,
        getattr(layer0, "weight_esimd", None),
        getattr(layer0, "scale_esimd", None),
        getattr(layer1, "weight_esimd", None),
        getattr(layer1, "scale_esimd", None),
        output0=buffers[0],
        output1=buffers[1],
    )
    if outputs is None:
        return None
    output0, output1 = outputs
    return (
        output0.reshape(input.shape[:-1] + (output0.shape[-1],)),
        output1.reshape(input.shape[:-1] + (output1.shape[-1],)),
    )
