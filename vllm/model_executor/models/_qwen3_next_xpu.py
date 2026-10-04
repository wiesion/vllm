# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded adapter for optional Qwen3Next XPU ESIMD post-processing."""

import importlib
import os
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

_DISABLE_ESIMD_QKV = os.environ.get("VLLM_XPU_DISABLE_ESIMD_QKV", "0") == "1"


def _load_esimd_module() -> object:
    return importlib.import_module("custom_esimd_kernels_vllm")


def _get_base_layer_for_no_lora(layer: object) -> object | None:
    base_layer = getattr(layer, "base_layer", None)
    if base_layer is None:
        return layer
    punica_wrapper = getattr(layer, "punica_wrapper", None)
    if not getattr(punica_wrapper, "no_lora", False):
        return None
    return base_layer


class Qwen3NextQkvPostprocessResult(NamedTuple):
    """Post-processed QKV tensors; ``gate`` has sigmoid already applied."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    gate: torch.Tensor | None


class _QkvPostprocessPlan(NamedTuple):
    op: Callable[..., object]
    q_size: int
    kv_size: int
    expected_width: int
    norm_wq: torch.Tensor
    norm_wk: torch.Tensor
    cos_sin_cache: torch.Tensor
    q_heads: int
    kv_heads: int
    rotary_dim: int
    attn_output_gate: bool


class _Fp8LinearPlan(NamedTuple):
    op: Callable[..., object]
    weight: torch.Tensor
    scale: torch.Tensor
    output: torch.Tensor
    block_size: tuple[int, int] | None


class _Int4LinearPlan(NamedTuple):
    op: Callable[..., object]
    weight: torch.Tensor
    scale: torch.Tensor
    output: torch.Tensor
    input_width: int


class Qwen3NextQkvEsimdAdapter:
    """Own the lazy optional import and the fused kernel's output-buffer ABI."""

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

    def try_postprocess(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        *,
        projection: Any,
        norm_wq: torch.Tensor,
        norm_wk: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        norm_epsilon: float,
        is_neox_style: bool,
        attn_output_gate: bool,
        text_only: bool,
    ) -> Qwen3NextQkvPostprocessResult | None:
        projection = _get_base_layer_for_no_lora(projection)
        if projection is None:
            return None
        plan = getattr(projection, "_esimd_qkv_postprocess_plan", None)
        if not isinstance(plan, _QkvPostprocessPlan):
            plan = None
        if plan is None and (
            _DISABLE_ESIMD_QKV
            or getattr(torch.compiler, "is_compiling", lambda: False)()
        ):
            return None

        rows = qkv.shape[0] if qkv.ndim == 2 else 0
        if positions.ndim == 1:
            plain_positions = positions
        elif text_only and positions.ndim == 2 and positions.shape == (3, rows):
            # V1 broadcasts text-only M-RoPE positions across all three rows.
            # Identical T/H/W coordinates reduce exactly to plain RoPE.
            plain_positions = positions[0]
        else:
            return None
        if not 1 <= rows <= 64 or plain_positions.shape[0] != rows:
            return None

        if plan is None:
            quant_method = getattr(projection, "quant_method", None)
            is_sym_int4 = getattr(quant_method, "is_sym_int4", False)
            weight = getattr(projection, "weight", None)
            weight_scale = getattr(projection, "weight_scale", None)
            weight_scale_inv = getattr(projection, "weight_scale_inv", None)
            weight_block_size = getattr(projection, "weight_block_size", None)
            is_runtime_fp8 = (
                isinstance(weight, torch.Tensor)
                and weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                and isinstance(weight_scale, torch.Tensor)
                and weight_scale.numel() == 1
            )
            is_block_fp8 = (
                isinstance(weight, torch.Tensor)
                and weight.dtype == torch.float8_e4m3fn
                and isinstance(weight_scale_inv, torch.Tensor)
                and weight_scale_inv.dtype == torch.float32
                and tuple(weight_block_size or ()) == (128, 128)
            )
            q_size = q_heads * head_dim
            kv_size = kv_heads * head_dim
            expected_width = q_size * (2 if attn_output_gate else 1) + 2 * kv_size
            tensors = (qkv, norm_wq, norm_wk, cos_sin_cache)
            if (
                not (is_sym_int4 or is_runtime_fp8 or is_block_fp8)
                or head_dim != 256
                or norm_epsilon != 1e-6
                or not is_neox_style
                or q_heads <= 0
                or kv_heads <= 0
                or rotary_dim <= 0
                or rotary_dim > head_dim
                or rotary_dim % 2 != 0
                or qkv.shape != (rows, expected_width)
                or plain_positions.dtype not in (torch.int32, torch.int64)
                or any(t.device.type != "xpu" for t in tensors)
                or any(t.device != qkv.device for t in (*tensors[1:], plain_positions))
                or any(t.dtype != torch.float16 for t in tensors)
                or norm_wq.shape != (head_dim,)
                or norm_wk.shape != (head_dim,)
                or cos_sin_cache.ndim != 2
                or cos_sin_cache.shape[0] == 0
                or cos_sin_cache.shape[1] != rotary_dim
                or any(not t.is_contiguous() for t in tensors)
            ):
                return None

            module = self._get_module()
            op = getattr(module, "esimd_qkv_split_norm_rope", None)
            if not callable(op):
                return None
            plan = _QkvPostprocessPlan(
                op=op,
                q_size=q_size,
                kv_size=kv_size,
                expected_width=expected_width,
                norm_wq=norm_wq,
                norm_wk=norm_wk,
                cos_sin_cache=cos_sin_cache,
                q_heads=q_heads,
                kv_heads=kv_heads,
                rotary_dim=rotary_dim,
                attn_output_gate=attn_output_gate,
            )
            projection._esimd_qkv_postprocess_plan = plan
        elif qkv.shape != (rows, plan.expected_width):
            return None

        buffers = getattr(projection, "_esimd_qkv_decode_buffers", None)
        if buffers is None or buffers[0].shape[0] < rows:
            capacity = 1 if rows == 1 else 64
            buffers = (
                torch.empty(
                    (capacity, plan.q_size),
                    dtype=torch.float16,
                    device=qkv.device,
                ),
                torch.empty(
                    (capacity, plan.kv_size),
                    dtype=torch.float16,
                    device=qkv.device,
                ),
                torch.empty(
                    (capacity, plan.kv_size),
                    dtype=torch.float16,
                    device=qkv.device,
                ),
                torch.empty(
                    (capacity, plan.q_size),
                    dtype=torch.float16,
                    device=qkv.device,
                ),
            )
            projection._esimd_qkv_decode_buffers = buffers
        q, k, v, gate_buffer = (buffer[:rows] for buffer in buffers)
        gate = gate_buffer if plan.attn_output_gate else None
        gate_out = gate
        if gate_out is None:
            gate_out = gate_buffer
        positions_i32 = plain_positions.to(dtype=torch.int32).contiguous()

        try:
            plan.op(
                qkv,
                q,
                gate_out,
                k,
                v,
                plan.norm_wq,
                plan.norm_wk,
                positions_i32,
                plan.q_heads,
                plan.kv_heads,
                plan.attn_output_gate,
                plan.rotary_dim,
                plan.cos_sin_cache,
            )
        except (AttributeError, RuntimeError, TypeError):
            return None
        return Qwen3NextQkvPostprocessResult(q, k, v, gate)


_QWEN3_NEXT_QKV_ESIMD_ADAPTER = Qwen3NextQkvEsimdAdapter()


def try_esimd_qkv_postprocess(*args, **kwargs) -> Qwen3NextQkvPostprocessResult | None:
    return _QWEN3_NEXT_QKV_ESIMD_ADAPTER.try_postprocess(*args, **kwargs)


def try_esimd_fp8_linear(
    x: torch.Tensor,
    projection: Any,
    buffer_name: str,
) -> torch.Tensor | None:
    projection = _get_base_layer_for_no_lora(projection)
    if projection is None:
        return None
    plan_name = f"{buffer_name}_plan"
    plan = getattr(projection, plan_name, None)
    if not isinstance(plan, _Fp8LinearPlan):
        plan = None
    if x.ndim != 2 or x.shape[0] != 1:
        return None
    if plan is None:
        if (
            _DISABLE_ESIMD_QKV
            or getattr(torch.compiler, "is_compiling", lambda: False)()
            or x.device.type != "xpu"
            or x.dtype != torch.float16
            or not x.is_contiguous()
        ):
            return None
        weight = getattr(projection, "weight", None)
        scale = getattr(projection, "weight_scale", None)
        block_scale = getattr(projection, "weight_scale_inv", None)
        block_size = tuple(getattr(projection, "weight_block_size", ()) or ())
        is_scalar_scale = isinstance(scale, torch.Tensor) and scale.numel() == 1
        is_block_scale = (
            isinstance(block_scale, torch.Tensor)
            and block_scale.dtype == torch.float32
            and block_size == (128, 128)
        )
        if (
            getattr(projection, "bias", None) is not None
            or not isinstance(weight, torch.Tensor)
            or weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
            or not weight.is_contiguous()
            or not (is_scalar_scale or is_block_scale)
        ):
            return None
        module = _QWEN3_NEXT_QKV_ESIMD_ADAPTER._get_module()
        if is_block_scale:
            if weight.dtype != torch.float8_e4m3fn or not isinstance(
                block_scale, torch.Tensor
            ):
                return None
            scale = block_scale
            expected_scale_shape = (
                (weight.shape[0] + block_size[0] - 1) // block_size[0],
                (weight.shape[1] + block_size[1] - 1) // block_size[1],
            )
            if (
                tuple(scale.shape) != expected_scale_shape
                or not scale.is_contiguous()
                or scale.device != x.device
                or weight.shape[1] % block_size[1] != 0
            ):
                return None
            op = getattr(module, "esimd_gemm_fp8_blockscale", None)
        else:
            if not isinstance(scale, torch.Tensor):
                return None
            block_size = ()
            op = getattr(module, "esimd_gemv_fp8_pert", None)
        if not callable(op):
            return None
        output = torch.empty((1, weight.shape[0]), dtype=torch.float16, device=x.device)
        setattr(projection, buffer_name, output)
        plan = _Fp8LinearPlan(
            op=op,
            weight=weight,
            scale=scale,
            output=output,
            block_size=(128, 128) if is_block_scale else None,
        )
        setattr(projection, plan_name, plan)
    if plan.block_size is None:
        plan.op(x, plan.weight, plan.scale, plan.output)
    else:
        plan.op(
            x,
            plan.weight,
            plan.scale,
            plan.output,
            plan.block_size[0],
            plan.block_size[1],
        )
    return plan.output


def try_esimd_int4_linear(
    x: torch.Tensor,
    projection: Any,
    buffer_name: str,
) -> torch.Tensor | None:
    """Run the cached M=1 sym-INT4 projection used by decode attention."""
    projection = _get_base_layer_for_no_lora(projection)
    if projection is None:
        return None
    plan_name = f"{buffer_name}_plan"
    plan = getattr(projection, plan_name, None)
    if not isinstance(plan, _Int4LinearPlan):
        plan = None
    if x.ndim != 2 or x.shape[0] != 1:
        return None
    if plan is None:
        int4_disabled = any(
            os.environ.get(name, "0").strip().upper() in {"1", "ON", "TRUE", "YES", "Y"}
            for name in ("DISABLE_ESIMD_INT4", "VLLM_XPU_DISABLE_ESIMD_INT4")
        )
        if (
            int4_disabled
            or getattr(torch.compiler, "is_compiling", lambda: False)()
            or x.device.type != "xpu"
            or x.dtype != torch.float16
            or not x.is_contiguous()
            or getattr(projection, "bias", None) is not None
            or not getattr(
                getattr(projection, "quant_method", None), "is_sym_int4", False
            )
        ):
            return None
        weight = getattr(projection, "weight_esimd", None)
        scale = getattr(projection, "scale_esimd", None)
        input_width = x.shape[1]
        if (
            not isinstance(weight, torch.Tensor)
            or not isinstance(scale, torch.Tensor)
            or weight.dtype != torch.uint8
            or scale.dtype != torch.float16
            or weight.device != x.device
            or scale.device != x.device
            or not weight.is_contiguous()
            or not scale.is_contiguous()
            or input_width % 128 != 0
            or weight.shape[1] * 2 != input_width
            or scale.shape != (weight.shape[0], input_width // 128)
        ):
            return None
        module = _QWEN3_NEXT_QKV_ESIMD_ADAPTER._get_module()
        op = getattr(module, "esimd_gemv_int4", None)
        if not callable(op):
            return None
        output = torch.empty((1, weight.shape[0]), dtype=torch.float16, device=x.device)
        setattr(projection, buffer_name, output)
        plan = _Int4LinearPlan(
            op=op,
            weight=weight,
            scale=scale,
            output=output,
            input_width=input_width,
        )
        setattr(projection, plan_name, plan)
    elif (
        x.shape[1] != plan.input_width
        or x.dtype != torch.float16
        or x.device != plan.weight.device
        or not x.is_contiguous()
    ):
        return None
    plan.op(x, plan.weight, plan.scale, plan.output)
    return plan.output
