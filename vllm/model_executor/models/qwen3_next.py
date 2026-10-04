# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

import importlib
import os
from collections.abc import Callable, Iterable
from itertools import islice
from typing import NamedTuple, cast

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    MoERunner,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.v1.attention.backend import AttentionType

from ._qwen3_next_xpu import (
    _get_base_layer_for_no_lora,
    try_esimd_fp8_linear,
    try_esimd_int4_linear,
    try_esimd_qkv_postprocess,
)
from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)


def _get_router_quant_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    if quant_config is not None and quant_config.get_name() == "sym_int4":
        return quant_config
    return None


def _enable_sym_int4_batch_esimd(*layers: object) -> None:
    for layer in layers:
        layer._sym_int4_allow_batch_esimd = True


KVCache = tuple[torch.Tensor, torch.Tensor]
_QWEN3_NEXT_PLAIN_MROPE = "_qwen3_next_plain_mrope"
_DISABLE_ESIMD_MOE_ROUTER = os.getenv("VLLM_XPU_DISABLE_ESIMD_MOE_ROUTER", "0") == "1"
_ESIMD_MOE_ROUTER_MAX_TOKENS = 17


class _EsimdMoePlan(NamedTuple):
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    shared_w13: torch.Tensor
    shared_w13_scale: torch.Tensor
    shared_w2: torch.Tensor
    shared_w2_scale: torch.Tensor
    shared_expert_gate_weight: torch.Tensor
    experts_per_token: int
    is_block: bool


class _EsimdInt4MoePlan(NamedTuple):
    full_op: Callable[..., torch.Tensor]
    tiny_op: Callable[..., torch.Tensor]
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    shared_w13: torch.Tensor
    shared_w2: torch.Tensor
    shared_expert_gate_weight: torch.Tensor
    experts_per_token: int


class _EsimdRouterPlan(NamedTuple):
    op: Callable[..., object]
    weight: torch.Tensor
    weight_scale: torch.Tensor | None


class _EsimdMoeSources(NamedTuple):
    experts: MoERunner
    routed_experts: RoutedExperts
    gate: ReplicatedLinear
    shared_expert_gate: ReplicatedLinear
    shared_gate_up_proj: MergedColumnParallelLinear
    shared_down_proj: RowParallelLinear


def _load_xpu_esimd_moe_ops():
    if not current_platform.is_xpu():
        return None
    try:
        return importlib.import_module("custom_esimd_kernels_vllm")
    except (ImportError, OSError):
        return None


def _runtime_plain_mrope_is_qkv_eligible() -> bool:
    if getattr(torch.compiler, "is_compiling", lambda: False)():
        return False
    if not is_forward_context_available():
        return False
    return get_forward_context().additional_kwargs.get(_QWEN3_NEXT_PLAIN_MROPE) is True


def _is_shared_expert_fse_compatible(quant_config) -> bool:
    """Check if shared expert can be fused with routed experts.

    FSE requires that shared and routed expert weights use the same
    quantization format. Returns False when the shared expert is
    excluded from quantization (e.g. float32 shared in an MXFP4 model)
    or has a different quant spec than routed experts.
    """
    if quant_config is None:
        return True
    # Quark stores its full config dict in quant_config.quant_config
    raw_config = getattr(quant_config, "quant_config", None)
    if not isinstance(raw_config, dict):
        return True
    exclude = raw_config.get("exclude", [])
    if not exclude:
        return True
    return not any("shared_expert." in str(e) for e in exclude)


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=_get_router_quant_config(quant_config),
            prefix=f"{prefix}.gate",
        )
        _enable_sym_int4_batch_esimd(self.gate)

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        _fse_requested = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        _fse_enabled = _fse_requested and _is_shared_expert_fse_compatible(quant_config)
        if _fse_requested and not _fse_enabled:
            logger.warning(
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but "
                "shared expert has a different quantization spec than routed "
                "experts. Falling back to non-fused shared expert path."
            )
        if _fse_enabled or config.shared_expert_intermediate_size <= 0:
            self.shared_expert = None
        else:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_expert",
            )
            _enable_sym_int4_batch_esimd(
                self.shared_expert.gate_up_proj,
                self.shared_expert.down_proj,
            )

        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=1 if self.shared_expert is None else None,
            shared_expert_gate=self.shared_expert_gate
            if self.shared_expert is None
            else None,
        )
        self._esimd_moe_ops = _load_xpu_esimd_moe_ops()
        speculative_config = vllm_config.speculative_config
        self._use_spec_fp8_router = (
            speculative_config is not None
            and speculative_config.use_qwen_mtp_or_dflash()
        )
        self._esimd_moe_enabled = (
            self._esimd_moe_ops is not None
            and os.getenv("VLLM_XPU_DISABLE_ESIMD_MOE", "0") != "1"
            and not parallel_config.enable_expert_parallel
            and self.shared_expert is not None
            and self.n_routed_experts % 16 == 0
        )
        quant_name = quant_config.get_name() if quant_config is not None else None
        self._esimd_int4_moe_enabled = (
            self._esimd_moe_ops is not None
            and quant_name == "sym_int4"
            and os.getenv("DISABLE_ESIMD_INT4", "0") != "1"
            and os.getenv("VLLM_XPU_DISABLE_ESIMD_INT4", "0") != "1"
            and os.getenv("VLLM_XPU_DISABLE_ESIMD_MOE", "0") != "1"
            and not parallel_config.enable_expert_parallel
            and self.shared_expert is not None
            and self.n_routed_experts % 16 == 0
        )
        self._esimd_moe_weight_eligible: bool | None = None
        self._esimd_moe_is_block = False
        if self._esimd_moe_enabled:
            logger.info_once(
                "Qwen3Next MoE is using the guarded ESIMD FP8 decode path."
            )
        self._gate_fp8_weight = None
        self._gate_fp8_scale = None
        self._gate_int4_weight = None
        self._gate_int4_scale = None
        self._gate_int4_op = None
        self._router_buf = None
        self._normed_buf = None
        self._esimd_moe_plan: _EsimdMoePlan | None = None
        self._esimd_int4_moe_plan: _EsimdInt4MoePlan | None = None
        self._esimd_router_plan: _EsimdRouterPlan | None = None
        self._batch_router_buffers: dict[int, torch.Tensor] = {}
        self._block_moe_output_buffers: dict[int, torch.Tensor] = {}

    def fused_post_attn_router(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "_gate_int4_op", None) is not None:
            self._gate_int4_op(
                hidden_states,
                residual,
                norm_weight,
                self._gate_int4_weight,
                self._gate_int4_scale,
                self._router_buf,
                self._normed_buf,
                eps,
            )
            return self._router_buf, self._normed_buf

        gate = cast(ReplicatedLinear | None, _get_base_layer_for_no_lora(self.gate))
        if gate is None:
            raise RuntimeError(
                "ESIMD fused router cannot run with an active gate LoRA."
            )
        quant_method = getattr(gate, "quant_method", None)
        if getattr(quant_method, "is_sym_int4", False):
            if self._router_buf is None:
                self._router_buf = torch.empty(
                    1,
                    self.n_routed_experts,
                    dtype=torch.float16,
                    device=hidden_states.device,
                )
                self._normed_buf = torch.empty_like(hidden_states)
            self._gate_int4_op = self._esimd_moe_ops.esimd_resadd_norm_gemv_int4_pert
            self._gate_int4_weight = gate.weight_esimd.view(torch.int32)
            self._gate_int4_scale = gate.scale_esimd
            self._gate_int4_op(
                hidden_states,
                residual,
                norm_weight,
                self._gate_int4_weight,
                self._gate_int4_scale,
                self._router_buf,
                self._normed_buf,
                eps,
            )
            return self._router_buf, self._normed_buf

        gate_weight, gate_scale = self._get_fp8_router_weight(gate)
        if self._router_buf is None:
            self._router_buf = torch.empty(
                1,
                self.n_routed_experts,
                dtype=torch.float16,
                device=hidden_states.device,
            )
            self._normed_buf = torch.empty_like(hidden_states)

        self._esimd_moe_ops.esimd_resadd_norm_gemv_fp8_pert(
            hidden_states,
            residual,
            norm_weight,
            gate_weight,
            gate_scale,
            self._router_buf,
            self._normed_buf,
            eps,
        )
        return self._router_buf, self._normed_buf

    def _can_forward_esimd_int4_moe(
        self,
        hidden_states: torch.Tensor,
        sources: _EsimdMoeSources | None = None,
    ) -> bool:
        if (
            not self._esimd_int4_moe_enabled
            or hidden_states.dtype != torch.float16
            or hidden_states.ndim != 2
            or not 1 <= hidden_states.shape[0] <= 64
        ):
            return False
        if self._esimd_int4_moe_plan is not None:
            return True
        if sources is None:
            sources = self._get_esimd_moe_sources()
        if sources is None:
            return False
        method = getattr(sources.routed_experts, "quant_method", None)
        gate_method = getattr(sources.gate, "quant_method", None)
        gate_weight = getattr(sources.gate, "weight_esimd", None)
        gate_scale = getattr(sources.gate, "scale_esimd", None)
        return bool(
            not _DISABLE_ESIMD_MOE_ROUTER
            and getattr(method, "is_sym_int4", False)
            and callable(getattr(method, "get_xpu_fused_moe_impl", None))
            and getattr(gate_method, "is_sym_int4", False)
            and isinstance(gate_weight, torch.Tensor)
            and gate_weight.dtype == torch.uint8
            and gate_weight.is_contiguous()
            and gate_weight.device == hidden_states.device
            and gate_weight.shape
            == (self.n_routed_experts, hidden_states.shape[-1] // 2)
            and isinstance(gate_scale, torch.Tensor)
            and gate_scale.dtype == torch.float16
            and gate_scale.is_contiguous()
            and gate_scale.device == hidden_states.device
            and gate_scale.shape
            == (self.n_routed_experts, hidden_states.shape[-1] // 128)
            and callable(
                getattr(
                    self._esimd_moe_ops,
                    "esimd_resadd_norm_gemv_int4_pert",
                    None,
                )
            )
            and sources.shared_gate_up_proj.weight.dtype == torch.float16
            and sources.shared_gate_up_proj.weight.is_contiguous()
            and sources.shared_down_proj.weight.dtype == torch.float16
            and sources.shared_down_proj.weight.is_contiguous()
            and sources.shared_expert_gate.weight.dtype == torch.float16
            and sources.shared_expert_gate.weight.is_contiguous()
            and callable(
                getattr(
                    self._esimd_moe_ops,
                    "moe_forward_full_cutlass_nmajor_int4",
                    None,
                )
            )
            and callable(
                getattr(
                    self._esimd_moe_ops,
                    "moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits",
                    None,
                )
            )
        )

    def _forward_esimd_int4_moe(
        self,
        hidden_states: torch.Tensor,
        num_tokens: int,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        plan = self._esimd_int4_moe_plan
        if plan is None:
            sources = self._get_esimd_moe_sources()
            if sources is None or not self._can_forward_esimd_int4_moe(
                hidden_states, sources
            ):
                return None
            method = sources.routed_experts.quant_method
            experts_per_token = sources.experts.moe_config.experts_per_token
            impl = method.get_xpu_fused_moe_impl(
                sources.routed_experts, experts_per_token
            )
            w13_scale = impl.gemm1_wei_scales
            w2_scale = impl.gemm2_wei_scales
            # The modular XPU WNA16 warmup may have already converted the
            # routed weights from packed uint4 to signed-s4 int8. A second
            # XpuFusedMoe wrapper then classifies those weights as unquantized
            # and leaves its scale aliases unset; the canonical layer scales
            # remain valid for the ESIMD CUTLASS path.
            if w13_scale is None:
                w13_scale = sources.routed_experts.w13_scales
            if w2_scale is None:
                w2_scale = sources.routed_experts.w2_scales
            w13 = impl.w13
            w2 = impl.w2
            if w13.dtype == torch.int8:
                w13 = w13.view(torch.uint8)
            if w2.dtype == torch.int8:
                w2 = w2.view(torch.uint8)
            plan = _EsimdInt4MoePlan(
                full_op=self._esimd_moe_ops.moe_forward_full_cutlass_nmajor_int4,
                tiny_op=self._esimd_moe_ops.moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits,
                w13=w13,
                w13_scale=w13_scale,
                w2=w2,
                w2_scale=w2_scale,
                shared_w13=sources.shared_gate_up_proj.weight,
                shared_w2=sources.shared_down_proj.weight,
                shared_expert_gate_weight=sources.shared_expert_gate.weight,
                experts_per_token=experts_per_token,
            )
            self._esimd_int4_moe_plan = plan

        if router_logits is None:
            sources = self._get_esimd_moe_sources()
            if sources is None:
                return None
            router_logits, _ = sources.gate(hidden_states)
        kernel_input = (
            hidden_states
            if hidden_states.is_contiguous()
            else hidden_states.contiguous()
        )
        op = plan.full_op if num_tokens == 1 else plan.tiny_op
        output = op(
            kernel_input,
            router_logits,
            plan.w13,
            plan.w13_scale,
            plan.w2,
            plan.w2_scale,
            plan.shared_w13,
            plan.shared_w2,
            plan.shared_expert_gate_weight,
            plan.experts_per_token,
            1,
            self.n_routed_experts,
        )
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output

    def _get_fp8_router_weight(
        self, gate: ReplicatedLinear
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._gate_fp8_weight is None:
            from vllm import _custom_ops as ops

            self._gate_fp8_weight, self._gate_fp8_scale = ops.scaled_fp8_quant(
                gate.weight.data.contiguous(), scale=None
            )
        assert self._gate_fp8_weight is not None
        assert self._gate_fp8_scale is not None
        return self._gate_fp8_weight, self._gate_fp8_scale

    def _forward_esimd_moe(
        self,
        hidden_states: torch.Tensor,
        num_tokens: int,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        sources = self._get_esimd_moe_sources()
        if sources is None or not self._can_forward_esimd_moe(hidden_states, sources):
            return None
        if self._esimd_moe_is_block and not 1 <= num_tokens <= 12:
            return None

        plan = self._esimd_moe_plan
        if plan is None:
            experts = sources.routed_experts
            scale_suffix = "_inv" if self._esimd_moe_is_block else ""
            plan = _EsimdMoePlan(
                w13=experts.w13_weight,
                w13_scale=getattr(experts, f"w13_weight_scale{scale_suffix}"),
                w2=experts.w2_weight,
                w2_scale=getattr(experts, f"w2_weight_scale{scale_suffix}"),
                shared_w13=sources.shared_gate_up_proj.weight,
                shared_w13_scale=getattr(
                    sources.shared_gate_up_proj,
                    f"weight_scale{scale_suffix}",
                ),
                shared_w2=sources.shared_down_proj.weight,
                shared_w2_scale=getattr(
                    sources.shared_down_proj,
                    f"weight_scale{scale_suffix}",
                ),
                shared_expert_gate_weight=sources.shared_expert_gate.weight,
                experts_per_token=sources.experts.moe_config.experts_per_token,
                is_block=self._esimd_moe_is_block,
            )
            self._esimd_moe_plan = plan

        if router_logits is None:
            router_logits = self._esimd_router_logits(
                hidden_states, num_tokens, sources.gate
            )
        if plan.is_block:
            output_buffer = self._block_moe_output_buffers.get(num_tokens)
            if output_buffer is None:
                output_buffer = torch.empty_like(hidden_states)
                self._block_moe_output_buffers[num_tokens] = output_buffer
            output = torch.ops.moe_ops.moe_forward_full_fp8_block(
                hidden_states,
                router_logits,
                output_buffer,
                plan.w13,
                plan.w13_scale,
                plan.shared_w13,
                plan.shared_w13_scale,
                plan.w2,
                plan.w2_scale,
                plan.shared_w2,
                plan.shared_w2_scale,
                plan.shared_expert_gate_weight,
                plan.experts_per_token,
                1,
                self.n_routed_experts,
            )
        else:
            moe_fn = (
                torch.ops.moe_ops.moe_forward_full
                if num_tokens == 1
                else torch.ops.moe_ops.moe_forward_full_v2
            )
            output = moe_fn(
                hidden_states,
                router_logits,
                plan.w13,
                plan.w13_scale,
                plan.shared_w13,
                plan.shared_w13_scale,
                plan.w2,
                plan.w2_scale,
                plan.shared_w2,
                plan.shared_w2_scale,
                plan.shared_expert_gate_weight,
                plan.experts_per_token,
                1,
                self.n_routed_experts,
            )
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output

    def _esimd_router_logits(
        self,
        hidden_states: torch.Tensor,
        num_tokens: int,
        gate: ReplicatedLinear | None = None,
    ) -> torch.Tensor:
        """Route via an ESIMD FP8 GEMM for small decode batches.

        Falls back to the standard gate() GEMM outside that range or when the
        fused router op / weight layout is not eligible.
        """
        if gate is None:
            gate = cast(
                ReplicatedLinear | None,
                _get_base_layer_for_no_lora(self.gate),
            )
            if gate is None:
                router_logits, _ = self.gate(hidden_states)
                return router_logits

        router_plan = self._esimd_router_plan
        max_tokens = (
            _ESIMD_MOE_ROUTER_MAX_TOKENS if self._use_spec_fp8_router else 8
        )
        if (
            router_plan is None
            and not _DISABLE_ESIMD_MOE_ROUTER
            and 2 <= num_tokens <= max_tokens
        ):
            gate_weight = gate.weight
            eligible = (
                hidden_states.dtype == torch.float16
                and hidden_states.is_contiguous()
                and gate_weight.dtype == torch.float16
                and gate_weight.is_contiguous()
                and gate.bias is None
            )
            if self._use_spec_fp8_router:
                router_op = getattr(
                    self._esimd_moe_ops, "esimd_gemm_fp8_pert", None
                )
                if callable(router_op) and eligible:
                    fp8_weight, fp8_scale = self._get_fp8_router_weight(gate)
                    router_plan = _EsimdRouterPlan(
                        op=router_op,
                        weight=fp8_weight,
                        weight_scale=fp8_scale,
                    )
            else:
                router_op = getattr(self._esimd_moe_ops, "esimd_gemv_fp16", None)
                if callable(router_op) and eligible:
                    router_plan = _EsimdRouterPlan(
                        op=router_op,
                        weight=gate_weight,
                        weight_scale=None,
                    )
            self._esimd_router_plan = router_plan

        if router_plan is not None and 2 <= num_tokens <= max_tokens:
            router_logits = self._batch_router_buffers.get(num_tokens)
            if router_logits is None:
                router_logits = torch.empty(
                    num_tokens,
                    self.n_routed_experts,
                    dtype=torch.float16,
                    device=hidden_states.device,
                )
                self._batch_router_buffers[num_tokens] = router_logits
            if router_plan.weight_scale is None:
                router_plan.op(hidden_states, router_plan.weight, router_logits)
            else:
                router_plan.op(
                    hidden_states,
                    router_plan.weight,
                    router_plan.weight_scale,
                    router_logits,
                )
            return router_logits

        router_logits, _ = gate(hidden_states)
        return router_logits

    def _get_esimd_moe_sources(self) -> _EsimdMoeSources | None:
        experts: object = self.experts
        base_experts = getattr(experts, "base_layer", None)
        if base_experts is not None:
            is_lora_active = getattr(experts, "is_lora_active", None)
            if not callable(is_lora_active) or is_lora_active():
                return None
            experts = base_experts

        shared_expert = self.shared_expert
        if shared_expert is None:
            return None
        gate = cast(ReplicatedLinear | None, _get_base_layer_for_no_lora(self.gate))
        shared_expert_gate = cast(
            ReplicatedLinear | None,
            _get_base_layer_for_no_lora(self.shared_expert_gate),
        )
        shared_gate_up_proj = cast(
            MergedColumnParallelLinear | None,
            _get_base_layer_for_no_lora(shared_expert.gate_up_proj),
        )
        shared_down_proj = cast(
            RowParallelLinear | None,
            _get_base_layer_for_no_lora(shared_expert.down_proj),
        )
        routed_experts = cast(
            RoutedExperts | None, getattr(experts, "routed_experts", None)
        )
        if (
            routed_experts is None
            or gate is None
            or shared_expert_gate is None
            or shared_gate_up_proj is None
            or shared_down_proj is None
        ):
            return None
        return _EsimdMoeSources(
            experts=cast(MoERunner, experts),
            routed_experts=routed_experts,
            gate=gate,
            shared_expert_gate=shared_expert_gate,
            shared_gate_up_proj=shared_gate_up_proj,
            shared_down_proj=shared_down_proj,
        )

    def _can_forward_esimd_moe(
        self,
        hidden_states: torch.Tensor,
        sources: _EsimdMoeSources | None = None,
    ) -> bool:
        if hidden_states.dtype != torch.float16:
            return False
        if sources is None:
            sources = self._get_esimd_moe_sources()
        if sources is None:
            return False
        if self._esimd_moe_weight_eligible is None:
            fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
            experts = sources.routed_experts
            shared_gate_up_proj = sources.shared_gate_up_proj
            shared_down_proj = sources.shared_down_proj
            scalar_scale_ready = (
                getattr(experts, "w13_weight", None) is not None
                and experts.w13_weight.dtype in fp8_dtypes
                and getattr(experts, "w13_weight_scale", None) is not None
                and getattr(experts, "w2_weight", None) is not None
                and experts.w2_weight.dtype in fp8_dtypes
                and getattr(experts, "w2_weight_scale", None) is not None
                and getattr(shared_gate_up_proj, "weight", None) is not None
                and shared_gate_up_proj.weight.dtype in fp8_dtypes
                and getattr(shared_gate_up_proj, "weight_scale", None) is not None
                and getattr(shared_down_proj, "weight", None) is not None
                and shared_down_proj.weight.dtype in fp8_dtypes
                and getattr(shared_down_proj, "weight_scale", None) is not None
                and getattr(sources.shared_expert_gate, "weight", None) is not None
            )
            block_scale_ready = (
                getattr(experts, "w13_weight", None) is not None
                and experts.w13_weight.dtype == torch.float8_e4m3fn
                and getattr(experts, "w13_weight_scale_inv", None) is not None
                and getattr(experts, "w2_weight", None) is not None
                and experts.w2_weight.dtype == torch.float8_e4m3fn
                and getattr(experts, "w2_weight_scale_inv", None) is not None
                and getattr(shared_gate_up_proj, "weight", None) is not None
                and shared_gate_up_proj.weight.dtype == torch.float8_e4m3fn
                and getattr(shared_gate_up_proj, "weight_scale_inv", None) is not None
                and getattr(shared_down_proj, "weight", None) is not None
                and shared_down_proj.weight.dtype == torch.float8_e4m3fn
                and getattr(shared_down_proj, "weight_scale_inv", None) is not None
                and getattr(sources.shared_expert_gate, "weight", None) is not None
                and callable(
                    getattr(
                        self._esimd_moe_ops,
                        "moe_forward_full_fp8_block",
                        None,
                    )
                )
            )
            self._esimd_moe_is_block = block_scale_ready
            self._esimd_moe_weight_eligible = scalar_scale_ready or block_scale_ready
            if not self._esimd_moe_weight_eligible:
                self._esimd_moe_enabled = False
        return self._esimd_moe_weight_eligible

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if (
            self._esimd_int4_moe_enabled
            and num_tokens <= 64
            and not self.is_sequence_parallel
        ):
            output = self._forward_esimd_int4_moe(hidden_states, num_tokens)
            if output is not None:
                return output.view(orig_shape)

        if (
            self._esimd_moe_enabled
            and num_tokens <= 128
            and not self.is_sequence_parallel
        ):
            output = self._forward_esimd_moe(hidden_states, num_tokens)
            if output is not None:
                return output.view(orig_shape)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(orig_shape)


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        _enable_sym_int4_batch_esimd(self.qkv_proj, self.o_proj)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        # Late-interaction retrieval models (e.g. ColQwen3.5) run BIDIRECTIONAL
        # attention on the full_attention layers; they set config.is_causal=False
        # via a VerifyAndUpdateConfig handler. Generation models leave is_causal
        # unset (-> causal/DECODER), so this is a no-op for them. Mirrors qwen3.py.
        attn_type = (
            AttentionType.DECODER
            if getattr(config, "is_causal", True)
            else AttentionType.ENCODER_ONLY
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Fuse the gated split + QK-RMSNorm + (partial) NeoX RoPE + gate copy.
        # TODO: support MRoPE
        mm_config = model_config.multimodal_config if model_config else None
        text_only = mm_config is None or mm_config.language_model_only
        self._esimd_qkv_text_only = text_only
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and text_only
        )

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return post-norm, post-RoPE (q, k, v) and the pre-sigmoid gate.

        Dispatches between the fused Triton kernel and the eager
        split + QK-RMSNorm + RoPE path. ``gate`` is ``None`` when output
        gating is disabled.
        """
        if self.use_fused_qk_norm_rope_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            # mRoPE passes positions as (3, n_tokens) for T/H/W. Fusion is only
            # enabled text-only, where the three rows are identical, so taking
            # the T row is exact. (1D positions pass through.)
            pos = positions[0] if positions.ndim == 2 else positions
            q, k, gate = fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                self.q_norm.weight.float() + 1.0,
                self.k_norm.weight.float() + 1.0,
                self.rotary_emb.cos_sin_cache,
                pos,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
            )
            return q, k, v, gate

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = try_esimd_int4_linear(
            hidden_states, self.qkv_proj, "_esimd_int4_qkv_projection_buffer"
        )
        if qkv is None:
            qkv = try_esimd_fp8_linear(
                hidden_states, self.qkv_proj, "_esimd_qkv_projection_buffer"
            )
        if qkv is None:
            qkv, _ = self.qkv_proj(hidden_states)
        text_only = self._esimd_qkv_text_only or _runtime_plain_mrope_is_qkv_eligible()
        fused_qkv = try_esimd_qkv_postprocess(
            qkv,
            positions,
            projection=self.qkv_proj,
            norm_wq=self.q_norm.weight,
            norm_wk=self.k_norm.weight,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
            q_heads=self.num_heads,
            kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            rotary_dim=self.rotary_emb.rotary_dim,
            norm_epsilon=self.q_norm.variance_epsilon,
            is_neox_style=getattr(self.rotary_emb, "is_neox_style", False),
            attn_output_gate=self.attn_output_gate,
            text_only=text_only,
        )
        if fused_qkv is None:
            q, k, v, gate = self._project_qkv_gate(qkv, positions)
        else:
            q, k, v, gate = fused_qkv
        attn_output = self.attn(q, k, v)
        if gate is not None:
            if fused_qkv is None:
                gate = torch.sigmoid(gate)
            attn_output = attn_output * gate
        output = try_esimd_int4_linear(
            attn_output, self.o_proj, "_esimd_int4_output_projection_buffer"
        )
        if output is None:
            output = try_esimd_fp8_linear(
                attn_output, self.o_proj, "_esimd_output_projection_buffer"
            )
        base_o_proj = getattr(self.o_proj, "base_layer", self.o_proj)
        if output is None:
            output, _ = self.o_proj(attn_output)
        elif base_o_proj.reduce_results and base_o_proj.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        is_moe_layer = (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        )
        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
            _enable_sym_int4_batch_esimd(
                self.linear_attn.in_proj_qkvz,
                self.linear_attn.in_proj_ba,
                self.linear_attn.out_proj,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if is_moe_layer:
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            _enable_sym_int4_batch_esimd(
                self.mlp.gate_up_proj,
                self.mlp.down_proj,
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
        self._init_esimd_moe_norm(vllm_config)

    def _init_esimd_moe_norm(self, vllm_config: VllmConfig) -> None:
        self._esimd_moe_norm_enabled = (
            isinstance(self.mlp, Qwen3NextSparseMoeBlock)
            and (self.mlp._esimd_moe_enabled or self.mlp._esimd_int4_moe_enabled)
            and not self.layer_scale
            and not self.use_attn_reduce_scatter_for_moe
            and vllm_config.lora_config is None
            and os.getenv("VLLM_XPU_DISABLE_ESIMD_MOE_NORM", "0") != "1"
        )
        self._post_norm_weight = None
        if self._esimd_moe_norm_enabled:
            logger.info_once(
                "Qwen3Next decode is using fused post-attention norm and "
                "MoE router GEMV."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        full_num_tokens = positions.shape[-1]
        decode_m1_xpu = (
            isinstance(self, Qwen3NextDecoderLayer)
            and current_platform.is_xpu()
            and hidden_states.ndim == 2
            and hidden_states.shape[0] == 1
            and hidden_states.dtype == torch.float16
            and not getattr(torch.compiler, "is_compiling", lambda: False)()
        )
        input_is_sequence_parallel = (
            self.use_attn_reduce_scatter_for_moe
            and residual is not None
            and hidden_states.shape[0] != full_num_tokens
        )

        if residual is None:
            residual = hidden_states
            if decode_m1_xpu:
                hidden_states = self.input_layernorm.forward_xpu(hidden_states)
            else:
                hidden_states = self.input_layernorm(hidden_states)
        else:
            if decode_m1_xpu:
                hidden_states, residual = self.input_layernorm.forward_xpu(
                    hidden_states, residual
                )
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if input_is_sequence_parallel:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if self.layer_type == "linear_attention":
            if decode_m1_xpu:
                hidden_states = self.linear_attn.forward_xpu(hidden_states)
            else:
                hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            if decode_m1_xpu:
                hidden_states = self.self_attn.forward(
                    hidden_states=hidden_states,
                    positions=positions,
                )
            else:
                hidden_states = self.self_attn(
                    hidden_states=hidden_states,
                    positions=positions,
                )
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        if self.use_attn_reduce_scatter_for_moe:
            tp_world_size = get_tensor_model_parallel_world_size()
            # small trick using minus, eg. -17 % 8 = 7
            sp_pad = (-hidden_states.shape[0]) % tp_world_size
            # pad if not divisible by world size
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            if not input_is_sequence_parallel:
                residual = sequence_parallel_chunk(residual)

        # Fully Connected
        if (
            self._esimd_moe_norm_enabled
            and residual is not None
            and hidden_states.shape[0] == 1
            and hidden_states.dtype == torch.float16
            and (
                self.mlp._can_forward_esimd_int4_moe(hidden_states)
                if self.mlp._esimd_int4_moe_enabled is True
                else self.mlp._can_forward_esimd_moe(hidden_states)
            )
        ):
            if self._post_norm_weight is None:
                self._post_norm_weight = (
                    (self.post_attention_layernorm.weight.data + 1.0)
                    .half()
                    .clone()
                    .contiguous()
                )
            router_logits, normed = self.mlp.fused_post_attn_router(
                hidden_states,
                residual,
                self._post_norm_weight,
                self.post_attention_layernorm.variance_epsilon,
            )
            if self.mlp._esimd_int4_moe_enabled is True:
                hidden_states = self.mlp._forward_esimd_int4_moe(
                    normed, 1, router_logits
                )
            else:
                hidden_states = self.mlp._forward_esimd_moe(normed, 1, router_logits)
            assert hidden_states is not None
        else:
            if (
                self._esimd_moe_norm_enabled
                and self.mlp._esimd_moe_enabled is not True
                and self.mlp._esimd_int4_moe_enabled is not True
            ):
                self._esimd_moe_norm_enabled = False
            if decode_m1_xpu:
                hidden_states, residual = self.post_attention_layernorm.forward_xpu(
                    hidden_states, residual
                )
            else:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
            if self.use_attn_reduce_scatter_for_moe:
                hidden_states = self.mlp(
                    hidden_states,
                    already_sequence_parallel=True,
                )
            elif decode_m1_xpu:
                hidden_states = self.mlp.forward(hidden_states)
            else:
                hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


def _all_gather_hidden_and_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    full_num_tokens: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if residual is None:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[:full_num_tokens]
        return hidden_states, None

    combined_states = torch.cat([hidden_states, residual], dim=-1)
    combined_states = tensor_model_parallel_all_gather(combined_states, 0)
    combined_states = combined_states[:full_num_tokens]
    hidden_states, residual = combined_states.split([hidden_size, hidden_size], dim=-1)
    return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module, EagleModelMixin):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        full_num_tokens = positions.shape[-1]
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if (
                hidden_states.shape[0] != full_num_tokens
                and not layer.use_attn_reduce_scatter_for_moe
            ):
                hidden_states, residual = _all_gather_hidden_and_residual(
                    hidden_states,
                    residual,
                    full_num_tokens,
                    self.config.hidden_size,
                )
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if (layer_idx + 1) in self.aux_hidden_state_layers and hidden_states.shape[
                0
            ] != full_num_tokens:
                hidden_states, residual = _all_gather_hidden_and_residual(
                    hidden_states,
                    residual,
                    full_num_tokens,
                    self.config.hidden_size,
                )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if hidden_states.shape[0] != full_num_tokens:
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                full_num_tokens,
                self.config.hidden_size,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0),
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class QwenNextMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
    SupportsEagle3,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkvz"],
        "in_proj_ba": ["in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights)
