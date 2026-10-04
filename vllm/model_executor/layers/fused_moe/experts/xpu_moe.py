# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kInt4Static32,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_xpu():
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe


def prepare_fp8_moe_layer_for_xpu(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w13_is_block = w13_scale is not None and w13_scale.ndim == 3
    w2_is_block = w2_scale is not None and w2_scale.ndim == 3
    if w13_is_block != w2_is_block:
        raise ValueError(
            "XPU FP8 MoE requires matching block-scale layouts for W13 and W2"
        )
    if w13_is_block:
        assert w13_scale is not None and w2_scale is not None
        if w13_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
            raise ValueError(
                "XPU block-FP8 MoE requires float32 block scales for W13 and W2"
            )
        for name, weight, scale in (
            ("W13", w13, w13_scale),
            ("W2", w2, w2_scale),
        ):
            if weight.shape[1] % 128 != 0 or weight.shape[2] % 128 != 0:
                raise ValueError(
                    "XPU block-FP8 grouped GEMM requires weight dimensions "
                    f"divisible by 128; got {name} shape {tuple(weight.shape)}"
                )
            expected = (
                weight.shape[0],
                weight.shape[1] // 128,
                weight.shape[2] // 128,
            )
            if tuple(scale.shape) != expected:
                raise ValueError(
                    f"XPU block-FP8 {name} scale shape {tuple(scale.shape)} "
                    f"does not match weight shape {tuple(weight.shape)}; "
                    f"expected {expected}"
                )
        # Keep the checkpoint-native [E, N, K] layout. The grouped GEMM
        # accepts this layout, and the fused block-FP8 decode kernel consumes
        # it directly; transposing here would make the decode fusion
        # ineligible or require a second persistent copy of every expert.
        w13 = w13.contiguous()
        w2 = w2.contiguous()
        logger.info_once(
            "Preparing XPU block-FP8 MoE weights in checkpoint-native "
            "[E, N, K] layout for grouped GEMM and fused decode"
        )
        return (
            w13,
            w13_scale.contiguous(),
            w2,
            w2_scale.contiguous(),
        )
    return (
        w13.transpose(-1, -2).contiguous(),
        w13_scale,
        w2.transpose(-1, -2).contiguous(),
        w2_scale,
    )


class XPUExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        self.fused_moe_impl: XpuFusedMoe | None = None

    def _ensure_fused_moe_impl(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk: int,
        activation: MoEActivation,
        expert_map: torch.Tensor | None,
    ):
        if self.fused_moe_impl is None:
            if (
                self.quant_config is not None
                and self.quant_config.weight_quant_dtype == "mxfp4"
            ):
                w1 = w1.view(torch.float4_e2m1fn_x2)
                w2 = w2.view(torch.float4_e2m1fn_x2)
            self.fused_moe_impl = XpuFusedMoe(
                w13=w1,
                w13_scales=self.w1_scale,
                w13_bias=self.w1_bias,
                w2=w2,
                w2_scales=self.w2_scale,
                w2_bias=self.w2_bias,
                n_experts_per_token=topk,
                activation=activation.value,
                num_experts=self.moe_config.num_local_experts,
                ep_rank=self.moe_config.ep_rank,
                ep_size=self.moe_config.ep_size,
                expert_map=expert_map,
                gemm1_clamp_limit=self.gemm1_clamp_limit,
            )
        return self.fused_moe_impl

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        topk = topk_ids.size(-1)
        impl = self._ensure_fused_moe_impl(w1, w2, topk, activation, expert_map)
        ctx = getattr(self, "_lora_context", None)
        has_active_lora = (
            None
            if ctx is None
            else getattr(ctx.punica_wrapper, "has_active_lora", None)
        )
        if (
            ctx is None
            or (
                has_active_lora is not None
                and not has_active_lora(ctx.adapter_enabled_cpu)
            )
            or (has_active_lora is None and ctx.punica_wrapper.no_lora)
        ):
            logger.info_once(
                "Using monolithic XPU quantized MoE path for a batch without "
                "active expert LoRA weights (%s)",
                type(self).__name__,
            )
            impl.apply(
                output=output,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
            return

        logger.info_once(
            "Using staged XPU quantized MoE LoRA path (%s)",
            type(self).__name__,
        )
        assert isinstance(self, LoRAExpertsMixin)
        assert not (ctx.fully_sharded and self.moe_config.ep_size > 1), (
            "XPU MoE LoRA does not support EP with fully-sharded LoRA"
        )
        if ctx.enable_moe_shared_loras:
            raise ValueError(
                "XPU quantized MoE LoRA does not support shared expert factors"
            )
        del a1q_scale, a2_scale, workspace13, workspace2
        del expert_tokens_meta, apply_router_weight_on_input

        num_tokens, hidden_size = hidden_states.shape
        num_moe_inputs = num_tokens * topk
        num_experts = impl.local_experts_num
        effective_expert_map = expert_map
        if effective_expert_map is None and impl.ep_size > 1:
            effective_expert_map = impl.expert_map

        remapped_hidden_states = torch.empty(
            (num_moe_inputs, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        rows_per_expert = torch.zeros(
            num_experts, dtype=torch.int32, device=hidden_states.device
        )
        route_to_permuted_row = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=hidden_states.device
        )
        torch.ops._moe_C.remap_hidden_states(
            hidden_states=hidden_states,
            hidden_states_scales=None,
            remapped_hidden_states=remapped_hidden_states,
            remapped_hidden_states_scales=None,
            expert_map=effective_expert_map,
            rows_per_expert=rows_per_expert,
            unpermuted_row_to_permuted_row=route_to_permuted_row,
            topk_ids=topk_ids,
            total_experts_num=impl.total_experts_num,
            local_experts_num=num_experts,
        )

        gemm1_output = torch.empty(
            (num_moe_inputs, 2 * impl.inter_size),
            dtype=output.dtype,
            device=output.device,
        )
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=remapped_hidden_states,
            ptr_A_scale=None,
            ptr_B=impl.w13,
            ptr_B_scale=impl.gemm1_wei_scales,
            ptr_bias=impl.w13_bias,
            ptr_D=gemm1_output,
            rows_per_expert=rows_per_expert,
            N=2 * impl.inter_size,
            K=hidden_size,
            num_experts=num_experts,
        )

        routing = route_to_permuted_row.reshape(-1)
        lora_input = (
            ctx.original_hidden_states
            if ctx.original_hidden_states is not None
            and ctx.original_hidden_states.shape[0] == hidden_states.shape[0]
            else hidden_states
        )
        lora_inter_size = ctx.w13_lora_b_stacked[0].shape[-2]
        if len(ctx.w13_lora_b_stacked) == 2:
            assert all(
                weight.shape[-2] == lora_inter_size for weight in ctx.w13_lora_b_stacked
            )
            if lora_inter_size > impl.inter_size:
                raise ValueError(
                    "XPU MoE LoRA intermediate size exceeds the base expert size: "
                    f"{lora_inter_size} > {impl.inter_size}"
                )
        logical_w13_width = sum(weight.shape[-2] for weight in ctx.w13_lora_b_stacked)
        if logical_w13_width > gemm1_output.shape[-1]:
            raise ValueError(
                "XPU MoE LoRA W13 output exceeds the base expert output: "
                f"{logical_w13_width} > {gemm1_output.shape[-1]}"
            )
        needs_w13_unpadding = logical_w13_width != gemm1_output.shape[-1]
        if needs_w13_unpadding:
            w13_lora_target = torch.zeros(
                (num_tokens, topk, logical_w13_width),
                dtype=gemm1_output.dtype,
                device=gemm1_output.device,
            )
        else:
            w13_lora_target = gemm1_output
        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        ) = self.apply_w13_lora(
            ctx,
            y=w13_lora_target,
            x=lora_input,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_map=effective_expert_map,
            w1=impl.w13,
            w2=impl.w2,
            num_tokens=num_tokens,
            top_k_num=topk,
            add_inputs=not needs_w13_unpadding,
            c_row_mapping=routing,
        )
        if needs_w13_unpadding:
            delta = w13_lora_target.view(num_moe_inputs, logical_w13_width)
            if len(ctx.w13_lora_b_stacked) == 2:
                gate_width = ctx.w13_lora_b_stacked[0].shape[-2]
                up_width = ctx.w13_lora_b_stacked[1].shape[-2]
                gemm1_output[:, :gate_width].add_(delta[:, :gate_width])
                gemm1_output[:, impl.inter_size : impl.inter_size + up_width].add_(
                    delta[:, gate_width:]
                )
            else:
                if logical_w13_width % 2 != 0:
                    raise ValueError(
                        "Packed XPU MoE LoRA W13 output must have an even width, "
                        f"got {logical_w13_width}"
                    )
                packed_half_width = logical_w13_width // 2
                if packed_half_width > impl.inter_size:
                    raise ValueError(
                        "Packed XPU MoE LoRA intermediate size exceeds the base "
                        f"expert size: {packed_half_width} > {impl.inter_size}"
                    )
                gemm1_output[:, :packed_half_width].add_(delta[:, :packed_half_width])
                gemm1_output[
                    :, impl.inter_size : impl.inter_size + packed_half_width
                ].add_(delta[:, packed_half_width:])

        act_output = torch.empty(
            (num_moe_inputs, impl.inter_size * impl.inter_size_scale),
            dtype=gemm1_output.dtype,
            device=gemm1_output.device,
        )
        apply_moe_activation(
            activation,
            act_output,
            gemm1_output,
            clamp_limit=self.gemm1_clamp_limit,
        )

        gemm2_output = torch.empty(
            (num_moe_inputs, hidden_size), dtype=output.dtype, device=output.device
        )
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=act_output,
            ptr_A_scale=None,
            ptr_B=impl.w2,
            ptr_B_scale=impl.gemm2_wei_scales,
            ptr_bias=impl.w2_bias,
            ptr_D=gemm2_output,
            rows_per_expert=rows_per_expert,
            N=hidden_size,
            K=impl.inter_size * impl.inter_size_scale,
            num_experts=num_experts,
        )
        self.apply_w2_lora(
            ctx,
            y=gemm2_output.view(num_tokens, topk, hidden_size),
            x=act_output[:, : ctx.w2_lora_a_stacked[0].shape[-1]].contiguous(),
            topk_weights=topk_weights,
            sorted_token_ids_lora=sorted_token_ids_lora,
            expert_ids_lora=expert_ids_lora,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            token_lora_mapping=token_lora_mapping,
            num_tokens=num_tokens,
            w1=impl.w13,
            w2=impl.w2,
            top_k_num=topk,
            row_mapping=routing,
            mul_routed_weight=False,
        )
        torch.ops._moe_C.moe_gather(
            output,
            gemm2_output,
            topk_weights,
            route_to_permuted_row,
            num_experts,
        )


class XPUExpertsFp8(LoRAExpertsMixin, XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsMxFp8(XPUExpertsFp8):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        assert quant_config.quant_dtype == "mxfp8"

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp8Static, None),
            (kMxfp8Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def supports_lora() -> bool:
        return False


class XPUExpertsBlockFp8(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsWNA16(LoRAExpertsMixin, XPUExperts):
    """W4A16 INT4-symmetric MoE backed by `xpu_fused_moe(is_int4=True)`.

    Weight layout when `is_int4=True` (per `xpu_fused_moe` docstring):
        w13: [num_experts, 2*inter_size, hidden_size]   contiguous int4-packed
        w13_scales: [num_experts, 2*inter_size, hidden_size // group_size]
        w2:  [num_experts, hidden_size, inter_size]     contiguous int4-packed
        w2_scales:  [num_experts, hidden_size, inter_size // group_size]

    Pairs with `INCXPULinearMethod` for the linear layers; together they
    cover full-attn + MoE on Intel XPU end-to-end without IPEX.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (kInt4Static, None),
            (kInt4Static32, None),
        )


class XPUExpertsMxFp4(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A
