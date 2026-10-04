# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.config.lora import LoRAConfig
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.layers.column_parallel_linear import (
    MergedColumnParallelLinearWithLoRA,
)
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.lora.layers.logits_processor import LogitsProcessorWithLoRA
from vllm.lora.layers.replicated_linear import ReplicatedLinearWithLoRA
from vllm.lora.layers.row_parallel_linear import RowParallelLinearWithLoRA
from vllm.lora.layers.utils import LoRAMapping, _get_lora_device
from vllm.lora.layers.vocal_parallel_embedding import VocabParallelEmbeddingWithLoRA
from vllm.lora.ops.triton_ops import fused_moe_lora_op
from vllm.lora.ops.triton_ops.fused_moe_lora_op import _get_xpu_direct_weight
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.lora.punica_wrapper.punica_xpu import PunicaWrapperXPU
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import xpu_moe
from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
    XPUExpertsBlockFp8,
    XPUExpertsFp8,
    XPUExpertsMxFp8,
    XPUExpertsWNA16,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)


def test_xpu_quantized_expert_lora_fallback_selection() -> None:
    """Only XPU quantizers with a split grouped-GEMM path accept MoE LoRA."""
    assert XPUExpertsFp8.supports_lora()
    assert XPUExpertsWNA16.supports_lora()
    assert not XPUExpertsMxFp8.supports_lora()
    assert not XPUExpertsBlockFp8.supports_lora()


def test_fused_moe_lora_rejects_round_robin_ep_placement() -> None:
    layer = object.__new__(FusedMoEWithLoRA)
    layer.base_layer = SimpleNamespace(expert_placement_strategy="round_robin")
    layer.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(use_ep=True))

    with pytest.raises(ValueError, match="requires linear expert placement"):
        layer._ep_check()


def test_xpu_two_pass_lora_can_return_an_unpadded_delta(monkeypatch) -> None:
    """The XPU two-pass path must support a delta-only padded-W13 staging buffer."""
    monkeypatch.setattr(fused_moe_lora_op.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(fused_moe_lora_op, "supports_tma", lambda device: False)
    monkeypatch.setattr(fused_moe_lora_op, "supports_pdl", lambda device: False)
    monkeypatch.setattr(
        fused_moe_lora_op,
        "_fused_moe_lora_shrink",
        lambda *args, **kwargs: None,
    )
    observed = {}

    def capture_expand(*args, **kwargs):
        observed["add_inputs"] = kwargs["add_inputs"]

    monkeypatch.setattr(
        fused_moe_lora_op,
        "_fused_moe_lora_expand",
        capture_expand,
    )

    fused_moe_lora_op._fused_moe_lora(
        output=torch.zeros(1, 2, 4),
        qcurr_hidden_states=torch.zeros(1, 3),
        lora_a_stacked=[torch.zeros(1, 1, 1, 3)],
        lora_b_stacked=[torch.zeros(1, 1, 4, 1)],
        topk_weights=torch.ones(1, 2),
        sorted_token_ids=None,
        expert_ids=torch.zeros(2, dtype=torch.int32),
        num_tokens_post_padded=None,
        token_lora_mapping=torch.zeros(1, dtype=torch.int32),
        max_lora_rank=1,
        top_k_num=2,
        lora_ids=torch.zeros(1, dtype=torch.int32),
        num_active_loras=torch.ones(1, dtype=torch.int32),
        adapter_enabled=torch.ones(1, dtype=torch.int32),
        shrink_block_size_m=16,
        shrink_block_size_n=16,
        shrink_block_size_k=16,
        shrink_group_size_m=1,
        shrink_num_warps=1,
        shrink_num_stages=1,
        shrink_split_k=1,
        expand_block_size_m=16,
        expand_block_size_n=16,
        expand_block_size_k=16,
        expand_group_size_m=1,
        expand_num_warps=1,
        expand_num_stages=1,
        expand_split_k=1,
        add_inputs=False,
    )

    assert observed == {"add_inputs": False}


def test_route_row_mapping_keeps_expanded_w2_input_rows(monkeypatch) -> None:
    """A route-to-permuted mapping addresses expanded W2 rows without top-k division."""
    observed = {}

    class FakeKernel:
        def __getitem__(self, grid):
            del grid

            def launch(*args, **kwargs):
                observed["token_mapping_factor"] = kwargs["token_mapping_factor"]

            return launch

    monkeypatch.setattr(fused_moe_lora_op, "_fused_moe_lora_kernel", FakeKernel())
    monkeypatch.setattr(
        fused_moe_lora_op,
        "_get_ptr",
        lambda weights, device: torch.zeros(1, dtype=torch.int64),
    )
    monkeypatch.setattr(fused_moe_lora_op.current_platform, "is_xpu", lambda: False)

    fused_moe_lora_op._fused_moe_lora_shrink(
        a_intermediate_cache1=torch.zeros(1, 1, 2, 1),
        qcurr_hidden_states=torch.zeros(2, 3),
        lora_a_stacked=[torch.zeros(1, 1, 1, 3)],
        topk_weights=torch.ones(1, 2),
        sorted_token_ids=None,
        expert_ids=torch.zeros(2, dtype=torch.int32),
        num_tokens_post_padded=None,
        token_lora_mapping=torch.zeros(1, dtype=torch.int32),
        top_k_num=2,
        lora_ids=torch.zeros(1, dtype=torch.int32),
        adapter_enabled=torch.ones(1, dtype=torch.int32),
        device=torch.device("cpu"),
        N=1,
        M=1,
        EM=32,
        K=3,
        num_tokens=2,
        num_experts=1,
        num_slices=1,
        block_size_m=16,
        block_size_n=16,
        block_size_k=16,
        group_size_m=1,
        num_warps=1,
        num_stages=1,
        split_k=1,
        num_active_loras=torch.ones(1, dtype=torch.int32),
        mul_routed_weight=False,
        a_row_mapping=torch.tensor([1, 0], dtype=torch.int32),
    )

    assert observed == {"token_mapping_factor": 1}


class _CpuPunicaMoE:
    """Numerical stand-in for the XPU Triton system boundary."""

    no_lora = False

    def __init__(self, token_lora_mapping: torch.Tensor):
        self.token_lora_mapping = token_lora_mapping
        self.topk_ids: torch.Tensor | None = None
        self.active_lora_indices = (0,)

    def has_active_lora(self, adapter_enabled_cpu: list[bool] | None = None) -> bool:
        if self.no_lora:
            return False
        if adapter_enabled_cpu is None:
            return True
        return any(adapter_enabled_cpu[index] for index in self.active_lora_indices)

    def add_lora_w13(
        self,
        y,
        x,
        lora_a_stacked,
        lora_b_stacked,
        topk_ids,
        topk_weights,
        expert_map,
        w1,
        w2,
        num_tokens,
        top_k_num,
        max_loras,
        adapter_enabled,
        local_num_experts,
        top_k,
        num_slices,
        fully_sharded,
        use_tuned_config,
        add_inputs=True,
        token_lora_mapping=None,
        c_row_mapping=None,
    ):
        del (
            topk_weights,
            expert_map,
            w1,
            w2,
            num_tokens,
            max_loras,
            adapter_enabled,
            local_num_experts,
            top_k,
            num_slices,
            fully_sharded,
            use_tuned_config,
            add_inputs,
        )
        self.topk_ids = topk_ids
        mapping = (
            self.token_lora_mapping
            if token_lora_mapping is None
            else token_lora_mapping
        )
        y_rows = y.view(-1, y.shape[-1])
        for route, expert in enumerate(topk_ids.flatten().tolist()):
            token = route // top_k_num
            adapter = int(mapping[token])
            dst = route if c_row_mapping is None else int(c_row_mapping[route])
            offset = 0
            for lora_a, lora_b in zip(lora_a_stacked, lora_b_stacked):
                delta = (x[token] @ lora_a[adapter, expert].T) @ lora_b[
                    adapter, expert
                ].T
                y_rows[dst, offset : offset + delta.numel()].add_(delta)
                offset += delta.numel()
        return None, None, None, mapping

    def add_lora_w2(
        self,
        y,
        x,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids_lora,
        expert_ids_lora,
        num_tokens_post_padded_lora,
        token_lora_mapping,
        num_tokens,
        w1,
        w2,
        top_k_num,
        max_loras,
        adapter_enabled,
        top_k,
        fully_sharded,
        tp_rank,
        use_tuned_config,
        add_inputs=True,
        row_mapping=None,
        mul_routed_weight=True,
    ):
        del (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            num_tokens,
            w1,
            w2,
            max_loras,
            adapter_enabled,
            top_k,
            fully_sharded,
            tp_rank,
            use_tuned_config,
            add_inputs,
        )
        assert self.topk_ids is not None
        assert x.shape[-1] == lora_a_stacked[0].shape[-1]
        y_rows = y.view(-1, y.shape[-1])
        lora_a, lora_b = lora_a_stacked[0], lora_b_stacked[0]
        for route, expert in enumerate(self.topk_ids.flatten().tolist()):
            token = route // top_k_num
            adapter = int(token_lora_mapping[token])
            row = route if row_mapping is None else int(row_mapping[route])
            delta = (x[row] @ lora_a[adapter, expert].T) @ lora_b[adapter, expert].T
            if mul_routed_weight:
                delta = delta * topk_weights.flatten()[route]
            y_rows[row].add_(delta)


def _install_cpu_xpu_ops(monkeypatch):
    def remap_hidden_states(**kwargs):
        hidden = kwargs["hidden_states"]
        topk_ids = kwargs["topk_ids"]
        remapped = kwargs["remapped_hidden_states"]
        rows_per_expert = kwargs["rows_per_expert"]
        route_to_row = kwargs["unpermuted_row_to_permuted_row"]
        next_row = 0
        for expert in range(kwargs["local_experts_num"]):
            routes = (topk_ids.flatten() == expert).nonzero().flatten()
            rows_per_expert[expert] = routes.numel()
            for route in routes.tolist():
                remapped[next_row].copy_(hidden[route // topk_ids.shape[1]])
                route_to_row.flatten()[route] = next_row
                next_row += 1

    def grouped_gemm(**kwargs):
        start = 0
        for expert, rows in enumerate(kwargs["rows_per_expert"].tolist()):
            end = start + rows
            kwargs["ptr_D"][start:end].copy_(
                kwargs["ptr_A"][start:end] @ kwargs["ptr_B"][expert].T
            )
            start = end

    def moe_gather(output, gemm2, topk_weights, route_to_row, num_experts):
        del num_experts
        output.zero_()
        for route, row in enumerate(route_to_row.flatten().tolist()):
            output[route // topk_weights.shape[1]].add_(
                gemm2[row] * topk_weights.flatten()[route]
            )

    def gelu_tanh_and_mul(output, input):
        width = output.shape[-1]
        output.copy_(F.gelu(input[:, :width], approximate="tanh") * input[:, width:])

    def gelu_and_mul(output, input):
        width = output.shape[-1]
        output.copy_(F.gelu(input[:, :width]) * input[:, width:])

    monkeypatch.setattr(torch.ops._moe_C, "remap_hidden_states", remap_hidden_states)
    monkeypatch.setattr(
        torch.ops._xpu_C, "cutlass_grouped_gemm_interface", grouped_gemm
    )
    monkeypatch.setattr(torch.ops._moe_C, "moe_gather", moe_gather)
    monkeypatch.setattr(torch.ops._C, "gelu_tanh_and_mul", gelu_tanh_and_mul)
    monkeypatch.setattr(torch.ops._C, "gelu_and_mul", gelu_and_mul)


@pytest.mark.parametrize(
    ("expert_cls", "padded_intermediate", "packed_w13_lora", "activation"),
    [
        (XPUExpertsFp8, False, False, MoEActivation.GELU_TANH),
        (XPUExpertsWNA16, True, False, MoEActivation.GELU),
        (XPUExpertsWNA16, True, True, MoEActivation.GELU),
    ],
)
def test_xpu_quantized_lora_apply_preserves_numeric_routing(
    monkeypatch,
    expert_cls,
    padded_intermediate,
    packed_w13_lora,
    activation,
) -> None:
    _install_cpu_xpu_ops(monkeypatch)

    w13 = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 1.0], [1.0, -0.5]],
        ]
    )
    w2 = torch.tensor([[[1.0], [2.0]], [[-1.0], [0.5]]])
    if padded_intermediate:
        zeros = torch.zeros_like(w13[:, :1])
        w13 = torch.cat((w13[:, :1], zeros, w13[:, 1:], zeros), dim=1)
        w2 = torch.cat((w2, torch.zeros_like(w2)), dim=2)

    converted_w13 = w13.clone()
    converted_w2 = w2.clone()
    if expert_cls is XPUExpertsWNA16:
        # Model the INT4 conversion boundary returning owned converted refs.
        converted_w13.add_(0.125)
        converted_w2.sub_(0.125)

    class FakeXpuFusedMoe:
        def __init__(self, **kwargs):
            self.w13 = converted_w13
            self.w2 = converted_w2
            self.gemm1_wei_scales = kwargs["w13_scales"]
            self.gemm2_wei_scales = kwargs["w2_scales"]
            self.w13_bias = kwargs["w13_bias"]
            self.w2_bias = kwargs["w2_bias"]
            self.inter_size = self.w13.shape[1] // 2
            self.inter_size_scale = 1
            self.num_experts = kwargs["num_experts"]
            self.local_experts_num = self.num_experts
            self.total_experts_num = self.num_experts
            self.ep_size = kwargs["ep_size"]
            self.expert_map = None

        def apply(self, output, hidden_states, topk_weights, topk_ids):
            del hidden_states, topk_weights, topk_ids
            output.fill_(-999)

    monkeypatch.setattr(xpu_moe, "XpuFusedMoe", FakeXpuFusedMoe, raising=False)

    token_lora_mapping = torch.tensor([0, 1], dtype=torch.int32)
    punica = _CpuPunicaMoE(token_lora_mapping)
    w13_a: tuple[torch.Tensor, ...] = (
        torch.tensor([[[[0.2, 0.1]], [[-0.1, 0.3]]], [[[0.4, 0.0]], [[0.1, -0.2]]]]),
        torch.tensor([[[[0.0, 0.3]], [[0.2, 0.1]]], [[[-0.2, 0.2]], [[0.3, 0.1]]]]),
    )
    w13_b: tuple[torch.Tensor, ...] = (
        torch.tensor([[[[0.5]], [[1.0]]], [[[0.25]], [[-0.5]]]]),
        torch.tensor([[[[1.0]], [[-0.25]]], [[[0.5]], [[0.75]]]]),
    )
    expected_w13_a = w13_a
    expected_w13_b = w13_b
    if packed_w13_lora:
        packed_b = torch.zeros(2, 2, 2, 2)
        packed_b[:, :, 0, 0] = w13_b[0].squeeze(-1).squeeze(-1)
        packed_b[:, :, 1, 1] = w13_b[1].squeeze(-1).squeeze(-1)
        w13_a = (torch.cat(w13_a, dim=-2),)
        w13_b = (packed_b,)
    w2_a = (torch.tensor([[[[0.4]], [[-0.2]]], [[[0.3]], [[0.5]]]]),)
    w2_b = (
        torch.tensor(
            [
                [[[0.5], [-0.25]], [[0.2], [0.4]]],
                [[[-0.1], [0.3]], [[0.6], [-0.2]]],
            ]
        ),
    )
    context = MoELoRAContext(
        w13_lora_a_stacked=w13_a,
        w13_lora_b_stacked=w13_b,
        w2_lora_a_stacked=w2_a,
        w2_lora_b_stacked=w2_b,
        adapter_enabled=torch.tensor([1, 1], dtype=torch.int32),
        max_loras=2,
        top_k=2,
        w13_num_slices=len(w13_b),
        fully_sharded=False,
        tp_rank=0,
        tp_size=1,
        local_num_experts=2,
        punica_wrapper=punica,
        use_tuned_config=False,
        local_token_lora_mapping=token_lora_mapping,
    )
    if packed_w13_lora:
        # Model an EP/DP dispatch whose gathered expert input has more rows than
        # the pre-dispatch context stash. The staged XPU path must use the
        # gathered hidden_states tensor in this case.
        context.original_hidden_states = torch.full((1, 2), 999.0)

    expert = object.__new__(expert_cls)
    expert.moe_config = SimpleNamespace(num_local_experts=2, ep_rank=0, ep_size=1)
    expert.quant_config = SimpleNamespace(
        weight_quant_dtype=(
            torch.uint8 if expert_cls is XPUExpertsWNA16 else torch.float8_e4m3fn
        ),
        gemm1_clamp_limit=None,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
    )
    expert.gemm1_clamp_limit = None
    expert.fused_moe_impl = None
    expert.set_lora_context(context)

    hidden = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    topk_ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
    output = torch.empty_like(hidden)
    original_apply_moe_activation = xpu_moe.apply_moe_activation
    activation_metadata = {}

    def capture_activation_metadata(*args, **kwargs):
        activation_metadata["topk_ids"] = kwargs.get("topk_ids")
        activation_metadata["expert_map"] = kwargs.get("expert_map")
        return original_apply_moe_activation(*args, **kwargs)

    monkeypatch.setattr(xpu_moe, "apply_moe_activation", capture_activation_metadata)
    expert.apply(
        output,
        hidden,
        w13,
        w2,
        topk_weights,
        topk_ids,
        activation,
        2,
        None,
        None,
        None,
        torch.empty(0),
        torch.empty(0),
        None,
        False,
    )
    assert activation_metadata == {"topk_ids": None, "expert_map": None}

    expected = torch.zeros_like(output)
    for token in range(hidden.shape[0]):
        adapter = int(token_lora_mapping[token])
        for slot, expert_id in enumerate(topk_ids[token].tolist()):
            pre_act = converted_w13[expert_id] @ hidden[token]
            inter_size = pre_act.numel() // 2
            for slice_id in range(2):
                dst = 0 if slice_id == 0 else inter_size
                pre_act[dst] += (
                    (hidden[token] @ expected_w13_a[slice_id][adapter, expert_id].T)
                    @ expected_w13_b[slice_id][adapter, expert_id].T
                ).squeeze()
            approximate = "tanh" if activation == MoEActivation.GELU_TANH else "none"
            activated = (
                F.gelu(pre_act[:inter_size], approximate=approximate)
                * pre_act[inter_size:]
            )
            route_out = converted_w2[expert_id] @ activated
            route_out += (activated[:1] @ w2_a[0][adapter, expert_id].T) @ w2_b[0][
                adapter, expert_id
            ].T
            expected[token] += route_out * topk_weights[token, slot]

    torch.testing.assert_close(output, expected)

    context.enable_moe_shared_loras = True
    with pytest.raises(ValueError, match="shared expert factors"):
        expert.apply(
            output,
            hidden,
            w13,
            w2,
            topk_weights,
            topk_ids,
            activation,
            2,
            None,
            None,
            None,
            torch.empty(0),
            torch.empty(0),
            None,
            False,
        )
    context.enable_moe_shared_loras = False

    context.adapter_enabled_cpu = [False, True]

    def unexpected_module_staged_remap(**kwargs):
        del kwargs
        raise AssertionError("inactive MoE module entered the staged LoRA path")

    monkeypatch.setattr(
        torch.ops._moe_C, "remap_hidden_states", unexpected_module_staged_remap
    )
    expert.apply(
        output,
        hidden,
        w13,
        w2,
        topk_weights,
        topk_ids,
        activation,
        2,
        None,
        None,
        None,
        torch.empty(0),
        torch.empty(0),
        None,
        False,
    )
    torch.testing.assert_close(output, torch.full_like(output, -999))

    # A subsequent base-only batch must retain the monolithic fast path even
    # after the split LoRA path initialized and used the converted weights.
    punica.no_lora = True

    def unexpected_staged_remap(**kwargs):
        del kwargs
        raise AssertionError("no-adapter batch entered the staged LoRA path")

    monkeypatch.setattr(
        torch.ops._moe_C, "remap_hidden_states", unexpected_staged_remap
    )
    expert.apply(
        output,
        hidden,
        w13,
        w2,
        topk_weights,
        topk_ids,
        activation,
        2,
        None,
        None,
        None,
        torch.empty(0),
        torch.empty(0),
        None,
        False,
    )
    torch.testing.assert_close(output, torch.full_like(output, -999))


def test_xpu_moe_lora_weights_use_adjacent_slice_storage(monkeypatch) -> None:
    from vllm.lora.layers import fused_moe as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer.current_platform, "is_xpu", lambda: True)
    layer = object.__new__(FusedMoEWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(
        num_local_experts=2,
        num_experts=2,
        hidden_dim=4,
        intermediate_size_per_partition=3,
        moe_parallel_config=SimpleNamespace(use_ep=False),
    )
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.device = torch.device("cpu")
    layer._w13_slices = 2
    config = LoRAConfig(
        max_loras=2,
        max_lora_rank=8,
        lora_dtype=torch.float32,
    )

    layer.create_lora_weights(2, config)

    for slices in (layer.w13_lora_a_stacked, layer.w13_lora_b_stacked):
        assert _get_xpu_direct_weight(list(slices)) is slices[0]
        assert slices[1].data_ptr() == (
            slices[0].data_ptr() + slices[0].numel() * slices[0].element_size()
        )


def test_punica_xpu_refreshes_no_lora_across_prefill_and_decode() -> None:
    wrapper = object.__new__(PunicaWrapperXPU)
    PunicaWrapperBase.__init__(wrapper, 8, 8, torch.device("cpu"))
    wrapper.lora_config = SimpleNamespace(specialize_active_lora=False)
    prepared = []
    wrapper.token_mapping_meta = SimpleNamespace(
        prepare_tensors=lambda tensor: prepared.append("token")
    )
    wrapper.prompt_mapping_meta = SimpleNamespace(
        prepare_tensors=lambda tensor: prepared.append("prompt")
    )

    cases = [
        (LoRAMapping((0, 0), (0,), is_prefill=True), True, ()),
        (LoRAMapping((1,), (1,), is_prefill=False), False, (0,)),
        (LoRAMapping((0,), (0,), is_prefill=False), True, ()),
        (LoRAMapping((1, 1), (1,), is_prefill=True), False, (0,)),
    ]
    for mapping, expected, active_lora_indices in cases:
        prepared.clear()
        wrapper.update_metadata(mapping, [1], 1, 16)
        assert wrapper.no_lora is expected
        assert wrapper.active_lora_indices == active_lora_indices
        assert prepared == ([] if expected else ["token", "prompt"])


def test_punica_xpu_skips_dense_lora_ops_for_base_only_batch(monkeypatch) -> None:
    from vllm.lora.punica_wrapper import punica_xpu

    wrapper = object.__new__(PunicaWrapperXPU)
    wrapper.no_lora = True

    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("base-only batch launched a dense LoRA kernel")

    monkeypatch.setattr(punica_xpu, "bgmv_expand", unexpected_kernel)
    monkeypatch.setattr(punica_xpu, "bgmv_shrink", unexpected_kernel)
    monkeypatch.setattr(punica_xpu, "bgmv_linear", unexpected_kernel)

    y = torch.ones(2, 4)
    x = torch.ones(2, 4)
    y_snapshot = y.clone()
    lora_a = torch.ones(1, 4, 2)
    lora_b = torch.ones(1, 4, 2)

    wrapper.add_lora_embedding(y, x, lora_b)
    wrapper.add_lora_linear(y, x, (lora_a,), (lora_b,), 1.0, (4,))
    wrapper.add_lora_logits(y, x, lora_a, lora_b, 1.0)

    assert torch.equal(y, y_snapshot)

    wrapper.no_lora = False
    wrapper.active_lora_indices = (0,)
    inactive_module = [False]
    wrapper.add_lora_embedding(y, x, lora_b, adapter_enabled_cpu=inactive_module)
    wrapper.add_lora_linear(
        y,
        x,
        (lora_a,),
        (lora_b,),
        1.0,
        (4,),
        adapter_enabled_cpu=inactive_module,
    )
    wrapper.add_lora_logits(
        y, x, lora_a, lora_b, 1.0, adapter_enabled_cpu=inactive_module
    )

    assert torch.equal(y, y_snapshot)


def test_xpu_logits_processor_passes_inactive_adapter_state(monkeypatch) -> None:
    from vllm.lora.layers import logits_processor as logits_processor_layer
    from vllm.lora.punica_wrapper import punica_xpu

    layer = object.__new__(LogitsProcessorWithLoRA)
    torch.nn.Module.__init__(layer)
    base_logits = torch.arange(8, dtype=torch.float32).view(2, 4)
    layer.base_layer = SimpleNamespace(
        vocab_size=4,
        _apply_head=lambda lm_head, hidden_states, embedding_bias: base_logits.clone(),
        _gather_logits=lambda logits: logits,
    )
    layer.hidden_size = 3
    layer.device = torch.device("cpu")
    layer.sharded_to_full_mapping = None
    layer.create_lora_weights(
        1,
        LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.float32),
    )

    wrapper = object.__new__(PunicaWrapperXPU)
    wrapper.no_lora = False
    wrapper.active_lora_indices = (0,)
    layer.set_mapping(wrapper)

    layer.set_lora(0, torch.ones(2, 3), torch.ones(4, 2))
    assert layer.adapter_enabled_cpu == [True]
    assert layer.adapter_ranks_cpu == [2]
    layer.reset_lora(0)
    assert layer.adapter_enabled_cpu == [False]
    assert layer.adapter_ranks_cpu == [0]

    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("inactive lm_head adapter launched a LoRA kernel")

    monkeypatch.setattr(punica_xpu, "bgmv_expand", unexpected_kernel)
    monkeypatch.setattr(punica_xpu, "bgmv_shrink", unexpected_kernel)
    monkeypatch.setattr(
        logits_processor_layer.current_platform, "can_update_inplace", lambda: True
    )

    result = layer._get_logits(
        hidden_states=torch.ones(2, 3),
        lm_head=SimpleNamespace(),
        embedding_bias=None,
    )

    torch.testing.assert_close(result, base_logits)


def test_punica_xpu_shrink_writes_directly_to_output(monkeypatch) -> None:
    from vllm.lora.punica_wrapper import punica_xpu

    wrapper = object.__new__(PunicaWrapperXPU)
    wrapper._token_lora_indices = torch.tensor([0, 0], dtype=torch.long)
    output = torch.zeros(2, 3)
    inputs = torch.ones(2, 4)
    weights = torch.ones(1, 1, 3, 4)

    def fake_bgmv_shrink(inputs_arg, weights_arg, output_arg, indices_arg, scale):
        assert inputs_arg is inputs
        assert weights_arg is weights
        assert output_arg is output
        assert torch.equal(indices_arg, wrapper._token_lora_indices)
        assert scale == 0.5
        output_arg.add_(7)

    monkeypatch.setattr(punica_xpu, "bgmv_shrink", fake_bgmv_shrink)

    wrapper._apply_shrink(output, inputs, weights, 0.5)

    torch.testing.assert_close(output, torch.full_like(output, 7))


def test_punica_xpu_shrink_stages_mixed_output_dtype(monkeypatch) -> None:
    from vllm.lora.punica_wrapper import punica_xpu

    wrapper = object.__new__(PunicaWrapperXPU)
    wrapper._token_lora_indices = torch.tensor([0, 0], dtype=torch.long)
    output = torch.empty(2, 3, dtype=torch.float64)
    inputs = torch.ones(2, 4)
    weights = torch.ones(1, 1, 3, 4)

    def fake_bgmv_shrink(inputs_arg, weights_arg, output_arg, indices_arg, scale):
        assert inputs_arg is inputs
        assert weights_arg is weights
        assert output_arg is not output
        assert output_arg.dtype == inputs.dtype
        assert torch.equal(indices_arg, wrapper._token_lora_indices)
        assert scale == 0.5
        output_arg.fill_(7)

    monkeypatch.setattr(punica_xpu, "bgmv_shrink", fake_bgmv_shrink)

    wrapper._apply_shrink(output, inputs, weights, 0.5)

    torch.testing.assert_close(output, torch.full_like(output, 7))


@pytest.mark.parametrize(
    ("active_lora_indices", "expected_rank"),
    [((1,), 64), ((0, 1), 128)],
)
def test_punica_xpu_uses_single_active_adapter_rank(
    monkeypatch, active_lora_indices, expected_rank
) -> None:
    from vllm.lora.punica_wrapper import punica_xpu

    wrapper = object.__new__(PunicaWrapperXPU)
    wrapper.no_lora = False
    wrapper.active_lora_indices = active_lora_indices
    wrapper._token_lora_indices = torch.tensor([1, 1], dtype=torch.long)
    monkeypatch.setattr(punica_xpu, "has_fused_lora_linear", lambda: False)
    output = torch.zeros(2, 4)
    inputs = torch.ones(2, 4)
    lora_a = torch.ones(2, 1, 128, 4)
    lora_b = torch.ones(2, 1, 4, 128)
    observed = {}

    def fake_bgmv_linear(
        inputs_arg,
        lora_a_arg,
        lora_b_arg,
        output_arg,
        indices_arg,
        output_slices_arg,
        active_rank,
        scaling,
        add_inputs=True,
    ):
        del (
            inputs_arg,
            lora_a_arg,
            lora_b_arg,
            output_arg,
            indices_arg,
            output_slices_arg,
            scaling,
            add_inputs,
        )
        observed["active_rank"] = active_rank

    monkeypatch.setattr(punica_xpu, "bgmv_linear", fake_bgmv_linear)

    wrapper.add_lora_linear(
        output,
        inputs,
        (lora_a,),
        (lora_b,),
        1.0,
        (4,),
        adapter_enabled_cpu=[True, True],
        adapter_ranks_cpu=[128, 64],
    )

    assert observed == {"active_rank": expected_rank}


@pytest.mark.parametrize("has_fused_op", [False, True])
def test_punica_xpu_selects_linear_index_dtype(monkeypatch, has_fused_op) -> None:
    from vllm.lora.punica_wrapper import punica_xpu

    wrapper = object.__new__(PunicaWrapperXPU)
    wrapper._token_lora_indices = torch.tensor([1, 0, 1], dtype=torch.long)
    wrapper.token_mapping_meta = SimpleNamespace(
        token_lora_mapping=torch.tensor([1, 0, 1], dtype=torch.int32)
    )
    monkeypatch.setattr(punica_xpu, "has_fused_lora_linear", lambda: has_fused_op)

    indices = wrapper._get_linear_lora_indices(torch.empty(2, 4))

    assert indices.dtype == (torch.int32 if has_fused_op else torch.long)
    assert indices.tolist() == [1, 0]


def test_xpu_lora_op_detection_handles_missing_namespace() -> None:
    from vllm.lora.ops.xpu_ops import lora_ops

    assert not lora_ops._has_xpu_op(SimpleNamespace(), "lora_linear")


def test_punica_xpu_uses_gemm_for_large_single_adapter(monkeypatch) -> None:
    from vllm.lora.punica_wrapper import punica_xpu

    wrapper = object.__new__(PunicaWrapperXPU)
    wrapper.no_lora = False
    wrapper.active_lora_indices = (1,)
    wrapper.single_lora_index = 1
    wrapper.is_prefill = True
    inputs = torch.randn(32, 4)
    output = torch.randn(32, 3)
    output_before = output.clone()
    lora_a = torch.randn(2, 1, 8, 4)
    lora_b = torch.randn(2, 1, 3, 8)

    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("large single-adapter batch used BGMV")

    monkeypatch.setattr(punica_xpu, "bgmv_shrink", unexpected_kernel)
    monkeypatch.setattr(punica_xpu, "bgmv_expand_slice", unexpected_kernel)
    monkeypatch.setattr(punica_xpu, "bgmv_linear", unexpected_kernel)
    monkeypatch.setattr(punica_xpu, "_SINGLE_LORA_GEMM_MIN_WORK", 0)

    wrapper.add_lora_linear(
        output,
        inputs,
        (lora_a,),
        (lora_b,),
        0.5,
        (3,),
        adapter_enabled_cpu=[True, True],
        adapter_ranks_cpu=[8, 4],
    )

    expected = output_before + 0.5 * inputs.mm(lora_a[1, 0, :4].t()).mm(
        lora_b[1, 0, :, :4].t()
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("is_prefill", "output_dtype"),
    [(False, torch.float32), (True, torch.float64)],
)
def test_punica_xpu_skips_single_adapter_gemm_when_unsupported(
    monkeypatch, is_prefill, output_dtype
) -> None:
    from vllm.lora.punica_wrapper import punica_xpu

    wrapper = object.__new__(PunicaWrapperXPU)
    wrapper.no_lora = False
    wrapper.active_lora_indices = (0,)
    wrapper.single_lora_index = 0
    wrapper.is_prefill = is_prefill
    wrapper._token_lora_indices = torch.tensor([0, 0], dtype=torch.long)
    inputs = torch.randn(2, 4)
    output = torch.randn(2, 3, dtype=output_dtype)
    lora_a = torch.randn(1, 1, 4, 4)
    lora_b = torch.randn(1, 1, 3, 4)
    observed = {}

    def fake_bgmv_linear(
        inputs_arg,
        lora_a_arg,
        lora_b_arg,
        output_arg,
        indices_arg,
        output_slices_arg,
        active_rank,
        scaling,
        add_inputs=True,
    ):
        del (
            inputs_arg,
            lora_a_arg,
            lora_b_arg,
            output_arg,
            indices_arg,
            output_slices_arg,
            active_rank,
            scaling,
            add_inputs,
        )
        observed["called"] = True

    monkeypatch.setattr(punica_xpu, "has_fused_lora_linear", lambda: False)
    monkeypatch.setattr(punica_xpu, "bgmv_linear", fake_bgmv_linear)
    monkeypatch.setattr(punica_xpu, "_SINGLE_LORA_GEMM_MIN_WORK", 0)

    wrapper.add_lora_linear(
        output,
        inputs,
        (lora_a,),
        (lora_b,),
        1.0,
        (3,),
        adapter_enabled_cpu=[True],
        adapter_ranks_cpu=[4],
    )

    assert observed == {"called": True}


def test_dense_lora_tracks_adapter_rank() -> None:
    layer = object.__new__(BaseLinearLayerWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.n_slices = 1
    layer.tp_size = 1
    layer.adapter_enabled_cpu = [False]
    layer.adapter_ranks_cpu = [0]
    layer.lora_a_stacked = (torch.zeros(1, 1, 128, 4),)
    layer.lora_b_stacked = (torch.zeros(1, 1, 4, 128),)
    lora_a = torch.ones(64, 4)
    lora_b = torch.ones(4, 64)

    layer.set_lora(0, lora_a, lora_b)

    assert layer.adapter_enabled_cpu == [True]
    assert layer.adapter_ranks_cpu == [64]

    layer.reset_lora(0)

    assert layer.adapter_enabled_cpu == [False]
    assert layer.adapter_ranks_cpu == [0]


@pytest.mark.parametrize(
    ("ranks", "expected_rank"),
    [((64, 64), 64), ((64, 128), 128)],
)
def test_packed_lora_tracks_common_adapter_rank(ranks, expected_rank) -> None:
    layer = object.__new__(MergedColumnParallelLinearWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.n_slices = 2
    layer.tp_size = 1
    layer.lora_config = SimpleNamespace(max_lora_rank=128)
    layer.adapter_enabled_cpu = [False]
    layer.adapter_ranks_cpu = [0]
    layer.lora_a_stacked = (
        torch.zeros(1, 1, 128, 4),
        torch.zeros(1, 1, 128, 4),
    )
    layer.lora_b_stacked = (
        torch.zeros(1, 1, 4, 128),
        torch.zeros(1, 1, 4, 128),
    )
    lora_a = [torch.ones(rank, 4) for rank in ranks]
    lora_b = [torch.ones(4, rank) for rank in ranks]

    layer.set_lora(0, lora_a, lora_b)

    assert layer.adapter_enabled_cpu == [True]
    assert layer.adapter_ranks_cpu == [expected_rank]


@pytest.mark.parametrize(
    "layer_cls",
    [RowParallelLinearWithLoRA, ReplicatedLinearWithLoRA],
)
def test_parallel_lora_uses_base_forward_for_inactive_batch(layer_cls) -> None:
    class BaseLayer(torch.nn.Module):
        def forward(self, inputs):
            return inputs + 1

    layer = object.__new__(layer_cls)
    torch.nn.Module.__init__(layer)
    layer.base_layer = BaseLayer()
    layer.adapter_enabled_cpu = [False]
    layer._has_active_lora = lambda _: False
    inputs = torch.zeros(2, 4)

    output = layer(inputs)

    torch.testing.assert_close(output, torch.ones_like(inputs))


def test_embedding_lora_uses_base_forward_for_inactive_batch() -> None:
    class BaseLayer(torch.nn.Module):
        def forward(self, inputs):
            return inputs + 1

    layer = object.__new__(VocabParallelEmbeddingWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.base_layer = BaseLayer()
    layer.adapter_enabled_cpu = [False]
    layer._has_active_lora = lambda _: False
    inputs = torch.zeros(2, 4)

    output = layer(inputs)

    torch.testing.assert_close(output, torch.ones_like(inputs))


def test_get_lora_device_skips_none_dense_weight() -> None:
    qweight = torch.empty(1)
    layer = SimpleNamespace(weight=None, qweight=qweight)
    assert _get_lora_device(layer) == qweight.device


def test_moe_lora_defaults_to_modular_before_weights_are_created() -> None:
    layer = object.__new__(FusedMoEWithLoRA)
    layer.punica_wrapper = SimpleNamespace(
        has_active_lora=lambda _: pytest.fail("adapter state is not initialized")
    )

    assert layer._has_active_moe_lora()


def test_inactive_moe_lora_uses_wrapped_monolithic_method() -> None:
    expected = torch.ones(2, 3)

    class MonolithicMethod:
        is_monolithic = True

        def apply_monolithic(self, **kwargs):
            assert kwargs["input_ids"] is None
            return expected

    method = object.__new__(FusedMoEModularMethod)
    method.old_quant_method = MonolithicMethod()
    method.moe_kernel = SimpleNamespace(is_monolithic=False)
    method.use_modular_kernel = lambda: False

    assert method.is_monolithic
    assert (
        method.apply_monolithic(
            layer=SimpleNamespace(),
            x=torch.empty(2, 3),
            router_logits=torch.empty(2, 4),
        )
        is expected
    )

    method.use_modular_kernel = lambda: True
    assert not method.is_monolithic
