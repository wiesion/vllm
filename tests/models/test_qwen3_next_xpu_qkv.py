# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.model_executor.models import _qwen3_next_xpu, qwen3_next
from vllm.model_executor.models._qwen3_next_xpu import Qwen3NextQkvEsimdAdapter


@pytest.mark.parametrize(
    ("quant_name", "expected"),
    [(None, None), ("fp8", None), ("sym_int4", "same")],
)
def test_qwen3_next_router_quantization_is_scoped_to_sym_int4(quant_name, expected):
    quant_config = None if quant_name is None else Mock()
    if quant_config is not None:
        quant_config.get_name.return_value = quant_name

    result = qwen3_next._get_router_quant_config(quant_config)

    if expected == "same":
        assert result is quant_config
    else:
        assert result is None


def test_qwen3_next_enables_batch_esimd_per_layer():
    layers = [SimpleNamespace(), SimpleNamespace()]

    qwen3_next._enable_sym_int4_batch_esimd(*layers)

    assert all(layer._sym_int4_allow_batch_esimd for layer in layers)


def test_qwen3_next_int4_qkv_postprocess_lazily_uses_esimd():
    loads = []
    calls = []

    def qkv_op(*args):
        calls.append(args)

    module = SimpleNamespace(esimd_qkv_split_norm_rope=qkv_op)

    def load_module():
        loads.append(module)
        return module

    adapter = Qwen3NextQkvEsimdAdapter(load_module)
    projection = SimpleNamespace(quant_method=SimpleNamespace(is_sym_int4=True))

    with FakeTensorMode():
        qkv = torch.empty((2, 1536), dtype=torch.float16, device="xpu")
        positions = torch.arange(2, dtype=torch.int64, device="xpu")
        norm_wq = torch.empty(256, dtype=torch.float16, device="xpu")
        norm_wk = torch.empty(256, dtype=torch.float16, device="xpu")
        cos_sin_cache = torch.empty((2048, 64), dtype=torch.float16, device="xpu")

        result = adapter.try_postprocess(
            qkv,
            positions,
            projection=projection,
            norm_wq=norm_wq,
            norm_wk=norm_wk,
            cos_sin_cache=cos_sin_cache,
            q_heads=2,
            kv_heads=1,
            head_dim=256,
            rotary_dim=64,
            norm_epsilon=1e-6,
            is_neox_style=True,
            attn_output_gate=True,
            text_only=True,
        )

    assert result is not None
    assert result.q.shape == (2, 512)
    assert result.gate is not None
    assert result.gate.shape == (2, 512)
    assert result.k.shape == (2, 256)
    assert result.v.shape == (2, 256)
    assert len(loads) == 1
    assert calls[0][0] is qkv
    assert calls[0][5] is norm_wq
    assert calls[0][6] is norm_wk
    assert calls[0][7].dtype == torch.int32
    assert calls[0][7].is_contiguous()
    assert calls[0][8:] == (2, 1, True, 64, cos_sin_cache)


def test_qwen3_next_runtime_fp8_qkv_postprocess_uses_same_output_contract():
    calls = []
    adapter = Qwen3NextQkvEsimdAdapter(
        lambda: SimpleNamespace(
            esimd_qkv_split_norm_rope=lambda *args: calls.append(args)
        )
    )
    with (
        FakeTensorMode(),
        patch.object(torch.compiler, "is_compiling", return_value=False) as compiling,
    ):
        projection = SimpleNamespace(
            quant_method=object(),
            weight=torch.empty((1536, 512), dtype=torch.float8_e4m3fn, device="xpu"),
            weight_scale=torch.ones(1, dtype=torch.float32, device="xpu"),
        )
        qkv = torch.empty((1, 1536), dtype=torch.float16, device="xpu")
        result = adapter.try_postprocess(
            qkv,
            torch.zeros(1, dtype=torch.int64, device="xpu"),
            projection=projection,
            norm_wq=torch.empty(256, dtype=torch.float16, device="xpu"),
            norm_wk=torch.empty(256, dtype=torch.float16, device="xpu"),
            cos_sin_cache=torch.empty((2048, 64), dtype=torch.float16, device="xpu"),
            q_heads=2,
            kv_heads=1,
            head_dim=256,
            rotary_dim=64,
            norm_epsilon=1e-6,
            is_neox_style=True,
            attn_output_gate=True,
            text_only=True,
        )
        second_result = adapter.try_postprocess(
            qkv,
            torch.zeros(1, dtype=torch.int64, device="xpu"),
            projection=projection,
            norm_wq=torch.empty(256, dtype=torch.float16, device="xpu"),
            norm_wk=torch.empty(256, dtype=torch.float16, device="xpu"),
            cos_sin_cache=torch.empty((2048, 64), dtype=torch.float16, device="xpu"),
            q_heads=2,
            kv_heads=1,
            head_dim=256,
            rotary_dim=64,
            norm_epsilon=1e-6,
            is_neox_style=True,
            attn_output_gate=True,
            text_only=True,
        )

    assert result is not None
    assert second_result is not None
    assert result.q.shape == (1, 512)
    assert result.k.shape == (1, 256)
    assert result.v.shape == (1, 256)
    assert result.gate is not None
    assert len(calls) == 2
    assert compiling.call_count == 1
    assert hasattr(projection, "_esimd_qkv_postprocess_plan")
    assert projection._esimd_qkv_decode_buffers[0].shape == (1, 512)


@pytest.mark.parametrize(
    ("active_lora", "expected"),
    [(False, True), (True, False)],
    ids=["inactive-lora", "active-lora"],
)
def test_qwen3_next_qkv_postprocess_handles_lora_wrapper(active_lora, expected):
    calls = []
    adapter = Qwen3NextQkvEsimdAdapter(
        lambda: SimpleNamespace(
            esimd_qkv_split_norm_rope=lambda *args: calls.append(args)
        )
    )
    base_projection = SimpleNamespace(quant_method=SimpleNamespace(is_sym_int4=True))
    projection = SimpleNamespace(
        base_layer=base_projection,
        punica_wrapper=SimpleNamespace(no_lora=not active_lora),
        is_lora_active=Mock(return_value=active_lora),
    )
    cached_result = None
    with FakeTensorMode():
        qkv = torch.empty((1, 1536), dtype=torch.float16, device="xpu")
        positions = torch.zeros(1, dtype=torch.int64, device="xpu")
        norm_wq = torch.empty(256, dtype=torch.float16, device="xpu")
        norm_wk = torch.empty(256, dtype=torch.float16, device="xpu")
        cos_sin_cache = torch.empty((2048, 64), dtype=torch.float16, device="xpu")

        def postprocess():
            return adapter.try_postprocess(
                qkv,
                positions,
                projection=projection,
                norm_wq=norm_wq,
                norm_wk=norm_wk,
                cos_sin_cache=cos_sin_cache,
                q_heads=2,
                kv_heads=1,
                head_dim=256,
                rotary_dim=64,
                norm_epsilon=1e-6,
                is_neox_style=True,
                attn_output_gate=True,
                text_only=True,
            )

        result = postprocess()
        if not active_lora:
            projection.punica_wrapper.no_lora = False
            projection.is_lora_active.return_value = True
            cached_result = postprocess()

    assert (result is not None) is expected
    assert bool(calls) is expected
    if not active_lora:
        assert cached_result is None
        assert len(calls) == 1
    projection.is_lora_active.assert_not_called()


@pytest.mark.parametrize(
    ("active_lora", "expected"),
    [(False, True), (True, False)],
    ids=["inactive-lora", "active-lora"],
)
def test_qwen3_next_fp8_linear_handles_lora_wrapper(active_lora, expected):
    op = Mock()
    with (
        FakeTensorMode(),
        patch.object(
            _qwen3_next_xpu._QWEN3_NEXT_QKV_ESIMD_ADAPTER,
            "_get_module",
            return_value=SimpleNamespace(esimd_gemv_fp8_pert=op),
        ),
    ):
        base_projection = SimpleNamespace(
            weight=torch.empty((16, 8), dtype=torch.float8_e4m3fn, device="xpu"),
            weight_scale=torch.ones(1, dtype=torch.float32, device="xpu"),
            bias=None,
        )
        projection = SimpleNamespace(
            base_layer=base_projection,
            punica_wrapper=SimpleNamespace(no_lora=not active_lora),
            is_lora_active=Mock(return_value=active_lora),
        )
        inputs = torch.empty((1, 8), dtype=torch.float16, device="xpu")
        output = _qwen3_next_xpu.try_esimd_fp8_linear(
            inputs, projection, "_test_output"
        )
        cached_output = None
        if not active_lora:
            projection.punica_wrapper.no_lora = False
            projection.is_lora_active.return_value = True
            cached_output = _qwen3_next_xpu.try_esimd_fp8_linear(
                inputs, projection, "_test_output"
            )

    assert (output is not None) is expected
    assert op.called is expected
    if not active_lora:
        assert cached_output is None
        op.assert_called_once()
    projection.is_lora_active.assert_not_called()


@pytest.mark.parametrize(
    ("active_lora", "expected"),
    [(False, True), (True, False)],
    ids=["inactive-lora", "active-lora"],
)
def test_qwen3_next_int4_linear_caches_decode_plan(active_lora, expected):
    op = Mock()
    with (
        FakeTensorMode(),
        patch.object(
            _qwen3_next_xpu._QWEN3_NEXT_QKV_ESIMD_ADAPTER,
            "_get_module",
            return_value=SimpleNamespace(esimd_gemv_int4=op),
        ),
        patch.object(torch.compiler, "is_compiling", return_value=False),
    ):
        base_projection = SimpleNamespace(
            quant_method=SimpleNamespace(is_sym_int4=True),
            weight_esimd=torch.empty((16, 64), dtype=torch.uint8, device="xpu"),
            scale_esimd=torch.ones((16, 1), dtype=torch.float16, device="xpu"),
            bias=None,
        )
        projection = SimpleNamespace(
            base_layer=base_projection,
            punica_wrapper=SimpleNamespace(no_lora=not active_lora),
        )
        inputs = torch.empty((1, 128), dtype=torch.float16, device="xpu")
        output = _qwen3_next_xpu.try_esimd_int4_linear(
            inputs, projection, "_test_int4_output"
        )
        cached_output = None
        if not active_lora:
            cached_output = _qwen3_next_xpu.try_esimd_int4_linear(
                inputs, projection, "_test_int4_output"
            )

    assert (output is not None) is expected
    assert op.called is expected
    if not active_lora:
        assert cached_output is output
        assert op.call_count == 2


def test_qwen3_next_block_fp8_qkv_postprocess_is_eligible():
    calls = []
    adapter = Qwen3NextQkvEsimdAdapter(
        lambda: SimpleNamespace(
            esimd_qkv_split_norm_rope=lambda *args: calls.append(args)
        )
    )
    with (
        FakeTensorMode(),
        patch.object(torch.compiler, "is_compiling", return_value=False),
    ):
        projection = SimpleNamespace(
            quant_method=object(),
            weight=torch.empty((1536, 512), dtype=torch.float8_e4m3fn, device="xpu"),
            weight_scale_inv=torch.ones((12, 4), dtype=torch.float32, device="xpu"),
            weight_block_size=(128, 128),
        )
        result = adapter.try_postprocess(
            torch.empty((1, 1536), dtype=torch.float16, device="xpu"),
            torch.zeros(1, dtype=torch.int64, device="xpu"),
            projection=projection,
            norm_wq=torch.empty(256, dtype=torch.float16, device="xpu"),
            norm_wk=torch.empty(256, dtype=torch.float16, device="xpu"),
            cos_sin_cache=torch.empty((2048, 64), dtype=torch.float16, device="xpu"),
            q_heads=2,
            kv_heads=1,
            head_dim=256,
            rotary_dim=64,
            norm_epsilon=1e-6,
            is_neox_style=True,
            attn_output_gate=True,
            text_only=True,
        )
    assert result is not None
    assert len(calls) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("text_only,should_dispatch", [(True, True), (False, False)])
def test_qwen3_next_mrope_positions_require_text_only_model(text_only, should_dispatch):
    calls = []
    adapter = Qwen3NextQkvEsimdAdapter(
        lambda: SimpleNamespace(
            esimd_qkv_split_norm_rope=lambda *args: calls.append(args)
        )
    )
    projection = SimpleNamespace(quant_method=SimpleNamespace(is_sym_int4=True))
    with FakeTensorMode():
        qkv = torch.empty((2, 1536), dtype=torch.float16, device="xpu")
        positions = torch.arange(2, dtype=torch.int64, device="xpu").expand(3, -1)
        result = adapter.try_postprocess(
            qkv,
            positions,
            projection=projection,
            norm_wq=torch.empty(256, dtype=torch.float16, device="xpu"),
            norm_wk=torch.empty(256, dtype=torch.float16, device="xpu"),
            cos_sin_cache=torch.empty((2048, 64), dtype=torch.float16, device="xpu"),
            q_heads=2,
            kv_heads=1,
            head_dim=256,
            rotary_dim=64,
            norm_epsilon=1e-6,
            is_neox_style=True,
            attn_output_gate=True,
            text_only=text_only,
        )

    assert (result is not None) is should_dispatch
    assert bool(calls) is should_dispatch
    if should_dispatch:
        assert calls[0][7].shape == (2,)


@pytest.mark.parametrize(
    "case",
    [
        "not_int4",
        "compiling",
        "m65",
        "multid_positions",
        "wrong_dtype",
        "wrong_device",
        "noncontiguous_qkv",
        "wrong_cache_layout",
    ],
)
def test_qwen3_next_qkv_ineligible_inputs_fall_back_before_loading(case):
    loads = []
    adapter = Qwen3NextQkvEsimdAdapter(lambda: loads.append(object()))
    projection = SimpleNamespace(
        quant_method=SimpleNamespace(is_sym_int4=case != "not_int4")
    )

    rows = 65 if case == "m65" else 2
    with (
        FakeTensorMode(),
        patch.object(torch.compiler, "is_compiling", return_value=case == "compiling"),
    ):
        if case == "noncontiguous_qkv":
            qkv = torch.empty((1536, rows), dtype=torch.float16, device="xpu").t()
        else:
            dtype = torch.bfloat16 if case == "wrong_dtype" else torch.float16
            device = "cpu" if case == "wrong_device" else "xpu"
            qkv = torch.empty((rows, 1536), dtype=dtype, device=device)
        positions = torch.empty(
            (1, rows) if case == "multid_positions" else (rows,),
            dtype=torch.int64,
            device="xpu",
        )
        norm_wq = torch.empty(256, dtype=torch.float16, device="xpu")
        norm_wk = torch.empty(256, dtype=torch.float16, device="xpu")
        cache_width = 128 if case == "wrong_cache_layout" else 64
        cos_sin_cache = torch.empty(
            (2048, cache_width), dtype=torch.float16, device="xpu"
        )

        result = adapter.try_postprocess(
            qkv,
            positions,
            projection=projection,
            norm_wq=norm_wq,
            norm_wk=norm_wk,
            cos_sin_cache=cos_sin_cache,
            q_heads=2,
            kv_heads=1,
            head_dim=256,
            rotary_dim=64,
            norm_epsilon=1e-6,
            is_neox_style=True,
            attn_output_gate=True,
            text_only=True,
        )

    assert result is None
    assert loads == []


def test_qwen3_next_qkv_unavailable_esimd_is_cached_and_falls_back():
    loads: list[None] = []

    def unavailable():
        loads.append(None)
        raise ImportError("optional ESIMD package is absent")

    adapter = Qwen3NextQkvEsimdAdapter(unavailable)
    projection = SimpleNamespace(quant_method=SimpleNamespace(is_sym_int4=True))
    with FakeTensorMode():
        kwargs = dict(
            projection=projection,
            norm_wq=torch.empty(256, dtype=torch.float16, device="xpu"),
            norm_wk=torch.empty(256, dtype=torch.float16, device="xpu"),
            cos_sin_cache=torch.empty((2048, 64), dtype=torch.float16, device="xpu"),
            q_heads=2,
            kv_heads=1,
            head_dim=256,
            rotary_dim=64,
            norm_epsilon=1e-6,
            is_neox_style=True,
            attn_output_gate=True,
            text_only=True,
        )
        qkv = torch.empty((2, 1536), dtype=torch.float16, device="xpu")
        positions = torch.arange(2, device="xpu")
        assert adapter.try_postprocess(qkv, positions, **kwargs) is None
        assert adapter.try_postprocess(qkv, positions, **kwargs) is None

    assert len(loads) == 1


def _attention_stub():
    qkv = torch.zeros(1, 4)
    return SimpleNamespace(
        qkv_proj=Mock(return_value=(qkv, None)),
        q_norm=SimpleNamespace(weight=torch.zeros(1), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.zeros(1), variance_epsilon=1e-6),
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.zeros(1, 1),
            rotary_dim=1,
            is_neox_style=True,
        ),
        num_heads=1,
        num_kv_heads=1,
        head_dim=1,
        attn_output_gate=True,
        _esimd_qkv_text_only=True,
        _project_qkv_gate=Mock(),
        attn=Mock(return_value=torch.ones(1, 2)),
        o_proj=Mock(side_effect=lambda value: (value, None)),
    )


def test_qwen3_next_attention_uses_kernel_activated_gate():
    layer = _attention_stub()
    q = torch.zeros(1, 2)
    k = torch.zeros(1, 1)
    v = torch.zeros(1, 1)
    activated_gate = torch.full((1, 2), 0.25)
    result = (q, k, v, activated_gate)
    hidden_states = torch.zeros(1, 2)
    positions = torch.zeros(1, dtype=torch.int64)

    with patch.object(qwen3_next, "try_esimd_qkv_postprocess", return_value=result):
        output = qwen3_next.Qwen3NextAttention.forward(layer, positions, hidden_states)

    layer._project_qkv_gate.assert_not_called()
    layer.attn.assert_called_once_with(q, k, v)
    torch.testing.assert_close(output, activated_gate)


def test_qwen3_next_attention_uses_lora_base_projection_collective():
    layer = _attention_stub()
    layer.o_proj = SimpleNamespace(
        base_layer=SimpleNamespace(reduce_results=True, tp_size=2)
    )
    q = torch.zeros(1, 2)
    k = torch.zeros(1, 1)
    v = torch.zeros(1, 1)
    projected = torch.ones(1, 2)
    reduced = torch.full((1, 2), 2.0)

    with (
        patch.object(
            qwen3_next,
            "try_esimd_fp8_linear",
            side_effect=[torch.zeros(1, 4), projected],
        ),
        patch.object(
            qwen3_next,
            "try_esimd_qkv_postprocess",
            return_value=(q, k, v, None),
        ),
        patch.object(
            qwen3_next,
            "tensor_model_parallel_all_reduce",
            return_value=reduced,
        ) as all_reduce,
    ):
        output = qwen3_next.Qwen3NextAttention.forward(
            layer,
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, 2),
        )

    all_reduce.assert_called_once_with(projected)
    assert output is reduced


def test_qwen3_next_attention_falls_back_to_current_qkv_projection():
    layer = _attention_stub()
    q = torch.zeros(1, 2)
    k = torch.zeros(1, 1)
    v = torch.zeros(1, 1)
    gate = torch.zeros(1, 2)
    layer._project_qkv_gate.return_value = (q, k, v, gate)
    hidden_states = torch.zeros(1, 2)
    positions = torch.zeros(1, dtype=torch.int64)

    with patch.object(qwen3_next, "try_esimd_qkv_postprocess", return_value=None):
        output = qwen3_next.Qwen3NextAttention.forward(layer, positions, hidden_states)

    layer._project_qkv_gate.assert_called_once_with(
        layer.qkv_proj.return_value[0], positions
    )
    torch.testing.assert_close(output, torch.full((1, 2), 0.5))


def test_qwen3_next_decode_falls_back_before_fused_norm_when_moe_ineligible():
    hidden_states = torch.zeros(1, 4, dtype=torch.float16)
    residual = torch.ones_like(hidden_states)
    normed = torch.full_like(hidden_states, 2)
    moe_output = torch.full_like(hidden_states, 3)
    post_norm = Mock(return_value=(normed, residual))
    post_norm.weight = SimpleNamespace(data=torch.zeros(4))
    post_norm.variance_epsilon = 1e-6
    mlp = Mock(return_value=moe_output)
    mlp._can_forward_esimd_moe.return_value = False
    mlp._esimd_moe_enabled = False
    mlp.fused_post_attn_router.return_value = (
        torch.zeros(1, 4),
        normed,
    )
    mlp._forward_esimd_moe.return_value = None
    layer = SimpleNamespace(
        use_attn_reduce_scatter_for_moe=False,
        input_layernorm=Mock(return_value=(hidden_states, residual)),
        layer_type="full_attention",
        self_attn=Mock(return_value=hidden_states),
        layer_scale=False,
        _esimd_moe_norm_enabled=True,
        _post_norm_weight=None,
        post_attention_layernorm=post_norm,
        mlp=mlp,
    )

    output, output_residual = qwen3_next.Qwen3NextDecoderLayer.forward(
        layer,
        hidden_states,
        residual,
        positions=torch.zeros(1, dtype=torch.int64),
    )

    mlp.fused_post_attn_router.assert_not_called()
    mlp._forward_esimd_moe.assert_not_called()
    post_norm.assert_called_once_with(hidden_states, residual)
    mlp.assert_called_once_with(normed)
    assert layer._esimd_moe_norm_enabled is False
    assert output is moe_output
    assert output_residual is residual


def test_qwen3_next_int4_fused_post_attn_router_uses_quantized_gate():
    op = Mock()
    hidden = torch.zeros(1, 128, dtype=torch.float16)
    residual = torch.ones_like(hidden)
    gate = SimpleNamespace(
        quant_method=SimpleNamespace(is_sym_int4=True),
        weight_esimd=torch.empty(256, 64, dtype=torch.uint8),
        scale_esimd=torch.ones(256, 1, dtype=torch.float16),
    )
    block = SimpleNamespace(
        gate=gate,
        n_routed_experts=256,
        _router_buf=None,
        _normed_buf=None,
        _gate_fp8_weight=None,
        _gate_fp8_scale=None,
        _esimd_moe_ops=SimpleNamespace(esimd_resadd_norm_gemv_int4_pert=op),
    )

    router, normed = qwen3_next.Qwen3NextSparseMoeBlock.fused_post_attn_router(
        block, hidden, residual, torch.ones(128), 1e-6
    )

    assert router.shape == (1, 256)
    assert normed.shape == hidden.shape
    assert op.call_args.args[3].dtype == torch.int32
    assert op.call_args.args[3].data_ptr() == gate.weight_esimd.data_ptr()
    assert op.call_args.args[4] is gate.scale_esimd


@pytest.mark.parametrize("num_tokens", [1, 4], ids=["full", "tiny"])
@pytest.mark.parametrize("impl_has_scales", [True, False], ids=["fresh", "prepacked"])
@pytest.mark.parametrize("contiguous", [True, False])
def test_qwen3_next_int4_esimd_moe_caches_signed_weights(
    num_tokens, impl_has_scales, contiguous
):
    hidden = torch.zeros(num_tokens, 128, dtype=torch.float16)
    if not contiguous:
        hidden = torch.zeros(num_tokens, 256, dtype=torch.float16)[:, ::2]
    logits = torch.zeros(num_tokens, 256, dtype=torch.float16)
    expected = torch.ones_like(hidden)
    full_op = Mock(return_value=expected)
    tiny_op = Mock(return_value=expected)
    impl = SimpleNamespace(
        w13=torch.empty(256, 256, 64, dtype=torch.int8),
        gemm1_wei_scales=(
            torch.ones(256, 256, 1, dtype=torch.float16) if impl_has_scales else None
        ),
        w2=torch.empty(256, 128, 128, dtype=torch.int8),
        gemm2_wei_scales=(
            torch.ones(256, 128, 1, dtype=torch.float16) if impl_has_scales else None
        ),
    )
    method = SimpleNamespace(
        is_sym_int4=True,
        get_xpu_fused_moe_impl=Mock(return_value=impl),
    )
    sources = qwen3_next._EsimdMoeSources(
        experts=SimpleNamespace(moe_config=SimpleNamespace(experts_per_token=8)),
        routed_experts=SimpleNamespace(
            quant_method=method,
            w13_scales=torch.ones(256, 256, 1, dtype=torch.float16),
            w2_scales=torch.ones(256, 128, 1, dtype=torch.float16),
        ),
        gate=SimpleNamespace(
            quant_method=SimpleNamespace(is_sym_int4=True),
            weight_esimd=torch.empty(256, 64, dtype=torch.uint8),
            scale_esimd=torch.ones(256, 1, dtype=torch.float16),
        ),
        shared_expert_gate=SimpleNamespace(
            weight=torch.empty(1, 128, dtype=torch.float16)
        ),
        shared_gate_up_proj=SimpleNamespace(
            weight=torch.empty(256, 128, dtype=torch.float16)
        ),
        shared_down_proj=SimpleNamespace(
            weight=torch.empty(128, 128, dtype=torch.float16)
        ),
    )
    block = SimpleNamespace(
        _esimd_int4_moe_enabled=True,
        _esimd_int4_moe_plan=None,
        _esimd_moe_ops=SimpleNamespace(
            esimd_resadd_norm_gemv_int4_pert=Mock(),
            moe_forward_full_cutlass_nmajor_int4=full_op,
            moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits=tiny_op,
        ),
        n_routed_experts=256,
        tp_size=1,
        _get_esimd_moe_sources=Mock(return_value=sources),
    )
    block._can_forward_esimd_int4_moe = (
        qwen3_next.Qwen3NextSparseMoeBlock._can_forward_esimd_int4_moe.__get__(block)
    )

    output = qwen3_next.Qwen3NextSparseMoeBlock._forward_esimd_int4_moe(
        block, hidden, num_tokens, logits
    )

    assert output is expected
    method.get_xpu_fused_moe_impl.assert_called_once()
    expected_op = full_op if num_tokens == 1 else tiny_op
    other_op = tiny_op if num_tokens == 1 else full_op
    expected_op.assert_called_once()
    other_op.assert_not_called()
    assert expected_op.call_args.args[0].is_contiguous()
    assert expected_op.call_args.args[2].dtype == torch.uint8
    assert expected_op.call_args.args[2].data_ptr() == impl.w13.data_ptr()
    assert expected_op.call_args.args[4].dtype == torch.uint8
    assert expected_op.call_args.args[4].data_ptr() == impl.w2.data_ptr()
    if impl_has_scales:
        assert expected_op.call_args.args[3] is impl.gemm1_wei_scales
        assert expected_op.call_args.args[5] is impl.gemm2_wei_scales
    else:
        assert expected_op.call_args.args[3] is sources.routed_experts.w13_scales
        assert expected_op.call_args.args[5] is sources.routed_experts.w2_scales


@pytest.mark.parametrize(
    ("scale_suffix", "expected"),
    [("", True), ("_inv", True)],
    ids=["per-tensor", "block"],
)
def test_qwen3_next_esimd_moe_eligibility_requires_supported_scales(
    scale_suffix, expected
):
    fp8_weight = torch.empty(1, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1)
    routed_experts = SimpleNamespace(
        w13_weight=fp8_weight,
        w2_weight=fp8_weight,
        **{
            f"w13_weight_scale{scale_suffix}": scale,
            f"w2_weight_scale{scale_suffix}": scale,
        },
    )
    projection = SimpleNamespace(
        weight=fp8_weight,
        **{f"weight_scale{scale_suffix}": scale},
    )
    block = SimpleNamespace(
        experts=SimpleNamespace(routed_experts=routed_experts),
        gate=SimpleNamespace(weight=torch.ones(1), bias=None),
        shared_expert_gate=SimpleNamespace(weight=torch.ones(1)),
        shared_expert=SimpleNamespace(
            gate_up_proj=projection,
            down_proj=projection,
        ),
        _esimd_moe_enabled=True,
        _esimd_moe_weight_eligible=None,
        _esimd_moe_ops=SimpleNamespace(moe_forward_full_fp8_block=lambda: None),
    )
    block._get_esimd_moe_sources = (
        qwen3_next.Qwen3NextSparseMoeBlock._get_esimd_moe_sources.__get__(block)
    )

    actual = qwen3_next.Qwen3NextSparseMoeBlock._can_forward_esimd_moe(
        block, torch.empty(1, 4, dtype=torch.float16)
    )

    assert actual is expected
    assert block._esimd_moe_enabled is expected


@pytest.mark.parametrize(
    ("active_lora", "expected"),
    [(False, True), (True, False)],
    ids=["inactive-lora", "active-lora"],
)
def test_qwen3_next_esimd_moe_handles_lora_wrapper(active_lora, expected):
    fp8_weight = torch.empty(1, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1)
    routed_experts = SimpleNamespace(
        w13_weight=fp8_weight,
        w13_weight_scale=scale,
        w2_weight=fp8_weight,
        w2_weight_scale=scale,
    )
    projection = SimpleNamespace(weight=fp8_weight, weight_scale=scale)
    block = SimpleNamespace(
        experts=SimpleNamespace(
            base_layer=SimpleNamespace(routed_experts=routed_experts),
            is_lora_active=Mock(return_value=active_lora),
        ),
        gate=SimpleNamespace(weight=torch.ones(1), bias=None),
        shared_expert_gate=SimpleNamespace(weight=torch.ones(1)),
        shared_expert=SimpleNamespace(
            gate_up_proj=projection,
            down_proj=projection,
        ),
        _esimd_moe_enabled=True,
        _esimd_moe_weight_eligible=None,
    )
    block._get_esimd_moe_sources = (
        qwen3_next.Qwen3NextSparseMoeBlock._get_esimd_moe_sources.__get__(block)
    )

    actual = qwen3_next.Qwen3NextSparseMoeBlock._can_forward_esimd_moe(
        block, torch.empty(1, 4, dtype=torch.float16)
    )

    assert actual is expected
    assert block._esimd_moe_enabled


@pytest.mark.parametrize(
    ("additional_kwargs", "expected"),
    [
        ({qwen3_next._QWEN3_NEXT_PLAIN_MROPE: True}, True),
        ({qwen3_next._QWEN3_NEXT_PLAIN_MROPE: False}, False),
        ({}, False),
    ],
)
def test_qwen3_next_attention_uses_runtime_plain_mrope_flag(
    additional_kwargs, expected
):
    layer = _attention_stub()
    layer._esimd_qkv_text_only = False
    layer._project_qkv_gate.return_value = (
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1, 2),
    )
    hidden_states = torch.zeros(1, 2)
    positions = torch.zeros((3, 1), dtype=torch.int64)
    forward_context = SimpleNamespace(additional_kwargs=additional_kwargs)

    with (
        patch.object(torch.compiler, "is_compiling", return_value=False),
        patch.object(qwen3_next, "is_forward_context_available", return_value=True),
        patch.object(qwen3_next, "get_forward_context", return_value=forward_context),
        patch.object(
            qwen3_next, "try_esimd_qkv_postprocess", return_value=None
        ) as postprocess,
    ):
        qwen3_next.Qwen3NextAttention.forward(layer, positions, hidden_states)

    assert postprocess.call_args.kwargs["text_only"] is expected


def test_qwen3_next_attention_does_not_read_forward_context_while_compiling():
    layer = _attention_stub()
    layer._esimd_qkv_text_only = False
    layer._project_qkv_gate.return_value = (
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1, 2),
    )

    with (
        patch.object(torch.compiler, "is_compiling", return_value=True),
        patch.object(qwen3_next, "is_forward_context_available") as available,
        patch.object(qwen3_next, "try_esimd_qkv_postprocess", return_value=None),
    ):
        qwen3_next.Qwen3NextAttention.forward(
            layer,
            torch.zeros((3, 1), dtype=torch.int64),
            torch.zeros(1, 2),
        )

    available.assert_not_called()


def test_qwen3_next_runtime_plain_mrope_defaults_false_without_context():
    with (
        patch.object(torch.compiler, "is_compiling", return_value=False),
        patch.object(qwen3_next, "is_forward_context_available", return_value=False),
        patch.object(qwen3_next, "get_forward_context") as get_context,
    ):
        assert not qwen3_next._runtime_plain_mrope_is_qkv_eligible()

    get_context.assert_not_called()


def _moe_stub(use_spec_fp8_router: bool = True):
    def fp8_weight(*shape):
        return torch.empty(shape, dtype=torch.float8_e4m3fn)

    experts = SimpleNamespace(
        routed_experts=SimpleNamespace(
            w13_weight=fp8_weight(2, 4, 4),
            w13_weight_scale=torch.ones(2),
            w2_weight=fp8_weight(2, 4, 4),
            w2_weight_scale=torch.ones(2),
        ),
        moe_config=SimpleNamespace(experts_per_token=2),
    )
    layer = SimpleNamespace(
        experts=experts,
        shared_expert=SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                weight=fp8_weight(4, 4), weight_scale=torch.ones(1)
            ),
            down_proj=SimpleNamespace(
                weight=fp8_weight(4, 4), weight_scale=torch.ones(1)
            ),
        ),
        shared_expert_gate=SimpleNamespace(weight=torch.ones(1)),
        gate=Mock(
            spec=["weight", "bias"],
            weight=torch.ones(2, 4, dtype=torch.float16),
            bias=None,
            return_value=(torch.zeros(8, 2, dtype=torch.float16), None),
        ),
        _esimd_moe_ops=SimpleNamespace(
            esimd_gemm_fp8_pert=Mock(),
            esimd_gemv_fp16=Mock(),
        ),
        _use_spec_fp8_router=use_spec_fp8_router,
        _batch_router_buffers={},
        _esimd_moe_plan=None,
        _esimd_router_plan=None,
        _gate_fp8_weight=fp8_weight(2, 4),
        _gate_fp8_scale=torch.ones(1, dtype=torch.float32),
        _esimd_moe_weight_eligible=None,
        _esimd_moe_is_block=False,
        _esimd_moe_enabled=True,
        _block_moe_output_buffers={},
        n_routed_experts=2,
        tp_size=1,
    )
    layer._get_esimd_moe_sources = (
        qwen3_next.Qwen3NextSparseMoeBlock._get_esimd_moe_sources.__get__(layer)
    )
    layer._can_forward_esimd_moe = (
        qwen3_next.Qwen3NextSparseMoeBlock._can_forward_esimd_moe.__get__(layer)
    )
    layer._esimd_router_logits = (
        qwen3_next.Qwen3NextSparseMoeBlock._esimd_router_logits.__get__(layer)
    )
    layer._get_fp8_router_weight = (
        qwen3_next.Qwen3NextSparseMoeBlock._get_fp8_router_weight.__get__(layer)
    )
    return layer


@pytest.mark.parametrize(
    "source_name",
    [
        "experts",
        "gate",
        "shared_expert_gate",
        "shared_gate_up_proj",
        "shared_down_proj",
    ],
)
def test_qwen3_next_cached_moe_plan_rechecks_lora_sources(source_name):
    layer = _moe_stub()
    if source_name == "experts":
        base_source = layer.experts
        active = Mock(return_value=False)
        wrapper = SimpleNamespace(
            base_layer=base_source,
            is_lora_active=active,
        )
        layer.experts = wrapper
    else:
        if source_name == "gate":
            owner, attr = layer, "gate"
        elif source_name == "shared_expert_gate":
            owner, attr = layer, "shared_expert_gate"
        elif source_name == "shared_gate_up_proj":
            owner, attr = layer.shared_expert, "gate_up_proj"
        else:
            owner, attr = layer.shared_expert, "down_proj"
        base_source = getattr(owner, attr)
        punica_wrapper = SimpleNamespace(no_lora=True)
        wrapper = SimpleNamespace(
            base_layer=base_source,
            punica_wrapper=punica_wrapper,
        )
        setattr(owner, attr, wrapper)

    hidden_states = torch.zeros(2, 4, dtype=torch.float16)
    moe_op = Mock(return_value=torch.zeros_like(hidden_states))
    with patch.object(
        torch.ops.moe_ops,
        "moe_forward_full_v2",
        moe_op,
        create=True,
    ):
        first = qwen3_next.Qwen3NextSparseMoeBlock._forward_esimd_moe(
            layer, hidden_states, 2
        )
        cached_plan = layer._esimd_moe_plan
        if source_name == "experts":
            active.return_value = True
        else:
            punica_wrapper.no_lora = False
        active_result = qwen3_next.Qwen3NextSparseMoeBlock._forward_esimd_moe(
            layer, hidden_states, 2
        )
        if source_name == "experts":
            active.return_value = False
        else:
            punica_wrapper.no_lora = True
        final = qwen3_next.Qwen3NextSparseMoeBlock._forward_esimd_moe(
            layer, hidden_states, 2
        )

    assert first is not None
    assert active_result is None
    assert final is not None
    assert layer._esimd_moe_plan is cached_plan
    assert layer._esimd_moe_enabled
    assert moe_op.call_count == 2


def test_qwen3_next_m8_router_uses_esimd_and_reuses_buffer():
    layer = _moe_stub()
    hidden_states = torch.zeros(8, 4, dtype=torch.float16)
    outputs = []

    def moe_forward(_hidden_states, router_logits, *_args):
        outputs.append(router_logits)
        return torch.zeros_like(_hidden_states)

    with (
        patch.object(qwen3_next, "_DISABLE_ESIMD_MOE_ROUTER", False),
        patch.object(
            torch.ops.moe_ops,
            "moe_forward_full_v2",
            side_effect=moe_forward,
            create=True,
        ),
    ):
        qwen3_next.Qwen3NextSparseMoeBlock._forward_esimd_moe(layer, hidden_states, 8)
        qwen3_next.Qwen3NextSparseMoeBlock._forward_esimd_moe(layer, hidden_states, 8)

    layer.gate.assert_not_called()
    assert layer._esimd_moe_ops.esimd_gemm_fp8_pert.call_count == 2
    assert outputs[0] is outputs[1]
    assert outputs[0] is layer._batch_router_buffers[8]


@pytest.mark.parametrize("num_tokens", [2, 4])
def test_qwen3_next_small_batch_router_uses_esimd(num_tokens):
    layer = _moe_stub()
    hidden_states = torch.zeros(num_tokens, 4, dtype=torch.float16)

    with (
        patch.object(qwen3_next, "_DISABLE_ESIMD_MOE_ROUTER", False),
        patch.object(
            torch.ops.moe_ops,
            "moe_forward_full_v2",
            return_value=torch.zeros_like(hidden_states),
            create=True,
        ),
    ):
        qwen3_next.Qwen3NextSparseMoeBlock._forward_esimd_moe(
            layer, hidden_states, num_tokens
        )

    layer.gate.assert_not_called()
    layer._esimd_moe_ops.esimd_gemm_fp8_pert.assert_called_once()
    assert num_tokens in layer._batch_router_buffers


@pytest.mark.parametrize("num_tokens", [1, 18])
def test_qwen3_next_router_outside_small_batch_uses_linear(num_tokens):
    layer = _moe_stub()
    hidden_states = torch.zeros(num_tokens, 4, dtype=torch.float16)
    layer.gate.return_value = (
        torch.zeros(num_tokens, 2, dtype=torch.float16),
        None,
    )
    moe_op = "moe_forward_full" if num_tokens == 1 else "moe_forward_full_v2"

    with (
        patch.object(qwen3_next, "_DISABLE_ESIMD_MOE_ROUTER", False),
        patch.object(
            torch.ops.moe_ops,
            moe_op,
            return_value=torch.zeros_like(hidden_states),
            create=True,
        ),
    ):
        qwen3_next.Qwen3NextSparseMoeBlock._forward_esimd_moe(
            layer, hidden_states, num_tokens
        )

    layer.gate.assert_called_once_with(hidden_states)
    layer._esimd_moe_ops.esimd_gemm_fp8_pert.assert_not_called()


def test_qwen3_next_block_moe_uses_fused_decode_and_reuses_output():
    def fp8_weight(*shape):
        return torch.empty(shape, dtype=torch.float8_e4m3fn)

    scale = torch.ones(2, 1, 1, dtype=torch.float32)
    projection = SimpleNamespace(
        weight=fp8_weight(4, 4),
        weight_scale_inv=torch.ones(1, 1, dtype=torch.float32),
    )
    routed = SimpleNamespace(
        w13_weight=fp8_weight(2, 4, 4),
        w13_weight_scale_inv=scale,
        w2_weight=fp8_weight(2, 4, 4),
        w2_weight_scale_inv=scale,
    )
    layer = SimpleNamespace(
        experts=SimpleNamespace(
            routed_experts=routed,
            moe_config=SimpleNamespace(experts_per_token=2),
        ),
        shared_expert=SimpleNamespace(
            gate_up_proj=projection,
            down_proj=projection,
        ),
        shared_expert_gate=SimpleNamespace(weight=torch.ones(1, 4)),
        gate=Mock(
            spec=["weight", "bias"],
            weight=torch.ones(2, 4, dtype=torch.float16),
            bias=None,
            return_value=(torch.zeros(4, 2, dtype=torch.float16), None),
        ),
        _esimd_moe_ops=SimpleNamespace(
            moe_forward_full_fp8_block=lambda *args: args[2],
            esimd_gemv_fp16=Mock(),
        ),
        _esimd_moe_enabled=True,
        _use_spec_fp8_router=False,
        _esimd_moe_weight_eligible=None,
        _esimd_moe_is_block=False,
        _esimd_moe_plan=None,
        _esimd_router_plan=None,
        _batch_router_buffers={},
        _block_moe_output_buffers={},
        n_routed_experts=2,
        tp_size=1,
    )
    layer._get_esimd_moe_sources = (
        qwen3_next.Qwen3NextSparseMoeBlock._get_esimd_moe_sources.__get__(layer)
    )
    layer._can_forward_esimd_moe = (
        qwen3_next.Qwen3NextSparseMoeBlock._can_forward_esimd_moe.__get__(layer)
    )
    layer._esimd_router_logits = (
        qwen3_next.Qwen3NextSparseMoeBlock._esimd_router_logits.__get__(layer)
    )
    hidden_states = torch.zeros(4, 4, dtype=torch.float16)
    outputs = []

    def fused_block(_hidden, _logits, output, *_args):
        outputs.append(output)
        return output

    with patch.object(
        torch.ops.moe_ops,
        "moe_forward_full_fp8_block",
        side_effect=fused_block,
        create=True,
    ):
        qwen3_next.Qwen3NextSparseMoeBlock._forward_esimd_moe(layer, hidden_states, 4)
        qwen3_next.Qwen3NextSparseMoeBlock._forward_esimd_moe(layer, hidden_states, 4)

    assert layer._esimd_moe_is_block
    assert outputs[0] is outputs[1]
    assert outputs[0] is layer._block_moe_output_buffers[4]
