# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract tests for the guarded ESIMD GDN decode adapter."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import _xpu_esimd, qwen_gdn_linear_attn

H = 1
HV = 1
K = 128
V = 128
CONV_WIDTH = 4
CONV_DIM = 2 * H * K + HV * V


def _metadata(num_decodes=1, num_prefills=0):
    return SimpleNamespace(
        num_prefills=num_prefills,
        num_prefill_tokens=0,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_decodes,
        spec_sequence_masks=None,
        non_spec_state_indices_tensor=torch.zeros(num_decodes, dtype=torch.int32),
    )


def _layer(gqa_interleaved_layout=False, ssm_dtype=torch.float16):
    return SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        num_k_heads=H,
        num_v_heads=HV,
        tp_size=1,
        head_k_dim=K,
        head_v_dim=V,
        conv_kernel_size=CONV_WIDTH,
        gqa_interleaved_layout=gqa_interleaved_layout,
        activation="silu",
        conv1d=SimpleNamespace(
            weight=torch.zeros(CONV_DIM, 1, CONV_WIDTH, dtype=torch.float16),
            bias=None,
        ),
        A_log=torch.zeros(HV, dtype=torch.float32),
        dt_bias=torch.zeros(HV, dtype=torch.float32),
        kv_cache=(
            torch.zeros(2, CONV_WIDTH - 1, CONV_DIM, dtype=torch.float16),
            torch.zeros(2, HV, V, K, dtype=ssm_dtype),
        ),
    )


def _inputs(num_tokens=1):
    qkvz = torch.zeros(num_tokens, 2 * H * K + 2 * HV * V, dtype=torch.float16)
    ba = torch.zeros(num_tokens, 2 * HV, dtype=torch.float16)
    output = torch.zeros(num_tokens, HV, V, dtype=torch.float16)
    z_out = torch.zeros_like(output)
    return qkvz, ba, output, z_out


def _context(layer, metadata):
    return SimpleNamespace(attn_metadata={layer.prefix: metadata})


def _spec_case(*, padded=False):
    local_h, local_hv, num_tokens = 8, 16, 5
    conv_dim = 2 * local_h * K + local_hv * V
    layer = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        num_k_heads=local_h,
        num_v_heads=local_hv,
        tp_size=1,
        head_k_dim=K,
        head_v_dim=V,
        conv_kernel_size=CONV_WIDTH,
        gqa_interleaved_layout=False,
        activation="silu",
        conv1d=SimpleNamespace(
            weight=torch.zeros(conv_dim, 1, CONV_WIDTH, dtype=torch.float16),
            bias=None,
        ),
        A_log=torch.zeros(local_hv),
        dt_bias=torch.zeros(local_hv),
        kv_cache=(
            torch.zeros(8, CONV_WIDTH + 3, conv_dim, dtype=torch.float16),
            torch.zeros(8, local_hv, V, K, dtype=torch.float16),
        ),
    )
    state_indices = torch.arange(
        (2 if padded else 1) * num_tokens, dtype=torch.int32
    ).reshape(-1, num_tokens)
    metadata = SimpleNamespace(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=num_tokens,
        num_actual_tokens=num_tokens,
        spec_sequence_masks=torch.tensor([True]),
        spec_state_indices_tensor=state_indices,
        spec_token_indx=torch.arange(num_tokens, dtype=torch.int32),
        num_accepted_tokens=torch.ones(2 if padded else 1, dtype=torch.int32),
    )
    qkvz = torch.zeros(
        num_tokens,
        2 * local_h * K + 2 * local_hv * V,
        dtype=torch.float16,
    )
    ba = torch.zeros(num_tokens, 2 * local_hv, dtype=torch.float16)
    output = torch.zeros(num_tokens, local_hv, V, dtype=torch.float16)
    return layer, metadata, qkvz, ba, output, torch.zeros_like(output)


def _fp8_projection(out_features, in_features, *, reduce_results=False):
    return SimpleNamespace(
        weight=torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones(1, dtype=torch.float32),
        bias=None,
        gather_output=False,
        reduce_results=reduce_results,
    )


def _block_fp8_projection(out_features, in_features, *, reduce_results=False):
    return SimpleNamespace(
        weight=torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.ones(
            (out_features + 127) // 128,
            (in_features + 127) // 128,
            dtype=torch.float32,
        ),
        weight_block_size=(128, 128),
        bias=None,
        gather_output=False,
        reduce_results=reduce_results,
    )


def _int4_projection(out_features, in_features, *, reduce_results=False):
    return SimpleNamespace(
        weight_esimd=torch.empty(out_features, in_features // 2, dtype=torch.uint8),
        scale_esimd=torch.ones(out_features, in_features // 128, dtype=torch.float16),
        quant_method=SimpleNamespace(is_sym_int4=True),
        bias=None,
        gather_output=False,
        reduce_results=reduce_results,
    )


def _lora_wrapper(base_layer, active):
    return SimpleNamespace(
        base_layer=base_layer,
        punica_wrapper=SimpleNamespace(no_lora=not active),
        is_lora_active=Mock(return_value=active),
    )


def test_esimd_gdn_fuses_scalar_scale_fp8_input_projections():
    hidden = torch.zeros(1, 512, dtype=torch.float16)
    qkvz_proj = _fp8_projection(768, 512)
    ba_proj = _fp8_projection(8, 512)
    op = Mock()

    with (
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            return_value=SimpleNamespace(esimd_gemv_fp8_pert_fused2=op),
        ) as load_module,
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
        patch.object(torch.compiler, "is_compiling", return_value=False) as compiling,
    ):
        result = _xpu_esimd.try_esimd_gdn_input_projections(hidden, qkvz_proj, ba_proj)
        second_result = _xpu_esimd.try_esimd_gdn_input_projections(
            hidden, qkvz_proj, ba_proj
        )

    assert result is not None
    assert second_result is not None
    qkvz, ba = result
    assert qkvz.shape == (1, 768)
    assert ba.shape == (1, 8)
    assert op.call_count == 2
    op.assert_called_with(
        hidden,
        qkvz_proj.weight,
        qkvz_proj.weight_scale,
        qkvz,
        ba_proj.weight,
        ba_proj.weight_scale,
        ba,
    )
    load_module.assert_called_once()
    assert compiling.call_count == 1
    assert second_result[0] is qkvz
    assert second_result[1] is ba


def test_esimd_gdn_fuses_block_fp8_qkvz_with_fp16_ba():
    hidden = torch.zeros(1, 512, dtype=torch.float16)
    qkvz_proj = _block_fp8_projection(768, 512)
    ba_proj = SimpleNamespace(
        weight=torch.empty(8, 512, dtype=torch.float16),
        bias=None,
        gather_output=False,
    )
    op = Mock()

    with (
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            return_value=SimpleNamespace(esimd_gemv_fp8_blockscale_fp16_fused2=op),
        ),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
        patch.object(torch.compiler, "is_compiling", return_value=False),
    ):
        result = _xpu_esimd.try_esimd_gdn_input_projections(hidden, qkvz_proj, ba_proj)

    assert result is not None
    qkvz, ba = result
    op.assert_called_once_with(
        hidden,
        qkvz_proj.weight,
        qkvz_proj.weight_scale_inv,
        qkvz,
        ba_proj.weight,
        ba,
        128,
        128,
    )


@pytest.mark.parametrize(
    ("active_lora", "expected"),
    [(False, True), (True, False)],
    ids=["inactive-lora", "active-lora"],
)
def test_esimd_gdn_fp16_ba_projection_handles_lora_wrapper(active_lora, expected):
    hidden = torch.zeros(2, 512, dtype=torch.float16)
    qkvz_proj = _lora_wrapper(_block_fp8_projection(768, 512), active_lora)
    ba_proj = _lora_wrapper(
        SimpleNamespace(
            weight=torch.empty(8, 512, dtype=torch.float16),
            bias=None,
            gather_output=False,
        ),
        active_lora,
    )
    op = Mock()
    cached_output = None

    with (
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            return_value=SimpleNamespace(esimd_gemv_fp16=op),
        ),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
        patch.object(torch.compiler, "is_compiling", return_value=False),
    ):
        output = _xpu_esimd.try_esimd_gdn_fp16_ba_projection(hidden, qkvz_proj, ba_proj)
        if not active_lora:
            qkvz_proj.punica_wrapper.no_lora = False
            ba_proj.punica_wrapper.no_lora = False
            cached_output = _xpu_esimd.try_esimd_gdn_fp16_ba_projection(
                hidden, qkvz_proj, ba_proj
            )

    assert (output is not None) is expected
    assert op.called is expected
    if not active_lora:
        assert cached_output is None
        op.assert_called_once()
    qkvz_proj.is_lora_active.assert_not_called()
    ba_proj.is_lora_active.assert_not_called()


@pytest.mark.parametrize(
    ("active_lora", "expected"),
    [(False, True), (True, False)],
    ids=["inactive-lora", "active-lora"],
)
def test_esimd_gdn_input_projections_handle_lora_wrapper(active_lora, expected):
    hidden = torch.zeros(1, 512, dtype=torch.float16)
    qkvz_base = _fp8_projection(768, 512)
    ba_base = _fp8_projection(8, 512)
    qkvz_proj = _lora_wrapper(qkvz_base, active_lora)
    ba_proj = _lora_wrapper(ba_base, active_lora)
    op = Mock()
    cached_result = None

    with (
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            return_value=SimpleNamespace(esimd_gemv_fp8_pert_fused2=op),
        ),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
    ):
        result = _xpu_esimd.try_esimd_gdn_input_projections(hidden, qkvz_proj, ba_proj)
        if not active_lora:
            qkvz_proj.punica_wrapper.no_lora = False
            ba_proj.punica_wrapper.no_lora = False
            qkvz_proj.is_lora_active.return_value = True
            ba_proj.is_lora_active.return_value = True
            cached_result = _xpu_esimd.try_esimd_gdn_input_projections(
                hidden, qkvz_proj, ba_proj
            )

    assert (result is not None) is expected
    assert op.called is expected
    if not active_lora:
        assert cached_result is None
        op.assert_called_once()
    qkvz_proj.is_lora_active.assert_not_called()
    ba_proj.is_lora_active.assert_not_called()


def test_esimd_gdn_fuses_gated_norm_and_fp8_output_projection():
    layer = SimpleNamespace(
        num_v_heads=4,
        tp_size=1,
        head_v_dim=128,
        norm=SimpleNamespace(
            weight=torch.ones(128),
            activation="silu",
            group_size=None,
            norm_before_gate=True,
            eps=1e-6,
        ),
        out_proj=_fp8_projection(512, 512, reduce_results=True),
    )
    core = torch.zeros(1, 4, 128, dtype=torch.float16)
    z = torch.ones_like(core)
    op = Mock()

    with (
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            return_value=SimpleNamespace(esimd_norm_gemv_fp8_pert=op),
        ) as load_module,
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
        patch.object(torch.compiler, "is_compiling", return_value=False) as compiling,
    ):
        output = _xpu_esimd.try_esimd_gdn_output_projection(layer, core, z)
        second_output = _xpu_esimd.try_esimd_gdn_output_projection(layer, core, z)

    assert output is not None
    assert second_output is output
    assert output.shape == (1, 512)
    assert op.call_count == 2
    load_module.assert_called_once()
    assert compiling.call_count == 1
    args = op.call_args.args
    assert args[0].data_ptr() == core.data_ptr()
    assert args[1].data_ptr() == z.data_ptr()
    assert args[2] is layer._esimd_gdn_norm_weight_fp16
    assert args[3] is layer.out_proj.weight
    assert args[4] is layer.out_proj.weight_scale
    assert args[5] is output
    assert args[6:] == (4, 128, pytest.approx(1e-6))


def test_esimd_gdn_fuses_gated_norm_and_block_fp8_output_projection():
    layer = SimpleNamespace(
        num_v_heads=4,
        tp_size=1,
        head_v_dim=128,
        norm=SimpleNamespace(
            weight=torch.ones(128),
            activation="silu",
            group_size=None,
            norm_before_gate=True,
            eps=1e-6,
        ),
        out_proj=_block_fp8_projection(512, 512, reduce_results=True),
    )
    core = torch.zeros(1, 4, 128, dtype=torch.float16)
    z = torch.ones_like(core)
    op = Mock()

    with (
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            return_value=SimpleNamespace(esimd_norm_gemv_fp8_blockscale=op),
        ),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
        patch.object(torch.compiler, "is_compiling", return_value=False),
    ):
        output = _xpu_esimd.try_esimd_gdn_output_projection(layer, core, z)

    assert output is not None
    assert op.call_args.args[3] is layer.out_proj.weight
    assert op.call_args.args[4] is layer.out_proj.weight_scale_inv


def test_esimd_gdn_fuses_gated_norm_and_int4_output_projection():
    layer = SimpleNamespace(
        num_v_heads=4,
        tp_size=1,
        head_v_dim=128,
        norm=SimpleNamespace(
            weight=torch.ones(128),
            activation="silu",
            group_size=None,
            norm_before_gate=True,
            eps=1e-6,
        ),
        out_proj=_int4_projection(512, 512, reduce_results=True),
    )
    core = torch.zeros(1, 4, 128, dtype=torch.float16)
    z = torch.ones_like(core)
    op = Mock()

    with (
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            return_value=SimpleNamespace(esimd_norm_gemv_int4_pert=op),
        ),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
        patch.object(torch.compiler, "is_compiling", return_value=False),
    ):
        output = _xpu_esimd.try_esimd_gdn_output_projection(layer, core, z)

    assert output is not None
    args = op.call_args.args
    assert args[3].dtype == torch.int32
    assert args[3].data_ptr() == layer.out_proj.weight_esimd.data_ptr()
    assert args[4] is layer.out_proj.scale_esimd


@pytest.mark.parametrize(
    ("active_lora", "expected"),
    [(False, True), (True, False)],
    ids=["inactive-lora", "active-lora"],
)
def test_esimd_gdn_output_projection_handles_lora_wrapper(active_lora, expected):
    base_projection = _fp8_projection(512, 512, reduce_results=True)
    projection = _lora_wrapper(base_projection, active_lora)
    layer = SimpleNamespace(
        num_v_heads=4,
        tp_size=1,
        head_v_dim=128,
        norm=SimpleNamespace(
            weight=torch.ones(128),
            activation="silu",
            group_size=None,
            norm_before_gate=True,
            eps=1e-6,
        ),
        out_proj=projection,
    )
    core = torch.zeros(1, 4, 128, dtype=torch.float16)
    op = Mock()
    cached_output = None

    with (
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            return_value=SimpleNamespace(esimd_norm_gemv_fp8_pert=op),
        ),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
    ):
        output = _xpu_esimd.try_esimd_gdn_output_projection(
            layer, core, torch.ones_like(core)
        )
        if not active_lora:
            projection.punica_wrapper.no_lora = False
            projection.is_lora_active.return_value = True
            cached_output = _xpu_esimd.try_esimd_gdn_output_projection(
                layer, core, torch.ones_like(core)
            )

    assert (output is not None) is expected
    assert op.called is expected
    if not active_lora:
        assert cached_output is None
        op.assert_called_once()
    projection.is_lora_active.assert_not_called()


def test_esimd_gdn_batched_gated_norm_matches_native_reference():
    """2 <= M <= 128 decode batch: ESIMD gated norm must match forward_static."""
    from vllm.model_executor.layers.layernorm import RMSNormGated

    torch.manual_seed(0)
    num_tokens, local_heads, head_dim = 16, 4, 128
    layer = SimpleNamespace(
        num_v_heads=local_heads,
        tp_size=1,
        head_v_dim=head_dim,
        norm=SimpleNamespace(
            weight=torch.rand(head_dim) + 0.5,
            activation="silu",
            group_size=None,
            norm_before_gate=True,
            eps=1e-6,
        ),
    )
    core = torch.randn(num_tokens, local_heads, head_dim, dtype=torch.float16)
    z = torch.randn_like(core)

    def fake_op(x_rows, z_rows, weight, out_rows, eps):
        out_rows.copy_(
            RMSNormGated.forward_static(
                x_rows,
                z_rows,
                weight,
                eps,
                x_rows.dtype,
                group_size=None,
                norm_before_gate=True,
                activation="silu",
            )
        )

    with (
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            return_value=SimpleNamespace(esimd_rms_norm_gated=fake_op),
        ) as load_module,
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
        patch.object(torch.compiler, "is_compiling", return_value=False),
    ):
        result = _xpu_esimd.try_esimd_gdn_output_norm_batched(layer, core, z)
        second_result = _xpu_esimd.try_esimd_gdn_output_norm_batched(layer, core, z)

    assert result is not None
    assert result.shape == (num_tokens, local_heads * head_dim)
    load_module.assert_called_once()
    assert torch.equal(result, second_result)

    expected = RMSNormGated.forward_static(
        core.reshape(num_tokens * local_heads, head_dim),
        z.reshape(num_tokens * local_heads, head_dim),
        layer.norm.weight,
        layer.norm.eps,
        core.dtype,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
    ).reshape(num_tokens, local_heads * head_dim)
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)


def test_esimd_gdn_batched_gated_norm_gate_falls_back_outside_range():
    layer = SimpleNamespace(
        num_v_heads=4,
        tp_size=1,
        head_v_dim=128,
        norm=SimpleNamespace(
            weight=torch.ones(128),
            activation="silu",
            group_size=None,
            norm_before_gate=True,
            eps=1e-6,
        ),
    )
    # M == 1 is handled by the fused M==1 tail, not this batched path.
    core = torch.zeros(1, 4, 128, dtype=torch.float16)
    z = torch.ones_like(core)
    with patch.object(_xpu_esimd, "_load_esimd_module") as load_module:
        assert _xpu_esimd.try_esimd_gdn_output_norm_batched(layer, core, z) is None
    load_module.assert_not_called()


def test_esimd_gdn_projection_gate_falls_back(monkeypatch):
    monkeypatch.setenv("VLLM_XPU_DISABLE_ESIMD_GDN_PROJ", "1")
    hidden = torch.zeros(1, 512, dtype=torch.float16)
    projection = _fp8_projection(512, 512)

    with patch.object(_xpu_esimd, "_load_esimd_module") as load_module:
        assert (
            _xpu_esimd.try_esimd_gdn_input_projections(hidden, projection, projection)
            is None
        )

    load_module.assert_not_called()


def test_qwen_gdn_decode_buffers_are_sliced_and_reused():
    layer = SimpleNamespace(
        num_v_heads=4,
        tp_size=1,
        head_v_dim=128,
        prefix="model.layers.0.linear_attn",
        in_proj_qkvz=object(),
        in_proj_ba=object(),
    )
    seen_outputs = []

    def core_decode(_layer, _qkvz, _ba, output, _z):
        seen_outputs.append(output)
        return True

    with (
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_input_projections",
            side_effect=lambda hidden, *_args: (
                torch.zeros(hidden.shape[0], 768, dtype=torch.float16),
                torch.zeros(hidden.shape[0], 8, dtype=torch.float16),
            ),
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_core_decode",
            side_effect=core_decode,
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_output_projection",
            side_effect=lambda _layer, output, _z: output.flatten(1),
        ),
    ):
        qwen_gdn_linear_attn.QwenGatedDeltaNetAttention.forward_xpu(
            layer, torch.zeros(4, 512, dtype=torch.float16)
        )
        qwen_gdn_linear_attn.QwenGatedDeltaNetAttention.forward_xpu(
            layer, torch.zeros(8, 512, dtype=torch.float16)
        )

    assert seen_outputs[0].shape == (4, 4, 128)
    assert seen_outputs[1].shape == (8, 4, 128)
    assert seen_outputs[0].untyped_storage().data_ptr() == (
        seen_outputs[1].untyped_storage().data_ptr()
    )
    assert layer._esimd_gdn_decode_buffers[0].shape == (128, 4, 128)


def test_qwen_gdn_reused_core_buffer_is_zeroed_before_fallback():
    layer = SimpleNamespace(
        num_v_heads=1,
        tp_size=1,
        head_v_dim=128,
        prefix="model.layers.0.linear_attn",
        in_proj_qkvz=object(),
        in_proj_ba=object(),
    )
    checked = []

    def fallback(output, *_args):
        checked.append(torch.count_nonzero(output).item())

    with (
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_input_projections",
            return_value=(
                torch.zeros(4, 384, dtype=torch.float16),
                torch.zeros(4, 2, dtype=torch.float16),
            ),
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_core_decode",
            side_effect=lambda _layer, _qkvz, _ba, output, _z: (
                output.fill_(1).numel() == 0
            ),
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_core_spec_decode",
            return_value=False,
        ),
        patch.object(
            torch.ops.vllm,
            "gdn_attention_core_xpu",
            side_effect=fallback,
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_output_projection",
            side_effect=lambda _layer, output, _z: output.flatten(1),
        ),
    ):
        qwen_gdn_linear_attn.QwenGatedDeltaNetAttention.forward_xpu(
            layer, torch.zeros(4, 512, dtype=torch.float16)
        )

    assert checked == [0]


def test_esimd_gdn_projections_fall_back_when_package_is_unavailable():
    hidden = torch.zeros(1, 512, dtype=torch.float16)
    projection = _fp8_projection(512, 512)

    with (
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            side_effect=ImportError("test-only missing package"),
        ),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
    ):
        assert (
            _xpu_esimd.try_esimd_gdn_input_projections(hidden, projection, projection)
            is None
        )


def test_esimd_gdn_decode_dispatches_sequential_op_without_xpu():
    layer = _layer(gqa_interleaved_layout=False)
    qkvz, ba, output, z_out = _inputs()
    op = Mock()
    module = SimpleNamespace(esimd_gdn_conv_fused_seq=op)

    with (
        patch.object(
            _xpu_esimd,
            "get_forward_context",
            return_value=_context(layer, _metadata()),
        ),
        patch.object(
            _xpu_esimd, "_load_esimd_module", return_value=module
        ) as load_module,
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
        patch.object(torch.compiler, "is_compiling", return_value=False) as compiling,
    ):
        assert _xpu_esimd.try_esimd_gdn_core_decode(layer, qkvz, ba, output, z_out)
        assert _xpu_esimd.try_esimd_gdn_core_decode(layer, qkvz, ba, output, z_out)

    assert op.call_count == 2
    load_module.assert_called_once()
    assert compiling.call_count == 1
    assert op.call_args.args[0] is qkvz
    assert op.call_args.args[7] is ba
    assert op.call_args.args[10] is output
    assert op.call_args.args[11] is z_out
    assert op.call_args.args[12:] == (1, H, HV, K, V, pytest.approx(K**-0.5))


def test_esimd_gdn_decode_uses_interleaved_op():
    layer = _layer(gqa_interleaved_layout=True)
    qkvz, ba, output, z_out = _inputs()
    interleaved = Mock()
    sequential = Mock()
    module = SimpleNamespace(
        esimd_gdn_conv_fused=interleaved,
        esimd_gdn_conv_fused_seq=sequential,
    )

    with (
        patch.object(
            _xpu_esimd,
            "get_forward_context",
            return_value=_context(layer, _metadata()),
        ),
        patch.object(_xpu_esimd, "_load_esimd_module", return_value=module),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
    ):
        assert _xpu_esimd.try_esimd_gdn_core_decode(layer, qkvz, ba, output, z_out)

    interleaved.assert_called_once()
    sequential.assert_not_called()


@pytest.mark.parametrize("padded", [False, True])
def test_esimd_gdn_spec_decode_reuses_active_metadata(padded):
    layer, metadata, qkvz, ba, output, z_out = _spec_case(padded=padded)
    op = Mock()

    with (
        patch.object(
            _xpu_esimd,
            "get_forward_context",
            return_value=_context(layer, metadata),
        ),
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            return_value=SimpleNamespace(esimd_gdn_conv_fused_seq_spec=op),
        ),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
        patch.object(torch.compiler, "is_compiling", return_value=False),
        patch.object(_xpu_esimd, "_SPEC_GDN_DISABLED", False),
    ):
        assert _xpu_esimd.try_esimd_gdn_core_spec_decode(
            layer, qkvz, ba, output, z_out
        )

    args = op.call_args.args
    if padded:
        assert args[4].untyped_storage().data_ptr() == (
            metadata.spec_state_indices_tensor.untyped_storage().data_ptr()
        )
        assert args[12].untyped_storage().data_ptr() == (
            metadata.num_accepted_tokens.untyped_storage().data_ptr()
        )
    else:
        assert args[4] is metadata.spec_state_indices_tensor
        assert args[12] is metadata.num_accepted_tokens
    assert args[11] is metadata.spec_token_indx
    assert args[13:] == (1, 5, 8, 16, K, V, pytest.approx(K**-0.5))


@pytest.mark.parametrize("disabled,local_v_heads", [(True, 16), (False, 20)])
def test_esimd_gdn_spec_decode_falls_back_before_loading_kernel(
    disabled, local_v_heads
):
    layer, metadata, qkvz, ba, output, z_out = _spec_case()
    if local_v_heads != 16:
        layer.num_v_heads = local_v_heads

    with (
        patch.object(
            _xpu_esimd,
            "get_forward_context",
            return_value=_context(layer, metadata),
        ),
        patch.object(_xpu_esimd, "_load_esimd_module") as load_module,
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
        patch.object(torch.compiler, "is_compiling", return_value=False),
        patch.object(_xpu_esimd, "_SPEC_GDN_DISABLED", disabled),
    ):
        assert not _xpu_esimd.try_esimd_gdn_core_spec_decode(
            layer, qkvz, ba, output, z_out
        )

    load_module.assert_not_called()


def test_esimd_gdn_decode_rejects_interleaved_geometry_over_wg32():
    layer = _layer(gqa_interleaved_layout=True)
    layer.num_k_heads = 8
    layer.num_v_heads = 16
    qkvz = torch.zeros(1, 2 * 8 * K + 2 * 16 * V, dtype=torch.float16)
    ba = torch.zeros(1, 2 * 16, dtype=torch.float16)
    output = torch.zeros(1, 16, V, dtype=torch.float16)
    z_out = torch.zeros_like(output)

    with (
        patch.object(
            _xpu_esimd,
            "get_forward_context",
            return_value=_context(layer, _metadata()),
        ),
        patch.object(_xpu_esimd, "_load_esimd_module") as load_module,
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
    ):
        assert not _xpu_esimd.try_esimd_gdn_core_decode(layer, qkvz, ba, output, z_out)

    load_module.assert_not_called()


def test_esimd_gdn_decode_falls_back_when_package_is_unavailable():
    layer = _layer()
    qkvz, ba, output, z_out = _inputs()

    with (
        patch.object(
            _xpu_esimd,
            "get_forward_context",
            return_value=_context(layer, _metadata()),
        ),
        patch.object(
            _xpu_esimd,
            "_load_esimd_module",
            side_effect=ImportError("test-only missing package"),
        ),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
    ):
        assert not _xpu_esimd.try_esimd_gdn_core_decode(layer, qkvz, ba, output, z_out)


def test_esimd_gdn_decode_rejects_out_of_range_mock_state_index():
    layer = _layer()
    qkvz, ba, output, z_out = _inputs()
    metadata = _metadata()
    metadata.non_spec_state_indices_tensor.fill_(2)

    with (
        patch.object(
            _xpu_esimd,
            "get_forward_context",
            return_value=_context(layer, metadata),
        ),
        patch.object(_xpu_esimd, "_load_esimd_module"),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
    ):
        assert not _xpu_esimd.try_esimd_gdn_core_decode(layer, qkvz, ba, output, z_out)


@pytest.mark.parametrize(
    "metadata,ssm_dtype,num_tokens",
    [
        (_metadata(num_prefills=1), torch.float16, 1),
        (_metadata(num_decodes=129), torch.float16, 129),
        (_metadata(), torch.float32, 1),
    ],
)
def test_esimd_gdn_decode_guard_failures_are_side_effect_free(
    metadata, ssm_dtype, num_tokens
):
    layer = _layer(ssm_dtype=ssm_dtype)
    qkvz, ba, output, z_out = _inputs(num_tokens)
    op = Mock()
    module = SimpleNamespace(esimd_gdn_conv_fused_seq=op)
    cache_before = tuple(state.clone() for state in layer.kv_cache)

    with (
        patch.object(
            _xpu_esimd,
            "get_forward_context",
            return_value=_context(layer, metadata),
        ),
        patch.object(_xpu_esimd, "_load_esimd_module", return_value=module),
        patch.object(_xpu_esimd, "_is_xpu_tensor", return_value=True),
    ):
        assert not _xpu_esimd.try_esimd_gdn_core_decode(layer, qkvz, ba, output, z_out)

    op.assert_not_called()
    for before, after in zip(cache_before, layer.kv_cache):
        torch.testing.assert_close(before, after)
    assert not hasattr(layer, "_esimd_gdn_conv_weight_fp16")


def test_forward_xpu_calls_native_core_when_esimd_declines():
    layer = SimpleNamespace(
        num_v_heads=HV,
        tp_size=1,
        head_v_dim=V,
        prefix="model.layers.0.linear_attn",
        in_proj_qkvz=Mock(return_value=(_inputs()[0], None)),
        in_proj_ba=Mock(return_value=(_inputs()[1], None)),
        norm=Mock(side_effect=lambda output, gate: output),
        out_proj=Mock(return_value=(torch.zeros(1, 1), None)),
    )
    hidden_states = torch.zeros(1, 1)
    native = Mock()

    with (
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_input_projections",
            return_value=None,
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_int4_fused2_layers",
            return_value=None,
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_core_decode",
            return_value=False,
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_core_spec_decode",
            return_value=False,
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_output_projection",
            return_value=None,
        ),
        patch.object(
            torch.ops.vllm,
            "gdn_attention_core_xpu",
            native,
            create=True,
        ),
    ):
        qwen_gdn_linear_attn.QwenGatedDeltaNetAttention.forward_xpu(
            layer, hidden_states
        )

    native.assert_called_once_with(
        native.call_args.args[0],
        native.call_args.args[1],
        layer.in_proj_qkvz.return_value[0],
        layer.in_proj_ba.return_value[0],
        layer.prefix,
    )


def test_forward_xpu_uses_fused_int4_projections():
    qkvz, ba, _, _ = _inputs()
    layer = SimpleNamespace(
        num_v_heads=HV,
        tp_size=1,
        head_v_dim=V,
        prefix="model.layers.0.linear_attn",
        in_proj_qkvz=Mock(),
        in_proj_ba=Mock(),
        norm=Mock(side_effect=lambda output, gate: output),
        out_proj=Mock(return_value=(torch.zeros(1, 1), None)),
    )
    hidden_states = torch.zeros(1, 1)

    with (
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_input_projections",
            return_value=None,
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_int4_fused2_layers",
            return_value=(qkvz, ba),
        ) as fused,
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_core_decode",
            return_value=True,
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_output_projection",
            return_value=None,
        ),
    ):
        qwen_gdn_linear_attn.QwenGatedDeltaNetAttention.forward_xpu(
            layer, hidden_states
        )

    fused.assert_called_once_with(hidden_states, layer.in_proj_qkvz, layer.in_proj_ba)
    layer.in_proj_qkvz.assert_not_called()
    layer.in_proj_ba.assert_not_called()


def test_forward_xpu_prefers_fused_fp8_projections():
    qkvz, ba, _, _ = _inputs()
    layer = SimpleNamespace(
        num_v_heads=HV,
        tp_size=1,
        head_v_dim=V,
        prefix="model.layers.0.linear_attn",
        in_proj_qkvz=Mock(),
        in_proj_ba=Mock(),
        norm=Mock(side_effect=lambda output, gate: output),
        out_proj=Mock(return_value=(torch.zeros(1, 1), None)),
    )
    hidden_states = torch.zeros(1, 1)

    with (
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_input_projections",
            return_value=(qkvz, ba),
        ) as fused,
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_int4_fused2_layers",
        ) as int4_fused,
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_core_decode",
            return_value=True,
        ),
        patch.object(
            qwen_gdn_linear_attn,
            "try_esimd_gdn_output_projection",
            return_value=torch.zeros(1, 1),
        ),
    ):
        qwen_gdn_linear_attn.QwenGatedDeltaNetAttention.forward_xpu(
            layer, hidden_states
        )

    fused.assert_called_once_with(hidden_states, layer.in_proj_qkvz, layer.in_proj_ba)
    int4_fused.assert_not_called()
    layer.in_proj_qkvz.assert_not_called()
    layer.in_proj_ba.assert_not_called()
