# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("model_cls", "expected_prefix"),
    [
        (Gemma4ForCausalLM, "model.layers.0.moe.experts"),
        (
            Gemma4ForConditionalGeneration,
            "language_model.model.layers.0.moe.experts",
        ),
    ],
)
@pytest.mark.parametrize("module_suffix", ["", ".base_layer"])
@pytest.mark.parametrize("lora_side", ["A", "B"])
def test_gemma4_moe_lora_mapping(model_cls, expected_prefix, module_suffix, lora_side):
    mapper = model_cls.hf_to_vllm_mapper.get_unstacked_mapper()
    module_name, is_lora_a = parse_fine_tuned_lora_name(
        "base_model.model.model.language_model.layers.0.experts"
        f"{module_suffix}.lora_{lora_side}.weight",
        mapper,
    )

    assert module_name == expected_prefix + module_suffix
    assert is_lora_a is (lora_side == "A")
    assert model_cls.is_3d_moe_weight


@pytest.mark.parametrize(
    ("model_cls", "expected_name"),
    [
        (
            Gemma4ForCausalLM,
            "model.layers.0.experts.gate_up_proj.weight",
        ),
        (
            Gemma4ForConditionalGeneration,
            "language_model.model.layers.0.experts.gate_up_proj.weight",
        ),
    ],
)
def test_gemma4_moe_lora_mapping_does_not_remap_base_weights(model_cls, expected_name):
    mapper = model_cls.hf_to_vllm_mapper

    assert (
        mapper._map_name("model.language_model.layers.0.experts.gate_up_proj.weight")
        == expected_name
    )
