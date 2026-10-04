# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5MoeForConditionalGeneration,
)


@pytest.mark.parametrize(
    ("peft_name", "expected_module"),
    [
        (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "language_model.model.layers.0.self_attn.q_proj",
        ),
        (
            "base_model.model.model.layers.0.mlp.experts.lora_A.weight",
            "language_model.model.layers.0.mlp.experts",
        ),
        (
            "base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight",
            "language_model.model.layers.0.mlp.experts.base_layer",
        ),
    ],
)
def test_qwen3_5_peft_lora_prefix_mapping(
    peft_name: str, expected_module: str
) -> None:
    mapper = (
        Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper.get_unstacked_mapper()
    )

    module_name, _ = parse_fine_tuned_lora_name(peft_name, mapper)

    assert module_name == expected_module
