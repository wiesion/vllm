# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker import gpu_model_runner, gpu_ubatch_wrapper
from vllm.v1.worker.gpu import model_runner as refactored_gpu_model_runner
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.ubatch_utils import UBatchSlice


def _eligibility_kwargs(num_tokens: int = 2):
    positions = np.tile(np.arange(max(num_tokens, 1)), (3, 1))
    return {
        "uses_mrope": True,
        "cudagraph_mode": CUDAGraphMode.NONE,
        "num_tokens_unpadded": num_tokens,
        "num_tokens_padded": num_tokens,
        "is_speculative": False,
        "allow_speculative": False,
        "mrope_positions": positions,
    }


def test_plain_mrope_eligibility_accepts_equal_rows():
    assert gpu_model_runner._is_qwen3_next_plain_mrope_eligible(**_eligibility_kwargs())


def test_plain_mrope_eligibility_accepts_speculative_equal_rows():
    kwargs = _eligibility_kwargs()
    kwargs["is_speculative"] = True
    kwargs["allow_speculative"] = True

    assert gpu_model_runner._is_qwen3_next_plain_mrope_eligible(**kwargs)


def test_plain_mrope_eligibility_rejects_other_speculative_methods():
    kwargs = _eligibility_kwargs()
    kwargs["is_speculative"] = True

    assert not gpu_model_runner._is_qwen3_next_plain_mrope_eligible(**kwargs)


@pytest.mark.parametrize(
    ("case", "updates"),
    [
        ("not_mrope", {"uses_mrope": False}),
        ("padded", {"num_tokens_padded": 3}),
        ("empty", {"num_tokens_unpadded": 0, "num_tokens_padded": 0}),
        ("over_64", {"num_tokens_unpadded": 65, "num_tokens_padded": 65}),
        ("capture", {"cudagraph_mode": CUDAGraphMode.FULL}),
    ],
)
def test_plain_mrope_eligibility_rejects_non_runtime_contexts(case, updates):
    kwargs = _eligibility_kwargs(65 if case == "over_64" else 2)
    kwargs.update(updates)

    assert not gpu_model_runner._is_qwen3_next_plain_mrope_eligible(**kwargs)


@pytest.mark.parametrize("row", [1, 2])
def test_plain_mrope_eligibility_rejects_unequal_rows(row):
    kwargs = _eligibility_kwargs()
    kwargs["mrope_positions"][row, 1] += 1

    assert not gpu_model_runner._is_qwen3_next_plain_mrope_eligible(**kwargs)


@pytest.mark.parametrize(
    (
        "uses_mrope",
        "cudagraph_mode",
        "num_tokens",
        "num_tokens_padded",
        "prefilling",
        "expected",
    ),
    [
        (True, CUDAGraphMode.NONE, 17, 17, False, True),
        (False, CUDAGraphMode.NONE, 17, 17, False, False),
        (True, CUDAGraphMode.FULL, 17, 17, False, False),
        (True, CUDAGraphMode.NONE, 17, 32, False, False),
        (True, CUDAGraphMode.NONE, 65, 65, False, False),
        (True, CUDAGraphMode.NONE, 17, 17, True, False),
    ],
)
def test_refactored_runner_sets_plain_mrope_context(
    uses_mrope,
    cudagraph_mode,
    num_tokens,
    num_tokens_padded,
    prefilling,
    expected,
):
    context = SimpleNamespace(additional_kwargs={})
    input_batch = SimpleNamespace(
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens_padded,
        is_prefilling_np=np.array([prefilling]),
    )
    with patch.object(
        refactored_gpu_model_runner,
        "get_forward_context",
        return_value=context,
    ):
        refactored_gpu_model_runner._set_qwen3_next_plain_mrope_context(
            enabled=True,
            uses_mrope=uses_mrope,
            cudagraph_mode=cudagraph_mode,
            input_batch=input_batch,
        )

    assert context.additional_kwargs[
        refactored_gpu_model_runner._QWEN3_NEXT_PLAIN_MROPE
    ] is expected


def test_refactored_runner_does_not_enable_plain_mrope_without_qwen_spec():
    context = SimpleNamespace(additional_kwargs={})
    input_batch = SimpleNamespace(
        num_tokens=17,
        num_tokens_after_padding=17,
        is_prefilling_np=np.array([False]),
    )
    with patch.object(
        refactored_gpu_model_runner,
        "get_forward_context",
        return_value=context,
    ):
        refactored_gpu_model_runner._set_qwen3_next_plain_mrope_context(
            enabled=False,
            uses_mrope=True,
            cudagraph_mode=CUDAGraphMode.NONE,
            input_batch=input_batch,
        )

    assert context.additional_kwargs[
        refactored_gpu_model_runner._QWEN3_NEXT_PLAIN_MROPE
    ] is False


def test_ubatch_contexts_propagate_additional_kwargs():
    wrapper = object.__new__(UBatchWrapper)
    wrapper.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2)
    )
    wrapper.comm_stream = object()
    wrapper.ready_barrier = object()
    wrapper.cudagraphs = {}
    wrapper.sm_control = nullcontext()
    wrapper.runnable = Mock()
    wrapper._run_ubatches = Mock(return_value="result")

    ubatch_slices = [
        UBatchSlice(slice(0, 1), slice(0, 1)),
        UBatchSlice(slice(1, 2), slice(1, 2)),
    ]
    additional_kwargs = {gpu_model_runner._QWEN3_NEXT_PLAIN_MROPE: True}
    forward_context = SimpleNamespace(
        batch_descriptor=object(),
        ubatch_slices=ubatch_slices,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        attn_metadata=[{}, {}],
        slot_mapping=[{}, {}],
        dp_metadata=object(),
        additional_kwargs=additional_kwargs,
    )
    ubatch_contexts = [SimpleNamespace(id=0), SimpleNamespace(id=1)]

    with (
        patch.object(
            gpu_ubatch_wrapper, "get_forward_context", return_value=forward_context
        ),
        patch.object(torch.cuda, "current_stream", return_value=object()),
        patch.object(
            gpu_ubatch_wrapper.DPMetadata, "make", side_effect=[object(), object()]
        ),
        patch.object(
            gpu_ubatch_wrapper,
            "create_forward_context",
            side_effect=[object(), object()],
        ) as create_context,
        patch.object(
            gpu_ubatch_wrapper,
            "make_ubatch_contexts",
            return_value=ubatch_contexts,
        ),
    ):
        result = wrapper(
            input_ids=torch.arange(2),
            positions=torch.arange(2).expand(3, -1),
            intermediate_tensors=None,
            inputs_embeds=None,
        )

    assert result == "result"
    assert len(create_context.call_args_list) == 2
    for call in create_context.call_args_list:
        assert call.kwargs["additional_kwargs"] is additional_kwargs
