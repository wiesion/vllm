# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch

from vllm.distributed.device_communicators import xpu_communicator
from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator
from vllm.platforms.xpu import XPUPlatform


def test_xpu_custom_op_collectives_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES", raising=False)
    assert XPUPlatform.use_custom_op_collectives()
    monkeypatch.setenv("VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES", "0")
    assert not XPUPlatform.use_custom_op_collectives()


@pytest.mark.parametrize("env_value", [None, "0"])
def test_all_reduce_returns_input_when_explicitly_skipped(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
):
    # `_SKIP_ALL_REDUCE` is cached at import time, so exercise the skip path
    # by patching the cached flag rather than the environment variable.
    if env_value is not None:
        monkeypatch.setenv("VLLM_XPU_INPLACE_ALLREDUCE", env_value)
    communicator = object.__new__(XpuCommunicator)
    communicator.device_group = object()
    input_tensor = torch.tensor([1.0])

    with (
        patch.object(xpu_communicator, "_SKIP_ALL_REDUCE", True),
        patch(
            "vllm.distributed.device_communicators.xpu_communicator.dist.all_reduce"
        ) as all_reduce,
    ):
        output = communicator.all_reduce(input_tensor)

    assert output is not input_tensor
    assert torch.equal(output, input_tensor)
    all_reduce.assert_not_called()


def test_all_reduce_is_in_place_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_XPU_INPLACE_ALLREDUCE", raising=False)
    communicator = object.__new__(XpuCommunicator)
    communicator.device_group = Mock()
    communicator.device_group.allreduce.return_value = None
    communicator._allreduce_options = Mock(
        reduceOp=torch.distributed.ReduceOp.SUM, asyncOp=False
    )
    input_tensor = torch.tensor([1.0])

    with patch.object(xpu_communicator, "_SKIP_ALL_REDUCE", False):
        output = communicator.all_reduce(input_tensor)

    assert output is input_tensor
    communicator.device_group.allreduce.assert_called_once()
    args = communicator.device_group.allreduce.call_args.args
    assert args[0] == [input_tensor]
    assert args[1].reduceOp == torch.distributed.ReduceOp.SUM
    assert not args[1].asyncOp


def test_all_reduce_can_be_forced_out_of_place(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_XPU_INPLACE_ALLREDUCE", "0")
    communicator = object.__new__(XpuCommunicator)
    communicator.device_group = Mock()
    communicator.device_group.allreduce.return_value = None
    communicator._allreduce_options = Mock(
        reduceOp=torch.distributed.ReduceOp.SUM, asyncOp=False
    )
    input_tensor = torch.tensor([1.0])

    with (
        patch.object(xpu_communicator, "_SKIP_ALL_REDUCE", False),
        patch.object(xpu_communicator, "_INPLACE_ALL_REDUCE", False),
        patch(
            "vllm.distributed.device_communicators.xpu_communicator.dist.all_reduce"
        ) as all_reduce,
    ):
        output = communicator.all_reduce(input_tensor)

    assert output is not input_tensor
    assert torch.equal(output, input_tensor)
    communicator.device_group.allreduce.assert_called_once()
    args = communicator.device_group.allreduce.call_args.args
    assert args[0] == [output]
    assert not args[1].asyncOp


def test_all_reduce_can_use_direct_process_group_collective():
    communicator = object.__new__(XpuCommunicator)
    communicator.device_group = Mock()
    communicator._allreduce_options = object()
    work = communicator.device_group.allreduce.return_value
    input_tensor = torch.tensor([1.0])

    with (
        patch.object(xpu_communicator, "_SKIP_ALL_REDUCE", False),
        patch.object(xpu_communicator, "_INPLACE_ALL_REDUCE", True),
        patch(
            "vllm.distributed.device_communicators.xpu_communicator.dist.all_reduce"
        ) as all_reduce,
    ):
        output = communicator.all_reduce(input_tensor)

    assert output is input_tensor
    communicator.device_group.allreduce.assert_called_once_with(
        [input_tensor], communicator._allreduce_options
    )
    work.wait.assert_called_once_with()
    all_reduce.assert_not_called()


def test_all_reduce_clones_under_torch_compile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_XPU_INPLACE_ALLREDUCE", raising=False)
    communicator = object.__new__(XpuCommunicator)
    communicator.device_group = object()
    input_tensor = torch.tensor([1.0])

    with (
        patch.object(xpu_communicator, "_SKIP_ALL_REDUCE", False),
        patch(
            "vllm.distributed.device_communicators.xpu_communicator."
            "torch.compiler.is_compiling",
            return_value=True,
        ),
        patch(
            "vllm.distributed.device_communicators.xpu_communicator.dist.all_reduce"
        ) as all_reduce,
    ):
        output = communicator.all_reduce(input_tensor)

    assert output is not input_tensor
    assert torch.equal(output, input_tensor)
    all_reduce.assert_called_once_with(output, group=communicator.device_group)
