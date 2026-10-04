# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import msgspec

from vllm.v1.engine import EngineCoreReadyResponse
from vllm.v1.engine.core_client import MPClient


def test_apply_ready_response_syncs_mamba_block_size():
    client = object.__new__(MPClient)
    client.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16, mamba_block_size=16, num_gpu_blocks=0
        ),
        model_config=SimpleNamespace(max_model_len=8192),
    )
    client.stats_update_address = None
    payload = msgspec.msgpack.encode(
        EngineCoreReadyResponse(
            max_model_len=8192,
            num_gpu_blocks=100,
            block_size=64,
            mamba_block_size=128,
            dp_stats_address=None,
            dtype="bfloat16",
            vllm_version="test",
            world_size=1,
            data_parallel_size=1,
        )
    )

    client._apply_ready_response(payload)

    assert client.vllm_config.cache_config.block_size == 64
    assert client.vllm_config.cache_config.mamba_block_size == 128


def test_legacy_ready_response_preserves_frontend_mamba_block_size():
    client = object.__new__(MPClient)
    client.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16, mamba_block_size=32, num_gpu_blocks=0
        ),
        model_config=SimpleNamespace(max_model_len=8192),
    )
    client.stats_update_address = None
    payload = msgspec.msgpack.encode(
        {
            "max_model_len": 8192,
            "num_gpu_blocks": 100,
            "block_size": 64,
            "dp_stats_address": None,
            "dtype": "bfloat16",
            "vllm_version": "test",
            "world_size": 1,
            "data_parallel_size": 1,
        }
    )

    client._apply_ready_response(payload)

    assert client.vllm_config.cache_config.block_size == 64
    assert client.vllm_config.cache_config.mamba_block_size == 32
