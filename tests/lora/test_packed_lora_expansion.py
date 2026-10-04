# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm.lora.layers import MergedColumnParallelLinearWithLoRA


def test_expand_packed_lora_preserves_unadapted_trailing_slices() -> None:
    layer = SimpleNamespace(n_slices=4, output_sizes=(2, 2, 4, 4))
    qkv_a = torch.rand(3, 8)
    qkv_b = torch.rand(8, 3)

    expanded_a, expanded_b = MergedColumnParallelLinearWithLoRA.expand_packed_lora(
        layer,
        [qkv_a, None],
        [qkv_b, None],
    )

    assert all(item is qkv_a for item in expanded_a[:3])
    assert expanded_a[-1] is None
    assert expanded_b[-1] is None
    torch.testing.assert_close(expanded_b[0], qkv_b[:2])
    torch.testing.assert_close(expanded_b[1], qkv_b[2:4])
    torch.testing.assert_close(expanded_b[2], qkv_b[4:])


def test_expand_packed_lora_preserves_unadapted_leading_slices() -> None:
    layer = SimpleNamespace(n_slices=4, output_sizes=(2, 2, 4, 4))
    z_a = torch.rand(3, 8)
    z_b = torch.rand(4, 3)

    expanded_a, expanded_b = MergedColumnParallelLinearWithLoRA.expand_packed_lora(
        layer,
        [None, z_a],
        [None, z_b],
    )

    assert expanded_a[:3] == [None, None, None]
    assert expanded_b[:3] == [None, None, None]
    assert expanded_a[-1] is z_a
    torch.testing.assert_close(expanded_b[-1], z_b)


def test_expand_packed_lora_preserves_unadapted_interior_slices() -> None:
    layer = SimpleNamespace(n_slices=4, output_sizes=(2, 2, 4, 4))
    q_a = torch.rand(3, 8)
    q_b = torch.rand(2, 3)
    z_a = torch.rand(3, 8)
    z_b = torch.rand(4, 3)

    expanded_a, expanded_b = MergedColumnParallelLinearWithLoRA.expand_packed_lora(
        layer,
        [q_a, None, z_a],
        [q_b, None, z_b],
    )

    assert expanded_a[0] is q_a
    torch.testing.assert_close(expanded_b[0], q_b)
    assert expanded_a[1:3] == [None, None]
    assert expanded_b[1:3] == [None, None]
    assert expanded_a[-1] is z_a
    torch.testing.assert_close(expanded_b[-1], z_b)
