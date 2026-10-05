# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 per-token-head KV cache helpers for the XPU FLASH_ATTN backend.

Cache layout ("block-local split"): every KV block holds exactly the bytes vLLM
budgets for `int8_per_token_head` (2 * block_size * num_kv_heads * (head_size + 4)),
arranged as
    [int8 K|V for all (slot, head) | K scales fp32 (slot, head) | V scales fp32]
so that every per-(block, head) K/V base address is 64-byte aligned for Xe 2D
block loads. The upstream Triton per-token-head writer writes it unchanged
through the strided views returned by `split_layout_views`.
"""

import torch

from vllm.triton_utils import tl, triton


def split_layout_views(
    kv_cache: torch.Tensor, num_kv_heads: int, head_size: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Views over a per-layer int8 KV cache tensor (block dim first).

    Returns key/value int8 [num_blocks, block_size, num_kv_heads, head_size] and
    k_scale/v_scale fp32 [num_blocks, block_size, num_kv_heads]. The block stride
    is taken from the tensor (it can exceed the page size for padded pages).
    """
    assert kv_cache.dtype == torch.int8
    num_blocks = kv_cache.shape[0]
    page_stride = kv_cache.stride(0)
    data_bytes = block_size * num_kv_heads * 2 * head_size
    scale_bytes = block_size * num_kv_heads * 4
    assert page_stride >= data_bytes + 2 * scale_bytes
    base = kv_cache.storage_offset()
    data = torch.as_strided(
        kv_cache,
        (num_blocks, block_size, num_kv_heads, 2 * head_size),
        (page_stride, num_kv_heads * 2 * head_size, 2 * head_size, 1),
        base,
    )
    assert base % 4 == 0 and page_stride % 4 == 0 and data_bytes % 4 == 0
    f32 = torch.empty(0, dtype=torch.float32, device=kv_cache.device).set_(
        kv_cache.untyped_storage()
    )

    def scales(offset_bytes: int) -> torch.Tensor:
        return torch.as_strided(
            f32,
            (num_blocks, block_size, num_kv_heads),
            (page_stride // 4, num_kv_heads, 1),
            (base + offset_bytes) // 4,
        )

    return (
        data[..., :head_size],
        data[..., head_size:],
        scales(data_bytes),
        scales(data_bytes + scale_bytes),
    )


@triton.jit
def _dequant_to_scratch_kernel(
    key_ptr,
    value_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    max_blocks,
    stride_bt,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_head,
    stride_s_blk,
    stride_s_slot,
    stride_s_head,
    stride_o_blk,
    stride_o_slot,
    stride_o_head,
    BLOCK_SIZE: tl.constexpr,
    SLOT_TILE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    pid = tl.program_id(0)  # scratch block = seq * max_blocks + j
    head = tl.program_id(1)
    slot0 = tl.program_id(2) * SLOT_TILE
    seq = pid // max_blocks
    j = pid % max_blocks
    seq_len = tl.load(seq_lens_ptr + seq)
    if j * BLOCK_SIZE >= seq_len:
        return
    # Slots past seq_len in the last block are written as zeros: the attention
    # kernel reads whole KV tiles, and 0 * garbage could be NaN.
    src = tl.load(block_table_ptr + seq * stride_bt + j).to(tl.int64)
    slot_1d = slot0 + tl.arange(0, SLOT_TILE)
    live = j * BLOCK_SIZE + slot_1d < seq_len
    slot = slot_1d[:, None]
    d = tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    mask = d < HEAD_SIZE
    load_mask = mask & live[:, None]
    kv_off = src * stride_kv_blk + slot * stride_kv_slot + head * stride_kv_head + d
    s_off = src * stride_s_blk + slot_1d * stride_s_slot + head * stride_s_head
    k = tl.load(key_ptr + kv_off, mask=load_mask, other=0).to(tl.float32)
    v = tl.load(value_ptr + kv_off, mask=load_mask, other=0).to(tl.float32)
    k = k * tl.load(k_scale_ptr + s_off, mask=live, other=0.0)[:, None]
    v = v * tl.load(v_scale_ptr + s_off, mask=live, other=0.0)[:, None]
    o = out_ptr + pid.to(tl.int64) * stride_o_blk + slot * stride_o_slot
    o += head * stride_o_head + d
    tl.store(o, k.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(o + HEAD_SIZE, v.to(out_ptr.dtype.element_ty), mask=mask)


def dequant_to_scratch(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize the live blocks of a batch into a packed fp16/bf16 paged cache.

    Returns (scratch [num_seqs * max_blocks, block_size, H, 2 * D] with K|V packed
    like the regular FLASH_ATTN cache, block table [num_seqs, max_blocks]).
    Slots past a sequence's length in its last block are zero; scratch blocks
    wholly past it are left uninitialized and are never read.
    """
    num_seqs = seq_lens.shape[0]
    _, block_size, num_kv_heads, head_size = key_cache.shape
    max_blocks = triton.cdiv(max_seq_len, block_size)
    scratch = torch.empty(
        (num_seqs * max_blocks, block_size, num_kv_heads, 2 * head_size),
        dtype=dtype,
        device=key_cache.device,
    )
    slot_tile = 16
    assert block_size % slot_tile == 0
    grid = (num_seqs * max_blocks, num_kv_heads, block_size // slot_tile)
    _dequant_to_scratch_kernel[grid](
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        block_table,
        seq_lens,
        scratch,
        max_blocks,
        block_table.stride(0),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        scratch.stride(0),
        scratch.stride(1),
        scratch.stride(2),
        BLOCK_SIZE=block_size,
        SLOT_TILE=slot_tile,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
    )
    new_block_table = torch.arange(
        num_seqs * max_blocks, dtype=torch.int32, device=key_cache.device
    ).view(num_seqs, max_blocks)
    return scratch, new_block_table
