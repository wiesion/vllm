# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int8_per_token_head KV cache on the XPU FLASH_ATTN backend.

Decode rows (incl. uniform spec-decode verification rows) run on the native
INT8 paged decode kernel; prefill rows are dequantized into a query-dtype
scratch cache and run on the regular kernel.
Outputs are compared with an fp32 reference on the same quantized cache
(prefill rows: K/V rounded to the query dtype first, as the scratch is).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_xpu():
    pytest.skip("XPU only", allow_module_level=True)

BLOCK = 64
CASES = {  # (seq_lens, q_lens), decodes first as the metadata builder orders
    "decode": ([17, 64, 200, 129], [1, 1, 1, 1]),
    "prefill": ([17, 64, 200, 129], [17, 33, 77, 129]),
    "mixed": ([64, 200, 1, 129, 300], [1, 1, 1, 40, 300]),
    "spec_decode": ([17, 64, 200, 129], [4, 4, 4, 4]),
    "spec_mixed": ([64, 200, 129, 300], [4, 4, 40, 300]),
}
SPEC_THRESHOLD = 4  # 1 + num_speculative_tokens, as the metadata builder sets


def _num_decodes(q_lens):
    """Leading uniform rows <= threshold (split_decodes_and_prefills with
    require_uniform=True); 0 for pure decode, which takes its own branch."""
    if max(q_lens) == 1 or q_lens[0] > SPEC_THRESHOLD:
        return 0
    nd = 0
    while nd < len(q_lens) and q_lens[nd] == q_lens[0]:
        nd += 1
    return nd


def _make_cache(seq_lens, hkv, d, dtype, gen):
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )
    from vllm.v1.attention.ops.xpu_int8_kv import split_layout_views
    from vllm.v1.kv_cache_interface import KVQuantMode

    nblk = [(n + BLOCK - 1) // BLOCK for n in seq_lens]
    num_blocks = sum(nblk) + 3
    # 0x7F is not a valid scale pattern: catches reads of unwritten slots.
    kv_cache = torch.full(
        (num_blocks, hkv, BLOCK, 2 * (d + 4)), 0x7F, dtype=torch.int8, device="xpu"
    )
    views = split_layout_views(kv_cache, hkv, d, BLOCK)
    perm = torch.randperm(num_blocks - 1, generator=gen, device="xpu") + 1
    bt = torch.zeros(len(seq_lens), max(nblk), dtype=torch.int32, device="xpu")
    c = 0
    for i, n in enumerate(nblk):
        bt[i, :n] = perm[c : c + n]
        c += n
    for i, n in enumerate(seq_lens):
        pos = torch.arange(n, device="xpu")
        slots = bt[i, pos // BLOCK].long() * BLOCK + pos % BLOCK
        k = (torch.randn(n, hkv, d, generator=gen, device="xpu") * 2).to(dtype)
        k[:, :, 7] *= 20  # outlier channel, as in real K
        v = torch.randn(n, hkv, d, generator=gen, device="xpu").to(dtype)
        triton_reshape_and_cache_flash_per_token_head_quant(
            k, v, *views, slots, kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD
        )
    return kv_cache, views, bt


def _dequant(views, bt_row, n):
    kc, vc, ks, vs = views
    blocks = bt_row[: (n + BLOCK - 1) // BLOCK].long()
    k = (kc[blocks].float() * ks[blocks][..., None]).flatten(0, 1)[:n]
    v = (vc[blocks].float() * vs[blocks][..., None]).flatten(0, 1)[:n]
    return k, v


def _attend(q, k, v, scale, q_pos):
    g = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    s = torch.einsum("qhd,khd->hqk", q, k) * scale
    mask = torch.arange(k.shape[0], device=q.device)[None, :] > q_pos[:, None]
    p = torch.softmax(s.masked_fill(mask[None], float("-inf")), -1)
    return torch.einsum("hqk,khd->qhd", p, v)


def _run(case, dtype, d, hq, hkv):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    seq_lens, q_lens = CASES[case]
    gen = torch.Generator(device="xpu").manual_seed(0)
    kv_cache, views, bt = _make_cache(seq_lens, hkv, d, dtype, gen)
    n = sum(q_lens)
    q = torch.randn(n, hq, d, generator=gen, device="xpu").to(dtype)
    out = torch.full_like(q, float("nan"))
    qsl = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32, device="xpu"
    )
    nd = _num_decodes(q_lens)
    md = SimpleNamespace(
        use_cascade=False,
        num_actual_tokens=n,
        sliding_window=None,
        causal=True,
        query_start_loc=qsl,
        max_query_len=max(q_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="xpu"),
        max_seq_len=max(seq_lens),
        block_table=bt,
        max_num_splits=0,
        num_decode_reqs=nd,
        num_decode_tokens=sum(q_lens[:nd]),
    )
    impl = SimpleNamespace(
        scale=d**-0.5,
        head_size=d,
        num_kv_heads=hkv,
        sliding_window=None,
        alibi_slopes=None,
        logits_soft_cap=0.0,
        vllm_flash_attn_version=2,
        sinks=None,
        _int8_kv_views=None,
    )
    impl._int8_kv_cache_views = lambda kv: FlashAttentionImpl._int8_kv_cache_views(
        impl, kv
    )
    impl._int8_kv_spec_decode_supported = lambda q_len: (
        FlashAttentionImpl._int8_kv_spec_decode_supported(impl, q_len)
    )
    FlashAttentionImpl._forward_int8_kv_xpu(impl, q, kv_cache, md, out)
    return seq_lens, q_lens, nd, views, bt, qsl, q, out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("d,hq,hkv", [(128, 32, 8), (256, 16, 4)])
@pytest.mark.parametrize("case", list(CASES))
@torch.inference_mode()
def test_int8_kv_forward(dtype, d, hq, hkv, case):
    seq_lens, q_lens, nd, views, bt, qsl, q, out = _run(case, dtype, d, hq, hkv)

    assert torch.isfinite(out).all()
    ref = []
    for b, (n_kv, n_q) in enumerate(zip(seq_lens, q_lens)):
        k, v = _dequant(views, bt[b], n_kv)
        if b >= nd and max(q_lens) > 1:  # prefill rows read a query-dtype scratch
            k, v = k.to(dtype).float(), v.to(dtype).float()
        q0 = int(qsl[b])
        pos = torch.arange(n_kv - n_q, n_kv, device="xpu")
        ref.append(_attend(q[q0 : q0 + n_q].float(), k, v, d**-0.5, pos))
    torch.testing.assert_close(out.float(), torch.cat(ref), atol=1.5e-2, rtol=1.5e-2)


@torch.inference_mode()
def test_dequant_to_scratch_zeroes_tail_slots():
    from vllm.v1.attention.ops.xpu_int8_kv import dequant_to_scratch

    seq_lens = [70, 5]  # partial last blocks
    gen = torch.Generator(device="xpu").manual_seed(0)
    _, views, bt = _make_cache(seq_lens, 4, 128, torch.bfloat16, gen)
    kc, vc, ks, vs = views
    sl = torch.tensor(seq_lens, dtype=torch.int32, device="xpu")
    scratch, sbt = dequant_to_scratch(
        kc, vc, ks, vs, bt, sl, max(seq_lens), torch.bfloat16
    )
    for b, n in enumerate(seq_lens):
        last = int(sbt[b, (n - 1) // BLOCK])
        tail = scratch[last, n % BLOCK :] if n % BLOCK else scratch[last, :0]
        assert torch.count_nonzero(tail) == 0
        k, v = _dequant(views, bt[b], n)
        got = scratch[sbt[b, : (n + BLOCK - 1) // BLOCK].long()].flatten(0, 1)[:n]
        torch.testing.assert_close(got[..., :128].float(), k.to(torch.bfloat16).float())
        torch.testing.assert_close(got[..., 128:].float(), v.to(torch.bfloat16).float())


@pytest.mark.parametrize("case", ["spec_decode", "spec_mixed"])
@pytest.mark.parametrize("fast_path", [True, False])
@torch.inference_mode()
def test_spec_decode_rows_skip_dequant(monkeypatch, case, fast_path):
    """Uniform verification rows must not dequantize the whole context every
    step; when the kernel's spec fast path cannot take them they fall back to
    the dequant path (and stay correct, see test_int8_kv_forward)."""
    from vllm_xpu_kernels import flash_attn_interface

    from vllm.v1.attention.ops import xpu_int8_kv

    if not fast_path:
        monkeypatch.setattr(flash_attn_interface, "_SPEC_DECODE_MAX_QLEN", 1)
    calls = []
    real = xpu_int8_kv.dequant_to_scratch

    def spy(kc, vc, ks, vs, block_table, *args):
        calls.append(block_table.shape[0])
        return real(kc, vc, ks, vs, block_table, *args)

    monkeypatch.setattr(xpu_int8_kv, "dequant_to_scratch", spy)
    seq_lens, q_lens, nd, *_, out = _run(case, torch.bfloat16, 128, 32, 8)
    assert torch.isfinite(out).all()
    if fast_path:
        expected = [len(seq_lens) - nd] if len(seq_lens) > nd else []
    else:
        expected = [len(seq_lens)]
    assert calls == expected
