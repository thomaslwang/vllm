# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere decode must read the same paged records the encoder writes.

The gather is validated against `quantize_and_insert_k_cache` (the in-tree
encoder) rather than a hand-rolled layout, so a record-layout change breaks the
test rather than silently returning wrong rows.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernels"
)

BYTES_PER_TOKEN = 584
HEAD_DIM = 512
BLOCK_SIZE = 32


def _build_cache(num_blocks, k_rows):
    """Encode `k_rows` [n, 512] bf16 into a paged cache at slots 0..n-1."""
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    cache = torch.zeros(
        (num_blocks, BLOCK_SIZE, BYTES_PER_TOKEN), dtype=torch.uint8, device="cuda"
    )
    slots = torch.arange(k_rows.shape[0], dtype=torch.int64, device="cuda")
    quantize_and_insert_k_cache(
        k_rows, cache, slots, block_size=BLOCK_SIZE, bytes_per_token=BYTES_PER_TOKEN
    )
    return cache


def _gather(cache, indices, total=None, col_offset=0, num_tokens=None):
    from vllm.models.deepseek_v41.ampere.sparse_decode import _gather_dequant

    num_tokens = num_tokens or indices.shape[0]
    width = indices.shape[1]
    total = total or width
    out = torch.zeros(
        (num_tokens * total, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    _gather_dequant(cache, indices.contiguous(), out, total, col_offset)
    return out.view(num_tokens, total, HEAD_DIM)


def test_gather_roundtrips_the_encoder():
    torch.manual_seed(0)
    n = 96
    k = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    cache = _build_cache(4, k)

    idx = torch.randperm(n, device="cuda")[:64].to(torch.int32).view(2, 32)
    got = _gather(cache, idx)

    want = k[idx.reshape(-1).long()].view(2, 32, HEAD_DIM)
    # FP8 block quantization on the leading 448 dims; the 64 RoPE dims are BF16.
    fp8_err = (got[..., :448].float() - want[..., :448].float()).abs().max()
    assert fp8_err / want[..., :448].abs().max() < 0.1, fp8_err
    torch.testing.assert_close(got[..., 448:], want[..., 448:], rtol=0, atol=0)


def test_padding_slots_are_zeroed_not_stale():
    torch.manual_seed(0)
    n = 64
    k = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    cache = _build_cache(2, k)

    idx = torch.arange(32, dtype=torch.int32, device="cuda").view(1, 32)
    idx[0, 16:] = -1
    got = _gather(cache, idx)
    assert torch.count_nonzero(got[0, 16:]) == 0
    assert torch.count_nonzero(got[0, :16]) > 0


def test_gather_writes_only_its_own_column_window():
    """Two sources must interleave into the union buffer without overwriting."""
    torch.manual_seed(0)
    n = 64
    k = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    cache = _build_cache(2, k)
    from vllm.models.deepseek_v41.ampere.sparse_decode import _gather_dequant

    num_tokens, w_a, w_b = 2, 8, 4
    total = w_a + w_b
    out = torch.zeros(
        (num_tokens * total, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    idx_a = torch.arange(num_tokens * w_a, dtype=torch.int32, device="cuda").view(
        num_tokens, w_a
    )
    idx_b = torch.arange(
        32, 32 + num_tokens * w_b, dtype=torch.int32, device="cuda"
    ).view(num_tokens, w_b)

    _gather_dequant(cache, idx_a, out, total, 0)
    _gather_dequant(cache, idx_b, out, total, w_a)

    view = out.view(num_tokens, total, HEAD_DIM)
    ref_a = _gather(cache, idx_a)
    ref_b = _gather(cache, idx_b)
    torch.testing.assert_close(view[:, :w_a], ref_a, rtol=0, atol=0)
    torch.testing.assert_close(view[:, w_a:], ref_b, rtol=0, atol=0)


def test_decode_matches_a_dense_reference():
    from vllm.models.deepseek_v41.ampere.sparse_decode import ampere_sparse_decode

    torch.manual_seed(0)
    n, num_tokens, h_q = 96, 3, 64
    k = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    cache = _build_cache(4, k)

    q = torch.randn(num_tokens, h_q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    swa = torch.randint(0, n, (num_tokens, 16), dtype=torch.int32, device="cuda")
    comp = torch.randint(0, n, (num_tokens, 32), dtype=torch.int32, device="cuda")
    swa[0, 8:] = -1  # a short row
    sink = torch.randn(h_q, dtype=torch.float32, device="cuda")
    sm_scale = HEAD_DIM**-0.5

    out = torch.zeros(num_tokens, h_q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    ampere_sparse_decode(q, cache, swa, cache, comp, sm_scale, sink, out)

    # Reference over exactly the rows the cache holds, so quantization cancels.
    rows = _gather(cache, torch.arange(n, dtype=torch.int32, device="cuda").view(1, n))
    rows = rows[0]  # [n, 512] dequantized
    want = torch.zeros(num_tokens, h_q, HEAD_DIM, dtype=torch.float32, device="cuda")
    for t in range(num_tokens):
        idx = torch.cat([swa[t], comp[t]])
        valid = idx >= 0
        kk = rows[idx[valid].long()].float()
        logits = (q[t].float() @ kk.T) * sm_scale
        logits = torch.cat([logits, sink.view(-1, 1)], dim=1)
        want[t] = torch.softmax(logits, dim=-1)[:, :-1] @ kk

    rel = (out.float() - want).abs().max() / want.abs().max()
    assert rel < 2e-2, f"max rel err {rel:.3e}"
