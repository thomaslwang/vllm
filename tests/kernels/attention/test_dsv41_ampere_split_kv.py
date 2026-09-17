# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV sparse attention must agree with the single-CTA-per-head path.

The split path carries its own running max and denominator per CTA and merges
them afterwards, so the cases that matter are the ones where a split sees no
usable key at all: a row trimmed mid-split, and a row that is entirely padding.
"""

import pytest
import torch
import triton

from vllm.models.deepseek_v41.ampere.sparse_attn import ampere_sparse_attn
from vllm.models.deepseek_v41.ampere.sparse_attn_split import (
    _BLOCK_H,
    _choose_splits,
    ampere_sparse_attn_split,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernels"
)

DIM = 512


def _dense_reference(q, kv, indices, sm_scale, attn_sink):
    num_tokens, h_q, _ = q.shape
    out = torch.zeros(num_tokens, h_q, DIM, dtype=torch.float32, device=q.device)
    for t in range(num_tokens):
        idx = indices[t, 0]
        valid = idx >= 0
        if not bool(valid.any()):
            continue
        k = kv[idx[valid].long(), 0].float()
        logits = (q[t].float() @ k.T) * sm_scale
        if attn_sink is not None:
            logits = torch.cat([logits, attn_sink.float().view(-1, 1)], dim=1)
        probs = torch.softmax(logits, dim=-1)
        if attn_sink is not None:
            probs = probs[:, :-1]
        out[t] = probs @ k
    return out


def _case(num_tokens, topk, *, sink=True, trim=None, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(num_tokens, 64, DIM, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(num_tokens * topk, 1, DIM, dtype=torch.bfloat16, device=dev)
    idx = torch.arange(
        num_tokens * topk, dtype=torch.int32, device=dev
    ).view(num_tokens, 1, topk)
    if trim is not None:
        for t, keep in enumerate(trim):
            idx[t, 0, keep:] = -1
    attn_sink = torch.randn(64, dtype=torch.float32, device=dev) if sink else None
    return q, kv, idx, DIM**-0.5, attn_sink


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-6)).item()


@pytest.mark.parametrize(
    "num_tokens,topk", [(1, 640), (1, 512), (2, 640), (8, 640), (1, 2048)]
)
def test_matches_dense_reference(num_tokens, topk):
    q, kv, idx, scale, sink = _case(num_tokens, topk)
    got = ampere_sparse_attn_split(q, kv, idx, scale, sink).float()
    assert _rel(got, _dense_reference(q, kv, idx, scale, sink)) < 2e-2


def test_no_worse_than_the_unsplit_path():
    """The fp32 merge should not lose accuracy relative to the single-CTA path."""
    q, kv, idx, scale, sink = _case(1, 640)
    ref = _dense_reference(q, kv, idx, scale, sink)
    split = _rel(ampere_sparse_attn_split(q, kv, idx, scale, sink).float(), ref)
    unsplit = _rel(ampere_sparse_attn(q, kv, idx, scale, sink).float(), ref)
    assert split <= unsplit * 1.5


def test_row_trimmed_mid_split():
    # 37 keys leaves most splits with nothing to do.
    q, kv, idx, scale, sink = _case(1, 640, trim=[37])
    got = ampere_sparse_attn_split(q, kv, idx, scale, sink).float()
    assert _rel(got, _dense_reference(q, kv, idx, scale, sink)) < 2e-2


def test_fully_padded_row_is_zero_not_nan():
    q, kv, idx, scale, sink = _case(2, 640, trim=[0, 640])
    got = ampere_sparse_attn_split(q, kv, idx, scale, sink)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[0]) == 0
    assert torch.count_nonzero(got[1]) > 0


def test_sink_is_actually_applied():
    q, kv, idx, scale, sink = _case(1, 640)
    with_sink = ampere_sparse_attn_split(q, kv, idx, scale, sink).float()
    without = ampere_sparse_attn_split(q, kv, idx, scale, None).float()
    assert not torch.allclose(with_sink, without, atol=1e-3)
    assert with_sink.abs().sum() < without.abs().sum()


@pytest.mark.parametrize(
    "num_tokens,topk,expected",
    # Empirical optima measured on an A800; the heuristic must keep matching.
    [(1, 640, 16), (2, 640, 16), (4, 640, 16), (8, 640, 8), (16, 640, 4),
     (32, 640, 2), (1, 2048, 32)],
)
def test_split_heuristic_tracks_measured_optimum(num_tokens, topk, expected):
    h_blocks = triton.cdiv(64, _BLOCK_H)
    assert _choose_splits(num_tokens, h_blocks, topk) == expected
