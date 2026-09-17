# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV sparse MLA decode for DeepSeek V4.1 on Ampere.

The shared BF16 sparse-MLA kernel parallelizes over tokens and head blocks
only, so a batch-1 decode step launches `ceil(64 / BLOCK_H) == 4` CTAs and
leaves an A800's 108 SMs 96% idle while one CTA walks all `topk / BLOCK_N`
key blocks serially. Measured under CUDA-graph replay at the V4.1 decode
shape (1 token, 64 heads, topk 640) that is 77 us; the same kernel does eight
times the work in 79 us at 8 tokens, which is the signature of a launch-bound
kernel rather than a throughput-bound one.

This splits the key range across CTAs as well (flash-decoding), each producing
a partial numerator with its own running max and denominator, and combines the
partials with the usual log-sum-exp merge. The per-head attention sink folds
into the combine step, so the separate elementwise pass the non-split path
needs disappears too.
"""

import torch

from vllm.triton_utils import LOG2E, tl, triton

_BLOCK_H = 16
_BLOCK_N = 16
# Measured on an A800: total CTAs around 256 is the optimum across batch sizes
# -- enough to fill 108 SMs with a couple of waves, before per-split fixed costs
# and the combine's reduction width start to dominate.
_TARGET_CTAS = 256
_MAX_SPLITS = 32


@triton.jit
def _split_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    acc_ptr,  # [T, H, S, DV] fp32 partial numerators
    emax_ptr,  # [T, H, S] fp32 running max (log2 domain)
    esum_ptr,  # [T, H, S] fp32 running denominator
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_idx_token,
    stride_acc_token,
    stride_acc_head,
    stride_acc_split,
    stride_st_token,
    stride_st_head,
    sm_scale,
    index_topk: tl.constexpr,
    per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_t = tl.program_id(0)
    cur_hb = tl.program_id(1)
    cur_s = tl.program_id(2)

    offs_h = cur_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + cur_t * stride_q_token + offs_h[:, None] * stride_q_head
    q = tl.load(q_base + offs_d[None, :], mask=mask_h[:, None], other=0.0)

    # A finite sentinel, not -inf: a split whose keys are all padding would
    # otherwise produce exp2(-inf - -inf) = NaN in the combine.
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    start = cur_s * per_split
    for n0 in range(start, tl.minimum(start + per_split, index_topk), BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        idx = tl.load(
            indices_ptr + cur_t * stride_idx_token + offs_n,
            mask=offs_n < index_topk,
            other=-1,
        )
        mask_kv = (idx >= 0) & (idx < seq_kv)
        k = tl.load(
            kv_ptr + idx[None, :] * stride_kv_token + offs_d[:, None],
            mask=mask_kv[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k.to(q.dtype)) * sm_scale
        qk = tl.where(mask_h[:, None] & mask_kv[None, :], qk, -1.0e30)

        v = tl.load(
            kv_ptr + idx[:, None] * stride_kv_token + offs_d[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        rescale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * rescale + tl.sum(p, 1)
        e_max = n_e_max

    base = (
        cur_t * stride_acc_token + offs_h * stride_acc_head + cur_s * stride_acc_split
    )
    tl.store(acc_ptr + base[:, None] + offs_d[None, :], acc, mask=mask_h[:, None])
    st = cur_t * stride_st_token + offs_h * stride_st_head + cur_s
    tl.store(emax_ptr + st, e_max, mask=mask_h)
    tl.store(esum_ptr + st, e_sum, mask=mask_h)


@triton.jit
def _combine_kernel(
    acc_ptr,
    emax_ptr,
    esum_ptr,
    sink_ptr,
    out_ptr,
    stride_acc_token,
    stride_acc_head,
    stride_acc_split,
    stride_st_token,
    stride_st_head,
    stride_out_token,
    stride_out_head,
    LOG2E_C: tl.constexpr,
    HAS_SINK: tl.constexpr,
    n_splits: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One CTA per (token, head).

    Reducing a whole head at once keeps the partials in a single
    `[n_splits, BLOCK_D]` tile. Looping over splits instead, with a
    `[BLOCK_H, BLOCK_D]` tile per iteration, spills once n_splits > 4 and costs
    an order of magnitude more than the split kernel it is combining.
    """
    cur_t = tl.program_id(0)
    cur_h = tl.program_id(1)

    splits = tl.arange(0, n_splits)
    st = cur_t * stride_st_token + cur_h * stride_st_head + splits
    emax = tl.load(emax_ptr + st)
    esum = tl.load(esum_ptr + st)

    m = tl.max(emax, 0)
    scale = tl.exp2(emax - m)
    denom = tl.sum(esum * scale, 0)

    if HAS_SINK:
        # The sink is a raw logit appended to the softmax denominator; the
        # running maxima live in the log2 domain, hence the LOG2E factor.
        denom += tl.exp2(tl.load(sink_ptr + cur_h) * LOG2E_C - m)

    offs_d = tl.arange(0, BLOCK_D)
    part = tl.load(
        acc_ptr
        + cur_t * stride_acc_token
        + cur_h * stride_acc_head
        + splits[:, None] * stride_acc_split
        + offs_d[None, :]
    )
    acc = tl.sum(part * scale[:, None], 0)

    # denom == 0 means every key in every split was padding.
    out = tl.where(denom > 0, acc / tl.where(denom > 0, denom, 1.0), 0.0)
    tl.store(
        out_ptr + cur_t * stride_out_token + cur_h * stride_out_head + offs_d,
        out.to(out_ptr.dtype.element_ty),
    )


def _choose_splits(num_tokens: int, h_blocks: int, topk: int) -> int:
    """Power-of-two split count that lands near `_TARGET_CTAS` total CTAs.

    Capped so each split still covers at least two key blocks; below that the
    per-split prologue (loading q) outweighs the work it saves.
    """
    n_blocks = max(1, triton.cdiv(topk, _BLOCK_N))
    want = _TARGET_CTAS // max(1, num_tokens * h_blocks)
    want = min(want, _MAX_SPLITS, max(1, n_blocks // 2))
    return 1 << (want.bit_length() - 1) if want > 1 else 1


def ampere_sparse_attn_split(
    q: torch.Tensor,  # [num_tokens, num_heads, dim]
    kv: torch.Tensor,  # [num_kv, 1, dim]
    indices: torch.Tensor,  # [num_tokens, 1, topk], -1 marks padding
    sm_scale: float,
    attn_sink: torch.Tensor | None,  # [num_heads]
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens, h_q, dim = q.shape
    topk = indices.shape[-1]
    assert kv.shape[-1] == dim, "q and kv head dims must match"

    h_blocks = triton.cdiv(h_q, _BLOCK_H)
    n_splits = _choose_splits(num_tokens, h_blocks, topk)
    per_split = triton.cdiv(triton.cdiv(topk, _BLOCK_N), n_splits) * _BLOCK_N

    acc = torch.empty(
        (num_tokens, h_q, n_splits, dim), dtype=torch.float32, device=q.device
    )
    emax = torch.empty(
        (num_tokens, h_q, n_splits), dtype=torch.float32, device=q.device
    )
    esum = torch.empty_like(emax)

    idx2d = indices.view(num_tokens, -1)
    _split_kernel[(num_tokens, h_blocks, n_splits)](
        q,
        kv,
        idx2d,
        acc,
        emax,
        esum,
        kv.shape[0],
        h_q,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        idx2d.stride(0),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        emax.stride(0),
        emax.stride(1),
        sm_scale * LOG2E,
        index_topk=topk,
        per_split=per_split,
        BLOCK_H=_BLOCK_H,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=dim,
        num_warps=4,
        num_stages=2,
    )

    if out is None:
        out = torch.empty((num_tokens, h_q, dim), dtype=q.dtype, device=q.device)
    _combine_kernel[(num_tokens, h_q)](
        acc,
        emax,
        esum,
        attn_sink,
        out,
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        emax.stride(0),
        emax.stride(1),
        out.stride(0),
        out.stride(1),
        LOG2E_C=LOG2E,
        HAS_SINK=attn_sink is not None,
        n_splits=n_splits,
        BLOCK_D=dim,
        num_warps=4,
    )
    return out
