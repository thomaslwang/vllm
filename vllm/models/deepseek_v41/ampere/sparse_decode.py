# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 sparse-MLA decode for DeepSeek V4.1 on Ampere.

Stands in for ``flash_mla_with_kvcache``, which is SM90a/SM100f only. That
kernel attends over two paged FP8 caches at once -- the sliding window and the
compressed KV -- each with its own ``-1``-padded global-slot index list, and
applies a per-head attention sink.

Ampere has no FlashMLA and no FP8 MMA, so this gathers the rows both index
lists name into one BF16 buffer, renumbers the indices to address it, and runs
a single sparse attention over the union. Gathering only the named rows keeps
the buffer proportional to ``tokens x topk`` rather than to the cache.
"""

import torch

from vllm.models.deepseek_v41.ampere.sparse_attn_split import (
    ampere_sparse_attn_split,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8_e4m3_portable import (
    e4m3_bytes_to_float,
    has_native_fp8e4nv,
)

# The V4 per-token record: 448 FP8 dims in 64-wide UE8M0 blocks, then 64 BF16
# RoPE dims, with a block's scales packed after all of its token data.
_V4_BYTES_PER_TOKEN = 584
_V4_FP8_DIM = 448
_V4_BF16_DIM = 64
_V4_SCALE_DIM = 8
_V4_QUANT_BLOCK = 64
_V4_TOKEN_DATA_SIZE = _V4_FP8_DIM + _V4_BF16_DIM * 2  # 576
_HEAD_DIM = 512


@triton.jit(do_not_specialize=["num_rows"])
def _gather_dequant_slots_kernel(
    cache_ptr,
    indices_ptr,
    out_ptr,
    num_rows,
    block_stride,
    WIDTH: tl.constexpr,
    TOTAL: tl.constexpr,
    COL_OFFSET: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    TOKEN_DATA_SIZE: tl.constexpr,
    FP8_DIM: tl.constexpr,
    BF16_DIM: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_QUANT_BLOCKS: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
):
    """Dequantize the paged rows named by `indices` into a dense BF16 buffer.

    Row `r` of `indices` is token `r // WIDTH`, column `r % WIDTH`, and lands in
    the union buffer at `token * TOTAL + COL_OFFSET + column`.
    """
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(indices_ptr + row).to(tl.int64)

    token = row // WIDTH
    column = row % WIDTH
    out_row_ptr = out_ptr + (token * TOTAL + COL_OFFSET + column) * HEAD_DIM
    cols = tl.arange(0, HEAD_DIM)

    if slot < 0:
        # Padding slot. The attention kernel masks it, but the buffer is reused
        # across steps so it must not keep a stale row.
        tl.store(out_row_ptr + cols, tl.zeros((HEAD_DIM,), dtype=tl.bfloat16))
    else:
        block_idx = slot // CACHE_BLOCK_SIZE
        pos_in_block = slot % CACHE_BLOCK_SIZE
        # int64 throughout: block_idx * block_stride overflows int32 on large
        # caches, the same wrap the paged MQA-logits kernel hit.
        cache_block_ptr = cache_ptr + block_idx * block_stride
        token_data_ptr = cache_block_ptr + pos_in_block * TOKEN_DATA_SIZE
        token_scale_ptr = (
            cache_block_ptr
            + CACHE_BLOCK_SIZE * TOKEN_DATA_SIZE
            + pos_in_block * SCALE_DIM
        )

        for qblock_idx in tl.static_range(N_QUANT_BLOCKS):
            qblock_start = qblock_idx * QUANT_BLOCK
            if qblock_start < FP8_DIM:
                offsets = qblock_start + tl.arange(0, QUANT_BLOCK)
                mask = offsets < FP8_DIM
                raw = tl.load(token_data_ptr + offsets, mask=mask, other=0)
                x = e4m3_bytes_to_float(raw, NATIVE_FP8)
                # UE8M0: the stored byte is the fp32 exponent field.
                encoded = tl.load(token_scale_ptr + qblock_idx)
                scale = tl.exp2(encoded.to(tl.float32) - 127.0)
                tl.store(out_row_ptr + offsets, (x * scale).to(tl.bfloat16), mask=mask)

        if BF16_DIM > 0:
            bf16_ptr = (token_data_ptr + FP8_DIM).to(tl.pointer_type(tl.bfloat16))
            tail = tl.arange(0, BF16_DIM)
            tl.store(out_row_ptr + FP8_DIM + tail, tl.load(bf16_ptr + tail))


def _gather_dequant(
    cache: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    total: int,
    col_offset: int,
) -> None:
    """Scatter-dequantize `indices` [tokens, width] into `out` [., 512] BF16."""
    bytes_per_token = cache.shape[-1]
    if bytes_per_token != _V4_BYTES_PER_TOKEN:
        raise NotImplementedError(
            f"DeepSeek V4.1 Ampere decode expects the {_V4_BYTES_PER_TOKEN}-byte "
            f"KV record, got {bytes_per_token}. The 528-byte MXFP8 record is "
            "selected only on SM100."
        )
    num_rows = indices.numel()
    if num_rows == 0:
        return
    _gather_dequant_slots_kernel[(num_rows,)](
        cache,
        indices,
        out,
        num_rows,
        cache.stride(0),
        WIDTH=indices.shape[1],
        TOTAL=total,
        COL_OFFSET=col_offset,
        CACHE_BLOCK_SIZE=cache.shape[1],
        TOKEN_DATA_SIZE=_V4_TOKEN_DATA_SIZE,
        FP8_DIM=_V4_FP8_DIM,
        BF16_DIM=_V4_BF16_DIM,
        SCALE_DIM=_V4_SCALE_DIM,
        QUANT_BLOCK=_V4_QUANT_BLOCK,
        HEAD_DIM=_HEAD_DIM,
        N_QUANT_BLOCKS=_V4_SCALE_DIM,
        NATIVE_FP8=has_native_fp8e4nv(),
        num_warps=4,
    )


def ampere_sparse_decode(
    q: torch.Tensor,  # [num_tokens, num_heads, 512]
    swa_cache: torch.Tensor,  # [num_blocks, block_size, bytes] uint8
    swa_indices: torch.Tensor,  # [num_tokens, topk_swa] global slots
    compressed_cache: torch.Tensor | None,
    topk_indices: torch.Tensor | None,  # [num_tokens, topk_c] global slots
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
) -> torch.Tensor:
    """Decode-step sparse MLA over the SWA and compressed caches together."""
    num_tokens = q.shape[0]
    parts = [(swa_cache, swa_indices.reshape(num_tokens, -1).contiguous())]
    if compressed_cache is not None and topk_indices is not None:
        parts.append(
            (compressed_cache, topk_indices.reshape(num_tokens, -1).contiguous())
        )

    total = sum(idx.shape[1] for _, idx in parts)
    kv = torch.empty(
        (num_tokens * total, _HEAD_DIM), dtype=torch.bfloat16, device=q.device
    )

    # Row (token, j) of the union lands at token * total + j, so the attention
    # indices are a fixed arange with each source's padding carried over.
    flat_indices = torch.arange(
        num_tokens * total, device=q.device, dtype=torch.int32
    ).view(num_tokens, total)

    col_offset = 0
    for cache, idx in parts:
        width = idx.shape[1]
        _gather_dequant(cache, idx, kv, total, col_offset)
        window = flat_indices[:, col_offset : col_offset + width]
        window.copy_(torch.where(idx >= 0, window, -1))
        col_offset += width

    return ampere_sparse_attn_split(
        q,
        kv.unsqueeze(1),
        flat_indices.unsqueeze(1),
        sm_scale,
        attn_sink,
        out=out,
    )
