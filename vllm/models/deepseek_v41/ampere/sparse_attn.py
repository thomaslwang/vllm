# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 sparse-MLA attention for DeepSeek V4.1 on Ampere.

Stands in for FlashMLA's ``flash_mla_sparse_fwd``, which is SM90a/SM100f only.
The gather-and-attend core is vLLM's existing platform-generic Triton kernel
(``triton_bf16_mla_sparse_interface``); this module adds the two things
DeepSeek V4.1 needs on top of it:

* **Variable per-row topk** comes for free -- ``combine_topk_swa_indices`` pads
  with ``-1`` and the kernel already masks non-negative indices -- so
  ``topk_length`` needs no kernel support.
* **Attention sinks** are folded in afterwards from the kernel's own
  ``lse``/``max_logits`` outputs rather than inside the kernel. With
  ``m = max_logits`` and ``S = exp(lse - m) = sum_j exp(qk_j - m)``, the kernel
  returns ``acc = (1/S) * sum_j exp(qk_j - m) v_j``, while the sink-aware result
  divides the same numerator by ``S + exp(sink - m)``.  Scaling ``acc`` by
  ``S / (S + exp(sink - m))`` is therefore exact.
"""

import torch

from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface


def ampere_sparse_attn(
    q: torch.Tensor,  # [num_tokens, num_heads, dim_qk]
    kv: torch.Tensor,  # [num_kv, 1, dim_qk]
    indices: torch.Tensor,  # [num_tokens, 1, topk], -1 marks padding
    sm_scale: float,
    attn_sink: torch.Tensor | None,  # [num_heads]
    out: torch.Tensor | None = None,
    d_v: int = 512,
) -> torch.Tensor:
    """Sparse MLA attention with per-head sinks, in BF16."""
    # The kernel tiles the head dim as `BLOCK_DMODEL + BLOCK_DPE`, and both must
    # be powers of two. V4.1 prefill hands us the 512-wide latent with RoPE
    # already folded in (448 + 64 would not tile), so there is no separate PE
    # block; V3.2-style 576-wide input splits off its trailing 64.
    block_dpe = q.shape[-1] - d_v
    acc, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v=d_v, block_dpe=block_dpe
    )

    # S == 0 means the row had no valid index; the kernel's acc is 0/0 there.
    exp_sum = torch.exp(lse - max_logits)
    empty = ~torch.isfinite(exp_sum) | (exp_sum <= 0)

    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).view(1, -1).expand_as(max_logits)
        scale = exp_sum / (exp_sum + torch.exp(sink - max_logits))
        acc = acc * scale.unsqueeze(-1).to(acc.dtype)

    acc = torch.where(empty.unsqueeze(-1), torch.zeros_like(acc), acc)

    if out is not None:
        out.copy_(acc)
        return out
    return acc
