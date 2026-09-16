# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 output projection for DeepSeek V4.1 on Ampere.

The CUDA path quantizes the rotated attention output to FP8 and contracts it
with ``wo_a`` through DeepGEMM's ``fp8_einsum``.  Neither half works on sm_80:
Triton cannot emit the FP8 cast, and Ampere has no FP8 MMA to contract with
even if it could.  BF16 ``bmm`` is the native fast path here, so this keeps the
inverse-RoPE fused kernel (with quantization switched off, which drops the FP8
branch at Triton compile time since ``QUANTIZE`` is a ``tl.constexpr``) and
dequantizes ``wo_a`` once on first use.
"""

import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)


def _bf16_bmm_weight(wo_a: nn.Module, n_groups: int, o_lora_rank: int) -> torch.Tensor:
    """Return ``wo_a``'s weight as BF16 ``[n_groups, o_lora_rank, in_dim]``.

    MXFP8 checkpoints reach here still quantized when the layer's kernel keeps
    the packed weight; dequantize once and cache it on the layer.
    """
    cached = getattr(wo_a, "_ampere_bf16_weight", None)
    if cached is not None:
        return cached

    weight = wo_a.weight.data
    if weight.dtype in (torch.float8_e4m3fn, torch.uint8):
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            dequant_mxfp8_to_bf16,
        )

        scale = getattr(wo_a, "weight_scale", None)
        if scale is None:
            scale = wo_a.weight_scale_inv
        weight = dequant_mxfp8_to_bf16(
            weight.view(torch.float8_e4m3fn).contiguous(), scale.data
        )
    weight = weight.to(torch.bfloat16).view(n_groups, o_lora_rank, -1).contiguous()

    wo_a._ampere_bf16_weight = weight
    return weight


def ampere_bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """Inverse RoPE + grouped ``wo_a`` + ``wo_b``, entirely in BF16."""
    o_proj_input, _ = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quantize=False,
    )

    weight = _bf16_bmm_weight(wo_a, n_groups, o_lora_rank)
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    torch.bmm(
        o_proj_input.transpose(0, 1),
        weight.transpose(1, 2),
        out=z.transpose(0, 1),
    )
    return wo_b(z.flatten(1))
