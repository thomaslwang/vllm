# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere (sm_80/sm_86) sparse-MLA attention for DeepSeek V4.1."""

import torch

from vllm.models.deepseek_v41.ampere.o_proj import ampere_bf16_o_proj
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention


class DeepseekV4AmpereAttention(DeepseekV4FlashMLAAttention):
    """Sparse MLA attention for DeepSeek V4.1 on Ampere.

    Ampere has no FP8/FP4 MMA and no FlashMLA kernels, so the narrow-precision
    output projection is replaced with a BF16 one.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The FP8 einsum recipe the CUDA base computes is unused on this path.
        self._einsum_recipe = None
        self._tma_aligned_scales = False

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return ampere_bf16_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
        )
