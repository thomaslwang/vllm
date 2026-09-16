# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere (sm_80/sm_86) sparse-MLA attention for DeepSeek V4.1."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

from vllm.models.deepseek_v41.ampere.o_proj import ampere_bf16_o_proj
from vllm.models.deepseek_v41.ampere.sparse_attn import ampere_sparse_attn
from vllm.models.deepseek_v41.ampere.sparse_decode import ampere_sparse_decode
from vllm.models.deepseek_v41.common.ops import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadata


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

    def _sparse_attn_fwd(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        ampere_sparse_attn(
            q,
            kv,
            indices,
            self.scale,
            self.attn_sink,
            out=out,
        )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # None for SWA-only layers
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        """Decode without FlashMLA's dual-cache kernel or its tile scheduler."""
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            topk_indices, _ = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                swa_metadata.is_valid_token[:num_decode_tokens],
            )

        ampere_sparse_decode(
            q,
            self.swa_cache_layer.kv_cache,
            swa_metadata.decode_swa_indices,
            None if swa_only else kv_cache,
            topk_indices,
            self.scale,
            self.attn_sink,
            output,
        )
