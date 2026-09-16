# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Ampere sparse-MLA stand-in must match FlashMLA's contract in BF16.

Covers the two things it adds over the shared Triton kernel: per-head attention
sinks, and rows whose index list is shorter than ``topk`` (``-1`` padded), plus
the degenerate all-padding row.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernels"
)

D_V = 512


def _reference(q, kv, indices, sm_scale, attn_sink):
    """Dense reference: gather, softmax with an appended sink logit, weight V."""
    num_tokens, h_q, _ = q.shape
    out = torch.zeros(num_tokens, h_q, D_V, dtype=torch.float32, device=q.device)
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
        out[t] = probs @ k[:, :D_V]
    return out


def _run(num_tokens, h_q, seq_kv, topk, *, sink, trim=None, seed=0, dim_qk=512):
    from vllm.models.deepseek_v41.ampere.sparse_attn import ampere_sparse_attn

    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(num_tokens, h_q, dim_qk, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(seq_kv, 1, dim_qk, dtype=torch.bfloat16, device=dev)
    indices = torch.randint(
        0, seq_kv, (num_tokens, 1, topk), dtype=torch.int32, device=dev
    )
    if trim is not None:
        for t, keep in enumerate(trim):
            indices[t, 0, keep:] = -1
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=dev) if sink else None
    sm_scale = dim_qk**-0.5

    got = ampere_sparse_attn(q, kv, indices, sm_scale, attn_sink)
    want = _reference(q, kv, indices, sm_scale, attn_sink)
    return got.float(), want


def _assert_close(got, want):
    denom = want.abs().max().clamp(min=1e-6)
    rel = (got - want).abs().max() / denom
    assert rel < 2e-2, f"max rel err {rel:.3e}"


# 512 is the V4.1 prefill latent (RoPE folded in); 576 splits off a PE block.
@pytest.mark.parametrize("dim_qk", [512, 576])
def test_matches_reference_without_sink(dim_qk):
    _assert_close(*_run(4, 64, 4096, 512, sink=False, dim_qk=dim_qk))


@pytest.mark.parametrize("dim_qk", [512, 576])
def test_matches_reference_with_sink(dim_qk):
    _assert_close(*_run(4, 64, 4096, 512, sink=True, dim_qk=dim_qk))


def test_sink_actually_changes_the_output():
    """A sink shrinks every row, otherwise the fold-in is a no-op bug."""
    with_sink, _ = _run(4, 64, 4096, 512, sink=True)
    without_sink, _ = _run(4, 64, 4096, 512, sink=False)
    assert not torch.allclose(with_sink, without_sink, atol=1e-3)
    assert (with_sink.abs().sum() < without_sink.abs().sum()).item()


def test_variable_topk_via_negative_padding():
    # Rows keep 512, 128, 16 and 0 of their indices.
    got, want = _run(4, 64, 4096, 512, sink=True, trim=[512, 128, 16, 0])
    _assert_close(got, want)


def test_row_with_no_valid_index_is_zero_not_nan():
    got, _ = _run(2, 64, 4096, 512, sink=True, trim=[0, 0])
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got) == 0
