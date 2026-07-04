# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts import trtllm_fp8_moe


def test_swiglu_params_omitted_for_older_flashinfer(monkeypatch):
    monkeypatch.setattr(
        trtllm_fp8_moe,
        "_flashinfer_moe_supports_swiglu_params",
        lambda fn_name: False,
    )
    kwargs: dict[str, object] = {}

    trtllm_fp8_moe._add_swiglu_params_if_supported(
        kwargs,
        "trtllm_fp8_block_scale_moe",
        None,
        None,
        None,
    )

    assert "gemm1_alpha" not in kwargs
    assert "gemm1_beta" not in kwargs
    assert "gemm1_clamp_limit" not in kwargs


def test_non_default_swiglu_params_require_newer_flashinfer(monkeypatch):
    monkeypatch.setattr(
        trtllm_fp8_moe,
        "_flashinfer_moe_supports_swiglu_params",
        lambda fn_name: False,
    )

    with pytest.raises(RuntimeError, match="per-expert SwiGLU parameters"):
        trtllm_fp8_moe._add_swiglu_params_if_supported(
            {},
            "trtllm_fp8_block_scale_moe",
            torch.tensor([1.0]),
            None,
            None,
        )


def test_swiglu_params_preserved_for_newer_flashinfer(monkeypatch):
    monkeypatch.setattr(
        trtllm_fp8_moe,
        "_flashinfer_moe_supports_swiglu_params",
        lambda fn_name: True,
    )
    kwargs: dict[str, object] = {}

    trtllm_fp8_moe._add_swiglu_params_if_supported(
        kwargs,
        "trtllm_fp8_block_scale_moe",
        None,
        None,
        None,
    )

    assert set(kwargs) == {"gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"}
