# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.models.deepseek_v4.nvidia import model as dsv4_model
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _vllm_config(
    cache_dtype: str = "fp8",
    backend: AttentionBackendEnum | None = None,
):
    return SimpleNamespace(
        attention_config=SimpleNamespace(backend=backend),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
    )


def test_select_dsv4_sm10_fp8_default_uses_flashinfer(monkeypatch):
    monkeypatch.setattr(
        dsv4_model.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(10, 0),
    )

    assert (
        dsv4_model._select_dsv4_attn_cls(_vllm_config())
        is dsv4_model.DeepseekV4FlashInferMLAAttention
    )


def test_select_dsv4_explicit_flashmla_stays_flashmla(monkeypatch):
    monkeypatch.setattr(
        dsv4_model.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(10, 0),
    )

    assert (
        dsv4_model._select_dsv4_attn_cls(
            _vllm_config(backend=AttentionBackendEnum.FLASHMLA_SPARSE_DSV4)
        )
        is dsv4_model.DeepseekV4FlashMLAAttention
    )


def test_select_dsv4_sm12_default_uses_flashinfer_sm120(monkeypatch):
    monkeypatch.setattr(
        dsv4_model.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(12, 0),
    )

    assert (
        dsv4_model._select_dsv4_attn_cls(_vllm_config())
        is dsv4_model.DeepseekV4FlashInferSM120Attention
    )
