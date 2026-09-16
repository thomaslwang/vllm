# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable E4M3 conversions must match torch everywhere the docs claim they do.

These run on any CUDA device; on sm_89+ they exercise the native ``fp8e4nv``
path and on sm_80/sm_86 the ``fp8e4b15`` bias-shift path, so the same assertions
pin down both.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.fp8_e4m3_portable import (
    e4m3_to_float,
    float_to_e4m3,
    has_native_fp8e4nv,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernels"
)


def _all_byte_patterns(device):
    codes = torch.arange(256, dtype=torch.uint8, device=device)
    return codes, codes.view(torch.float8_e4m3fn).float()


def test_decode_is_bit_exact_on_every_finite_pattern():
    codes, ref = _all_byte_patterns("cuda")
    got = e4m3_to_float(codes)
    finite = torch.isfinite(ref)
    # 254 of 256: 0x7F and 0xFF are E4M3's NaNs.
    assert int(finite.sum()) == 254
    torch.testing.assert_close(got[finite], ref[finite], rtol=0, atol=0)


def test_decode_handles_non_power_of_two_and_masked_tail():
    torch.manual_seed(0)
    for n in (1, 3, 1023, 1025, 4097):
        codes = torch.randint(0, 256, (n,), dtype=torch.uint8, device="cuda")
        ref = codes.view(torch.float8_e4m3fn).float()
        got = e4m3_to_float(codes)
        finite = torch.isfinite(ref)
        torch.testing.assert_close(got[finite], ref[finite], rtol=0, atol=0)


def test_decode_preserves_shape_and_large_offsets():
    # Exceeds a single launch block many times over; also checks 2-D shapes.
    codes = torch.randint(0, 256, (4096, 512), dtype=torch.uint8, device="cuda")
    got = e4m3_to_float(codes)
    assert got.shape == codes.shape
    ref = codes.view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(ref)
    torch.testing.assert_close(got[finite], ref[finite], rtol=0, atol=0)


def test_encode_matches_torch_on_non_tie_values():
    torch.manual_seed(0)
    # Random floats never land on exact ties, so these must agree byte for byte.
    for scale in (0.001, 1.0, 100.0, 1e4):
        x = torch.randn(8192, device="cuda") * scale
        got = float_to_e4m3(x).view(torch.uint8)
        want = x.to(torch.float8_e4m3fn).view(torch.uint8)
        assert torch.equal(got, want), f"scale={scale}"


def test_encode_saturates_over_range_and_keeps_signed_zero():
    x = torch.tensor(
        [448.0, 1e4, 1e30, float("inf"), -448.0, -1e4, float("-inf"), 0.0, -0.0],
        device="cuda",
    )
    got = float_to_e4m3(x).view(torch.uint8)
    want = x.to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(got, want)


def test_encode_roundtrips_every_finite_value():
    codes, vals = _all_byte_patterns("cuda")
    finite = torch.isfinite(vals)
    got = float_to_e4m3(vals[finite]).view(torch.uint8)
    assert torch.equal(got, codes[finite])


def test_encode_tie_rounding_is_documented_behaviour():
    """Ties round half away from zero off the native path, half to even on it."""
    _, vals = _all_byte_patterns("cuda")
    finite = torch.isfinite(vals)
    pos = sorted({v for v in vals[finite].tolist() if v >= 0})
    mids = torch.tensor(
        [(a + b) / 2 for a, b in zip(pos, pos[1:])], device="cuda", dtype=torch.float32
    )
    mids = torch.cat([mids, -mids])

    got = float_to_e4m3(mids).float()
    want = mids.to(torch.float8_e4m3fn).float()

    if has_native_fp8e4nv():
        assert torch.equal(got, want)
    else:
        # Never toward zero. (torch may round a midpoint down to exactly 0;
        # we round it up to the smallest subnormal, so compare signs only
        # where torch kept one.)
        assert (got.abs() >= want.abs()).all()
        nonzero = want != 0
        assert torch.equal(got[nonzero].sign(), want[nonzero].sign())
        # Each result is still a representable E4M3 value, one step at most away.
        neighbours = torch.tensor(sorted(set(vals[finite].tolist())), device="cuda")
        assert torch.isin(got, neighbours).all()
