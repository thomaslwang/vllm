# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable OCP E4M3 <-> float conversions for Triton on pre-SM89 GPUs.

Triton's CUDA backend refuses the ``fp8e4nv`` *type* on sm_80/sm_86 -- not just
conversions to and from it, but any use at all, including ``bitcast=True``::

    ValueError: type fp8e4nv not supported in this architecture.
                The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')

That blocks every DeepSeek-V4/V4.1 FP8 KV-cache kernel on Ampere, since the
cache records are OCP E4M3 bytes.

``fp8e4b15`` *is* accepted, and it differs from OCP E4M3 only in exponent bias
(15 vs 7).  Reinterpreting E4M3 bits as e4b15 therefore yields exactly the right
value scaled by ``2**-8``, for normals and subnormals alike -- so a single
multiply by 256 recovers it.  Encoding is the same identity run backwards.

Accuracy (measured against ``torch`` on sm_80, see
``tests/kernels/attention/test_fp8_e4m3_portable.py``):

* ``e4m3_bytes_to_float`` is **bit-exact** on all 254 finite E4M3 byte patterns.
  Only 0x7F/0xFF -- E4M3's NaNs -- differ, and no well-formed cache encoder
  emits them.
* ``float_to_e4m3_bytes`` agrees with ``Tensor.to(torch.float8_e4m3fn)`` on
  every non-tie value, on over-range saturation, on subnormals and on negatives.
  It differs on **exact ties**, where it rounds half away from zero while IEEE
  rounds half to even (126 of the 252 midpoints between adjacent finite E4M3
  values), and it saturates NaN to 448 instead of propagating NaN.  Both are
  sub-ULP for KV-cache quantization, but callers that need strict RNE should not
  use this path.

On sm_89 and newer the native ``fp8e4nv`` conversions are used instead, so these
helpers are safe to call unconditionally.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def has_native_fp8e4nv() -> bool:
    """True when Triton can lower ``fp8e4nv`` on this device (sm_89+)."""
    if not current_platform.is_cuda():
        return True
    cap = current_platform.get_device_capability()
    if cap is None:
        return True
    return (cap.major, cap.minor) >= (8, 9)


@triton.jit
def e4m3_bytes_to_float(x_u8, NATIVE_FP8: tl.constexpr):
    """Decode OCP E4M3 bytes to fp32. ``x_u8`` is a uint8 tile.

    Bit-exact for every finite E4M3 encoding on both paths.
    """
    if NATIVE_FP8:
        return x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    # E4M3 bias is 7, e4b15's is 15, so the reinterpreted bits read 2**-8 low.
    return x_u8.to(tl.float8e4b15, bitcast=True).to(tl.float32) * 256.0


@triton.jit
def float_to_e4m3_bytes(x, NATIVE_FP8: tl.constexpr):
    """Encode a float tile to OCP E4M3 bytes (uint8).

    The non-native path rounds exact ties away from zero rather than to even;
    see the module docstring.
    """
    if NATIVE_FP8:
        return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    # 0.00390625 == 2**-8, undoing the bias difference before the cast.
    return (x * 0.00390625).to(tl.float8e4b15).to(tl.uint8, bitcast=True)


# ---------------------------------------------------------------------------
# Thin tensor-level wrappers, mainly for tests and one-off conversions.
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["n"])
def _decode_kernel(src, dst, n, NATIVE_FP8: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    b = tl.load(src + off, mask=mask, other=0)
    tl.store(dst + off, e4m3_bytes_to_float(b, NATIVE_FP8), mask=mask)


@triton.jit(do_not_specialize=["n"])
def _encode_kernel(src, dst, n, NATIVE_FP8: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    x = tl.load(src + off, mask=mask, other=0.0)
    tl.store(dst + off, float_to_e4m3_bytes(x, NATIVE_FP8), mask=mask)


def e4m3_to_float(x: torch.Tensor) -> torch.Tensor:
    """Decode a uint8 / float8_e4m3fn tensor to fp32."""
    if x.dtype == torch.float8_e4m3fn:
        x = x.view(torch.uint8)
    assert x.dtype == torch.uint8, f"expected uint8 bytes, got {x.dtype}"
    x = x.contiguous()
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    n = x.numel()
    BLOCK = 1024
    _decode_kernel[(triton.cdiv(n, BLOCK),)](
        x, out, n, NATIVE_FP8=has_native_fp8e4nv(), BLOCK=BLOCK
    )
    return out


def float_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Encode a float tensor to OCP E4M3, returned as float8_e4m3fn."""
    x = x.contiguous().to(torch.float32)
    out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    n = x.numel()
    BLOCK = 1024
    _encode_kernel[(triton.cdiv(n, BLOCK),)](
        x, out, n, NATIVE_FP8=has_native_fp8e4nv(), BLOCK=BLOCK
    )
    return out.view(torch.float8_e4m3fn)
