# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 backend for Ampere (sm_80/sm_86).

Ampere has no FP8 or FP4 tensor-core instruction and no FlashMLA/DeepGEMM
support, so this package replaces the narrow-precision and FlashMLA-bound parts
of the CUDA path with BF16 equivalents that Ampere runs natively.
"""
