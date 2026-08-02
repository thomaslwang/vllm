# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Island-aware hierarchical allreduce for multi-island PCIe topologies.

Targets boxes like 2x4 A100/A800 PCIe (two PIX islands bridged by the CPU
interconnect), where the NCCL 8-GPU ring pays the cross-socket latency on
every hop. Strategy per rank, all inside one Triton kernel launch:

  1. publish: copy the local input into this rank's IPC-shared slot,
     release-fence, bump the phase-A flag.
  2. island reduce: spin on the island peers' phase-A flags, then read the
     island's slots directly over P2P and reduce -> island partial, publish
     to the partial slot with a phase-B flag.
  3. cross exchange: spin on the counterpart rank's phase-B flag (the rank
     with the same island-local index in the other island), read its partial
     over P2P, add, write the final result. Cross-socket traffic is exactly
     one message per rank instead of the ring's repeated crossings.

Latency-bound small messages only (decode hidden states); large payloads
should stay on NCCL. Buffers are IPC-registered once; flags use monotonically
increasing sequence tokens so no zeroing is needed between calls.
"""

from collections.abc import Sequence

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_MAX_ELEMS = 64 * 1024  # 128KB bf16 cap; decode messages are ~8-32KB


@triton.jit
def _hier_all_reduce_kernel(
    inp_ptr,
    out_ptr,
    ptrs_ptr,  # [world] int64 device pointers to each rank's data slot
    partial_ptrs_ptr,  # [world] int64 pointers to each rank's partial slot
    flag_ptrs_ptr,  # [world] int64 pointers to each rank's flag pair
    rank: tl.constexpr,
    island_base: tl.constexpr,  # first rank of this island
    island_size: tl.constexpr,
    counterpart: tl.constexpr,  # same-index rank in the other island
    numel,
    token,  # sequence token for this call (int32, monotonically increasing)
    BLOCK: tl.constexpr,
):
    my_slot = tl.cast(tl.load(ptrs_ptr + rank), tl.pointer_type(tl.bfloat16))
    my_partial = tl.cast(
        tl.load(partial_ptrs_ptr + rank), tl.pointer_type(tl.float32)
    )
    my_flags = tl.cast(
        tl.load(flag_ptrs_ptr + rank), tl.pointer_type(tl.int32)
    )

    # Phase A: publish the whole input, then raise the phase-A flag once.
    for off in range(0, numel, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
        tl.store(my_slot + offs, x, mask=mask)
    tl.debug_barrier()
    tl.atomic_xchg(my_flags + 0, token, sem="release")

    # Wait for all island peers' phase-A flags (once, before touching data).
    for i in tl.static_range(island_size):
        peer = island_base + i
        if peer != rank:
            peer_flags = tl.cast(
                tl.load(flag_ptrs_ptr + peer), tl.pointer_type(tl.int32)
            )
            while tl.atomic_add(peer_flags + 0, 0, sem="acquire") < token:
                pass

    # Phase B: island reduce over P2P, publish partial, raise phase-B flag.
    for off in range(0, numel, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < numel
        acc = tl.load(my_slot + offs, mask=mask, other=0.0).to(tl.float32)
        for i in tl.static_range(island_size):
            peer = island_base + i
            if peer != rank:
                peer_slot = tl.cast(
                    tl.load(ptrs_ptr + peer), tl.pointer_type(tl.bfloat16)
                )
                acc += tl.load(peer_slot + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
        tl.store(my_partial + offs, acc, mask=mask)
    tl.debug_barrier()
    tl.atomic_xchg(my_flags + 1, token, sem="release")

    # Phase C: single cross-island exchange with the counterpart rank.
    cp_flags = tl.cast(
        tl.load(flag_ptrs_ptr + counterpart), tl.pointer_type(tl.int32)
    )
    while tl.atomic_add(cp_flags + 1, 0, sem="acquire") < token:
        pass
    cp_partial = tl.cast(
        tl.load(partial_ptrs_ptr + counterpart), tl.pointer_type(tl.float32)
    )
    for off in range(0, numel, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < numel
        acc = tl.load(my_partial + offs, mask=mask, other=0.0)
        acc += tl.load(cp_partial + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


class HierarchicalAllReduce:
    """Two-level island-aware allreduce over IPC-shared buffers.

    Args:
        group: torch.distributed group covering all ranks on this node.
        device: this rank's CUDA device.
        islands: rank partition, e.g. [[0,1,2,3],[4,5,6,7]]. Exactly two
            islands of equal size are supported.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        islands: Sequence[Sequence[int]],
    ) -> None:
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        assert len(islands) == 2 and len(islands[0]) == len(islands[1]), (
            "HierarchicalAllReduce supports exactly two equal islands"
        )
        self.islands = [list(i) for i in islands]
        me = self.rank
        self.island_idx = 0 if me in self.islands[0] else 1
        island = self.islands[self.island_idx]
        other = self.islands[1 - self.island_idx]
        self.island_base = min(island)
        self.island_size = len(island)
        self.counterpart = other[island.index(me)]

        # data slot (bf16), partial slot (fp32), flags (2 x int32)
        self._data = torch.zeros(_MAX_ELEMS, dtype=torch.bfloat16, device=device)
        self._partial = torch.zeros(_MAX_ELEMS, dtype=torch.float32, device=device)
        self._flags = torch.zeros(8, dtype=torch.int32, device=device)
        self._token = 0

        self._data_ptrs = self._exchange_ptrs(self._data)
        self._partial_ptrs = self._exchange_ptrs(self._partial)
        self._flag_ptrs = self._exchange_ptrs(self._flags)

    def _exchange_ptrs(self, t: torch.Tensor) -> torch.Tensor:
        """Share `t` with all ranks via CUDA IPC; return device ptr array."""
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        handle_info = t.untyped_storage()._share_cuda_()
        obj_list = [None] * self.world_size
        dist.all_gather_object(obj_list, (self.rank, handle_info, t.dtype, t.numel()), group=self.group)
        ptrs = torch.zeros(self.world_size, dtype=torch.int64, device=self.device)
        opened = []
        for rank, info, dtype, numel in obj_list:
            if rank == self.rank:
                ptrs[rank] = t.data_ptr()
                continue
            storage = type(t.untyped_storage())._new_shared_cuda(*info)
            peer_t = torch.tensor([], dtype=dtype, device=self.device).set_(
                storage
            )
            opened.append(peer_t)
            ptrs[rank] = peer_t.data_ptr()
        if not hasattr(self, "_opened"):
            self._opened = []
        self._opened.extend(opened)
        return ptrs

    def should_use(self, inp: torch.Tensor) -> bool:
        return (
            inp.dtype == torch.bfloat16
            and inp.is_contiguous()
            and inp.numel() <= _MAX_ELEMS
        )

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor | None = None):
        if out is None:
            out = torch.empty_like(inp)
        self._token += 1
        numel = inp.numel()
        _hier_all_reduce_kernel[(1,)](
            inp.view(-1),
            out.view(-1),
            self._data_ptrs,
            self._partial_ptrs,
            self._flag_ptrs,
            rank=self.rank,
            island_base=self.island_base,
            island_size=self.island_size,
            counterpart=self.counterpart,
            numel=numel,
            token=self._token,
            BLOCK=min(8192, triton.next_power_of_2(numel)),
            num_warps=16,
        )
        return out
