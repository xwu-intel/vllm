# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
XPU DeepSymm Sequence-Parallel PrepareFinalize for MoE.

When MoeSPFusionPass is active, the MoE runner receives a local TP chunk.
This PrepareFinalize uses DeepSymm SymmBuffer fused kernels:
  - prepare(): fused all_gather + remap (allgather_local_permute_fusion)
  - finalize(): fused unpermute + reduce_scatter (unpermute_reducescatter_fusion)
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_tp_group
from vllm.distributed.device_communicators.all2all import (
    DeepSymmAll2AllManager,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


class XPUDeepSymmPrepareFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Sequence-parallel PrepareFinalize backed by DeepSymm SymmBuffer.

    prepare(): fused all_gather + permute to expert-grouped layout via
    SymmBuffer.allgather_local_permute_fusion. Returns pre-permuted tensor
    with rows_per_expert metadata for grouped GEMM.

    finalize(): fused unpermute + reduce_scatter via
    SymmBuffer.unpermute_reducescatter_fusion. Produces local-chunk output.
    """

    all2all_manager: DeepSymmAll2AllManager

    def __init__(self, all2all_manager: DeepSymmAll2AllManager):
        super().__init__()
        self.all2all_manager = all2all_manager

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return 1

    def output_is_reduced(self) -> bool:
        return True

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        num_rows = a1.size(0)
        hidden_size = a1.size(1)
        n_experts_per_token = topk_ids.size(1)
        tp_size = get_tp_group().world_size
        total_rows = num_rows * tp_size

        sbuf = self.all2all_manager.get_sbuf(hidden_size, n_experts_per_token)

        remapped_hidden_states = torch.empty(
            (total_rows * n_experts_per_token, hidden_size),
            dtype=a1.dtype,
            device=a1.device,
        )

        _, symm_handle = sbuf.allgather_local_permute_fusion(
            hidden_shard=a1,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_experts=num_experts,
            remap_hidden_states=remapped_hidden_states,
        )
        self.all2all_manager.symm_handle = symm_handle

        rows_per_expert = symm_handle.rows_per_expert
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=rows_per_expert,
            expert_num_tokens_cpu=None,
        )

        # topk_ids/topk_weights are no longer needed for permutation
        # (already applied by allgather_local_permute_fusion), but the
        # modular kernel interface requires them for workspace allocation.
        # Return gathered versions for shape consistency.
        gathered_topk_ids = torch.empty(
            (total_rows, n_experts_per_token),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        gathered_topk_weights = torch.empty(
            (total_rows, n_experts_per_token),
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )

        return (
            remapped_hidden_states,
            None,
            expert_tokens_meta,
            gathered_topk_ids,
            gathered_topk_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        symm_handle = self.all2all_manager.symm_handle
        assert symm_handle is not None
        sbuf = self.all2all_manager.get_sbuf(
            output.size(-1), topk_ids.size(1)
        )

        sbuf.unpermute_reducescatter_fusion(
            expert_output=fused_expert_output,
            handle=symm_handle,
            output=output,
        )
