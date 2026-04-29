# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.model_executor.kernels.linear import (  # noqa: E501
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


class XPUFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFP8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in {kFp8StaticChannelSym, kFp8StaticTensorSym}:
            return (
                False,
                "XPUFP8ScaledMM only support per-channel and per-tensor quantization",
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUFP8ScaledMM only support FP8 weight dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        replace_parameter(layer, "weight", layer.weight.data.t())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        pass


class XPUFP8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """XPU FP8 block-scaled GEMM kernel.

    XPU has no native block-scaled FP8 GEMM op, so this kernel dequantizes
    block-scaled FP8 weights to BF16 at load time and uses standard
    torch.nn.functional.linear for inference.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFp8BlockScaledMM only supports XPU"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Apply base processing: padding and optional fnuz conversion.
        super().process_weights_after_loading(layer)

        params = self._get_layer_params(layer)
        weight = params.weight  # FP8, shape [N, K]
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        scale_attr_name = (
            params.WEIGHT_SCALE
            if params.weight_scale_inv is None
            else params.WEIGHT_SCALE_INV
        )

        # Dequantize block-scaled FP8 to BF16.
        # weight_group_shape is GroupShape(block_n, block_k), e.g. (128, 128).
        N, K = weight.shape
        block_n = self.weight_group_shape.row
        block_k = self.weight_group_shape.col

        s = weight_scale.to(torch.float32)  # [N // block_n, K // block_k]
        s_exp = torch.repeat_interleave(s, block_n, dim=0)
        s_exp = torch.repeat_interleave(s_exp, block_k, dim=1)
        s_exp = s_exp[:N, :K]  # crop to actual weight dims (after padding)

        dequant_weight = weight.to(torch.float32) * s_exp
        replace_parameter(layer, params.WEIGHT, dequant_weight.to(torch.bfloat16))
        # Scale no longer needed after dequantization.
        replace_parameter(
            layer, scale_attr_name, torch.empty(0, device=weight.device)
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight  # BF16 [N, K]
        output_shape = [*x.shape[:-1], weight.shape[0]]
        x_2d = x.view(-1, x.shape[-1])
        out = torch.nn.functional.linear(x_2d.to(weight.dtype), weight, bias)
        return out.to(self.config.out_dtype).view(*output_shape)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # Not reached — apply_weights is fully overridden.
        raise NotImplementedError
