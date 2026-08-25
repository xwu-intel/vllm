# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import moe_runner


def test_all_gather_padding_mask_is_temporary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_is_padding = torch.tensor([False, True])
    gathered_is_padding = torch.tensor([False, True, True, True])
    forward_context = SimpleNamespace(is_padding=local_is_padding)
    tp_group = SimpleNamespace(all_gather=lambda tensor, dim: gathered_is_padding)

    monkeypatch.setattr(moe_runner, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(moe_runner, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr("vllm.distributed.get_tp_group", lambda: tp_group)

    with moe_runner._all_gather_padding_mask(enabled=True):
        assert forward_context.is_padding is gathered_is_padding

    assert forward_context.is_padding is local_is_padding