"""Pinned MiVOLO model factory adapted for the repository's timm runtime.

The upstream general-purpose factory depends on private timm helpers that
were removed after the MiVOLO release. This adapter only exposes the
architecture entry point used by the pinned MiVOLO V2 checkpoint.
"""

from __future__ import annotations

from typing import Any

from timm.layers import set_layer_config
from timm.models._registry import model_entrypoint

from mivolo.model import mivolo_model as _mivolo_model


def create_model(
    model_name: str,
    pretrained: bool = False,
    scriptable: bool | None = None,
    exportable: bool | None = None,
    no_jit: bool | None = None,
    **kwargs: Any,
) -> Any:
    """Create the registered MiVOLO architecture without fetching weights."""
    if model_name != "mivolo_d1_384":
        raise ValueError(f"unsupported pinned MiVOLO architecture: {model_name}")
    create_architecture = model_entrypoint(model_name)
    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        return create_architecture(pretrained=pretrained, **kwargs)


__all__ = ["create_model"]
