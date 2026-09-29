"""One import that makes the calibration chain work on fused-QKV architectures.

`import phi4.inject` (or exec this file) right after a model is loaded, then call
`maybe_unfuse(model)`. It is a no-op for llama and qwen3 -- the guard is the presence of a fused
`qkv_proj` where the chain expects `k_proj`, not a model-name match, so a future fused-QKV model
gets the same treatment without another edit here.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def maybe_unfuse(model, *, verbose: bool = True) -> bool:
    """Unfuse qkv_proj if this architecture packs it. Returns True if anything changed."""
    try:
        layers = model.model.layers
    except AttributeError:
        return False
    if not len(layers):
        return False
    sa = layers[0].self_attn
    if hasattr(sa, "k_proj"):
        return False                      # llama / qwen3 shape: nothing to do
    if not hasattr(sa, "qkv_proj"):
        raise RuntimeError(
            f"{type(sa).__name__} has neither k_proj nor qkv_proj; the calibration chain reaches "
            "for the three projections by name and cannot proceed on this architecture."
        )
    from phi3_shim import unfuse_qkv_
    unfuse_qkv_(model, verbose=verbose)
    return True
