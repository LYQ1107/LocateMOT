"""Runtime compatibility shims kept outside the pinned official checkout."""

from __future__ import annotations

import inspect


def patch_transformers_causal_mask() -> dict[str, object]:
    """Bridge the pinned GroundAnything call to newer Transformers APIs.

    GroundAnything commit 8a3b43a calls ``create_causal_mask`` with the
    ``cache_position`` keyword.  Transformers 5.17 removed that keyword while
    retaining the same no-cache semantics used by the DLM training path.  The
    wrapper only drops that keyword; all official arguments and mask math stay
    unchanged.  The receipt records whether this compatibility path was used.
    """
    import transformers.masking_utils as masking_utils

    function = masking_utils.create_causal_mask
    if "cache_position" in inspect.signature(function).parameters:
        return {"patched": False, "reason": "runtime accepts cache_position"}
    if getattr(function, "_locatemot_g2_cache_position_compat", False):
        return {"patched": True, "reason": "already patched"}

    def compatible(*args, cache_position=None, **kwargs):
        del cache_position
        return function(*args, **kwargs)

    compatible._locatemot_g2_cache_position_compat = True
    masking_utils.create_causal_mask = compatible
    return {
        "patched": True,
        "reason": "drop cache_position for Transformers runtime without that keyword; no-cache DLM path",
        "original_signature": str(inspect.signature(function)),
    }
