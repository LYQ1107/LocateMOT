"""Fresh L88 sidecar namespace over the frozen, audited L86 architecture.

L88 changes the detector-derived Z1 input only.  This thin subclass keeps the
L86/L87-A sidecar parameterization and objective byte-for-byte conceptually,
while making the checkpoint namespace and attribution explicit.  It is
initialized from a fresh seed; no L86/L87 weights are loaded.
"""
from __future__ import annotations

from locatemot.models.l86_full_rmot import L86Config
from locatemot.models.l86_full_rmot import L86FullRMOT as _L86FullRMOT


class L88FullRMOT(_L86FullRMOT):
    """L88 sidecar with the unchanged L86/L87-A architecture."""

    pass


__all__ = ["L86Config", "L88FullRMOT"]
