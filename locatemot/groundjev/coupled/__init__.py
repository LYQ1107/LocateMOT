"""Single-backbone GroundJEV coupling primitives.

The coupled path keeps GroundAnything as the only visual backbone.  It exposes
the shared visual state explicitly so the typed identity readout, lifecycle
controller, and causal memory writer can be audited independently of the
GroundAnything training service.
"""

from .shared_ground_state import SharedGroundState
from .dynamic_identity_slots import IdentitySlot, build_dynamic_slots
from .typed_decisions import TypedDecision, TypedDecisionBatch
from .identity_memory import IdentityMemory, IdentityMemoryStore
from .memory_tokens import IdentityMemoryTokenProjector
from .groundjev_model import GroundJEVCoupledModel
from .g2_feature_bridge import G2FeatureBridge

__all__ = [
    "SharedGroundState",
    "IdentitySlot",
    "build_dynamic_slots",
    "TypedDecision",
    "TypedDecisionBatch",
    "IdentityMemory",
    "IdentityMemoryStore",
    "IdentityMemoryTokenProjector",
    "GroundJEVCoupledModel",
    "G2FeatureBridge",
]
