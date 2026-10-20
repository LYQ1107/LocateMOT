"""Single-backbone GroundJEV components.

The coupled path keeps GroundAnything's visual evidence as the only expensive
per-frame encoder.  Typed identity decisions, lifecycle control and memory
updates consume the resulting state; they do not load a second VLM.
"""

from .shared_ground_state import SharedCandidate, SharedGroundState
from .identity_memory import IdentityMemoryState, TrackMemory
from .dynamic_identity_slots import DynamicIdentitySlots, IdentitySlot, build_dynamic_slots
from .g2_jev_readout import G2JEVReadout
from .groundjev_model import GroundJEVModel
from .g2_shared_runtime import CoupledGroundJEVRuntime
from .causal_memory import CausalMemoryInput, build_causal_memory_input, causal_prefill_with_memory

__all__ = [
    "SharedCandidate", "SharedGroundState", "TrackMemory", "IdentityMemoryState",
    "IdentitySlot", "DynamicIdentitySlots", "build_dynamic_slots", "G2JEVReadout",
    "GroundJEVModel", "CoupledGroundJEVRuntime",
    "CausalMemoryInput", "build_causal_memory_input", "causal_prefill_with_memory",
]
