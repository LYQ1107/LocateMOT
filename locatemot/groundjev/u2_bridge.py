"""B2: real GroundAnything ROI features into the unchanged original U2 core."""
import torch
from torch import nn

from locatemot.unified.models.u2.shared_track_core import SharedTrackCore


class GroundU2Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560, 256))
        self.core = SharedTrackCore(hidden_dim=256, max_age=30)

    def candidates(self, features, boxes, scores):
        visual = self.visual(features)
        return {'boxes': boxes, 'scores': scores, 'id_visual': visual,
                'semantic_visual': visual, 'decoder_query': visual}


def candidate_tensors(rows):
    """Missing native confidence uses the existing unscored=1 convention."""
    return (torch.tensor([r['native_region_feature'] for r in rows], dtype=torch.float32).reshape(-1, 2560),
            torch.tensor([r['box'] for r in rows], dtype=torch.float32).reshape(-1, 4),
            torch.tensor([r['native_score'] if r.get('native_score') is not None else 1.
                          for r in rows], dtype=torch.float32))
