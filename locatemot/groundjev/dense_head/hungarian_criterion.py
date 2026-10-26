"""Class-agnostic Hungarian set prediction with focal objectness and GIoU."""
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def cxcywh_to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), -1)


def xyxy_to_cxcywh(boxes):
    return torch.cat(((boxes[..., :2] + boxes[..., 2:]) / 2,
                      boxes[..., 2:] - boxes[..., :2]), -1)


def box_iou(a, b):
    intersection = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    union = area_a[:, None] + area_b[None] - intersection
    return intersection / union.clamp_min(1e-8), union


def generalized_iou(a, b):
    iou, union = box_iou(a, b)
    cover = (torch.maximum(a[:, None, 2:], b[None, :, 2:]) -
             torch.minimum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    return iou - (cover - union) / cover.clamp_min(1e-8)


class HungarianCriterion(nn.Module):
    def forward(self, prediction, boxes, ignored=None, exhaustive=True):
        # One image at a time supports different native spatial grid sizes.
        logits, predicted = prediction["logits"][0].float(), prediction["boxes"][0].float()
        n = len(boxes)
        labels = torch.zeros_like(logits)
        pred_xyxy = cxcywh_to_xyxy(predicted)
        active = torch.full_like(labels, bool(exhaustive), dtype=torch.bool)
        if ignored is not None and len(ignored):
            # Unmatched objects inside ignore regions are not reliable negatives.
            lo = torch.maximum(pred_xyxy[:, None, :2], ignored[None, :, :2])
            hi = torch.minimum(pred_xyxy[:, None, 2:], ignored[None, :, 2:])
            inter = (hi - lo).clamp_min(0).prod(-1)
            area = (pred_xyxy[:, 2:] - pred_xyxy[:, :2]).clamp_min(0).prod(-1)
            active &= (inter / area[:, None].clamp_min(1e-8)).max(1).values < .5
        if n:
            target = xyxy_to_cxcywh(boxes)
            with torch.no_grad():
                cost = -logits.sigmoid()[:, None] + 5 * torch.cdist(predicted, target, p=1)
                cost -= 2 * generalized_iou(pred_xyxy, boxes)
                a, b = linear_sum_assignment(cost.cpu().numpy())
                a = torch.as_tensor(a, device=logits.device)
                b = torch.as_tensor(b, device=logits.device)
            labels[a] = 1
            active[a] = True
            l1 = F.l1_loss(predicted[a], target[b], reduction="sum") / n
            giou = (1 - generalized_iou(pred_xyxy[a], boxes[b]).diag()).sum() / n
        else:
            l1 = giou = predicted.sum() * 0
        probability = logits.sigmoid()
        pt = probability * labels + (1 - probability) * (1 - labels)
        alpha = .25 * labels + .75 * (1 - labels)
        objectness = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") *
                      (1 - pt).square() * alpha * active).sum() / max(n, 1)
        total = objectness + 5 * l1 + 2 * giou
        for auxiliary in prediction.get("auxiliary", []):
            total = total + .5 * self.forward(auxiliary, boxes, ignored, exhaustive)["loss"]
        return {"loss": total, "objectness": objectness, "L1": l1, "GIoU": giou}
