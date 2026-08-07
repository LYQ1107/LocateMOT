"""Full-video online tracker for T0-T6 with a shared birth/lifecycle shell.

All variants consume the exact same candidate set per frame. Birth and
lifecycle (tentative -> min_hits -> ACTIVE; lost_age -> max_age -> TERMINATED)
are shared evaluation infrastructure, not a proposed component.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from locatemot.models.track_decoder.features import category_hash_embedding
from locatemot.tracking.association import (
    center_dist_matrix,
    hungarian_max,
    hungarian_with_no_match,
    iou_matrix,
)
from locatemot.tracking.memory import init_memory, update_ema
from locatemot.tracking.motion import KalmanBoxTracker7
from locatemot.tracking.track_state import ACTIVE, LOST, TENTATIVE, TERMINATED, Obs, TrackState
from locatemot.tracking.trajectory_buffer import build_window_tensors, normalize_geom

CAT_EMBED = None


def _cat_embed():
    global CAT_EMBED
    if CAT_EMBED is None:
        CAT_EMBED = category_hash_embedding("person", 32)
    return CAT_EMBED


def _feat_dict(frame_cache_features: dict, idx: int, box: np.ndarray, image_size) -> Dict[str, np.ndarray]:
    f = frame_cache_features
    geom = normalize_geom(box, image_size)
    return {
        "pbd": np.asarray(f["pbd_coord_mean_last"][idx], dtype=np.float32),
        "pbd_be": np.asarray(f["pbd_box_end_last"][idx], dtype=np.float32),
        "region": np.asarray(f["region"][idx], dtype=np.float32),
        "geom": geom,
        "gen": float(f["gen_score"][idx]) if "gen_score" in f else 0.0,
    }


def _raw_feature_arrays(feats: List[Dict[str, np.ndarray]], device):
    def stack(key, shape, dtype=torch.float32):
        arr = np.zeros((len(feats),) + shape, dtype=np.float32)
        for i, f in enumerate(feats):
            if f.get(key) is not None:
                arr[i] = np.asarray(f[key], dtype=np.float32)
        return torch.as_tensor(arr, dtype=dtype, device=device)

    return {
        "pbd": stack("pbd", (2048,)),
        "pbd_be": stack("pbd_be", (2048,)),
        "region": stack("region", (4608,)),
        "geom": stack("geom", (5,)),
        "gen": torch.as_tensor([[f.get("gen", 0.0) for f in feats]], dtype=torch.float32, device=device),
    }


class OnlineTracker:
    """Online full-video tracker.

    variant in {T0, T1, T2, T3, T4, T5, T6}. b6 is the frozen RelationTrackDecoder
    model; temporal modules are passed for T3+.
    """

    def __init__(
        self,
        variant: str,
        b6=None,
        trajectory_encoder=None,
        motion_predictor=None,
        memory_fusion=None,
        motion_residual_head=None,
        reactivation_head=None,
        device: str = "cpu",
        iou_threshold: float = 0.3,
        no_match_theta: float = -3.5,
        no_match_bias: float = 0.0,
        max_age: int = 30,
        min_hits: int = 3,
        memory_conf_threshold: float = 0.5,
        image_size=(1280, 720),
    ):
        self.variant = variant
        self.b6 = b6
        self.traj_enc = trajectory_encoder
        self.motion_pred = motion_predictor
        self.memory_fusion = memory_fusion
        self.motion_head = motion_residual_head
        self.react_head = reactivation_head
        self.device = torch.device(device)
        self.iou_threshold = iou_threshold
        self.no_match_theta = no_match_theta
        self.no_match_bias = no_match_bias
        self.max_age = max_age
        self.min_hits = min_hits
        self.memory_conf_threshold = memory_conf_threshold
        self.image_size = image_size
        self.tracks: List[TrackState] = []
        self._next_id = 0
        self.frame_count = 0
        self._motion_kalman_delta_t = 3

    def reset(self):
        self.tracks = []
        self._next_id = 0
        self.frame_count = 0

    # ------------------------------------------------------------------ utils
    def _new_id(self):
        self._next_id += 1
        return self._next_id

    def _active_tracks(self):
        return [t for t in self.tracks if t.is_active]

    def _track_boxes(self, tracks):
        return np.asarray([t.last_box for t in tracks], dtype=np.float64).reshape(-1, 4)

    def _candidate_boxes(self, candidates):
        return np.asarray([c["box"] for c in candidates], dtype=np.float64).reshape(-1, 4)

    def _b6_batch(self, ref_feats, cur_feats, ref_boxes, cur_boxes, gap):
        cat = _cat_embed()
        M = len(ref_feats)
        N = len(cur_feats)
        R = _raw_feature_arrays(ref_feats, self.device)
        C = _raw_feature_arrays(cur_feats, self.device)
        return {
            "ref_pbd": R["pbd"].unsqueeze(0),
            "ref_pbd_be": R["pbd_be"].unsqueeze(0),
            "ref_region": R["region"].unsqueeze(0),
            "ref_geom": R["geom"].unsqueeze(0),
            "ref_gen": R["gen"],
            "ref_cat": cat.unsqueeze(0).unsqueeze(0).expand(1, M, 32).clone(),
            "ref_mask": torch.ones(1, M, dtype=torch.bool, device=self.device),
            "ref_boxes": torch.as_tensor(np.asarray(ref_boxes, dtype=np.float32).reshape(1, M, 4), device=self.device),
            "cur_pbd": C["pbd"].unsqueeze(0),
            "cur_pbd_be": C["pbd_be"].unsqueeze(0),
            "cur_region": C["region"].unsqueeze(0),
            "cur_geom": C["geom"].unsqueeze(0),
            "cur_gen": C["gen"],
            "cur_cat": cat.unsqueeze(0).unsqueeze(0).expand(1, N, 32).clone(),
            "cur_mask": torch.ones(1, N, dtype=torch.bool, device=self.device),
            "cur_boxes": torch.as_tensor(np.asarray(cur_boxes, dtype=np.float32).reshape(1, N, 4), device=self.device),
            "gap": torch.as_tensor([[float(gap)]], dtype=torch.float32, device=self.device),
        }

    def _trajectory_refs(self, tracks, cur_frame):
        """T3/T4/T5/T6: fused raw-space reference features per track."""
        if self.traj_enc is None:
            return [t.history[-1].features if t.history else t.last_features for t in tracks]
        win = build_window_tensors(tracks, self.traj_enc.max_k, cur_frame, device=self.device)
        with torch.no_grad():
            out = self.traj_enc(
                win["pbd"], win["region"], win["geom"], win["gen"],
                win["gaps"], win["mask"],
            )
        refs = []
        for b in range(len(tracks)):
            refs.append({
                "pbd": out["pbd"][b].cpu().numpy(),
                "pbd_be": _last_pbd_be(tracks[b]),
                "region": out["region"][b].cpu().numpy(),
                "geom": out["geom"][b].cpu().numpy(),
                "gen": float(out["gen"][b]),
            })
        return refs

    def _memory_refs(self, tracks, traj_refs, cur_frame):
        """T5/T6: MemoryFusion(anchor, ema, trajectory) in raw space."""
        refs = []
        for trk, tj in zip(tracks, traj_refs):
            if trk.anchor_features is None or self.memory_fusion is None:
                refs.append(tj)
                continue
            with torch.no_grad():
                anchor = _raw_feature_arrays([trk.anchor_features], self.device)
                ema = _raw_feature_arrays([trk.ema_features or trk.anchor_features], self.device)
                traj = _raw_feature_arrays([tj], self.device)
                geom = torch.as_tensor(trk.last_box, dtype=torch.float32, device=self.device).reshape(1, 4)
                geom_n = normalize_geom(trk.last_box, self.image_size)
                geom_t = torch.as_tensor(geom_n, dtype=torch.float32, device=self.device).reshape(1, 5)
                conf = torch.as_tensor([trk.confidence], dtype=torch.float32, device=self.device)
                out = self.memory_fusion(
                    anchor["pbd"], ema["pbd"], traj["pbd"],
                    anchor["region"], ema["region"], traj["region"],
                    geom_t, traj["gen"], conf,
                )
            refs.append({
                "pbd": out["pbd"][0].cpu().numpy(),
                "pbd_be": tj["pbd_be"],
                "region": out["region"][0].cpu().numpy(),
                "geom": tj["geom"],
                "gen": tj["gen"],
            })
        return refs

    def _motion_predictions(self, tracks, cur_frame):
        """Returns [B,4] predicted boxes for tracks with >=1 obs."""
        if self.motion_pred is None:
            return None
        L = self.motion_pred.window
        boxes = []
        gaps = []
        for trk in tracks:
            obs = [o for o in trk.history if o.frame < cur_frame][-L:]
            if not obs:
                boxes.append(np.zeros(4, dtype=np.float32))
                gaps.append([1.0] * L)
                continue
            bx = np.stack([o.box for o in obs] + [trk.last_box] * (L - len(obs)))[-L:]
            gap_arr = np.array([float(cur_frame - o.frame) for o in obs] + [1.0] * (L - len(obs)))[-L:]
            boxes.append(bx)
            gaps.append(gap_arr)
        b = torch.as_tensor(np.stack(boxes), dtype=torch.float32, device=self.device)
        g = torch.as_tensor(np.stack(gaps), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            delta = self.motion_pred(b, g)
            last = b[:, -1]
            lw = (last[:, 2] - last[:, 0]).clamp(min=1e-3)
            lh = (last[:, 3] - last[:, 1]).clamp(min=1e-3)
            lcx = (last[:, 0] + last[:, 2]) / 2
            lcy = (last[:, 1] + last[:, 3]) / 2
            gap = g[:, -1].clamp(min=1.0)
            cx = lcx + delta[:, 0] * lw * gap
            cy = lcy + delta[:, 1] * lh * gap
            w = lw * torch.exp((delta[:, 2] * gap).clamp(-3, 3))
            h = lh * torch.exp((delta[:, 3] * gap).clamp(-3, 3))
            pred = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        return pred.cpu().numpy()

    # ------------------------------------------------------------------ frame
    def process_frame(self, frame_id: int, candidates: List[dict]) -> List[dict]:
        """candidates: [{'box': np.ndarray(4), 'features': dict, ...}].

        Returns [{track_id, box}] for confirmed tracks output this frame.
        """
        self.frame_count += 1
        cur_boxes = self._candidate_boxes(candidates)
        cur_feats = [c["features"] for c in candidates]
        tracks = self._active_tracks()
        n = len(cur_boxes)

        if self.variant == "T0":
            assigns = self._associate_t0(tracks, cur_boxes)
        elif self.variant == "T1":
            assigns = self._associate_t1(tracks, cur_boxes)
        else:
            assigns = self._associate_learned(tracks, cur_feats, cur_boxes, frame_id)

        matched_det = set()
        matched_trk = set()
        for trk_idx, cand_idx, score in assigns:
            trk = tracks[trk_idx]
            cand = candidates[cand_idx]
            self._update_track(trk, frame_id, cand, match_score=score)
            matched_det.add(cand_idx)
            matched_trk.add(trk_idx)

        for i, trk in enumerate(tracks):
            if i in matched_trk:
                continue
            trk.lost_age += 1
            trk.age += 1
            if trk.kalman is not None:
                trk.kalman.update(None)
            if trk.lost_age > self.max_age:
                trk.status = TERMINATED
            elif trk.lost_age >= 2 and trk.status == ACTIVE:
                trk.status = LOST
            elif trk.status == TENTATIVE and trk.lost_age > self.max_age:
                trk.status = TERMINATED

        # births: unmatched candidates -> tentative tracks (shared shell)
        for i, cand in enumerate(candidates):
            if i in matched_det:
                continue
            self._birth(frame_id, cand)

        self.tracks = [t for t in self.tracks if t.status != TERMINATED]
        out = []
        for trk in self.tracks:
            if trk.status == ACTIVE and trk.last_frame == frame_id:
                out.append({"track_id": trk.track_id, "box": trk.last_box.copy()})
        return out

    def _birth(self, frame_id, cand):
        feats = cand["features"]
        trk = TrackState(
            track_id=self._new_id(),
            last_box=np.asarray(cand["box"], dtype=np.float64),
            last_frame=frame_id,
            status=TENTATIVE,
            hits=1,
            age=1,
            birth_frame=frame_id,
            last_features=feats,
            history=[Obs(frame_id, np.asarray(cand["box"], dtype=np.float64), feats,
                         float(feats.get("gen", 0.0)))],
            confidence=float(feats.get("gen", 0.0)),
        )
        if self.variant == "T1":
            trk.kalman = KalmanBoxTracker7(np.asarray(cand["box"], dtype=np.float64))
        self.tracks.append(trk)

    def _update_track(self, trk, frame_id, cand, match_score=0.0):
        new_box = np.asarray(cand["box"], dtype=np.float64)
        new_feats = cand["features"]
        if trk.last_frame == frame_id:
            return
        if trk.history:
            trk.prev_box = trk.last_box
        trk.last_box = new_box
        trk.last_frame = frame_id
        trk.lost_age = 0
        trk.age += 1
        trk.hits += 1
        trk.last_features = new_feats
        trk.confidence = float(new_feats.get("gen", 0.0))
        trk.history.append(Obs(frame_id, new_box, new_feats, float(new_feats.get("gen", 0.0))))
        if trk.kalman is not None:
            trk.kalman.update(np.concatenate([new_box, [1.0]]))
        # memory writes only on confirmed matches (shared T5/T6 rule)
        if (self.variant in ("T5", "T6") and trk.hits >= self.min_hits
                and match_score >= self.memory_conf_threshold):
            if trk.anchor_features is None:
                trk.anchor_features, trk.ema_features = init_memory(new_feats, trk.confidence)
            else:
                trk.ema_features = update_ema(trk.ema_features or trk.anchor_features, new_feats, alpha=0.5)
        if trk.status == TENTATIVE and trk.hits >= self.min_hits:
            trk.status = ACTIVE

    # ------------------------------------------------------------ association
    def _associate_t0(self, tracks, cur_boxes):
        if not tracks or len(cur_boxes) == 0:
            return []
        iou = iou_matrix(self._track_boxes(tracks), cur_boxes)
        return [(r, c, float(iou[r, c])) for r, c in hungarian_max(iou, self.iou_threshold)]

    def _associate_t1(self, tracks, cur_boxes):
        if not tracks or len(cur_boxes) == 0:
            return []
        preds = np.stack([
            trk.kalman.predict() if trk.kalman is not None else trk.last_box
            for trk in tracks
        ])
        iou = iou_matrix(preds, cur_boxes)
        first = hungarian_max(iou, self.iou_threshold)
        used_d = {c for _, c in first}
        used_t = {r for r, _ in first}
        left_t = [i for i in range(len(tracks)) if i not in used_t]
        left_d = [j for j in range(len(cur_boxes)) if j not in used_d]
        out = [(r, c, float(iou[r, c])) for r, c in first]
        if left_t and left_d:
            last_boxes = np.stack([tracks[i].last_box for i in left_t])
            left_boxes = cur_boxes[left_d]
            iou2 = iou_matrix(last_boxes, left_boxes)
            second = hungarian_max(iou2, self.iou_threshold)
            for ri, ci in second:
                out.append((left_t[ri], left_d[ci], float(iou2[ri, ci])))
        return out

    def _associate_learned(self, tracks, cur_feats, cur_boxes, frame_id):
        if not tracks or len(cur_boxes) == 0:
            return []
        ref_feats = self._trajectory_refs(tracks, frame_id) if self.variant in ("T3", "T4", "T5", "T6") else [
            trk.history[-1].features if trk.history else trk.last_features for trk in tracks
        ]
        if self.variant in ("T5", "T6"):
            ref_feats = self._memory_refs(tracks, ref_feats, frame_id)
        ref_boxes = self._track_boxes(tracks)
        # frozen B6 gap is per-sample; group tracks by quantized gap so each
        # track uses its true history gap while keeping within-group competition
        gaps = [max(1, frame_id - (trk.history[-1].frame if trk.history else frame_id - 1))
                for trk in tracks]
        quant = _quantize_gap(gaps)
        match_rows = {}
        nm_rows = {}
        ref_out_rows = {}
        cur_out_rows = {}
        with torch.no_grad():
            for g in sorted(set(quant)):
                idxs = [i for i in range(len(tracks)) if quant[i] == g]
                batch = self._b6_batch(
                    [ref_feats[i] for i in idxs],
                    cur_feats,
                    [ref_boxes[i] for i in idxs],
                    cur_boxes,
                    g,
                )
                pred = self.b6(batch)
                match_rows[g] = pred["match_logits"][0].cpu().numpy()
                nm_rows[g] = pred["no_match_logits"][0].cpu().numpy()
                ref_out_rows[g] = pred["ref_feats"][0].cpu().numpy()
                cur_out_rows[g] = pred["cur_feats"][0].cpu().numpy()
        match = np.vstack([match_rows[g] for g in sorted(set(quant))])
        nm = np.concatenate([nm_rows[g] for g in sorted(set(quant))])
        ref_out = np.vstack([ref_out_rows[g] for g in sorted(set(quant))])
        cur_out = cur_out_rows[sorted(set(quant))[0]]  # candidates identical across groups
        if self.variant in ("T4", "T5", "T6") and self.motion_head is not None:
            match = self._apply_motion(match, tracks, cur_boxes, frame_id)
        if self.variant == "T6" and self.react_head is not None:
            match = self._apply_reactivation(
                match, tracks, cur_feats, cur_boxes, frame_id, ref_out, cur_out, ref_feats)
        nm = nm - self.no_match_theta - self.no_match_bias
        return self._decode_assignments(match, nm)

    def _decode_assignments(self, match, nm):
        out = []
        for r, c in hungarian_with_no_match(match, nm):
            if isinstance(c, str):
                continue
            out.append((r, int(c.split(":")[1]), float(match[r, int(c.split(":")[1])])))
        return out

    def _apply_motion(self, match, tracks, cur_boxes, frame_id):
        pred_boxes = self._motion_predictions(tracks, frame_id)
        cur_boxes_a = np.asarray(cur_boxes, dtype=np.float64)
        last_boxes = self._track_boxes(tracks)
        iou_last = iou_matrix(last_boxes, cur_boxes_a)
        iou_pred = iou_matrix(pred_boxes, cur_boxes_a)
        cd_last = center_dist_matrix(last_boxes, cur_boxes_a)
        cd_pred = center_dist_matrix(pred_boxes, cur_boxes_a)
        diag = float(np.hypot(self.image_size[0], self.image_size[1]) + 1e-6)
        M, N = match.shape
        gaps = np.asarray([
            max(1, frame_id - (tracks[i].history[-1].frame if tracks[i].history else frame_id - 1))
            for i in range(M)
        ], dtype=np.float32)
        gap_mat = np.tile(np.log1p(gaps)[:, None], (1, N))
        x = np.stack([
            match,
            iou_last,
            iou_pred,
            cd_pred / diag,
            np.abs(cd_pred - cd_last) / diag,
            gap_mat,
        ], axis=-1).reshape(M * N, 6).astype(np.float32)
        with torch.no_grad():
            resid = self.motion_head(torch.as_tensor(x, device=self.device)).cpu().numpy().reshape(M, N)
        match += resid
        return match

    def _apply_reactivation(
        self, match, tracks, cur_feats, cur_boxes, frame_id, ref_out, cur_out, ref_feats
    ):
        pred_boxes = self._motion_predictions(tracks, frame_id)
        cur_boxes_a = np.asarray(cur_boxes, dtype=np.float64)
        cd_pred = center_dist_matrix(pred_boxes, cur_boxes_a)
        diag = float(np.hypot(self.image_size[0], self.image_size[1]) + 1e-6)
        rt = ref_out / (np.linalg.norm(ref_out, axis=1, keepdims=True) + 1e-6)
        ct = cur_out / (np.linalg.norm(cur_out, axis=1, keepdims=True) + 1e-6)
        traj_cos = rt @ ct.T
        ref_be = np.stack([f.get("pbd_be", np.zeros(2048, dtype=np.float32)) for f in ref_feats])
        cur_be = np.stack([f.get("pbd_be", np.zeros(2048, dtype=np.float32)) for f in cur_feats])
        rb = ref_be / (np.linalg.norm(ref_be, axis=1, keepdims=True) + 1e-6)
        cb = cur_be / (np.linalg.norm(cur_be, axis=1, keepdims=True) + 1e-6)
        pbd_cos = rb @ cb.T
        M, N = match.shape
        lost = np.asarray([t.lost_age for t in tracks], dtype=np.float32)
        lost_mask = (lost >= 2)[:, None].astype(np.float32)
        gap_mat = np.tile(np.log1p(np.maximum(lost, 1))[:, None], (1, N))
        iou_pred = iou_matrix(pred_boxes, cur_boxes_a)
        x = np.stack([
            match,
            traj_cos,
            pbd_cos,
            iou_pred,
            cd_pred / diag,
            gap_mat,
        ], axis=-1).reshape(M * N, 6).astype(np.float32)
        with torch.no_grad():
            resid = self.react_head(torch.as_tensor(x, device=self.device)).cpu().numpy().reshape(M, N)
        match += resid * lost_mask
        return match


def _last_pbd_be(track):
    if track.history:
        f = track.history[-1].features
        if f.get("pbd_be") is not None:
            return f["pbd_be"]
    if track.last_features and track.last_features.get("pbd_be") is not None:
        return track.last_features["pbd_be"]
    return np.zeros(2048, dtype=np.float32)


def _quantize_gap(gaps):
    """Map each gap to a representative value used as B6's per-sample gap."""
    reps = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    out = []
    for g in gaps:
        out.append(min(reps, key=lambda r: abs(r - g)))
    return out
