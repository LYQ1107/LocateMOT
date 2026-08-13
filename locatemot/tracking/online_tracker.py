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
from locatemot.models.l1d_association import L1DAssociator, compute_affinity_features
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
        "gen": torch.as_tensor([[float(f.get("gen", 0.0)) for f in feats]], dtype=torch.float32, device=device),
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
        ua=None,
        l1d=None,
        l5=None,
        l5b=None,
        uidm=None,
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
        output_all_candidates: bool = False,
        spec_idx: int = 0,
    ):
        self.variant = variant
        self.b6 = b6
        self.ua = ua
        self.l1d = l1d
        self.l5 = l5
        self.l5b = l5b
        self.uidm = uidm
        self.l1d_weights = (0.7, 0.3, 0.0)
        self.l1d_threshold = 0.3
        self.l1d_delta_scale = 0.6
        self.l1d_rel_threshold = 0.0
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
        self.output_all_candidates = output_all_candidates
        self.spec_idx = spec_idx
        self.tracks: List[TrackState] = []
        self._next_id = 0
        self.frame_count = 0
        self._motion_kalman_delta_t = 3
        self._l5b_slots = {}
        self._next_birth_state = None
        self.uidm_new_margin = 0.0
        self._uidm_birth = {}

    def reset(self):
        self.tracks = []
        self._next_id = 0
        self.frame_count = 0
        self._uidm_birth = {}

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
        cat = _cat_embed().to(self.device)
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

        if self.variant in ("T0", "C0"):
            assigns = self._associate_t0(tracks, cur_boxes)
        elif self.variant in ("T1", "C1"):
            assigns = self._associate_t1(tracks, cur_boxes)
        elif self.variant == "C2":
            assigns = self._associate_pbd(tracks, cur_feats, cur_boxes)
        elif self.variant == "C3":
            assigns = self._associate_iou_pbd(tracks, cur_feats, cur_boxes)
        elif self.variant == "L1D":
            assigns = self._associate_l1d(tracks, cur_feats, cur_boxes, frame_id)
        elif self.variant == "UA":
            assigns = self._associate_ua(tracks, cur_feats, cur_boxes, frame_id)
        elif self.variant == "L5":
            assigns = self._associate_l5(tracks, cur_feats, cur_boxes, frame_id)
        elif self.variant == "L5B":
            assigns = self._associate_l5b(tracks, cur_feats, cur_boxes, frame_id)
        elif self.variant == "UIDM":
            assigns = self._associate_uidm(tracks, cur_feats, cur_boxes,
                                           frame_id)
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
            if self.variant == "UIDM" and trk.uidm_state is not None:
                gap = max(1, frame_id - trk.last_frame)
                with torch.no_grad():
                    h = torch.as_tensor(trk.uidm_state["h"],
                                        device=self.device).float()
                    gp = torch.as_tensor([gap], device=self.device).float()
                    h = self.uidm.memory.decay(h, gp).squeeze(0)
                trk.uidm_state["h"] = h.cpu().numpy()
                trk.uidm_state["alive"] = (
                    trk.uidm_state["alive"] - 1.0)
                if trk.uidm_state["alive"] < 0.0:
                    trk.status = TERMINATED
            if trk.kalman is not None:
                trk.kalman.update(None)
            if trk.lost_age > self.max_age:
                trk.status = TERMINATED
            elif trk.lost_age >= 2 and trk.status == ACTIVE:
                trk.status = LOST
            elif trk.status == TENTATIVE and trk.lost_age > self.max_age:
                trk.status = TERMINATED

        # births: unmatched candidates -> tentative tracks (shared shell)
        born = []
        for i, cand in enumerate(candidates):
            if i in matched_det:
                continue
            if self.variant == "UIDM":
                bs = self._uidm_birth.pop(i, None)
                if bs is not None:
                    self._next_birth_state = bs
                else:
                    self._next_birth_state = None
            born.append(self._birth(frame_id, cand, cand_idx=i))
        self._next_birth_state = None
        self._uidm_birth = {}

        self.tracks = [t for t in self.tracks if t.status != TERMINATED]
        if self.output_all_candidates:
            cand_to_trk = {}
            for trk_idx, cand_idx, _score in assigns:
                cand_to_trk[cand_idx] = tracks[trk_idx]
            out = []
            born_iter = iter(born)
            for i, cand in enumerate(candidates):
                trk = cand_to_trk.get(i)
                if trk is None:
                    trk = next(born_iter)
                out.append({
                    "track_id": trk.track_id, "box": cand["box"].copy(),
                    "score": float(cand["features"].get("gen", 1.0)),
                })
            return out
        out = []
        for trk in self.tracks:
            if trk.status == ACTIVE and trk.last_frame == frame_id:
                out.append({"track_id": trk.track_id, "box": trk.last_box.copy()})
        return out

    def _birth(self, frame_id, cand, cand_idx=-1):
        feats = cand["features"]
        status = ACTIVE if self.output_all_candidates else TENTATIVE
        trk = TrackState(
            track_id=self._new_id(),
            last_box=np.asarray(cand["box"], dtype=np.float64),
            last_frame=frame_id,
            status=status,
            hits=1,
            age=1,
            birth_frame=frame_id,
            last_features=feats,
            history=[Obs(frame_id, np.asarray(cand["box"], dtype=np.float64), feats,
                         float(feats.get("gen", 0.0)))],
            confidence=float(feats.get("gen", 0.0)),
        )
        if self.variant in ("T1", "C1", "L1D", "L5"):
            trk.kalman = KalmanBoxTracker7(np.asarray(cand["box"], dtype=np.float64))
        if self.variant in ("L1D", "L5", "L5B"):
            trk.anchor_features = feats
        if self.variant == "L5B":
            trk.slot = int(self._l5b_slots.get(cand_idx, -1))
        if self.variant == "UIDM":
            trk.anchor_features = feats
            if getattr(self, "_next_birth_state", None) is not None:
                bs = self._next_birth_state
                trk.uidm_state = {
                    "h": bs["h"], "anchor": bs["anchor"],
                    "ref_pbd": bs["ref_pbd"], "anchor_pbd": bs["anchor_pbd"],
                    "alive": 1.0,
                }
            else:
                trk.uidm_state = None
        self.tracks.append(trk)
        return trk

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

    # ---------------- L1-C baselines (C2/C3) and UA ----------------
    def _feat_matrix(self, feats, key="pbd_be"):
        M = len(feats)
        arr = np.zeros((M, 2048), dtype=np.float32)
        for i, f in enumerate(feats):
            v = f.get(key)
            if v is not None:
                arr[i] = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(n, 1e-6)

    def _associate_pbd(self, tracks, cur_feats, cur_boxes):
        if not tracks or len(cur_boxes) == 0:
            return []
        refs = [trk.history[-1].features if trk.history else trk.last_features
                for trk in tracks]
        ref = self._feat_matrix(refs)
        cur = self._feat_matrix(cur_feats)
        sim = ref @ cur.T
        thresh = getattr(self, "pbd_thresh", 0.3)
        return [(r, c, float(sim[r, c])) for r, c in hungarian_max(sim, thresh)]

    def _associate_iou_pbd(self, tracks, cur_feats, cur_boxes):
        if not tracks or len(cur_boxes) == 0:
            return []
        refs = [trk.history[-1].features if trk.history else trk.last_features
                for trk in tracks]
        ref = self._feat_matrix(refs)
        cur = self._feat_matrix(cur_feats)
        sim = ref @ cur.T
        iou = iou_matrix(self._track_boxes(tracks), np.asarray(cur_boxes, dtype=np.float64))
        wi = getattr(self, "iou_w", 0.5)
        wp = getattr(self, "pbd_w", 0.5)
        cost = wi * iou + wp * sim
        thresh = getattr(self, "c3_thresh", 0.3)
        return [(r, c, float(cost[r, c])) for r, c in hungarian_max(cost, thresh)]

    def _associate_ua(self, tracks, cur_feats, cur_boxes, frame_id):
        if not tracks or len(cur_boxes) == 0:
            return []
        if self.ua is None:
            raise RuntimeError("UA variant requires ua model")
        win = build_window_tensors(tracks, self.ua.max_k, frame_id, device=self.device)
        T = len(tracks)
        N = len(cur_feats)
        K = self.ua.max_k
        cur_pbd = np.zeros((1, N, 2048), np.float32)
        cur_reg = np.zeros((1, N, 4608), np.float32)
        cur_geom = np.zeros((1, N, 5), np.float32)
        cur_norm = np.zeros((1, N, 4), np.float32)
        cur_gen = np.zeros((1, N, 1), np.float32)
        iw, ih = self.image_size
        diag = float(np.hypot(iw, ih)) + 1e-6
        for i, f in enumerate(cur_feats):
            cur_pbd[0, i] = np.asarray(f.get("pbd", np.zeros(2048, np.float32)), dtype=np.float32)
            cur_reg[0, i] = np.asarray(f.get("region", np.zeros(4608, np.float32)), dtype=np.float32)
            cur_geom[0, i] = np.asarray(f.get("geom", np.zeros(5, np.float32)), dtype=np.float32)
            bx = np.asarray(cur_boxes[i], dtype=np.float64)
            cur_norm[0, i] = [bx[0] / iw, bx[1] / ih, bx[2] / iw, bx[3] / ih]
            cur_gen[0, i, 0] = float(f.get("gen", 0.0))
        trk_last = np.zeros((1, T, 4), np.float32)
        trk_mask = (~win["mask"]).to(self.device).unsqueeze(0)  # [1,T,K] True=valid
        trk_valid = trk_mask.any(dim=-1)
        gap = torch.zeros(1, T, dtype=torch.float32, device=self.device)
        for b in range(T):
            if trk_valid[0, b]:
                last_box = tracks[b].last_box
                trk_last[0, b] = [last_box[0] / iw, last_box[1] / ih,
                                  last_box[2] / iw, last_box[3] / ih]
                last_frame = tracks[b].history[-1].frame if tracks[b].history else frame_id - 1
                gap[0, b] = max(1, frame_id - last_frame)
        ref_pbd = win["pbd"][:, -1, :].unsqueeze(0)
        ref_reg = win["region"][:, -1, :].unsqueeze(0)
        rp = torch.nn.functional.normalize(ref_pbd, dim=-1)
        cp = torch.nn.functional.normalize(torch.as_tensor(cur_pbd, device=self.device), dim=-1)
        rr = torch.nn.functional.normalize(ref_reg, dim=-1)
        cr = torch.nn.functional.normalize(torch.as_tensor(cur_reg, device=self.device), dim=-1)
        pbd_cos = torch.bmm(rp, cp.transpose(1, 2))
        region_cos = torch.bmm(rr, cr.transpose(1, 2))
        batch = {
            "cur_pbd": torch.as_tensor(cur_pbd, device=self.device),
            "cur_region": torch.as_tensor(cur_reg, device=self.device),
            "cur_geom": torch.as_tensor(cur_geom, device=self.device),
            "cur_gen": torch.as_tensor(cur_gen, device=self.device),
            "cur_norm_geom": torch.as_tensor(cur_norm, device=self.device),
            "cur_mask": torch.ones(1, N, dtype=torch.bool, device=self.device),
            "trk_pbd": win["pbd"].unsqueeze(0).to(self.device),
            "trk_region": win["region"].unsqueeze(0).to(self.device),
            "trk_geom": win["geom"].unsqueeze(0).to(self.device),
            "trk_gen": win["gen"].unsqueeze(0).to(self.device),
            "trk_times": win["gaps"].long().unsqueeze(0).to(self.device),
            "trk_mask": trk_mask,
            "trk_last_geom": torch.as_tensor(trk_last, device=self.device),
            "trk_valid": trk_valid,
            "gap": gap,
            "pbd_cos": pbd_cos,
            "region_cos": region_cos,
        }
        with torch.no_grad():
            pred = self.ua(batch)
        scores = pred["scores"][0].cpu().numpy()  # [N,T]
        new_logits = pred["new_logits"][0].cpu().numpy() - float(
            getattr(self, "new_margin", 0.0))
        return self._decode_ua(scores, new_logits)

    def _associate_l1d(self, tracks, cur_feats, cur_boxes, frame_id):
        if not tracks or len(cur_boxes) == 0:
            return []
        if self.l1d is None:
            raise RuntimeError("L1D variant requires l1d model")
        T = len(tracks)
        N = len(cur_feats)
        tb = np.zeros((T, 4), np.float64)
        pb = np.zeros((T, 4), np.float64)
        pred_boxes = np.zeros((T, 4), np.float64)
        gaps = np.zeros(T, np.float32)
        ages = np.zeros(T, np.float32)
        hits = np.zeros(T, np.float32)
        ref = np.zeros((T, 2048), np.float32)
        anchor = np.zeros((T, 2048), np.float32)
        for i, trk in enumerate(tracks):
            tb[i] = trk.last_box
            pb[i] = trk.prev_box if trk.prev_box is not None else trk.last_box
            pred_boxes[i] = trk.kalman.predict() if trk.kalman is not None else trk.last_box
            last_frame = trk.history[-1].frame if trk.history else frame_id - 1
            gaps[i] = max(1, frame_id - last_frame)
            ages[i] = trk.age
            hits[i] = trk.hits
            f = trk.history[-1].features if trk.history else trk.last_features
            if f and f.get("pbd_be") is not None:
                ref[i] = np.asarray(f["pbd_be"], dtype=np.float32)
            if trk.anchor_features and trk.anchor_features.get("pbd_be") is not None:
                anchor[i] = np.asarray(trk.anchor_features["pbd_be"], dtype=np.float32)
            else:
                anchor[i] = ref[i]
        cb = np.asarray(cur_boxes, dtype=np.float64).reshape(N, 4)
        cp = np.zeros((N, 2048), np.float32)
        cg = np.zeros(N, np.float32)
        for i, f in enumerate(cur_feats):
            if f.get("pbd_be") is not None:
                cp[i] = np.asarray(f["pbd_be"], dtype=np.float32)
            cg[i] = float(f.get("gen", 0.0))
        feats = compute_affinity_features(
            tb, cb, ref, anchor, cp, cg, gaps, ages, hits, pb,
            self.l1d_weights, self.image_size,
            motion_pred_boxes=pred_boxes)
        batch = {
            "pair_feats": torch.as_tensor(feats["pair_feats"][None], device=self.device),
            "track_feats": torch.as_tensor(feats["track_feats"][None], device=self.device),
            "cand_feats": torch.as_tensor(feats["cand_feats"][None], device=self.device),
            "base": torch.as_tensor(feats["base"][None], device=self.device),
            "trk_mask": torch.ones(1, T, dtype=torch.bool, device=self.device),
            "cand_mask": torch.ones(1, N, dtype=torch.bool, device=self.device),
        }
        if getattr(self.l1d, "use_spec", False):
            batch["spec"] = torch.full(
                (1,), int(self.spec_idx), dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred = self.l1d(batch)
            final = pred["final"][0].cpu().numpy()
        if self.l1d_rel_threshold > 0.0 or abs(self.l1d_delta_scale - 0.6) > 1e-6:
            base = feats["base"]
            rel = pred["reliability"][0].cpu().numpy()
            delta = pred["delta"][0].cpu().numpy()
            gate = rel * (rel >= self.l1d_rel_threshold)
            final = base + (self.l1d_delta_scale / 0.6) * gate[:, None] * delta
        return [(r, c, float(final[r, c]))
                for r, c in hungarian_max(final, self.l1d_threshold)]

    def _associate_l5(self, tracks, cur_feats, cur_boxes, frame_id):
        """Stage L5 Route A: temporal identity state + set-level decoder.

        Reuses the L1DK base affinity (IoU + PBD cosine + motion IoU) and
        adds a bounded residual from the L5 temporal associator.  History is
        taken from TrackState.history (may contain association errors at
        inference, exactly as trained on u0 source).
        """
        if not tracks or len(cur_boxes) == 0:
            return []
        if self.l5 is None:
            raise RuntimeError("L5 variant requires l5 model")
        model = self.l5
        K = getattr(model, "max_obs", 16)
        image_size = tuple(self.image_size)
        iw, ih = image_size
        T = len(tracks)
        N = len(cur_boxes)
        obs_pbd = np.zeros((T, K, 2048), np.float32)
        obs_feat = np.zeros((T, K, 9), np.float32)
        obs_mask = np.zeros((T, K), bool)
        tb = np.zeros((T, 4), np.float64)
        pb = np.zeros((T, 4), np.float64)
        pred_boxes = np.zeros((T, 4), np.float64)
        gaps = np.zeros(T, np.float32)
        ages = np.zeros(T, np.float32)
        hits = np.zeros(T, np.float32)
        ref = np.zeros((T, 2048), np.float32)
        anchor = np.zeros((T, 2048), np.float32)
        for i, trk in enumerate(tracks):
            tb[i] = trk.last_box
            pb[i] = trk.prev_box if trk.prev_box is not None else trk.last_box
            pred_boxes[i] = trk.kalman.predict() if trk.kalman is not None \
                else trk.last_box
            last_frame = trk.history[-1].frame if trk.history else frame_id - 1
            gaps[i] = max(1, frame_id - last_frame)
            ages[i] = trk.age
            hits[i] = trk.hits
            obs = [o for o in trk.history if o.frame < frame_id][-K:]
            start = K - len(obs)
            prev_center = None
            for j, o in enumerate(obs):
                k = start + j
                obs_mask[i, k] = True
                f = o.features
                pbd = np.asarray(f.get("pbd_be", np.zeros(2048, np.float32)),
                                 np.float32).reshape(2048)
                obs_pbd[i, k] = pbd
                b = np.asarray(o.box, np.float64)
                obs_feat[i, k, 0:4] = [b[0] / iw, b[1] / ih, b[2] / iw,
                                       b[3] / ih]
                center = np.asarray([(b[0] + b[2]) / 2.0,
                                     (b[1] + b[3]) / 2.0])
                if prev_center is not None:
                    obs_feat[i, k, 4:6] = (center - prev_center) / (iw, ih)
                prev_center = center
                obs_feat[i, k, 6] = float(o.gen_score or 0.0)
                obs_feat[i, k, 7] = float(np.log1p(N))
                obs_feat[i, k, 8] = float(frame_id - o.frame)
            if obs:
                last_obs = obs[-1].features
                ref[i] = np.asarray(last_obs.get("pbd_be",
                                                 np.zeros(2048, np.float32)),
                                    np.float32)
            if trk.anchor_features is not None:
                anchor[i] = np.asarray(trk.anchor_features.get(
                    "pbd_be", np.zeros(2048, np.float32)), np.float32)
            else:
                anchor[i] = ref[i]
        cb = np.asarray(cur_boxes, np.float64).reshape(N, 4)
        cp = np.zeros((N, 2048), np.float32)
        cg = np.zeros(N, np.float32)
        for i, f in enumerate(cur_feats):
            if f.get("pbd_be") is not None:
                cp[i] = np.asarray(f["pbd_be"], np.float32)
            cg[i] = float(f.get("gen", 0.0))
        feats = compute_affinity_features(
            tb, cb, ref, anchor, cp, cg, gaps, ages, hits, pb,
            self.l1d_weights, image_size,
            motion_pred_boxes=pred_boxes)
        batch = {
            "obs_pbd": torch.as_tensor(obs_pbd[None], device=self.device),
            "obs_feat": torch.as_tensor(obs_feat[None], device=self.device),
            "obs_mask": torch.as_tensor(obs_mask[None], device=self.device),
            "cand_pbd": torch.as_tensor(cp[None], device=self.device),
            "cand_feat": torch.as_tensor(feats["cand_feats"][None],
                                         device=self.device),
            "pair_feats": torch.as_tensor(feats["pair_feats"][None],
                                          device=self.device),
            "track_feats": torch.as_tensor(feats["track_feats"][None],
                                           device=self.device),
            "base": torch.as_tensor(feats["base"][None], device=self.device),
            "trk_mask": torch.ones(1, T, dtype=torch.bool,
                                   device=self.device),
            "cand_mask": torch.ones(1, N, dtype=torch.bool,
                                    device=self.device),
        }
        if getattr(model, "n_spec", 0) > 0:
            batch["spec"] = torch.full((1,), int(self.spec_idx),
                                       dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred = model(batch)
            final = pred["final"][0].cpu().numpy()
        return [(r, c, float(final[r, c]))
                for r, c in hungarian_max(final, self.l1d_threshold)]

    def _associate_l5b(self, tracks, cur_feats, cur_boxes, frame_id):
        """Stage L5 Route B: candidate -> sequence-local identity slot.

        Each track carries a persistent slot (assigned at birth).  A candidate
        is matched to the track with the same predicted slot, or born as NEW.
        One-to-one via Hungarian on the extended (slots + NEW) matrix.
        """
        if not tracks or len(cur_boxes) == 0:
            return []
        model = self.l5b
        if model is None:
            raise RuntimeError("L5B variant requires l5b model")
        # reuse the L5 evidence builder: call _associate_l5 internals via a
        # shared helper to keep tensor construction identical.
        batch = self._build_l5_batch(tracks, cur_feats, cur_boxes, frame_id)
        if batch is None:
            return []
        with torch.no_grad():
            pred = model(batch)
            logits = pred["slot_logits"][0].cpu().numpy()  # [N,G+1]
        N = logits.shape[0]
        G = max(8, 2 * len(tracks) + 4)
        G = min(G, logits.shape[1] - 1)
        slots = np.asarray([t.slot for t in tracks], np.int64)
        cost = np.full((N, len(tracks) + N), 1e6, dtype=np.float64)
        for j in range(N):
            for i, s in enumerate(slots):
                if 0 <= s <= G:
                    cost[j, i] = -float(logits[j, s])
            cost[j, len(tracks) + j] = -float(logits[j, G])
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        assigns = []
        born_slots = {}
        for r, c in zip(rows, cols):
            if c < len(tracks):
                assigns.append((int(c), int(r),
                                float(-cost[r, c])))
            else:
                cand_slot = int(np.argmax(logits[r, :G + 1]))
                born_slots[int(r)] = cand_slot
        self._l5b_slots = born_slots
        return assigns

    def _build_l5_batch(self, tracks, cur_feats, cur_boxes, frame_id):
        """Shared tensor construction for L5 / L5B variants."""
        model = self.l5 if self.l5 is not None else self.l5b
        if model is None:
            return None
        K = getattr(model, "max_obs", 16)
        image_size = tuple(self.image_size)
        iw, ih = image_size
        T = len(tracks)
        N = len(cur_boxes)
        obs_pbd = np.zeros((T, K, 2048), np.float32)
        obs_feat = np.zeros((T, K, 9), np.float32)
        obs_mask = np.zeros((T, K), bool)
        tb = np.zeros((T, 4), np.float64)
        pb = np.zeros((T, 4), np.float64)
        pred_boxes = np.zeros((T, 4), np.float64)
        gaps = np.zeros(T, np.float32)
        ages = np.zeros(T, np.float32)
        hits = np.zeros(T, np.float32)
        ref = np.zeros((T, 2048), np.float32)
        anchor = np.zeros((T, 2048), np.float32)
        for i, trk in enumerate(tracks):
            tb[i] = trk.last_box
            pb[i] = trk.prev_box if trk.prev_box is not None else trk.last_box
            pred_boxes[i] = trk.kalman.predict() if trk.kalman is not None \
                else trk.last_box
            last_frame = trk.history[-1].frame if trk.history else frame_id - 1
            gaps[i] = max(1, frame_id - last_frame)
            ages[i] = trk.age
            hits[i] = trk.hits
            obs = [o for o in trk.history if o.frame < frame_id][-K:]
            start = K - len(obs)
            prev_center = None
            for j, o in enumerate(obs):
                k = start + j
                obs_mask[i, k] = True
                f = o.features
                pbd = np.asarray(f.get("pbd_be", np.zeros(2048, np.float32)),
                                 np.float32).reshape(2048)
                obs_pbd[i, k] = pbd
                b = np.asarray(o.box, np.float64)
                obs_feat[i, k, 0:4] = [b[0] / iw, b[1] / ih, b[2] / iw,
                                       b[3] / ih]
                center = np.asarray([(b[0] + b[2]) / 2.0,
                                     (b[1] + b[3]) / 2.0])
                if prev_center is not None:
                    obs_feat[i, k, 4:6] = (center - prev_center) / (iw, ih)
                prev_center = center
                obs_feat[i, k, 6] = float(o.gen_score or 0.0)
                obs_feat[i, k, 7] = float(np.log1p(N))
                obs_feat[i, k, 8] = float(frame_id - o.frame)
            if obs:
                last_obs = obs[-1].features
                ref[i] = np.asarray(last_obs.get("pbd_be",
                                                  np.zeros(2048, np.float32)),
                                    np.float32)
            if trk.anchor_features is not None:
                anchor[i] = np.asarray(trk.anchor_features.get(
                    "pbd_be", np.zeros(2048, np.float32)), np.float32)
            else:
                anchor[i] = ref[i]
        cb = np.asarray(cur_boxes, np.float64).reshape(N, 4)
        cp = np.zeros((N, 2048), np.float32)
        cg = np.zeros(N, np.float32)
        for i, f in enumerate(cur_feats):
            if f.get("pbd_be") is not None:
                cp[i] = np.asarray(f["pbd_be"], np.float32)
            cg[i] = float(f.get("gen", 0.0))
        feats = compute_affinity_features(
            tb, cb, ref, anchor, cp, cg, gaps, ages, hits, pb,
            self.l1d_weights, image_size,
            motion_pred_boxes=pred_boxes)
        return {
            "obs_pbd": torch.as_tensor(obs_pbd[None], device=self.device),
            "obs_feat": torch.as_tensor(obs_feat[None], device=self.device),
            "obs_mask": torch.as_tensor(obs_mask[None], device=self.device),
            "cand_pbd": torch.as_tensor(cp[None], device=self.device),
            "cand_feat": torch.as_tensor(feats["cand_feats"][None],
                                         device=self.device),
            "pair_feats": torch.as_tensor(feats["pair_feats"][None],
                                          device=self.device),
            "track_feats": torch.as_tensor(feats["track_feats"][None],
                                           device=self.device),
            "base": torch.as_tensor(feats["base"][None], device=self.device),
            "trk_mask": torch.ones(1, T, dtype=torch.bool,
                                   device=self.device),
            "cand_mask": torch.ones(1, N, dtype=torch.bool,
                                    device=self.device),
        }

    def _associate_uidm(self, tracks, cur_feats, cur_boxes, frame_id):
        """Stage L6 UIDM: learned causal identity dynamics association.

        Uses each track's persistent model state h_i (updated by the model
        itself), set-level interaction with current candidates, and an
        extended transition matrix (existing ID / NEW / NO-MATCH).
        """
        model = self.uidm
        if model is None:
            raise RuntimeError("UIDM variant requires uidm model")
        from locatemot.models.l6_uidm import decode_lsa
        T = len(tracks)
        N = len(cur_boxes)
        image_size = tuple(self.image_size)
        iw, ih = image_size
        d = model.d_model
        app_dim = getattr(model, "app_dim", 2048)
        if T == 0 and N == 0:
            return []
        T = max(1, T)
        tb = np.zeros((T, 4), np.float64)
        pb = np.zeros((T, 4), np.float64)
        pred_boxes = np.zeros((T, 4), np.float64)
        gaps = np.zeros(T, np.float32)
        ages = np.zeros(T, np.float32)
        hits = np.zeros(T, np.float32)
        ref = np.zeros((T, app_dim), np.float32)
        anchor = np.zeros((T, app_dim), np.float32)
        h = np.zeros((T, d), np.float32)
        alive = np.zeros(T, np.float32)
        for i, trk in enumerate(tracks):
            tb[i] = trk.last_box
            pb[i] = trk.prev_box if trk.prev_box is not None else trk.last_box
            pred_boxes[i] = trk.kalman.predict() if trk.kalman is not None \
                else trk.last_box
            gaps[i] = max(1, frame_id - trk.last_frame)
            ages[i] = trk.age
            hits[i] = trk.hits
            if trk.uidm_state is not None:
                h[i] = np.asarray(trk.uidm_state["h"], np.float32)
                ref[i] = np.asarray(trk.uidm_state["ref_pbd"], np.float32)
                anchor[i] = np.asarray(trk.uidm_state["anchor_pbd"],
                                       np.float32)
                alive[i] = float(trk.uidm_state["alive"])
            f = trk.history[-1].features if trk.history else trk.last_features
            if f and f.get("pbd_be") is not None and \
                    trk.uidm_state is None:
                ref[i] = np.asarray(f["pbd_be"], np.float32)
            if trk.anchor_features is not None and \
                    trk.uidm_state is None:
                anchor[i] = np.asarray(
                    trk.anchor_features.get("pbd_be", ref[i]), np.float32)
            else:
                anchor[i] = ref[i]
        cb = np.asarray(cur_boxes, dtype=np.float64).reshape(N, 4)
        cp = np.zeros((N, app_dim), np.float32)
        cg = np.zeros(N, np.float32)
        for i, f in enumerate(cur_feats):
            if f.get("pbd_be") is not None:
                cp[i] = np.asarray(f["pbd_be"], dtype=np.float32)
            cg[i] = float(f.get("gen", 0.0))
        feats = compute_affinity_features(
            tb, cb, ref, anchor, cp, cg, gaps, ages, hits, pb,
            self.l1d_weights, image_size,
            motion_pred_boxes=pred_boxes, app_dim=app_dim)
        batch = {
            "trk_tok": torch.as_tensor(h[None], device=self.device),
            "cand_pbd": torch.as_tensor(cp[None], device=self.device),
            "cand_feat": torch.as_tensor(
                feats["cand_feats"][None], device=self.device),
            "pair_feats": torch.as_tensor(
                feats["pair_feats"][None], device=self.device),
            "track_feats": torch.as_tensor(
                feats["track_feats"][None], device=self.device),
            "cand_mask": torch.ones(1, N, dtype=torch.bool,
                                    device=self.device),
            "trk_mask": torch.zeros(1, T, dtype=torch.bool,
                                    device=self.device),
            "gap": torch.as_tensor(gaps[None], device=self.device),
        }
        batch["trk_mask"][0, :len(tracks)] = True
        with torch.no_grad():
            pred = model.forward_frame(batch)
            pair = pred["pair_logits"][0, :len(tracks), :].cpu().numpy()
            nm = pred["no_match"][0, :len(tracks)].cpu().numpy()
            nw = pred["new"][0, :N].cpu().numpy()
            cand_tok = pred["cand_tok"][0].cpu().numpy()
            trk_tok = pred["trk_tok"][0].cpu().numpy()
            alive_pre = pred["alive_pre"][0].cpu().numpy()
        self._uidm_birth = {}
        if len(tracks) == 0:
            for j in range(N):
                with torch.no_grad():
                    h_init = model.memory.init(
                        torch.as_tensor(cand_tok[j], device=self.device)
                    ).cpu().numpy()
                self._uidm_birth[j] = {
                    "h": np.asarray(h_init, np.float32),
                    "anchor": cand_tok[j].astype(np.float32),
                    "ref_pbd": cp[j].astype(np.float32),
                    "anchor_pbd": cp[j].astype(np.float32),
                    "alive": 1.0,
                }
            return []
        matches, births = decode_lsa(
            pair, nm, nw - float(getattr(self, "uidm_new_margin", 0.0)))
        # update matched track states
        for t, j, _score in matches:
            trk = tracks[t]
            if trk.uidm_state is None:
                trk.uidm_state = {
                    "h": h[t], "anchor": anchor[t],
                    "ref_pbd": ref[t], "anchor_pbd": anchor[t],
                    "alive": 0.0,
                }
            with torch.no_grad():
                h_t = torch.as_tensor(trk.uidm_state["h"],
                                      device=self.device).float()
                obs = torch.as_tensor(cand_tok[j], device=self.device).float()
                ctx = torch.as_tensor(trk_tok[t], device=self.device).float()
                new_h = model.memory.update(h_t, obs, ctx).cpu().numpy()
            trk.uidm_state["h"] = new_h.astype(np.float32)
            trk.uidm_state["ref_pbd"] = cp[j].astype(np.float32)
            trk.uidm_state["alive"] = float(alive_pre[t]) + 2.0
        # decay is handled by the shared unmatched loop; births stashed
        for j in births:
            with torch.no_grad():
                h_init = model.memory.init(
                    torch.as_tensor(cand_tok[j], device=self.device)
                ).cpu().numpy()
            self._uidm_birth[j] = {
                "h": h_init.astype(np.float32),
                "anchor": cand_tok[j].astype(np.float32),
                "ref_pbd": cp[j].astype(np.float32),
                "anchor_pbd": cp[j].astype(np.float32),
                "alive": 1.0,
            }
        return [(t, j, float(pair[t, j])) for t, j, _ in matches]

    def _decode_ua(self, scores, new_logits):
        """Candidates rows -> tracks or NEW; one-to-one; all candidates output."""
        from scipy.optimize import linear_sum_assignment
        N, T = scores.shape
        if T == 0:
            return []
        cost = -scores.astype(np.float64)
        dummies = np.full((N, N), 1e6, dtype=np.float64)
        for i in range(N):
            dummies[i, i] = -float(new_logits[i])
        cost = np.concatenate([cost, dummies], axis=1)
        rows, cols = linear_sum_assignment(cost)
        out = []
        for r, c in zip(rows, cols):
            if c < T:
                out.append((int(c), int(r), float(scores[r, c])))
        return out

    def _decode_assignments(self, match, nm):
        out = []
        for r, c in hungarian_with_no_match(match, nm):
            if c == "NO_MATCH":
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
