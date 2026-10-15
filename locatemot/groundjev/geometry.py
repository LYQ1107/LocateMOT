"""Batched FP64 geometry with the scalar tracker's operation order."""


def overlap_matrix(candidate_boxes, predicted_boxes):
    import numpy as np
    current = np.asarray(candidate_boxes, dtype=np.float64).reshape(-1, 4)
    history = np.asarray(predicted_boxes, dtype=np.float64).reshape(-1, 4)
    width = np.maximum(0., np.minimum(current[:, None, 2], history[None, :, 2]) -
                       np.maximum(current[:, None, 0], history[None, :, 0]))
    height = np.maximum(0., np.minimum(current[:, None, 3], history[None, :, 3]) -
                        np.maximum(current[:, None, 1], history[None, :, 1]))
    current_area = np.maximum(0., current[:, 2]-current[:, 0])*np.maximum(0., current[:, 3]-current[:, 1])
    history_area = np.maximum(0., history[:, 2]-history[:, 0])*np.maximum(0., history[:, 3]-history[:, 1])
    intersection = width*height
    return intersection/np.maximum(current_area[:, None]+history_area[None, :]-intersection, 1e-8)


def rank_tracks(group, tracks, predicted_boxes, limit):
    if not tracks:
        return []
    if group:
        scores = overlap_matrix([c['box'] for c in group], predicted_boxes).max(axis=0)
    else:
        scores = [0.]*len(tracks)
    indices = sorted(range(len(tracks)), key=lambda i: (-float(scores[i]), tracks[i].lost_age, tracks[i].track_id))
    return [tracks[i] for i in indices[:limit]]
