"""In-memory two-frame dataset adapter for official TrackEval.

Subclasses MotChallenge2DBox so that preprocessing, IoU similarity, metric
computation and COMBINED_SEQ aggregation remain 100% official TrackEval
(commit 12c8791b). Only the raw data source is replaced by in-memory arrays.
"""
from __future__ import annotations

import os

import numpy as np

from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
from trackeval.utils import TrackEvalException


class TwoFrameLocateMOT(MotChallenge2DBox):
    """Evaluate a set of two-frame pair sequences built by tools/l0d_trackeval.py.

    DATA_FILE: npz with keys:
      seq_ids (S,) int
      gt_boxes0 (S,M0,4) xyxy, gt_ids0 (S,M0)
      gt_boxes1 (S,M1,4) xyxy, gt_ids1 (S,M1)
      tr_boxes0 (S,K0,4) xyxy, tr_ids0 (S,K0)
      tr_boxes1 (S,K1,4) xyxy, tr_ids1 (S,K1)
    Boxes are converted to MOT xywh inside _load_raw_file.
    """

    @staticmethod
    def get_default_dataset_config():
        code_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_config = {
            'DATA_FILE': os.path.join(code_path, 'data/two_frame_pairs.npz'),
            'SEQ_IDS': None,
            'PRINT_CONFIG': True,
            'DO_PREPROC': False,
            'TRACKERS_TO_EVAL': ['model'],
            'TRACKER_DISPLAY_NAMES': ['model'],
            'OUTPUT_FOLDER': None,
            'TRACKER_SUB_FOLDER': '',
            'OUTPUT_SUB_FOLDER': '',
        }
        return default_config

    def __init__(self, config=None):
        self.config = {}
        self.config.update(self.get_default_dataset_config())
        if config:
            self.config.update(config)
        self.gt_fol = self.config['DATA_FILE']
        self.tracker_fol = self.config['DATA_FILE']
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = False
        self.do_preproc = self.config['DO_PREPROC']
        self.output_fol = self.config['OUTPUT_FOLDER'] or os.path.dirname(self.config['DATA_FILE'])
        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']
        self.benchmark = 'TWFRAME'
        self.valid_classes = ['pedestrian']
        self.class_list = ['pedestrian']
        self.class_name_to_class_id = {
            'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4,
            'motorbike': 5, 'non_mot_vehicle': 6, 'static_person': 7,
            'distractor': 8, 'occluder': 9, 'occluder_on_ground': 10,
            'occluder_full': 11, 'reflection': 12, 'crowd': 13}
        self.valid_class_numbers = list(self.class_name_to_class_id.values())
        data = np.load(self.config['DATA_FILE'], allow_pickle=True)
        self.data = data
        seq_ids = data['seq_ids']
        if self.config['SEQ_IDS'] is not None:
            seq_ids = np.asarray(self.config['SEQ_IDS'])
        self.seq_list = [f'seq_{int(s)}' for s in seq_ids]
        self.seq_lengths = {s: 2 for s in self.seq_list}
        self.tracker_list = self.config['TRACKERS_TO_EVAL']
        self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))

    def _get_seq_info(self):
        return self.seq_list, self.seq_lengths

    def _load_raw_file(self, tracker, seq, is_gt):
        idx = int(seq.split('_')[1])
        pos = int(np.where(self.data['seq_ids'] == idx)[0][0])
        num_timesteps = 2
        if is_gt:
            boxes0 = self.data['gt_boxes0'][pos]
            ids0 = self.data['gt_ids0'][pos]
            boxes1 = self.data['gt_boxes1'][pos]
            ids1 = self.data['gt_ids1'][pos]
        else:
            boxes0 = self.data['tr_boxes0'][pos]
            ids0 = self.data['tr_ids0'][pos]
            boxes1 = self.data['tr_boxes1'][pos]
            ids1 = self.data['tr_ids1'][pos]
        # filter padding rows (id == 0 marks padded slots in the npz)
        mask0 = ids0 > 0
        mask1 = ids1 > 0
        boxes0, ids0 = boxes0[mask0], ids0[mask0]
        boxes1, ids1 = boxes1[mask1], ids1[mask1]
        raw_data = {
            'ids': [ids0.astype(int), ids1.astype(int)],
            'classes': [np.ones(len(ids0), dtype=int), np.ones(len(ids1), dtype=int)],
            'dets': [self._xyxy_to_xywh(boxes0), self._xyxy_to_xywh(boxes1)],
            'num_timesteps': num_timesteps,
            'seq': seq,
        }
        if is_gt:
            raw_data['gt_crowd_ignore_regions'] = [
                np.empty((0, 4)), np.empty((0, 4))]
            raw_data['gt_extras'] = [
                {'zero_marked': np.ones(len(ids0), dtype=int)},
                {'zero_marked': np.ones(len(ids1), dtype=int)},
            ]
            key_map = {'ids': 'gt_ids', 'classes': 'gt_classes', 'dets': 'gt_dets'}
        else:
            raw_data['tracker_confidences'] = [np.ones(len(ids0)), np.ones(len(ids1))]
            key_map = {'ids': 'tracker_ids', 'classes': 'tracker_classes', 'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        return raw_data

    @staticmethod
    def _xyxy_to_xywh(boxes):
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.size == 0:
            return np.empty((0, 4))
        out = boxes.copy()
        out[:, 2] = boxes[:, 2] - boxes[:, 0]
        out[:, 3] = boxes[:, 3] - boxes[:, 1]
        return out
