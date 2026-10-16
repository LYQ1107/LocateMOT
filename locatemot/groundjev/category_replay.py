"""Reuse native image-only categories, never another run's identity or memory."""
from copy import deepcopy
import json
from pathlib import Path

PROTOCOL = 'causal-same-checkpoint-native-category-replay-v1'


def categories_from_frame(row, candidates, checkpoint, vocabulary):
    metadata = row['state']['decision_metadata']
    assert Path(metadata['checkpoint']).resolve() == Path(checkpoint).resolve()
    assert metadata['GT_read'] is False
    assert [c['box'] for c in row['candidates']] == [c['box'] for c in candidates]
    by_index = {e['candidate']: e['decision']['category_decision']
        for e in row['state']['events'] if 'candidate' in e}
    assert set(by_index) == set(range(len(candidates)))
    categories = [deepcopy(by_index[i]) for i in range(len(candidates))]
    assert all(c['native_probabilities'] and c['vocabulary_size'] == len(vocabulary) for c in categories)
    assert all(c['label'] == '__unknown__' or c['label'] in vocabulary for c in categories)
    return categories, deepcopy(metadata['category_decision'])


class CausalNativeCategoryReplay:
    def __init__(self, source_run, checkpoint, manifest, *, allow_partial_source_for_probe=False):
        self.source = Path(source_run).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.vocabulary = manifest['vocabulary']
        self.source_run_complete = not allow_partial_source_for_probe
        if self.source_run_complete:
            result = json.loads((self.source/'result.json').read_text())
            assert result['backend'] == 'full' and result['category_policy'] == 'native_hierarchical'
            assert result['all_input_frames_processed'] and result['frames'] == len(manifest['images'])
            assert Path(result['checkpoint']).resolve() == self.checkpoint
            assert result['specification']['vocabulary'] == self.vocabulary
            assert not result['gt_read_by_runtime'] and not result['future_frames_read_by_runtime']
        self.stream = (self.source/'frame_decisions.jsonl').open()
        self.current = None
        self.frames_reused = 0
        self.categories_reused = 0
        self.last_requested_frame = None

    def begin_frame(self, image_path, frame):
        assert self.last_requested_frame is None or frame > self.last_requested_frame
        # A resumed consumer may already have restored a past prefix. Advance
        # through that prefix only; a future source row is never used.
        for line in self.stream:
            row = json.loads(line)
            if row['frame'] < frame:
                continue
            assert row['frame'] == frame
            assert Path(row['image_path']).resolve() == Path(image_path).resolve()
            self.current = row
            self.last_requested_frame = frame
            return
        raise ValueError(f'Native category source lacks current frame {frame}')

    def classify(self, image, candidates, vocabulary):
        from PIL import Image
        assert self.current is not None and vocabulary == self.vocabulary
        with Image.open(self.current['image_path']) as original:
            assert original.size == image.size
        categories, recorded_report = categories_from_frame(self.current, candidates, self.checkpoint, vocabulary)
        report = {k: v for k, v in recorded_report.items() if k != 'batches'}
        report.update(batches=[], recorded_native_report=recorded_report,
            category_replay=dict(protocol=PROTOCOL, source_run=str(self.source),
                frame=self.current['frame'], current_image_only=True,
                identity_or_memory_from_source_used=False, actual_native_category_calls=0,
                recorded_timings_are_source_provenance_not_new_compute=True))
        self.frames_reused += 1
        self.categories_reused += len(categories)
        self.current = None
        return categories, report

    def summary(self):
        return dict(protocol=PROTOCOL, source_run=str(self.source), checkpoint=str(self.checkpoint),
            frames_reused=self.frames_reused, candidates_reused=self.categories_reused,
            actual_native_category_calls=0, all_recorded_root_and_conditional_leaf_probabilities_retained=True,
            source_complete_before_production_replay=self.source_run_complete,
            timing_comparable_to_uncached_inference=False)

    def close(self):
        self.stream.close()
