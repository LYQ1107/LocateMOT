"""Bounded native-frame archives; optional exact FP32 ROI tensors beside JSONL."""
import json
from pathlib import Path


class CandidateArchiveWriter:
    def __init__(self, path, packed=False):
        self.path = Path(path)
        self.index = self.path.open('w')
        self.packed = packed
        self.feature_bytes = 0
        self.feature_files = 0
        self.final_bytes = None
        if packed:
            self.directory = self.path.parent/'native_region_features'
            self.directory.mkdir()

    def __enter__(self):
        return self

    def write(self, row):
        if self.packed:
            import numpy as np
            vectors = [c['native_region_feature'] for c in row['candidates']]
            regions = np.asarray(vectors, dtype=np.float32).reshape(len(vectors), 2560)
            # The real native service emits regions.float().tolist(). Reject
            # any precision loss rather than quantizing an unexpected input.
            if regions.tolist() != vectors:
                raise ValueError('Native FP32 ROI archive would change an input value')
            target = self.directory/f"regions_{row['frame']:08d}.npz"
            temporary = target.with_suffix('.tmp')
            with temporary.open('wb') as file:
                np.savez_compressed(file, regions=regions)
            temporary.replace(target)
            self.feature_bytes += target.stat().st_size
            self.feature_files += 1
            row = dict(row, candidates=[{k: v for k, v in c.items() if k != 'native_region_feature'}
                for c in row['candidates']], native_region_features_npz=str(target.relative_to(self.path.parent)))
        self.index.write(json.dumps(row)+'\n')

    def flush(self):
        self.index.flush()

    def tell(self):
        return self.final_bytes if self.index.closed else self.index.tell()+self.feature_bytes

    def __exit__(self, *exception):
        self.flush()
        self.final_bytes = self.index.tell()+self.feature_bytes
        self.index.close()


def decode_candidate_row(row, directory):
    if 'native_region_features_npz' in row:
        import numpy as np
        row = dict(row)
        path = Path(directory)/row.pop('native_region_features_npz')
        with np.load(path, allow_pickle=False) as archive:
            vectors = archive['regions'].tolist()
        if len(vectors) != len(row['candidates']):
            raise ValueError('Packed native ROI rows differ from the candidate count')
        row['candidates'] = [dict(c, native_region_feature=v) for c, v in zip(row['candidates'], vectors)]
    return row
