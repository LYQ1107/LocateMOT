"""Stream existing legal FIT observations; labels stay external to actor state."""
import json
from pathlib import Path


def rmot_fit_observations(plan):
    import numpy as np
    import torch
    from PIL import Image
    from tools.groundjev.export_rmot_full_fit_groups import context_for_manifest
    from tools.groundjev.audit_global_identity_labels import frame_support
    source=Path(plan['immutable_source_root']);index=json.loads((source/plan['source_index']).read_text())
    label_root=Path('/data2/user/LocateMOT/data/refer_kitti_v2/labels_with_ids/image_02')
    for name,unit in index['units'].items():
        if unit['task']!='rmot':continue
        manifest=json.loads((source/unit['manifest']).read_text())
        assert manifest['scope']=='all_available_fit_frames_independent_native_query_rollout_not_benchmark'
        # Existing exporter asserts legal FIT membership, expression/source
        # identity and normalized top-left xywh projection before opening labels.
        context=context_for_manifest(manifest,source/unit['original_run'],label_root,0)
        with (source/unit['original_run']/'frame_decisions.jsonl').open() as trace:
            for row in map(json.loads,trace):
                number=row['frame'];candidates=row['candidates']
                assert row['image_path']==manifest['images'][manifest['frame_numbers'].index(number)]
                with Image.open(row['image_path']) as image:size=image.size
                with np.load(source/unit['frames'][str(number)]['feature_path'],allow_pickle=False) as archive:
                    features=archive['regions']
                assert features.dtype==np.float32 and len(features)==len(candidates)
                candidates=[dict(c,native_region_feature=features[i]) for i,c in enumerate(candidates)]
                truth=[dict(identity=g['gt_id'],box=g['box'],visibility=None) for g in context['truth'][number]]
                matrix,matched,support=frame_support(candidates,truth)
                unique=[truth[s[0]]['identity'] if len(s)==1 else -1 for s in support]
                quality=[float(matrix[i,s[0]]) if len(s)==1 else 0. for i,s in enumerate(support)]
                labels=dict(action_options=[[] for _ in candidates],unique_GT=torch.tensor(unique,dtype=torch.long),
                    quality=torch.tensor(quality),GT_supervision_only=True,full_truth=truth,support=support,matched=matched)
                yield dict(unit=name,frame=number,timestamp=row['state']['timestamp'],image_size=size,
                    candidates=candidates,labels=labels,manifest=manifest,source_row=row,
                    source_histories_used_as_model_inputs=False)


def mixed_fit_observations(plan):
    from itertools import groupby
    from tools.groundjev.global_identity_data import training_frames
    # Interleave complete causal units in their source order, proportional to
    # 43 MOT /18 RMOT units; never mix two videos inside one predicted state.
    generators=[iter(groupby(training_frames(plan),lambda f:f['unit'])),
                iter(groupby(rmot_fit_observations(plan),lambda f:f['unit']))]
    used=[0,0];sizes=[43,18]
    for _ in range(sum(sizes)):
        candidates=[i for i in range(2) if used[i]<sizes[i]]
        selected=min(candidates,key=lambda i:used[i]/sizes[i])
        _,frames=next(generators[selected]);yield from frames;used[selected]+=1
    assert used==sizes
