"""U1R2 expression adaptation from the selected class-aware U1-A model."""

_base_ = './u1_generic_adaptation.py'

load_from = '/data2/user/LocateMOT/outputs/unified/u1r2/generic_refinement_epoch1/epoch_1.pth'
adapt_root = '/data2/user/LocateMOT/outputs/unified/u1r2'
lang_model_name = '/data2/user/LocateMOT/weights/bert-base-uncased'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomSamplingNegPos', tokenizer_name=lang_model_name, num_sample_negative=0, max_tokens=256, full_sampling_prob=1.0),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'text', 'custom_entities', 'tokens_positive', 'dataset_mode')),
]


def _vg_dataset(path):
    return dict(
        type='ODVGDataset', data_root='', ann_file=adapt_root + '/' + path,
        data_prefix=dict(img=''), filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline, return_classes=True, backend_args=None)


def _od_dataset(path):
    return dict(
        type='ODVGDataset', data_root='', ann_file=adapt_root + '/' + path,
        label_map_file=adapt_root + '/generic_label_map.json',
        data_prefix=dict(img=''), filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline, return_classes=True, backend_args=None)


train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[
        _od_dataset('expression_mix_generic.jsonl'),
        _vg_dataset('expression_mix_v1.jsonl'),
        _vg_dataset('expression_mix_v2.jsonl'),
    ]),
)

val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
max_epochs = 1
find_unused_parameters = True
model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=True, static_graph=True)
param_scheduler = [dict(type='MultiStepLR', begin=0, end=1, by_epoch=True, milestones=[], gamma=0.1)]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.0),
        'language_model': dict(lr_mult=0.0),
        'backbone': dict(lr_mult=0.1),
        'encoder': dict(lr_mult=0.6),
        'neck': dict(lr_mult=0.6),
    }),
)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))
