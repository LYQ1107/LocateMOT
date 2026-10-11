"""U1R2 bounded generic refinement, starting from the frozen U1-A epoch4.

This config intentionally uses OptimWrapper (no AMP/BF16).  TF32 is enabled
by the launcher/audit environment where supported; all trainable gradients
are checked separately before a run is accepted.
"""

_base_ = './u1_generic_adaptation.py'

load_from = '/data2/user/reference_envs/u1_generic_adaptation_20261001/valid_work_bf16_batch2_frozen01_noacc/epoch_4.pth'
adapt_root = '/data2/user/LocateMOT/outputs/unified/u1r2'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomSamplingNegPos', tokenizer_name='/data2/user/LocateMOT/weights/bert-base-uncased', num_sample_negative=0, max_tokens=256, full_sampling_prob=1.0),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'text', 'custom_entities', 'tokens_positive', 'dataset_mode')),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ODVGDataset',
        data_root='',
        ann_file=adapt_root + '/generic_train_manifest.jsonl',
        label_map_file=adapt_root + '/generic_label_map.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        return_classes=True,
        backend_args=None,
    ),
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
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
    static_graph=True,
)
param_scheduler = [dict(type='MultiStepLR', begin=0, end=1, by_epoch=True, milestones=[], gamma=0.1)]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.0),
        'language_model': dict(lr_mult=0.0),
        'backbone': dict(lr_mult=0.1),
        'encoder': dict(lr_mult=0.5),
        'neck': dict(lr_mult=0.5),
    }),
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
)
