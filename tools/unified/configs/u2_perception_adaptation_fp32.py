"""Registered 20k-row U2-P replay, at most two FP32 epochs from immutable U1R2."""

_base_ = './u1r2_expression_adaptation_fp32.py'

custom_imports = dict(imports=['locatemot.unified.runtime.perception_adaptation'], allow_failed_imports=False)
load_from = '/data2/user/LocateMOT/outputs/unified/u1r2/expression_adaptation_epoch1/epoch_1.pth'
model = dict(type='U2PerceptionGroundingDINO', backbone=dict(frozen_stages=2))

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # 20,000 rows divide exactly into 8 ranks x batch 2. Ratio buckets can
    # create singleton tail batches and change the registered step budget.
    batch_sampler=None,
    dataset=dict(
        type='ODVGDataset', data_root='',
        ann_file='/data2/user/LocateMOT/outputs/unified/u2/perception_adaptation/replay_train.jsonl',
        data_prefix=dict(img=''), filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline, return_classes=True, backend_args=None),
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)
max_epochs = 2
randomness = dict(seed=20261012, diff_rank_seed=False)
param_scheduler = [dict(type='MultiStepLR', begin=0, end=2, by_epoch=True, milestones=[], gamma=0.1)]
optim_wrapper = dict(
    _delete_=True, type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.0),
        'language_model': dict(lr_mult=0.0),
        'backbone': dict(lr_mult=0.1),
    }),
)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))
