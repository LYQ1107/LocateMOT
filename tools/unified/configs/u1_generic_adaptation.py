"""Controlled MM-GroundingDINO-B generic adaptation config.

The registered U1-A freeze policy freezes Swin stages 0 and 1, keeps BERT
frozen, and trains the multimodal/detection path in BF16.  The per-GPU
microbatch is two, giving an effective global batch of 16 on the eight-GPU
DDP run without changing the official 800x1333 input.  Accumulation is kept
at one because PyTorch's static-graph reducer is incompatible with the
checkpointed model under multi-step accumulation.
"""

_base_ = '../../../third_party/mmdetection/configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py'

load_from = '/data2/user/LocateMOT/weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'
lang_model_name = '/data2/user/LocateMOT/weights/bert-base-uncased'
adapt_root = '/data2/user/reference_envs/u1_generic_adaptation_20261001'

model = dict(
    # SwinTransformer freezes patch_embed plus stages [0, 1] when this is 2.
    backbone=dict(frozen_stages=2),
    language_model=dict(name=lang_model_name),
    bbox_head=dict(contrastive_cfg=dict(log_scale=None)),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomSamplingNegPos', tokenizer_name=lang_model_name, num_sample_negative=0, max_tokens=256, full_sampling_prob=1.0),
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
        ann_file=adapt_root + '/legal_fit_odvg.jsonl',
        label_map_file=adapt_root + '/legal_category_map.json',
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
    type='AmpOptimWrapper',
    accumulative_counts=1,
    dtype='bfloat16',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.0),
        'backbone': dict(lr_mult=0.1),
        'language_model': dict(lr_mult=0.0),
    }),
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
)
