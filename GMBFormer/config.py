# ==================================================================
# custom_gmb — GMB at C3 (small-image only, 512x512)
#   - GMB 插在 C3 (320 ch, 32×32 in 512 input)
#   - NDVI threshold τ = 0.6 (归一化后 [0,1] 尺度，对应原始 NDVI ≈ 0.2)
#   - Memory bank size S = 64
#   - EMA momentum α = 0.99
#   - Cross-Attention heads = 8
#   - Backbone receives RGB only, NDVI decoupled as gate signal
# ==================================================================

crop_size = (512, 512)

custom_imports = dict(
    imports=[
        'GMBFormer',
        'mmseg.datasets.transforms.rgba_transforms',
    ],
    allow_failed_imports=False
)

# ------------------------------------------------------------------
# data_preprocessor: 4-channel RGBA (RGB + NDVI in alpha)
# ------------------------------------------------------------------
data_preprocessor = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=False,
    mean=[123.675, 116.28, 103.53, 0.0],
    std=[58.395, 57.12, 57.375, 255.0],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

# ------------------------------------------------------------------
# model
# ------------------------------------------------------------------
model = dict(
    type='GMBEncoderDecoder',
    data_preprocessor=data_preprocessor,

    use_ndvi_channel=True,
    ndvi_channel_idx=3,
    backbone_rgb_only=True,

    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 8, 27, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth'
        )
    ),

    decode_head=dict(
        type='SegformerGMBHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=0.5),
        ],
        # ── GMB 核心超参（挂在 C3 上）──────────────────────────────
        memory_size=64,      # 记忆库容量
        momentum=0.99,       # EMA 动量
        ndvi_thresh=0.6,     # NDVI 门控阈值（归一化后 [0,1]，对应原始 NDVI ≈ 0.2）
        memory_heads=8,
        ndvi_channel_idx=3,
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# ------------------------------------------------------------------
# metainfo
# ------------------------------------------------------------------
metainfo = dict(
    classes=('background', 'greenland'),
    palette=[[0, 0, 0], [0, 128, 0]]
)

# ------------------------------------------------------------------
# dataset: VOC-style, 512x512 pre-cropped patches
# ------------------------------------------------------------------
dataset_type = 'PascalVOCDataset'
data_root = r'G:\DATA\Potsdam\2classes\3band\tree_lowvegetation\data'

train_pipeline = [
    dict(type='LoadRGBAImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', direction=['horizontal', 'vertical'], prob=0.5),
    dict(type='PhotoMetricDistortionRGBA',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadRGBAImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Segmentation/train.txt',
        data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
        metainfo=metainfo,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Segmentation/val.txt',
        data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
        metainfo=metainfo,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

# ------------------------------------------------------------------
# optimizer / scheduler
# ------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            gmb=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0)
        )
    )
)

param_scheduler = [
    dict(type='LinearLR', begin=0, end=1500, by_epoch=False, start_factor=1e-6),
    dict(type='PolyLR', begin=1500, end=320000, by_epoch=False, eta_min=0.0, power=1.0),
]

# ------------------------------------------------------------------
# runtime
# ------------------------------------------------------------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=320000, val_interval=32000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False,
                    interval=32000, max_keep_ckpts=20),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=2000),
)

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

launcher = 'none'
load_from = None
resume = False
log_level = 'INFO'
log_processor = dict(by_epoch=False)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', name='visualizer', alpha=0.5,
    vis_backends=vis_backends
)

work_dir = './work_dirs/GMBFormer'
