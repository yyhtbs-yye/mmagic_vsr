_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/basicvsr_test_config.py'
]

experiment_name = 'fstrn-pp_c64n7_8xb1-600k_reds4'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'
scale = 4

# model settings
model = dict(
    type='NaiveVSR',
    generator=dict(
        type='FSTRNet',),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(fix_iter=300_000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], frames_per_clip=99),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], frames_per_clip=99),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(type='PackInputs')
]

data_root = '/workspace/mmagic/datasets/REDS'

train_dataloader = dict(
    num_workers=8,
    batch_size=8,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root=data_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_train.txt',
        depth=2,
        num_input_frames=7,
        num_output_frames=1,
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=8,
    batch_size=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root=data_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_val.txt',
        depth=2,
        num_input_frames=7,
        num_output_frames=1,
        fixed_seq_len=50,
        pipeline=val_pipeline))

val_evaluator = dict(
    type='Evaluator', metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])

default_hooks = dict(checkpoint=dict(out_dir=save_dir))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300_000, val_interval=100)
val_cfg = dict(type='MultiValLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99)),
    paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)}))

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    eta_min=1e-7)
