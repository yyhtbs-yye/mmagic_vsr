_base_ = '../_base_/models/base_edvr.py'

experiment_name = 'edvrm_8xb4-600k_reds'
save_dir = './work_dirs'
work_dir = f'./work_dirs/{experiment_name}'

scale = 4

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], frames_per_clip=99),
    dict(type='TemporalReverse', keys='img_path', reverse_ratio=0),
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb'),
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
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection_circle'),
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb'),
    dict(type='PackInputs')
]


# model settings
pretrain_generator_url = (
    'https://download.openmmlab.com/mmediting/restorers/edvr/'
    'edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth')
model = dict(
    type='EDVR',
    generator=dict(
        type='EDVRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_frames=9,
        deform_groups=8,
        num_blocks_extraction=5,
        num_blocks_reconstruction=10,
        center_frame_idx=2,
        with_tsa=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrain_generator_url,
            prefix='generator.')),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='sum'),
    train_cfg=dict(tsa_iter=5000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

data_root = 'datasets/REDS'

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
        num_input_frames=9,
        num_output_frames=1,
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root=data_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_val.txt',
        depth=2,
        num_input_frames=9,
        num_output_frames=1,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[50000, 100000, 150000, 150000, 150000],
    restart_weights=[1, 1, 1, 1, 1],
    eta_min=1e-7)

find_unused_parameters = True
