# dataset settings
dataset_type = 'FaceAttr'
img_norm_cfg = dict(
    mean=[130.66653389, 111.08275716, 103.82772657],
    std=[67.69638518, 61.18537887, 58.46574405],
    to_rgb=True)

train_pipeline = [
    # dict(type='RandomCrop', size=32, padding=4),
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(512, -1)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(512, -1)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/face_attr_1221',
        ann_file='data/face_attr_1221/train.txt',
        pipeline=train_pipeline,
        # classes=['male', 'female']
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/face_attr_1221',
        ann_file='data/face_attr_1221/val.txt',
        pipeline=test_pipeline,
        test_mode=True,
        # classes=['male', 'female']
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        test_mode=True,
        # classes=['male', 'female']
        ),
)
evaluation = dict(
    interval=1, metric=['mAP', 'CP', 'OP', 'CR', 'OR', 'CF1', 'OF1'])
