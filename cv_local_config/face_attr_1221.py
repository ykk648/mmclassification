_base_ = [
    './_base_/default_runtime.py',
    './datasets/face_attr.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV3', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=10,
        in_channels=576,
        # mid_channels=[1280],
        # act_cfg=dict(type='HSwish'),
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1.0),
        # loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        # topk=(1,)
    ),
)

load_from = '/workspace/codes/mmclassification/work_dirs/face_attr_1221/epoch_30.pth'
# load_from = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth'

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[15, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)
