_base_ = "./configs/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py"


# https://mmengine.readthedocs.io/en/latest/api/visualization.html
_base_.visualizer.vis_backends = [
dict(type='LocalVisBackend'),
dict(type='TensorboardVisBackend'),
dict(type='WandbVisBackend', init_kwargs={
        'project': "MM-Manacus",
        "reinit": True,}),]


# ========================training configurations======================
work_dir = './work_dirs/rtmdet_s_pipeswitch'
max_epochs = 100
interval = 5
# Batch size of a single GPU during training
train_batch_size_per_gpu = 16
val_batch_size_per_gpu = train_batch_size_per_gpu

# -----data related-----
data_root = '/home/rahul/workspace/eeb/manacus-project/manacus-dynamics/dataset/fcat/coco/fcat-manacus-v2/'
# Path of train annotation file
train_ann_file = 'annotations/train.json'
train_data_prefix = 'train/images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/validation.json'
val_data_prefix = 'validation/images/'  # Prefix of val image path


class_names = ("Male", "Female", "Unknown", ) # dataset category name
num_classes = len(class_names)                # Number of classes for classification
# metainfo is a configuration that must be passed to the dataloader, otherwise it is invalid
# palette is a display color for category at visualization
# The palette length must be greater than or equal to the length of the classes
metainfo = dict(classes=class_names, palette=[[51,221,255], [240,120,240], [250,250,55]])

# load COCO pre-trained weight from checkpoint
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
)

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes
            )
        )
    )


# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.5
img_scale = _base_.img_scale

# ratio range for random resize
random_resize_ratio_range = (0.5, 2.0)
# Number of cached images in mosaic
mosaic_max_cached_images = 40
# Number of cached images in mixup
mixup_max_cached_images = 20

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=mosaic_max_cached_images,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=random_resize_ratio_range,  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=mixup_max_cached_images),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=random_resize_ratio_range,  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]


# COCO data loader
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + val_ann_file, classwise=True)
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - _base_.num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2)
]