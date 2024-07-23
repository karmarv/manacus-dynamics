_base_ = "./configs/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py"


# https://mmengine.readthedocs.io/en/latest/api/visualization.html
_base_.visualizer.vis_backends = [
dict(type='LocalVisBackend'),
dict(type='TensorboardVisBackend'),
dict(type='WandbVisBackend', init_kwargs={
        'project': "MM-Manacus",
        "reinit": True,}),]



max_epochs = 100
interval = 5
# Batch size of a single GPU during training
train_batch_size_per_gpu = 8
val_batch_size_per_gpu = train_batch_size_per_gpu

# -----data related-----
data_root = '/home/rahul/workspace/eeb/manacus-project/manacus-dynamics/dataset/fcat/coco/fcat-manacus-v1/'
# Path of train annotation file
train_ann_file = 'annotations/train.json'
train_data_prefix = 'train/images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/val.json'
val_data_prefix = 'val/images/'  # Prefix of val image path


class_names = ("Male", "Female", "Unknown", ) # dataset category name
num_classes = len(class_names)               # Number of classes for classification
# metainfo is a configuration that must be passed to the dataloader, otherwise it is invalid
# palette is a display color for category at visualization
# The palette length must be greater than or equal to the length of the classes
metainfo = dict(classes=class_names, palette=[[51,221,255], [240,120,240], [250,250,55]])


# load COCO pre-trained weight from https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet
load_from = 'https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-0a8c901a.pth'  # noqa

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

# RDD COCO data loader
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))
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