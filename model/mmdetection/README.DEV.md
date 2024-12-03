
## Environment Setup with CUDA
- Setup an Ubuntu 22.04 instance with a NVIDIA GPU on AWS EC2(https://aws.amazon.com/ec2/instance-types/g4/)
  ```

  ```

- Python 3.9 Installation 
  - Miniconda v23.1.0 - https://docs.conda.io/projects/miniconda/en/latest/
  ```bash
  conda env remove -n mmdet118 -y
  conda create -n mmdet118 python=3.8 -y
  conda activate mmdet118
  pip install -r requirements.dev.txt

  pip install torch==2.0.0 torchvision==0.15.1  --index-url https://download.pytorch.org/whl/cu118

  ```
- Install CUDA on this instance
  ```
  # CUDA 11.8
  #wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
  #sudo sh cuda_11.8.0_520.61.05_linux.run

  export CUDA_HOME="/usr/local/cuda-11.8"
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```
- Additional distributed training issues: https://github.com/open-mmlab/mmdetection/issues/6534


## MMDetection - RTMDet


- MMDetection Installation: https://mmdetection.readthedocs.io/en/latest/get_started.html
  ```bash
  mim install albumentations --no-binary qudida,albumentations
  mim install mmengine
  #pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
  pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

  # Install MMDetection
  pip install -v -e .
  ```
  - [ERROR-Broken] MMCV from source and not from `mim install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html`
    ```bash
    mim uninstall mmcv mmdet
    git clone https://github.com/open-mmlab/mmcv.git && cd mmcv
    # https://github.com/open-mmlab/mmcv/releases/tag/v2.0.1
    git checkout v2.0.1
    pip install -r requirements/optional.txt
    pip install -e . -v
    rm -rf mmcv/
    ```
- Test PyTorch version
  ```
  python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME)'
  ```
- Weights and Bias dashboard
  ```bash
  # After running wandb login, enter the API Keys obtained from your project, and the login is successful.
  wandb login 
  ```

### Exp

#### Run - RTMDet-S
```
# 1 GPU # XXX hours for b2 100 epochs
CUDA_VISIBLE_DEVICES=0 PORT=29601 ./tools/dist_train.sh rtmdet_s_manacus.py 1
```
>
```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.539
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.789
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.679
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.553
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.621
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.652
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.559
07/25 14:25:13 - mmengine - INFO -
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| Male     | 0.555 | 0.796  | 0.699  | nan   | 0.553 | 0.574 |
| Female   | 0.523 | 0.781  | 0.658  | nan   | 0.553 | 0.315 |
| Unknown  | nan   | nan    | nan    | nan   | nan   | nan   |
+----------+-------+--------+--------+-------+-------+-------+
```

#### Run - RTMDet-M 
- Run 1 - b48 300 epochs. Estimated 5-6 days
  ```
  CUDA_VISIBLE_DEVICES=0,1 PORT=29601 ./tools/dist_train.sh rtmdet_m_manacus_r1_allaug.py 2
  ```
  - The learning collapsed after e30. Reference: https://github.com/open-mmlab/mmdetection/issues/2942
- Run 2 - halved the learning rate and training for e100. Estimated 1 day
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_m_manacus_r1_allaug.py 4
  ```
  - [In-progress]
- Run 3 - tenth the learning rate and training for e100. Estimated 1 day
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_m_manacus_r1_allaug.py 4
  ```
- Run 4 - removed switch, allaug, b16, halved the learning rate and training for e200. Estimated 2 day
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_m_manacus_r1_noswitch.py 4
  ```
    - Weights work_dirs/rtmdet_m_r1_noswitch_allaug_b16_e200/best_coco_bbox_mAP_epoch_135.pth 
    ```
    +----------+-------+--------+--------+-------+-------+-------+
    | category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
    +----------+-------+--------+--------+-------+-------+-------+
    | Male     | 0.748 | 0.96   | 0.844  | 0.013 | 0.706 | 0.806 |
    | Female   | 0.74  | 0.971  | 0.871  | 0.0   | 0.725 | 0.78  |
    | Unknown  | 0.0   | 0.0    | 0.0    | 0.0   | 0.0   | nan   |
    +----------+-------+--------+--------+-------+-------+-------+
    ```
  - Test
    ```
    CUDA_VISIBLE_DEVICES=0 python ./tools/test.py rtmdet_m_manacus_r1_noswitch.py work_dirs/rtmdet_m_r1_noswitch_allaug_b16_e200/best_coco_bbox_mAP_epoch_135.pth --show-dir show-dir
    ```
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.496
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.643
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.573
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.004
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.791
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.519
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.525
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.540
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.058
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.537
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.827
    12/02 15:28:54 - mmengine - INFO -
    +----------+-------+--------+--------+-------+-------+-------+
    | category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
    +----------+-------+--------+--------+-------+-------+-------+
    | Male     | 0.747 | 0.959  | 0.843  | 0.013 | 0.703 | 0.805 |
    | Female   | 0.74  | 0.971  | 0.875  | 0.0   | 0.728 | 0.777 |
    | Unknown  | 0.0   | 0.0    | 0.0    | 0.0   | 0.0   | nan   |
    +----------+-------+--------+--------+-------+-------+-------+
    12/02 15:28:54 - mmengine - INFO - bbox_mAP_copypaste: 0.496 0.643 0.573 0.004 0.477 0.791
    12/02 15:28:58 - mmengine - INFO - Epoch(test) [1201/1201]    coco/Male_precision: 0.7470  coco/Female_precision: 0.7400  coco/Unknown_precision: 0.0000  coco/bbox_mAP: 0.4960  coco/bbox_mAP_50: 0.6430  coco/bbox_mAP_75: 0.5730  coco/bbox_mAP_s: 0.0040  coco/bbox_mAP_m: 0.4770  coco/bbox_mAP_l: 0.7910  data_time: 4.1103  time: 4.2241
    ```
- Run 5 - removed switch, noaug, b16, halved the learning rate and training for e200. Estimated 2 day
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29602 ./tools/dist_train.sh rtmdet_m_manacus_r1_noswitch.py 4
  ```
### Deploy for inference

- `v0.1` MMYolo release https://github.com/karmarv/manacus-dynamics/releases/tag/v0.1
  ```
  pip install onnx onnx-simplifier
  python ./projects/easydeploy/tools/export_onnx.py rtmdet_s_manacus.py \
    ./work_dirs/rtmdet_s_manacus_r1/epoch_100.pth \
    --work-dir ./work_dirs/rtmdet_s_manacus_r1/deploy/ \
    --img-size 640 640 --batch 1 --device cpu \
    --iou-threshold 0.65 \
    --score-threshold 0.25
  ```
  - export logs
    ```
    Export ONNX with bbox decoder and NMS ...
    Loads checkpoint by local backend from path: ./work_dirs/rtmdet_s_manacus_r1/epoch_100.pth
    ============= Diagnostic Run torch.onnx.export version 2.0.1+cu118 =============
    verbose: False, log level: Level.ERROR
    ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

    ONNX export success, save into ./work_dirs/rtmdet_s_manacus_r1/deploy/epoch_100.onnx
    ```
  - Netron image of the exported ONNX model - [./work_dirs/rtmdet_s_manacus_r1/deploy/epoch_100.onnx.png](./work_dirs/rtmdet_s_manacus_r1/deploy/epoch_100.onnx.png)

- `v0.2` MMDet release - https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmdet.html#install-mmdeploy
  ```
  python tools/deploy.py \
    configs/mmdet/detection/detection_onnxruntime_dynamic.py \
    ../mmdetection/work_dirs/rtmdet_m_r1_noswitch_allaug_b16_e200/rtmdet_m_manacus_r1_noswitch.py \
    ../mmdetection/work_dirs/rtmdet_m_r1_noswitch_allaug_b16_e200/best_coco_bbox_mAP_epoch_135.pth \
    ../../inference/mm/deploy/frame_000830.PNG \
    --work-dir ../mmdetection/work_dirs/rtmdet_m_r1_noswitch_allaug_b16_e200/ort \
    --device cpu \
    --show \
    --dump-info

  cp ../mmdetection/work_dirs/rtmdet_m_r1_noswitch_allaug_b16_e200/ort/end2end.onnx ../../inference/mm/deploy/rtmdet_m_r1_noswitch_allaug_b16_e135.onnx
  ```