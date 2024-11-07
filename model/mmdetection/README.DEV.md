
## Environment Setup with CUDA
- Setup an Ubuntu 22.04 instance with a NVIDIA GPU on AWS EC2(https://aws.amazon.com/ec2/instance-types/g4/)
  ```

  ```
- Install CUDA on this instance
  ```
  # CUDA 11.8
  #wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
  #sudo sh cuda_11.8.0_520.61.05_linux.run

  export CUDA_HOME="/usr/local/cuda-11.8"
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

  # CUDA 12.1
  wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
  sudo sh cuda_12.1.0_530.30.02_linux.run

  export CUDA_HOME="/usr/local/cuda-12.1"
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```


## MMDetection - RTMDet
- Python 3.9 Installation 
  - Miniconda v23.1.0 - https://docs.conda.io/projects/miniconda/en/latest/
  ```bash
  conda env remove -n mmdet121 -y
  conda create -n mmdet121 python=3.9 -y
  conda activate mmdet121
  pip install -r requirements.dev.txt
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
- MMDetection Installation: https://mmdetection.readthedocs.io/en/latest/get_started.html
  ```bash
  mim install albumentations --no-binary qudida,albumentations
  # Install MMDetection
  mim install -v -e .
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
```
# 1 GPU # XXX hours for b2 100 epochs
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh rtmdet_m_manacus.py 4
```

### Deploy for inference
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
