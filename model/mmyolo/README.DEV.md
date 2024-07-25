


## MMYolo - RTMDet
- Python 3.8.18 Installation via Miniconda v23.1.0 - https://docs.conda.io/projects/miniconda/en/latest/
  ```bash
  conda env remove -n mmyolo
  conda create -n mmyolo python=3.9
  conda activate mmyolo
  pip install -r requirements.txt
  ```
- Installation: https://github.com/open-mmlab/mmyolo/blob/main/docs/en/get_started/installation.md 
  ```bash
  pip install -U openmim wandb future tensorboard prettytable
  mim install "mmengine>=0.6.0" "mmcv>=2.0.0rc4,<2.1.0" "mmdet>=3.0.0,<4.0.0"
  mim install albumentations --no-binary qudida,albumentations
  # Install MMYOLO
  mim install -v -e .
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
