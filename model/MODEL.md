
# Model

- Python environment
    ```
    pip install -r requirements.txt
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install Pillow==9.5.0
    ```

## (1.) Yolov7 - `Stage 1` manacus detection model

Training logs on W&B - https://wandb.ai/karmar/Yv7-Manacus

```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
# train p5 models
python train.py --workers 16 --device 0 --batch-size 16 --data data/manacus.yaml --img 640 640 --cfg cfg/training/yolov7-manacus.yaml --weights 'yolov7.pt' --name yv7-manacus --hyp data/hyp.scratch.p5.yaml
```
- [r1] Initial fake pseudo labels based model evaluated on test set. 300 training epochs completed in 10.434 hours.
    ```log
      Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 
        all         335         338       0.367       0.687       0.354       0.284
       Male         335          92       0.243       0.477       0.225       0.117
     Female         335          24       0.216       0.583       0.152      0.0658
    Unknown         335         222       0.644           1       0.684       0.669
    ```
- [r2] SME labeled dataset based model evaluated on test set. 300 training epochs completed in 9.379 hours.
    ```log
      Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95:
        all         337         342       0.991       0.652       0.671       0.551
       Male         337         238       0.995       0.996       0.997       0.852
     Female         337          99       0.979        0.96       0.979       0.786
    Unknown         337           5           1           0      0.0382      0.0143
    ```
  - Sample prediction result @[../dataset/ebird/samples/test/test_batch2_pred.png](../dataset/ebird/samples/test/test_batch2_pred.png)    
- [r3] SME labeled dataset based model with added albumentation and cutouts strategies.
  ```log
      Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95:
        all         337         342       0.976       0.652       0.671       0.543
       Male         337         238       0.996       0.987       0.995       0.846
     Female         337          99       0.932       0.969       0.971       0.758
    Unknown         337           5           1           0      0.0471      0.0257
  ```
- [r4] SME labeled dataset based model with added multiscale to previous augmentations. Needs a bigger GPU to compute.
  ```bash
  python train.py --workers 16 --device 0 --batch-size 16 --data data/manacus.yaml --img 640 640 --multi-scale --cfg cfg/training/yolov7-manacus.yaml --weights 'yolov7.pt' --name yv7-manacus --hyp data/hyp.scratch.p5.yaml
  ```
  ```log
    Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95:
      all         337         342       0.989       0.651       0.661       0.554
     Male         337         238       0.999       0.983       0.996       0.858
   Female         337          99        0.97        0.97       0.977       0.797
  Unknown         337           5           1           0      0.0106      0.0085
  ```
- [r5] v7x model with added multiscale. Needs a bigger GPU to compute.
  ```bash
  # train p5 models
  python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 4,5,6,7 --sync-bn --batch-size 64 --data data/manacus.yaml --img 640 640 --multi-scale --cfg cfg/training/yolov7x-manacus.yaml --weights 'yolov7x.pt' --name yv7x-manacus --hyp data/hyp.scratch.p5.yaml
  ```
  ```log
  # In-progress
  ```
  
##### Inference 
- Given a video file report the detections in frames
  ```bash
  python detect.py --weights runs/train/r3-ebird-aug/weights/best.pt --conf 0.55 --img-size 640 --save-txt --save-conf --source "../../../data-fcat-sample-trap-videos/Full-length-clip-5_copulation.MP4"
  python detect.py --weights runs/train/r3-ebird-aug/weights/best.pt --conf 0.55 --img-size 640 --save-txt --save-conf --source "../../../data-fcat-sample-trap-videos/Full-length-clip-1_female-visitation.MP4"
  ```

##### Yolov9 - `Stage 1` manacus detection model

Training logs on W&B - https://wandb.ai/karmar/Yv9-Manacus

```bash
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt
# train yolov9 models
python train_dual.py --workers 8 --device 0 --batch 16 --data data/manacus.yaml --img 640 --cfg models/detect/yolov9-c-manacus.yaml --weights 'yolov9-c-converted.pt' --name yv9-c-manacus --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 15
```
- [r1] Yv9-c model trained on SME labeled dataset 
  ```log
  # In-progress
  ```



## `Stage 2` Transfer learning camera trap manacus detection model

### (.) Dataset Preparation
- fcat-manacus-v1: Sample 10 videos male/female stationary track labeling
  - Link: http://vader.ece.ucsb.edu:8080/projects/6?page=1
  - COCO 1.0 project export format for model evaluation
- fcat-manacus-v2: TODO

### (.) Yolov7 - `Stage 2` manacus detection model

Training logs on W&B - https://wandb.ai/karmar/Yv7-Manacus

> Looks like some issue with data [Learning is not happening]

- [r6-scratch] Initial 10 video sample track labels based model validated on 2 videos. 300 training epochs completed in 10.434 hours.
  ```bash
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
  # train p5 models
  python train.py --workers 16 --device 0 --batch-size 16 --data data/manacus-fcat.yaml --img 640 640 --cfg cfg/training/yolov7-manacus-fcat.yaml --weights 'yolov7.pt' --name r6-fcat-init --hyp data/hyp.scratch.p5.yaml
  ```
  - No learning happening
    ```log
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
    49/299     14.3G   0.02266  0.002145 0.0008067   0.02561         9       640: 100%|████████████████████████████████████████████| 182/182 [01:26<00:00,  2.10it/s]
    Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|████████████████████████████████████████████| 23/23 [00:06<00:00,  3.63it/s]
      all         726         354      0.0291      0.0196     0.00102    0.000207
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
    50/299     14.3G   0.02254  0.002002 0.0006905   0.02524         7       640: 100%|███████████████████████████████████████████| 182/182 [01:28<00:00,  2.06it/s]
    Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|████████████████████████████████████████████| 23/23 [00:06<00:00,  3.46it/s]
      all         726         354      0.0525      0.0196     0.00186     0.00021
    ```
  ```bash
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
  # train p5 models
  python train.py --workers 16 --device 0 --batch-size 4 --data data/manacus-fcat.yaml --img 1280 1280 --cfg cfg/training/yolov7-manacus-fcat.yaml --weights 'yolov7.pt' --name  r6-fcat-init-1280w-v2 --hyp data/hyp.scratch.p5.yaml
  ```
  - No learning happening
    ```log
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
    20/299     12.8G   0.02386  0.003066 0.0008518   0.02778         5      1280: 100%|███████████████████████████████████████████| 726/726 [04:08<00:00,  2.92it/s]
    Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|████████████████████████████████████████████| 91/91 [00:13<00:00,  6.92it/s]
      all         726         354      0.0622      0.0196     0.00236    0.000535
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
    21/299     12.8G   0.02408  0.003029  0.000979   0.02809         0      1280: 100%|███████████████████████████████████████████| 726/726 [04:10<00:00,  2.90it/s]
    Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|████████████████████████████████████████████| 91/91 [00:13<00:00,  6.89it/s]
      all         726         354       0.119      0.0196     0.00516    0.000544
    ```
    ```log
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
   100/299     13.2G    0.0168    0.0022 0.0004092   0.01941         5      1280: 100%|██████████████████████████████████████████████| 726/726 [05:54<00:00,  2.05it/s]
    Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|██████████████████████████████████| 91/91 [00:17<00:00,  5.09it/s]
      all         726         354      0.0216      0.0196    0.000765    0.000155
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
   101/299     13.2G   0.01714  0.002182 0.0003572   0.01968         5      1280: 100%|██████████████████████████████████████████████| 726/726 [05:55<00:00,  2.04it/s]
    Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|██████████████████████████████████| 91/91 [00:17<00:00,  5.08it/s]
      all         726         354      0.0223      0.0196    0.000784    0.000159
    ```
  - [TODO] Attempt with small objects modification
  ```bash
  python train.py --workers 8 --device 0 --batch-size 4 --epochs 100 --data data/manacus-fcat.yaml --img 1280 1280 --cfg cfg/training/yolov7-manacus-fcat-so.yaml --weights 'yolov7.pt' --name  r6-fcat-so-1280w-v3 --hyp data/hyp.scratch.p5.yaml
  ```
  - [TODO] Attempt with yolov7-w6 model
  ```bash
  python train.py --workers 8 --device 0 --batch-size 4 --epochs 100 --data data/manacus-fcat.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-manacus-fcat.yaml --weights 'yolov7.pt' --name  r6-fcat-w6-1280w-v4 --hyp data/hyp.scratch.p5.yaml
  ```

- [r7-transfer]
  ```bash
  # train p5 models - model/yolov7/runs/train/r3-ebird-aug/weights/best_289.pt
  python train.py --workers 8 --device 0 --batch-size 4 --data data/manacus-fcat.yaml --img 1280 1280 --cfg cfg/training/yolov7-manacus-fcat.yaml --weights 'runs/train/r3-ebird-aug/weights/best_289.pt' --name r7-fcat-tx-1280w --hyp data/hyp.scratch.p5.yaml