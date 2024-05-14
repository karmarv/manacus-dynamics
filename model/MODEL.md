
# Model

- Python environment
    ```
    pip install -r requirements.txt
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install Pillow==9.5.0
    ```

### (1.) Yolov7 - `Stage 1` manacus detection model

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
  python detect.py --weights runs/train/r2-ebird-sme/weights/best.pt --conf 0.55 --img-size 640 --save-txt --save-conf --source "../../../data-fcat-sample-trap-videos/copulation-1.mp4"
  ```

### (2.) Yolov9 - `Stage 1` manacus detection model

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

