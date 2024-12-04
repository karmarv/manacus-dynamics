
# Model

- Python environment
    ```
    conda create -n yv7 python=3.9 -y
    conda activate yv7
    pip install -r requirements.txt
    pip install Pillow==9.5.0
    ```

## Step (1) - Environment Setup with CUDA 11.8
- Setup an Ubuntu 20.04 or 22.04 instance with a NVIDIA GPU on AWS EC2(https://aws.amazon.com/ec2/instance-types/g4/)
- Install CUDA 11.8 on this instance
  ```
  wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
  sudo sh cuda_11.8.0_520.61.05_linux.run
  rm cuda_11.8.0_520.61.05_linux.run
  ```
- Environment variables check and PyTorch
  ```bash
  export CUDA_HOME="/usr/local/cuda-11.8"
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

  # Ensure download for the 11.8 version
  pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
  ```
- Test
  ```bash
  python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME)'
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




## `Stage 2` Transfer learning camera trap manacus detection model

### (.) Dataset Preparation
- fcat-manacus-v2: Sample 10 videos male/female stationary track labeling
  - Link: http://vader.ece.ucsb.edu:8080/projects/6?page=1
  - COCO 1.0 project export format for model evaluation
- fcat-manacus-v3: 239 video labeling
- fcat-manacus-v4: Interpolated extraction of track labels
- fcat-manacus-v5: eBird data combined with v4 dataset

### (.) Yolov7 - `Stage 2` manacus detection model

Training logs on W&B - https://wandb.ai/karmar/Yv7-Manacus

> Looks like some issue with data [Learning is not happening]

- [r6-scratch] Initial 10 video sample track labels based model validated on 2 videos. 300 training epochs completed in 10.434 hours.
  ```bash
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
  # train p5 models
  python train.py --workers 16 --device 0 --batch-size 16 --data data/manacus-fcat.yaml --img 640 640 --cfg cfg/training/yolov7-manacus.yaml --weights 'yolov7.pt' --name r6-fcat-b16-640w --hyp data/hyp.scratch.p5.yaml --epochs 200
  ```
  - logs general --cfg cfg/training/yolov7-manacus.yaml
    ```log
    Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95
      all         197         241       0.918       0.949       0.959       0.801
     Male         197         128       0.924       0.969        0.95       0.774
   Female         197         113       0.912       0.929       0.967       0.828
    ```
  - logs small objects --cfg cfg/training/yolov7-manacus-so.yaml
    ```log
      all         197         241       0.904       0.921        0.91       0.739
     Male         197         128       0.891       0.956        0.94       0.725
   Female         197         113       0.917       0.885        0.88       0.754
    ```

- Attempt with resolution modification
  ```bash
  python train.py --workers 8 --device 0 --batch-size 4 --data data/manacus-fcat.yaml --img 1280 1280 --cfg cfg/training/yolov7-manacus.yaml --weights 'yolov7.pt' --name  r6-fcat-b4-1280w --hyp data/hyp.scratch.p5.yaml --epochs 200
  ```
  - logs
  ```log
     all         197         241       0.898       0.919       0.924       0.734
    Male         197         128       0.897       0.961       0.951       0.723
  Female         197         113         0.9       0.876       0.897       0.746
  ```
  
  - [TODO] Attempt with yolov7-w6 model
    - train p6 models with multiple error fixed in utils/loss.py [reference](https://stackoverflow.com/questions/74372636/indices-should-be-either-on-cpu-or-on-the-same-device-as-the-indexed-tensor)
  ```bash
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt
  python train_aux.py --workers 8 --device 0 --batch-size 4 --data data/manacus-fcat.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-manacus.yaml --weights 'yolov7-w6.pt' --name  r6-fcat-b4-1280w-w6 --hyp data/hyp.scratch.p5.yaml --epochs 200 
  ```
  ```log
     all         197         241       0.906       0.961       0.957       0.799
    Male         197         128       0.912       0.975       0.949       0.767
  Female         197         113       0.899       0.947       0.965       0.831
  ```


--- 

Training on larger video labeled dataset: v4 -> ./dataset/fcat/yolo/fcat-manacus-v4-inter/
```
export CUDA_VISIBLE_DEVICES=0,1,2,3  

```
- https://wandb.ai/karmar/Yv7-Manacus

- [v4-r1-fcat-b16-640w] train v7 models with V4-inter YOLO dataset [Nov/06/2024]
  ```bash
  python train.py --workers 8 --device 0 --batch-size 16 --data data/manacus-fcat.yaml --img 640 640 --cfg cfg/training/yolov7-manacus.yaml --weights 'yolov7.pt' --name d4-r1-fcat-b16-640w --hyp data/hyp.scratch.p5.yaml --epochs 200
  ```
  - 🚀 View run at https://wandb.ai/karmar/Yv7-Manacus/runs/87iczpy9
  ```
    Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95
      all       19209       26106       0.966       0.963       0.983       0.857
     Male       19209       13256       0.973       0.967       0.989       0.915
   Female       19209       12831        0.98       0.974        0.99       0.905
  Unknown       19209          19       0.946       0.947        0.97       0.751
  200 epochs completed in 112.720 hours.
  ```

- [v4-r2-fcat-b64-640w] train v7 models with V4-inter YOLO dataset [Nov/06/2024]
  ```bash
  python train.py --workers 8 --device 1 --batch-size 64 --data data/manacus-fcat.yaml --img 640 640 --cfg cfg/training/yolov7-manacus.yaml --weights 'yolov7.pt' --name d4-r2-fcat-b64-640w --hyp data/hyp.scratch.p5.yaml --epochs 200
  ```
  -  🚀 View run at https://wandb.ai/karmar/Yv7-Manacus/runs/uwp1hxco
  ```
    Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 
      all       19209       26106       0.949       0.967       0.959       0.755
     Male       19209       13256       0.947       0.974       0.978       0.829
   Female       19209       12831       0.953       0.981       0.983        0.83
  Unknown       19209          19       0.947       0.947       0.917       0.605
  200 epochs completed in 111.019 hours
  ```

- [v4-r3-fcat-b16-1280w] train v7 models with V4-inter YOLO dataset [Nov/06/2024]
  ```bash
  python train.py --workers 8 --device 2 --batch-size 16 --data data/manacus-fcat.yaml --img 1280 1280 --cfg cfg/training/yolov7-manacus.yaml --weights 'yolov7.pt' --name d4-r3-fcat-b16-1280w --hyp data/hyp.scratch.p5.yaml --epochs 200
  ```
  - 🚀 View run at https://wandb.ai/karmar/Yv7-Manacus/runs/cgv3l6zu

- [v4-r4-fcat-b16-1280w-w6] train yolov7-w6 models with V4-inter YOLO dataset [Nov/06/2024]
  ```bash
  python train_aux.py --workers 8 --device 3 --batch-size 16 --data data/manacus-fcat.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-manacus.yaml --weights 'yolov7-w6.pt' --name d4-r4-fcat-b16-1280w-w6 --hyp data/hyp.scratch.p5.yaml --epochs 200 
  ```
  -  🚀 View run at https://wandb.ai/karmar/Yv7-Manacus/runs/sakogsq


Training on larger video labeled dataset: v5 -> ./dataset/fcat/yolo/fcat-manacus-v5-fcat-ebird/
- [v5-r1-fcat-b16-640w] train v7 models with v5 YOLO dataset [Dec/04/2024]
  ```bash
  python train.py --workers 8 --device 0 --batch-size 16 --data data/manacus-fcat-v5.yaml --img 640 640 --cfg cfg/training/yolov7-manacus.yaml --weights 'yolov7.pt' --name v5-r1-fcat-b16-640w --hyp data/hyp.scratch.p5.yaml --epochs 200
  ```
  - 🚀 View run at https://wandb.ai/karmar/Yv7-Manacus/runs/ek3vy5vc

- [v5-r3-fcat-b16-1280w] train v7 models with V5 YOLO dataset [Dec/04/2024]
  ```bash
  python train.py --workers 8 --device 2 --batch-size 12 --data data/manacus-fcat-v5.yaml --img 1280 1280 --cfg cfg/training/yolov7-manacus.yaml --weights 'yolov7.pt' --name v5-r3-fcat-b12-1280w --hyp data/hyp.scratch.p5.yaml --epochs 200
  ```