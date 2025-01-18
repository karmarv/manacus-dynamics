
### Docker Environment
- Run
  ```bash
  t=ultralytics/ultralytics:latest && docker pull $t && docker run -it --ipc=host --gpus all -v "$(pwd)":"/ultralytics/code" -v "/home/rahul/workspace/vision/manacus-dynamics/dataset/fcat":"/datasets" $t
  ```

### Python Env 
  ```
  conda env remove -n ulyolo -y
  conda create -n ulyolo python=3.10 -y
  conda activate ulyolo

  pip install -r requirements.txt
  wandb login
  ```
- Environment variables
  ```
  #export CUDA_HOME="/usr/local/cuda-12.6"
  export CUDA_HOME="/usr/local/cuda-11.8"
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```
- Test
  ```bash
  python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME)'
  ```

### YoloV11-Medium
#### Run 1
- Config: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
    - Download weights
    ```bash
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
    ```
- Train
  - Script
  ```bash
  python train_yolo11m.py
  ```
    - Val Run (10 epochs)
    ```bash
    Ultralytics 8.3.61 🚀 Python-3.10.16 torch-2.4.1+cu124 CUDA:3 (NVIDIA RTX 6000 Ada Generation, 48539MiB)
    YOLO11m summary (fused): 303 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
    val: Scanning /home/rahul/workspace/vision/manacus-dynamics/dataset/fcat/yolo/fcat-manacus-v5-fcat-ebird/labels/val.cache... 19546 images, 0 backgrounds, 0 corrupt: 100%|██████████| 19546/19546 [00:00<?, ?it/
      Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1222/1222 [01:20<00:00, 15.27it/s]
        all      19546      26448       0.97      0.642      0.669      0.524
       Male      13393      13494      0.946      0.967      0.976      0.775
     Female      12732      12930      0.964      0.959      0.981      0.771
    Unknown         24         24          1          0     0.0516      0.025
    ```
    - Others
    ```bash
    Results saved to /home/rahul/workspace/vision/manacus-dynamics/model/ultralytics/ul-yolo/y11m-dv5-r15/weights
    Predict:         yolo predict task=detect model=ul-yolo/y11m-dv5-r15/weights/best.onnx imgsz=640
    Validate:        yolo val task=detect model=ul-yolo/y11m-dv5-r15/weights/best.onnx imgsz=640 data=coco8-fcat-v5.yaml
    Visualize:       https://netron.app
    Export ONNX:  ul-yolo/y11m-dv5-r15/weights/best.onnx
    ```
  - CLI
  ```bash
  yolo train data=coco8-fcat-v5.yaml model=yolo11m.pt project="ul-yolo" epochs=10
  ```
    - Val
    ```bash
    10 epochs completed in 8.133 hours.
    Optimizer stripped from ul-yolo/train/weights/last.pt, 40.5MB
    Optimizer stripped from ul-yolo/train/weights/best.pt, 40.5MB

    Validating ul-yolo/train/weights/best.pt...
    Ultralytics 8.3.61 🚀 Python-3.10.16 torch-2.4.1+cu124 CUDA:0 (NVIDIA RTX 6000 Ada Generation, 48539MiB)
    YOLO11m summary (fused): 303 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
      Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 611/611 [03:43<00:00,  2.73it/s]
        all      19546      26448      0.952      0.725      0.784      0.592
       Male      13393      13494      0.917      0.981      0.979       0.77
     Female      12732      12930      0.939      0.981      0.983      0.761
    Unknown         24         24          1      0.214      0.391      0.244
    Speed: 0.4ms preprocess, 1.7ms inference, 0.0ms loss, 2.0ms postprocess per image
    ```

#### Run 2
- Config: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
    - Download weights
    ```bash
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt
    ```
- Train (100 epochs)
  - CLI 
  ```bash
  # Fri Jan 17 07:45:26 PM PST 2025
  yolo train data=coco8-fcat-v5.yaml model=yolo11m.pt project="ul-yolo" name="train" epochs=100
  ```

### YoloV11-Large

#### Run 1
- Train (10 epochs)
  - CLI 
  ```bash
  # Fri Jan 17 07:45:26 PM PST 2025
  yolo train data=coco8-fcat-v5.yaml model=yolo11l.pt project="ul-yolo" name="y11l-dv5-default-train" epochs=10
  ```
  ```bash
  10 epochs completed in 7.354 hours.
  Optimizer stripped from ul-yolo/y11l-dv5-default-train/weights/last.pt, 51.2MB
  Optimizer stripped from ul-yolo/y11l-dv5-default-train/weights/best.pt, 51.2MB

  Validating ul-yolo/y11l-dv5-default-train/weights/best.pt...
  Ultralytics 8.3.61 🚀 Python-3.10.16 torch-2.4.1+cu124 CUDA:0 (NVIDIA RTX 6000 Ada Generation, 48539MiB)
  YOLO11l summary (fused): 464 layers, 25,281,625 parameters, 0 gradients, 86.6 GFLOPs
    Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 611/611 [02:36<00:00,  3.92it/s]
      all      19546      26448      0.901      0.698      0.721      0.574
     Male      13393      13494      0.922      0.983      0.982       0.84
   Female      12732      12930      0.932      0.985      0.985      0.836
  Unknown         24         24      0.849      0.125      0.195     0.0456
  Speed: 0.4ms preprocess, 1.8ms inference, 0.0ms loss, 1.3ms postprocess per image
  Results saved to ul-yolo/y11l-dv5-default-train
  ```

#### Run 2
- Train (100 epochs)
  - CLI 
  ```bash
  # Fri Jan 17 07:45:26 PM PST 2025
  yolo train data=coco8-fcat-v5.yaml model=yolo11l.pt project="ul-yolo" name="y11l-dv5-e100-train" epochs=100 device=3
  ```


# Issues 
- Ultralytics Yolo11 Distributed training config `device=2,3` not working with Pytorch 2.4.1 + CUDA 11.8 or 12.4/12.6