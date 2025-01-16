
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
  export CUDA_HOME="/usr/local/cuda-12.6"
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```
- Test
  ```bash
  python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME)'
  ```

### YoloV11
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
  - CLI
  ```bash
  yolo train data=coco8-fcat-v5.yaml model=yolo11m.pt project="ul-yolo" epochs=10
  ```

### YoloV10

#### Run 1
- 