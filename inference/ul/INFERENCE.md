## Inference for Manacus Male/Female detection in camera trap videos


#### 1. Create a python environment 

- Python Installation instructions via Miniconda - https://docs.conda.io/projects/miniconda/en/latest/
- Create a virtual environment named "infer" for this analysis
  ```bash
  conda env remove -n infer
  conda create -n infer python=3.10 -y
  conda activate infer
  ```
- Install the necessary packages
  ```bash
  pip install -r requirements.txt
  ```
  - GPU related [instructions on onnruntime-gpu](https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-12x)
    ```bash
    pip uninstall onnxruntime
    pip install onnxruntime-gpu
    export CUDA_VISIBLE_DEVICES=0
    ```

#### 2. Configure the inference script
- python yolo_infer.py --help

  | Argument (with Default)                            | Description                                              |
  | :------------------------------------------------- | :------------------------------------------------------- |
  | `--model "./deploy/y11m-dv6-e25-im1280.onnx"`      | Path where ONNX model is located                         |
  | `--video VIDEO_FILEPATH`                           | Path where video file is located                         |
  | `--image IMAGE_FILEPATH`                           | Path where image file is located (Not in use for videos) |
  | `--out-suffix "v03.result"`                        | Result filename suffix                                   |
  | `--out-path "./results"`                           | Result output folder                                     |
  | `--conf-thres 0.5`                                 | Confidence threshold                                     |
  | `--iou-thres 0.5`                                  | NMS IoU threshold                                        |
  | `--view-debug`                                     | Write qualitative intermediate results                   |

- Ensure that the `*.onnx` model file is downloaded in the deploy folder.
  ```bash
  # Model medium and large for Image Size = 640 
  wget https://github.com/karmarv/manacus-dynamics/releases/download/v0.3/v03-y11m-dv6-e25-im640.onnx   -O ./deploy/v03-y11m-dv6-e25-im640.onnx
  wget https://github.com/karmarv/manacus-dynamics/releases/download/v0.3/v03-y11l-dv6-e25-im640.onnx   -O ./deploy/v03-y11l-dv6-e25-im640.onnx
  # Model medium and large for Image Size = 1280
  wget https://github.com/karmarv/manacus-dynamics/releases/download/v0.3/v03-y11m-dv6-e25-im1280.onnx  -O ./deploy/v03-y11m-dv6-e25-im1280.onnx
  wget https://github.com/karmarv/manacus-dynamics/releases/download/v0.3/v03-y11l-dv6-e25-im1280.onnx  -O ./deploy/v03-y11l-dv6-e25-im1280.onnx
  ``` 
  - The `medium model with image size 640` is faster while `large model with image size 1280` is slower.


#### 3. Run inference 

Inference code path: `cd ~/manacus-dynamics/inference/ul`

(a.) Image model inference on a sample image in deploy folder
```bash
time python yolo_infer.py --view-debug --model "./deploy/v03-y11m-dv6-e25-im640.onnx" --out-path "./results/results-y11m-dv6-e25-im640" --image "./deploy/frame_000830.PNG"
## Alternate large model use example
time python yolo_infer.py --view-debug --model "./deploy/v03-y11l-dv6-e25-im1280.onnx" --out-path "./results/results-y11l-dv6-e25-im1280" --image "./deploy/frame_000830.PNG"
```
- expected output at [./results/results-y11l-dv6-e25-im1280/frame_000830.PNG.v03.result.csv](./results/results-y11l-dv6-e25-im1280/frame_000830.PNG.v03.result.csv)
  ```log
  2025-02-01 10:41:35.672658 - Process Image: ./deploy/frame_000830.PNG
  2025-02-01 10:41:51.993074 - 2 results written to ./results/results-y11l-dv6-e25-im1280/frame_000830.PNG.v03.result.csv
  
  real    0m19.439s
  ```

(b.) Video model inference on a frame by frame basis for a sample video in deploy folder
```bash
time python yolo_infer.py --view-debug --model "./deploy/v03-y11m-dv6-e25-im640.onnx" --out-path "./results/results-y11m-dv6-e25-im640" --video "./deploy/LM.P4_1.8.22-1.13.22_0127.MP4"
## Alternate large model use example
time python yolo_infer.py --view-debug --model "./deploy/v03-y11l-dv6-e25-im1280.onnx" --out-path "./results/results-y11l-dv6-e25-im1280" --video "./deploy/LM.P4_1.8.22-1.13.22_0127.MP4"
```
- expected output at [./results/LM.P4_1.8.22-1.13.22_0127.MP4.v02.result.csv](./results/LM.P4_1.8.22-1.13.22_0127.MP4.v02.result.csv)
  ```log
  2025-02-01 10:43:44.572118 - Process Video: ./deploy/LM.P4_1.8.22-1.13.22_0127.MP4
  FPS:60.00, (Frames: 1815),       Video:./deploy/LM.P4_1.8.22-1.13.22_0127.MP4
  100%|██████████████████████████████████████████| 1815/1815 [09:27<00:00,  3.20it/s]
  2025-02-01 10:53:12.709435 - 1238 results written to ./results/results-y11l-dv6-e25-im1280/LM.P4_1.8.22-1.13.22_0127.MP4

  real    9m31.435s
  ```

(c.) Run inference on multiple videos using a bash script
- Prepare video list and run bash script with the respective model weights
```bash
TEST_VIDEOS_PATH="/home/rahul/workspace/vision/manacus-dynamics/dataset/fcat/box/test_videos"
find ${TEST_VIDEOS_PATH} -type f > run_batch_test_videos.list
```
```bash
bash run_batch.bash run_batch_test_videos.list
```
- expect output files in results folder as configured in the bash script
