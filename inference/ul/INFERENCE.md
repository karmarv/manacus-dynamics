## Inference for Manacus Male/Female detection in camera trap videos


#### 1. Create a python environment 

- Python 3.9 Installation instructions via Miniconda - https://docs.conda.io/projects/miniconda/en/latest/
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

#### 2. Configure the inference script
- Ensure that the `*.onnx` file is available locally in the deploy folder.
  ```bash
  wget [TODO]
  ``` 


#### 3. Run inference 
(a.) Image model inference on a sample image in deploy folder
```bash
time python yolo_infer.py --view-debug --image "./deploy/frame_000830.PNG"
```
- expected output at [./results/frame_000830.PNG.v02.result.jpg](./results/frame_000830.PNG.v02.result.jpg)
  ```log
  
  2025-01-17 17:32:27.786762 - Process Image: ./deploy/frame_000830.PNG
  2025-01-17 17:32:28.281782 - 2 results written to ./results/frame_000830.PNG.v02.result.csv

  real    0m2.827s
  user    1m3.455s
  sys     0m1.776s
  ``` 
(b.) Video model inference on a frame by frame basis for a sample video in deploy folder
```bash
cd ~/manacus-dynamics/inference/ul
time python yolo_infer.py --view-debug --video "./deploy/LM.P4_1.8.22-1.13.22_0127.MP4"
```
- expected output at [./results/LM.P4_1.8.22-1.13.22_0127.MP4.v02.result.csv](./results/LM.P4_1.8.22-1.13.22_0127.MP4.v02.result.csv)
  ```log
  2025-01-17 17:42:35.070423 - Process Video: ./deploy/LM.P4_1.8.22-1.13.22_0127.MP4
  FPS:60.00, (Frames: 1815),       Video:./deploy/LM.P4_1.8.22-1.13.22_0127.MP4
  100%|████████████████████████████████████████████████████████| 1815/1815 [09:19<00:00,  3.24it/s]
  2025-01-17 17:51:55.233192 - 1185 results written to ./results/LM.P4_1.8.22-1.13.22_0127.MP4

  real    9m22.585s
  user    956m21.971s
  sys     1m44.975s
  ```
(c.) Run inference on multiple videos using bash script
- Prepare video list and run bash script
```bash
TEST_VIDEOS_PATH="/home/rahul/workspace/vision/manacus-dynamics/dataset/fcat/box/test_videos"
find ${TEST_VIDEOS_PATH} -type f > run_batch_test_videos.list
```
```bash
bash run_batch.bash run_batch_test_videos.list
```
- expect output files in results folder
